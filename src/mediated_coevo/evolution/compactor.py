"""Build compact, structured payloads for HistoryEntry.

The Reflector feeds on ``HistoryEntry.payload`` (``MediatorSignal`` /
``PlannerSignal``), not on raw blobs.

Two entry points for mediator signals:

- ``deterministic_mediator_signal()`` — a free function that builds a
  ``MediatorSignal`` with **no LLM call**. Used for conditions that do
  not have a ``MediatorAgent`` (conditions 2-4), and as the short-path
  fallback inside ``MediatorAgent.compact_feedback``.
- ``MediatorAgent.compact_feedback()`` (in ``agents/mediator.py``) —
  uses the mediator's own LLM client for a one-shot compaction call on
  long feedback, and delegates to ``deterministic_mediator_signal`` on
  the short path or on LLM failure.

Planner signals are always built deterministically here — there is no
LLM call involved in turning a ``SkillUpdate`` into a ``PlannerSignal``.

The helpers below (``first_sentence``, ``head_tail_text``,
``abstraction_level_str``) are shared between this module and
``MediatorAgent.compact_feedback``.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import TYPE_CHECKING

from mediated_coevo.models.history_signals import MediatorSignal, PlannerSignal
from mediated_coevo.token_budget import count_text_tokens, fit_text_to_tokens

if TYPE_CHECKING:
    from mediated_coevo.llm.client import LLMClient
    from mediated_coevo.models.report import MediatorReport
    from mediated_coevo.models.skill import SkillUpdate
    from mediated_coevo.models.trace import ExecutionTrace

logger = logging.getLogger(__name__)


# Feedback text shorter than this is kept verbatim — an LLM call would be wasted.
RAW_PASSTHROUGH_CHARS = 800

# Target compacted evidence length (LLM hint + fallback cap).
TARGET_EVIDENCE_CHARS = 700

# Target compacted headline length (LLM hint + fallback cap).
TARGET_HEADLINE_CHARS = 160

# Head + tail lines kept when excerpting a unified diff.
DIFF_EXCERPT_HEAD_LINES = 12
DIFF_EXCERPT_TAIL_LINES = 12


COMPACTOR_SYSTEM_PROMPT = """\
You are a log compactor. Your job is to condense a long mediator report
into a structured JSON object with exactly two fields:

- "headline": ONE sentence capturing the key observation or decision
  the mediator is communicating.
- "evidence": 2-4 sentences of the most diagnostic text from the report,
  quoted verbatim where possible. Prefer concrete error messages,
  failing assertions, or specific recommendations over generic framing.

Respond with ONLY a JSON object — no prose, no markdown fences."""


CONTEXT_COMPACTOR_SYSTEM_PROMPT = """\
You are a log compactor. Condense long execution context for a planner prompt.

Return JSON with exactly two string fields:
- "headline": ONE sentence naming the most important signal.
- "evidence": 2-4 concise sentences preserving concrete error messages,
  failing assertions, command names, paths, or verifier details where relevant.

Respond with ONLY a JSON object — no prose, no markdown fences."""


async def compact_text_for_context(
    text: str,
    *,
    llm_client: LLMClient | None = None,
    label: str = "context",
    model: str,
    budget_tokens: int | None = None,
    completion_tokens: int = 600,
    condition_name: str | None = None,
) -> str:
    """Compact long prompt context with the existing compactor fallback rules."""
    raw = text.strip()
    if len(raw) <= RAW_PASSTHROUGH_CHARS and (
        budget_tokens is None or count_text_tokens(model, raw) <= budget_tokens
    ):
        return raw

    if llm_client is None:
        if budget_tokens is not None:
            excerpt = head_tail_text(raw, TARGET_EVIDENCE_CHARS)
            return fit_text_to_tokens(model, excerpt, budget_tokens)
        return head_tail_text(raw, TARGET_EVIDENCE_CHARS)

    try:
        from mediated_coevo.utils import parse_json_object

        prompt_raw = raw
        prompt_budget = None
        if budget_tokens is not None:
            prompt_raw = fit_text_to_tokens(llm_client.model, raw, budget_tokens)
            prompt_budget = max(1, budget_tokens + 500)

        response = await llm_client.complete(
            messages=[
                {"role": "system", "content": CONTEXT_COMPACTOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"## {label} ({len(raw)} chars)\n\n"
                        f"{prompt_raw}\n\n"
                        f"Keep evidence to about {TARGET_EVIDENCE_CHARS} "
                        f"characters and headline to about {TARGET_HEADLINE_CHARS}."
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=completion_tokens,
            budget_label="compactor.context",
            prompt_budget=prompt_budget,
            budget_overflow_strategy="head_tail",
            condition_name=condition_name,
        )
        parsed = parse_json_object(str(response.get("content", "")))
        headline = str(parsed.get("headline", "")).strip()
        evidence = str(parsed.get("evidence", "")).strip()
        compacted = "\n".join(part for part in [headline, evidence] if part)
        if compacted:
            if budget_tokens is not None:
                return fit_text_to_tokens(model, compacted, budget_tokens)
            return compacted
    except Exception as e:
        logger.warning(
            "Context compaction LLM call failed for %s (%s); using fallback excerpt.",
            label,
            e,
        )

    if budget_tokens is not None:
        excerpt = head_tail_text(raw, TARGET_EVIDENCE_CHARS)
        return fit_text_to_tokens(model, excerpt, budget_tokens)
    return head_tail_text(raw, TARGET_EVIDENCE_CHARS)


def deterministic_mediator_signal(
    feedback: str,
    report: "MediatorReport | None" = None,
) -> MediatorSignal:
    raw_length = len(feedback)
    evidence = (
        feedback
        if raw_length <= RAW_PASSTHROUGH_CHARS
        else head_tail_text(feedback, TARGET_EVIDENCE_CHARS)
    )
    return MediatorSignal(
        headline=first_sentence(feedback, TARGET_HEADLINE_CHARS),
        evidence=evidence,
        abstraction_level=abstraction_level_str(report),
        withheld=report.withheld if report else False,
        mediator_reasoning=report.reasoning if report else "",
        raw_length=raw_length,
    )


def build_planner_signal(update: SkillUpdate) -> PlannerSignal:
    added, removed, excerpt = _diff_parts(update.old_content, update.new_content)
    return PlannerSignal(
        reasoning=update.reasoning,
        lines_added=added,
        lines_removed=removed,
        diff_excerpt=excerpt,
    )

def first_sentence(text: str, max_chars: int) -> str:
    """Return the first sentence or line of text, bounded by max_chars."""
    stripped = text.strip()
    if not stripped:
        return ""
    match = re.search(r"[.!?\n]", stripped)
    cut = match.start() + 1 if match else len(stripped)
    sentence = stripped[:cut].strip()
    if len(sentence) > max_chars:
        sentence = sentence[: max_chars - 1].rstrip() + "…"
    return sentence


def trace_header_summary(
    trace: "ExecutionTrace",
    *,
    include_source_task: bool = False,
) -> str:
    """Format the leading 'iter=X reward=Y STATUS' prefix for a trace summary."""
    if trace.status != "ok":
        status = f"{trace.status.upper()}({trace.error_kind or 'unknown'})"
    elif trace.exit_code == 0:
        status = "OK"
    else:
        status = f"FAIL(exit={trace.exit_code})"
    reward = f"{trace.reward:.2f}" if trace.reward is not None else "n/a"
    prefix = f"source_task={trace.task_id} " if include_source_task else ""
    return f"{prefix}iter={trace.iteration} reward={reward} {status}"


def head_tail_text(text: str, budget: int) -> str:
    """Head + tail excerpt with a gap marker (fallback for evidence)."""
    if len(text) <= budget:
        return text
    half = max(1, (budget - 10) // 2)
    return f"{text[:half].rstrip()}\n…\n{text[-half:].lstrip()}"


def abstraction_level_str(report: "MediatorReport | None") -> str:
    """Return the report's abstraction_level as a string, or ''."""
    if report is None:
        return ""
    return report.abstraction_level.value


# ── Planner diff helper (internal) ──────────────────────────────────────

def _diff_parts(old: str, new: str) -> tuple[int, int, str]:
    """Compute a unified diff, return (added, removed, head+tail excerpt)."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            old_lines, new_lines, fromfile="before", tofile="after", n=2
        )
    )
    added = sum(
        1 for ln in diff_lines if ln.startswith("+") and not ln.startswith("+++")
    )
    removed = sum(
        1 for ln in diff_lines if ln.startswith("-") and not ln.startswith("---")
    )

    head_lines, tail_lines = DIFF_EXCERPT_HEAD_LINES, DIFF_EXCERPT_TAIL_LINES
    if len(diff_lines) <= head_lines + tail_lines:
        excerpt = "".join(diff_lines)
    else:
        gap = f"... ({len(diff_lines) - head_lines - tail_lines} more diff lines) ...\n"
        excerpt = "".join(diff_lines[:head_lines]) + gap + "".join(diff_lines[-tail_lines:])
    return added, removed, excerpt
