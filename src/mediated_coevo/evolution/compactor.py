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
from mediated_coevo.prompt_text import PromptText
from mediated_coevo.runtime.token_budget import TokenBudgetExceeded, count_text_tokens

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


COMPACTOR_SYSTEM_PROMPT = PromptText.COMPACTOR_SYSTEM
CONTEXT_COMPACTOR_SYSTEM_PROMPT = PromptText.CONTEXT_COMPACTOR_SYSTEM
CONTEXT_COMPACTOR_ATTEMPTS = 3


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
    """Compact long prompt context without hard truncation fallbacks."""
    raw = text.strip()
    if len(raw) <= RAW_PASSTHROUGH_CHARS and (
        budget_tokens is None or count_text_tokens(model, raw) <= budget_tokens
    ):
        return raw

    if llm_client is None:
        if budget_tokens is not None:
            raise TokenBudgetExceeded(
                f"{label} requires compaction but no LLM client is configured"
            )
        return raw

    from mediated_coevo.core.utils import parse_json_object

    last_error: Exception | None = None
    last_compacted = ""
    for attempt in range(1, CONTEXT_COMPACTOR_ATTEMPTS + 1):
        try:
            response = await llm_client.complete(
                messages=[
                    {"role": "system", "content": PromptText.CONTEXT_COMPACTOR_SYSTEM},
                    {
                        "role": "user",
                        "content": PromptText.context_compactor_user(
                            label=label,
                            raw_length=len(raw),
                            prompt_raw=raw,
                            target_evidence_chars=TARGET_EVIDENCE_CHARS,
                            target_headline_chars=TARGET_HEADLINE_CHARS,
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=completion_tokens,
                budget_label="compactor.context",
                budget_overflow_strategy="none",
                condition_name=condition_name,
            )
            parsed = parse_json_object(str(response.get("content", "")))
            headline = str(parsed.get("headline", "")).strip()
            evidence = str(parsed.get("evidence", "")).strip()
            compacted = "\n".join(part for part in [headline, evidence] if part)
            if not compacted:
                raise ValueError("compactor returned empty content")
            last_compacted = compacted
            compacted_tokens = count_text_tokens(model, compacted)
            if budget_tokens is None or compacted_tokens <= budget_tokens:
                return compacted
            last_error = TokenBudgetExceeded(
                f"{label} compaction attempt {attempt} used "
                f"{compacted_tokens} tokens, budget={budget_tokens}"
            )
        except Exception as e:
            last_error = e
            logger.warning(
                "Context compaction attempt %s/%s failed for %s: %s",
                attempt,
                CONTEXT_COMPACTOR_ATTEMPTS,
                label,
                e,
            )

    if last_compacted:
        logger.warning(
            "Context compaction for %s remained over budget after %s attempts; "
            "returning the last compacted output without fallback truncation.",
            label,
            CONTEXT_COMPACTOR_ATTEMPTS,
        )
        return last_compacted

    raise TokenBudgetExceeded(
        f"{label} compaction failed after {CONTEXT_COMPACTOR_ATTEMPTS} attempts"
    ) from last_error


def deterministic_mediator_signal(
    report: MediatorReport,
) -> MediatorSignal:
    feedback = report.exposed_content or ""
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
        withheld=report.withheld,
        mediator_reasoning=report.reasoning,
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
    trace: ExecutionTrace,
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


def abstraction_level_str(report: MediatorReport) -> str:
    """Return the report's abstraction_level as a string."""
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
        excerpt = (
            "".join(diff_lines[:head_lines]) + gap + "".join(diff_lines[-tail_lines:])
        )
    return added, removed, excerpt
