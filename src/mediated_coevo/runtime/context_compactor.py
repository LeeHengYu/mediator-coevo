"""Prompt-context compaction helpers."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from mediated_coevo.prompt_text import PromptText
from mediated_coevo.runtime.token_budget import TokenBudgetExceeded, count_text_tokens

if TYPE_CHECKING:
    from mediated_coevo.llm.client import LLMClient
    from mediated_coevo.models.trace import ExecutionTrace

logger = logging.getLogger(__name__)

RAW_PASSTHROUGH_CHARS = 800
TARGET_EVIDENCE_CHARS = 700
TARGET_HEADLINE_CHARS = 160
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
        except Exception as error:
            last_error = error
            logger.warning(
                "Context compaction attempt %s/%s failed for %s: %s",
                attempt,
                CONTEXT_COMPACTOR_ATTEMPTS,
                label,
                error,
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
    """Format the leading iteration, reward, and status trace summary."""
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
    """Return a bounded head-and-tail excerpt with a gap marker."""
    if len(text) <= budget:
        return text
    half = max(1, (budget - 10) // 2)
    return f"{text[:half].rstrip()}\n…\n{text[-half:].lstrip()}"
