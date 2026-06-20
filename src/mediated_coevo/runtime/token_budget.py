"""Token counting, packing, and budget telemetry helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

OverflowStrategy = Literal["none", "head_tail", "drop_oldest", "section_pack"]


class TokenBudgetExceeded(ValueError):
    """Raised when required prompt content cannot fit a configured budget."""


class TokenCountingError(RuntimeError):
    """Raised when LiteLLM cannot provide an exact token count."""


class TokenBudgetEvent(BaseModel):
    """Serialized token budget usage for one repo-controlled LLM call."""

    label: str
    model: str
    condition_name: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    budget_limit: int = 0
    budget_overflow_strategy: str = "none"


@dataclass(frozen=True)
class BudgetSection:
    """A named prompt section with an optional per-section cap."""

    name: str
    content: str
    required: bool = False
    max_tokens: int | None = None
    overflow_strategy: OverflowStrategy = "head_tail"


def count_text_tokens(model: str, text: str) -> int:
    """Count text tokens using LiteLLM, falling back to a local tokenizer."""
    if not text:
        return 0
    if not model:
        raise TokenCountingError("A model name is required for token counting")
    try:
        import litellm

        return int(litellm.token_counter(model=model, text=text))
    except Exception as litellm_error:
        logger.debug(
            "LiteLLM token counting failed for model=%r text; using tokenizer fallback",
            model,
            exc_info=litellm_error,
        )
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception as tokenizer_error:
        raise TokenCountingError(
            f"Token counting failed for model={model!r} text"
        ) from tokenizer_error


def count_message_tokens(model: str, messages: list[dict[str, Any]]) -> int:
    """Count chat message tokens using LiteLLM, falling back to a local tokenizer."""
    if not messages:
        return 0
    if not model:
        raise TokenCountingError("A model name is required for token counting")
    try:
        import litellm

        return int(litellm.token_counter(model=model, messages=messages))
    except Exception as litellm_error:
        logger.debug(
            "LiteLLM token counting failed for model=%r messages; using tokenizer fallback",
            model,
            exc_info=litellm_error,
        )
    try:
        return _count_messages_with_tiktoken(model, messages)
    except Exception as tokenizer_error:
        raise TokenCountingError(
            f"Token counting failed for model={model!r} messages"
        ) from tokenizer_error


def _count_messages_with_tiktoken(model: str, messages: list[dict[str, Any]]) -> int:
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    total = 3
    for message in messages:
        total += 4
        for key, value in message.items():
            if value is None:
                continue
            content = _message_value_to_text(value)
            if key not in {"role", "content", "name"}:
                content = f"{key}: {content}"
            total += len(encoding.encode(content))
            if key == "name":
                total += 1
    return total


def _message_value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def fit_text_to_tokens(
    model: str,
    text: str,
    max_tokens: int,
    *,
    marker: str = "\n...\n",
) -> str:
    """Return text that fits max_tokens, preserving head and tail."""
    if max_tokens <= 0 or not text:
        return ""
    if count_text_tokens(model, text) <= max_tokens:
        return text
    if count_text_tokens(model, marker) > max_tokens:
        return ""

    lo = 0
    hi = max(1, len(text) // 2)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = (
            f"{text[:mid].rstrip()}{marker}{text[-mid:].lstrip()}" if mid else marker
        )
        if count_text_tokens(model, candidate) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1

    if best:
        return best

    # Tiny budget fallback: keep a prefix that fits.
    lo = 0
    hi = len(text)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip()
        if count_text_tokens(model, candidate) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def pack_sections(
    model: str,
    sections: list[BudgetSection],
    budget_limit: int,
    *,
    separator: str = "\n\n",
) -> str:
    """Pack prompt sections deterministically within a total token budget."""
    if budget_limit <= 0:
        raise TokenBudgetExceeded("Prompt budget must be positive")

    packed: list[tuple[BudgetSection, str]] = []
    for section in sections:
        raw_content = section.content.strip()
        content = raw_content
        if section.max_tokens is not None:
            content = _fit_text_to_tokens_with_strategy(
                model,
                content,
                section.max_tokens,
                section.overflow_strategy,
            )
            if section.required and raw_content and not content:
                raise TokenBudgetExceeded(
                    f"Required section {section.name!r} cannot fit "
                    f"max_tokens={section.max_tokens} with "
                    f"overflow_strategy={section.overflow_strategy!r}"
                )
        packed.append((section, content))

    required_text = separator.join(
        content for section, content in packed if section.required
    )
    required_tokens = count_text_tokens(model, required_text)
    if required_tokens > budget_limit:
        raise TokenBudgetExceeded(
            f"Required prompt sections use {required_tokens} tokens, budget={budget_limit}"
        )

    selected: list[str] = []
    for index, (section, content) in enumerate(packed):
        if not content:
            continue
        candidate = separator.join([*selected, content])
        if count_text_tokens(model, candidate) <= budget_limit:
            if section.required or _optional_candidate_fits(
                model,
                selected,
                content,
                packed[index + 1 :],
                budget_limit,
                separator,
            ):
                selected.append(content)
                continue
        if section.required:
            raise TokenBudgetExceeded(
                f"Required section {section.name!r} exceeds remaining budget"
            )

        truncated = _fit_optional_content_to_remaining_budget(
            model,
            selected,
            content,
            packed[index + 1 :],
            budget_limit,
            separator,
            section.overflow_strategy,
        )
        if truncated:
            selected.append(truncated)

    return separator.join(selected)


def validate_messages_fit(
    *,
    model: str,
    messages: list[dict[str, Any]],
    budget_limit: int,
    label: str,
) -> int:
    """Return prompt tokens or raise if messages exceed budget."""
    prompt_tokens = count_message_tokens(model, messages)
    if prompt_tokens > budget_limit:
        raise TokenBudgetExceeded(
            f"{label} prompt has {prompt_tokens} tokens, budget={budget_limit}"
        )
    return prompt_tokens


def _remaining_text_budget(
    model: str,
    selected: list[str],
    budget_limit: int,
    separator: str,
) -> int:
    used = count_text_tokens(model, separator.join(selected)) if selected else 0
    sep_tokens = count_text_tokens(model, separator) if selected else 0
    return max(0, budget_limit - used - sep_tokens)


def _fit_text_to_tokens_with_strategy(
    model: str,
    text: str,
    max_tokens: int,
    strategy: OverflowStrategy,
) -> str:
    if max_tokens <= 0 or not text:
        return ""
    if count_text_tokens(model, text) <= max_tokens:
        return text
    if strategy == "head_tail":
        return fit_text_to_tokens(model, text, max_tokens)
    if strategy == "section_pack":
        return _fit_section_units_to_tokens(model, text, max_tokens)
    if strategy == "drop_oldest":
        return _fit_drop_oldest_text_to_tokens(model, text, max_tokens)
    if strategy == "none":
        return ""
    return fit_text_to_tokens(model, text, max_tokens)


def _fit_drop_oldest_text_to_tokens(model: str, text: str, max_tokens: int) -> str:
    """Drop oldest line-level units first, preserving the newest suffix."""
    units = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(units) > 1:
        selected: list[str] = []
        for unit in reversed(units):
            candidate = "\n".join([unit, *selected])
            if count_text_tokens(model, candidate) > max_tokens:
                if not selected:
                    return _fit_suffix_to_tokens(model, unit, max_tokens)
                break
            selected.insert(0, unit)
        return "\n".join(selected)

    return _fit_suffix_to_tokens(model, text, max_tokens)


def _fit_prefix_after_units(
    model: str,
    selected: list[str],
    unit: str,
    max_tokens: int,
) -> str:
    lo = 0
    hi = len(unit)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        prefix = unit[:mid].rstrip()
        candidate = "\n".join([*selected, prefix]) if prefix else "\n".join(selected)
        if count_text_tokens(model, candidate) <= max_tokens:
            best = prefix
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _fit_suffix_to_tokens(model: str, text: str, max_tokens: int) -> str:
    lo = 0
    hi = len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[-mid:].lstrip() if mid else ""
        if count_text_tokens(model, candidate) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _fit_section_units_to_tokens(model: str, text: str, max_tokens: int) -> str:
    """Pack complete line-level units from the front of a section."""
    units = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not units:
        return ""

    selected: list[str] = []
    for unit in units:
        candidate = "\n".join([*selected, unit])
        if count_text_tokens(model, candidate) > max_tokens:
            prefix = _fit_prefix_after_units(model, selected, unit, max_tokens)
            if prefix:
                selected.append(prefix)
            break
        selected.append(unit)
    return "\n".join(selected)


def _fit_optional_content_to_remaining_budget(
    model: str,
    selected: list[str],
    content: str,
    remaining_packed: list[tuple[BudgetSection, str]],
    budget_limit: int,
    separator: str,
    strategy: OverflowStrategy,
) -> str:
    remaining = _remaining_text_budget_for_optional(
        model,
        selected,
        remaining_packed,
        budget_limit,
        separator,
    )
    if remaining <= 0:
        return ""

    lo = 0
    hi = remaining
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = _fit_text_to_tokens_with_strategy(model, content, mid, strategy)
        if not candidate:
            lo = mid + 1
            continue
        if _optional_candidate_fits(
            model,
            selected,
            candidate,
            remaining_packed,
            budget_limit,
            separator,
        ):
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _optional_candidate_fits(
    model: str,
    selected: list[str],
    candidate_content: str,
    remaining_packed: list[tuple[BudgetSection, str]],
    budget_limit: int,
    separator: str,
) -> bool:
    later_required = [
        content for section, content in remaining_packed if section.required and content
    ]
    candidate = separator.join([*selected, candidate_content, *later_required])
    return count_text_tokens(model, candidate) <= budget_limit


def _remaining_text_budget_for_optional(
    model: str,
    selected: list[str],
    remaining_packed: list[tuple[BudgetSection, str]],
    budget_limit: int,
    separator: str,
) -> int:
    later_required = [
        content for section, content in remaining_packed if section.required and content
    ]
    if not later_required:
        return _remaining_text_budget(model, selected, budget_limit, separator)

    required_baseline = separator.join([*selected, *later_required])
    reserved_tokens = count_text_tokens(model, required_baseline)
    if reserved_tokens >= budget_limit:
        return 0
    sep_tokens = count_text_tokens(model, separator)
    return max(0, budget_limit - reserved_tokens - sep_tokens)
