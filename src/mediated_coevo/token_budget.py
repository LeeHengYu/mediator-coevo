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
        candidate = f"{text[:mid].rstrip()}{marker}{text[-mid:].lstrip()}" if mid else marker
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
        content = section.content.strip()
        if section.max_tokens is not None:
            content = fit_text_to_tokens(model, content, section.max_tokens)
        packed.append((section, content))

    required_text = separator.join(content for section, content in packed if section.required)
    required_tokens = count_text_tokens(model, required_text)
    if required_tokens > budget_limit:
        raise TokenBudgetExceeded(
            f"Required prompt sections use {required_tokens} tokens, budget={budget_limit}"
        )

    selected: list[str] = []
    for section, content in packed:
        if not content:
            continue
        candidate = separator.join([*selected, content])
        if count_text_tokens(model, candidate) <= budget_limit:
            selected.append(content)
            continue
        if section.required:
            raise TokenBudgetExceeded(
                f"Required section {section.name!r} exceeds remaining budget"
            )

        remaining = _remaining_text_budget(model, selected, budget_limit, separator)
        if remaining <= 0:
            continue
        truncated = fit_text_to_tokens(model, content, remaining)
        if truncated:
            candidate = separator.join([*selected, truncated])
            if count_text_tokens(model, candidate) <= budget_limit:
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
