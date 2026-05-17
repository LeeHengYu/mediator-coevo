"""LLM client via LiteLLM/OpenRouter routing.

Adapted from OpenSpace's openspace/llm/client.py — single class handles
chat completions through LiteLLM. This project supports OpenRouter as the
model credential boundary.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Mapping
from typing import Any, Protocol, TypedDict, cast

import litellm
from litellm.types.utils import ModelResponse

from mediated_coevo.runtime.token_budget import (
    TokenBudgetEvent,
    validate_messages_fit,
)

# Suppress litellm debug noise
litellm.set_verbose = False
litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


class CompletionResult(TypedDict):
    """Typed return value of LLMClient.complete()."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    raw: dict[str, Any]


class LLMUsageError(RuntimeError):
    """Raised when a LiteLLM response does not include required usage data."""


class LLMCredentialError(RuntimeError):
    """Raised when required OpenRouter model credentials are unavailable."""


def validate_openrouter_credentials(
    environ: Mapping[str, str] | None = None,
) -> None:
    """Fail clearly when the only supported model credential is missing."""
    source = os.environ if environ is None else environ
    api_key = source.get(OPENROUTER_API_KEY_ENV, "")
    if not api_key.strip():
        raise LLMCredentialError(
            f"{OPENROUTER_API_KEY_ENV} is required for all model calls. "
            f"Export {OPENROUTER_API_KEY_ENV} before running experiments."
        )


class _UsageCounts(Protocol):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _UsageResponse(Protocol):
    usage: _UsageCounts | None


class LLMClient:
    """Single-round LLM call via LiteLLM/OpenRouter.

    Usage:
        client = LLMClient(model="openrouter/anthropic/claude-opus-4")
        result = await client.complete("Hello, world!")
        print(result["content"])
    """

    def __init__(
        self,
        model: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 120.0,
        rate_limit_delay: float = 0.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_call_time = 0.0
        self._token_events: list[TokenBudgetEvent] = []

    # ── Message normalization (from OpenSpace) ──

    @staticmethod
    def _merge_consecutive_system_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge consecutive system messages into one.

        Some providers (e.g. MiniMax) reject requests with multiple
        consecutive same-role messages. Merging is safe for all providers.
        """
        if not messages:
            return messages
        merged: list[dict[str, Any]] = []
        for msg in messages:
            if (
                merged
                and msg.get("role") == "system"
                and merged[-1].get("role") == "system"
            ):
                merged[-1] = {
                    "role": "system",
                    "content": (
                        merged[-1].get("content", "") + "\n\n" + msg.get("content", "")
                    ),
                }
            else:
                merged.append(msg.copy())
        return merged

    async def _rate_limit(self) -> None:
        if self.rate_limit_delay > 0:
            elapsed = time.time() - self._last_call_time
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
            self._last_call_time = time.time()

    async def _call_with_retry(self, **kwargs: Any) -> ModelResponse:
        """Call litellm.acompletion with exponential backoff on retryable errors."""
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return await asyncio.wait_for(
                    litellm.acompletion(**kwargs),
                    timeout=self.timeout,
                )
            except TimeoutError:
                last_exc = TimeoutError(f"LLM call timed out after {self.timeout}s")
                logger.warning("Timeout (attempt %d/%d)", attempt + 1, self.max_retries)
            except Exception as e:
                last_exc = e
                error_str = str(e).lower()
                retryable = any(
                    kw in error_str
                    for kw in [
                        "rate limit",
                        "429",
                        "overloaded",
                        "500",
                        "502",
                        "503",
                        "504",
                        "connection",
                        "timeout",
                    ]
                )
                if not retryable or attempt >= self.max_retries - 1:
                    raise
                backoff = self.retry_delay * (2**attempt)
                logger.warning(
                    "Retryable error (attempt %d/%d), waiting %.1fs: %s",
                    attempt + 1,
                    self.max_retries,
                    backoff,
                    e,
                )
                await asyncio.sleep(backoff)

        raise last_exc  # type: ignore[misc]

    async def complete(
        self,
        messages: list[dict[str, Any]] | str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        budget_label: str | None = None,
        prompt_budget: int | None = None,
        budget_overflow_strategy: str = "none",
        condition_name: str | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Single-round LLM completion.

        Args:
            messages: OpenAI-format message list, or a plain string
                      (auto-wrapped as a user message).
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            {
                "content": str,
                "input_tokens": int,
                "output_tokens": int,
                "model": str,
                "raw": dict,
            }
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        messages = self._merge_consecutive_system_messages(messages)
        model = kwargs.pop("model", self.model)
        label = budget_label or "llm.complete"
        if prompt_budget is not None:
            validate_messages_fit(
                model=model,
                messages=messages,
                budget_limit=prompt_budget,
                label=label,
            )

        await self._rate_limit()

        response = await self._call_with_retry(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        choice = response.choices[0]
        usage = cast(_UsageResponse, response).usage
        if usage is None:
            raise LLMUsageError("LiteLLM response did not include usage accounting")
        input_tokens = int(usage.prompt_tokens)
        output_tokens = int(usage.completion_tokens)
        total_tokens = int(usage.total_tokens)
        if budget_label or prompt_budget is not None:
            self._token_events.append(
                TokenBudgetEvent(
                    label=label,
                    model=model,
                    condition_name=condition_name,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=total_tokens,
                    budget_limit=prompt_budget or 0,
                    budget_overflow_strategy=budget_overflow_strategy,
                )
            )

        return {
            "content": choice.message.content or "",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": self.model,
            "raw": response.model_dump(),
        }

    def drain_token_events(self) -> list[TokenBudgetEvent]:
        """Return and clear accumulated token budget telemetry."""
        events = list(self._token_events)
        self._token_events.clear()
        return events
