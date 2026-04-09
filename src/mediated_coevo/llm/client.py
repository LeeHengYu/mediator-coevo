"""Universal LLM client via litellm.

Adapted from OpenSpace's openspace/llm/client.py — single class handles
any provider (Anthropic, Google, OpenAI) through litellm routing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TypedDict

import litellm
from litellm.types.utils import ModelResponse, Usage

# Suppress litellm debug noise
litellm.set_verbose = False
litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)


class CompletionResult(TypedDict):
    """Typed return value of LLMClient.complete()."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    raw: dict[str, Any]


class LLMClient:
    """Single-round LLM call via litellm. One class, any provider.

    Usage:
        client = LLMClient(model="anthropic/claude-opus-4")
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
                        merged[-1].get("content", "")
                        + "\n\n"
                        + msg.get("content", "")
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
            except asyncio.TimeoutError:
                last_exc = TimeoutError(
                    f"LLM call timed out after {self.timeout}s"
                )
                logger.warning(
                    "Timeout (attempt %d/%d)", attempt + 1, self.max_retries
                )
            except Exception as e:
                last_exc = e
                error_str = str(e).lower()
                retryable = any(
                    kw in error_str
                    for kw in [
                        "rate limit", "429", "overloaded", "500", "502",
                        "503", "504", "connection", "timeout",
                    ]
                )
                if not retryable or attempt >= self.max_retries - 1:
                    raise
                backoff = self.retry_delay * (2 ** attempt)
                logger.warning(
                    "Retryable error (attempt %d/%d), waiting %.1fs: %s",
                    attempt + 1, self.max_retries, backoff, e,
                )
                await asyncio.sleep(backoff)

        raise last_exc  # type: ignore[misc]

    async def complete(
        self,
        messages: list[dict[str, Any]] | str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
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

        await self._rate_limit()

        response = await self._call_with_retry(
            model=kwargs.pop("model", self.model),
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        choice = response.choices[0]
        usage = response.usage or Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        return {
            "content": choice.message.content or "",
            "input_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "completion_tokens", 0),
            "model": self.model,
            "raw": response.model_dump() if hasattr(response, "model_dump") else {},
        }
