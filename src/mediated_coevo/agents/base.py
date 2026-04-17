"""Base agent class.

Provides shared infrastructure (LLM client, step counter, JSON parsing)
for the three role-specific agents. Each subclass defines its own entry
point — there is no uniform `process` contract across roles.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from mediated_coevo.llm.client import CompletionResult, LLMClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in the mediated co-evolution system."""

    def __init__(self, name: str, llm_client: LLMClient) -> None:
        self._name = name
        self._llm_client = llm_client
        self._step = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def llm_client(self) -> LLMClient:
        return self._llm_client

    @property
    def step(self) -> int:
        return self._step

    @property
    @abstractmethod
    def role(self) -> str:
        """One of 'planner', 'executor', 'mediator'."""
        ...

    async def get_llm_response(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CompletionResult:
        """Convenience: delegate to the underlying LLMClient."""
        return await self._llm_client.complete(messages=messages, **kwargs)

    def increment_step(self) -> None:
        self._step += 1

    def response_to_dict(self, response_text: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown fences."""
        from mediated_coevo.utils import parse_json_object

        result = parse_json_object(response_text)
        if not result:
            logger.error("%s: Failed to parse response as JSON", self.name)
            return {"error": "Failed to parse response", "raw": response_text}
        return result

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, step={self.step})>"
