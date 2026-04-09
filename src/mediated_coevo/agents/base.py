"""Base agent class.

Adapted from OpenSpace's openspace/agents/base.py — provides the core
contract (process + construct_messages) that all agents implement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from mediated_coevo.llm.client import CompletionResult, LLMClient

logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    """Structured response from any agent LLM call."""

    content: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class BaseAgent(ABC):
    """Base class for all agents in the mediated co-evolution system.

    Each agent wraps an LLMClient and exposes two abstract methods:
      - construct_messages(): build provider-agnostic message list from context
      - process(): main entry point — plan, execute, or mediate
    """

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

    @abstractmethod
    def construct_messages(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Build the LLM message list from context.

        Each agent constructs its own system prompt + user message(s).
        """
        ...

    @abstractmethod
    async def process(self, context: dict[str, Any]) -> dict[str, Any]:
        """Main entry point. Takes context dict, returns structured result.

        Typical flow: construct_messages() → get_llm_response() → parse.
        """
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
