"""Executor agent — Gemini.

Runs tasks in a sandboxed environment (Harbor) using current skills.
Produces execution traces with stdout, stderr, test results, and reward.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from .base import BaseAgent

if TYPE_CHECKING:
    from mediated_coevo.llm.client import LLMClient
    from mediated_coevo.models.task import TaskSpec
    from mediated_coevo.models.trace import ExecutionTrace

logger = logging.getLogger(__name__)

EXECUTOR_SYSTEM_PROMPT = """\
You are the Executor in a multi-agent skill co-evolution system.

Your responsibilities:
1. Execute the task exactly as instructed by the Planner.
2. Follow any skill guidelines provided to you.
3. Report your results, including any errors encountered.

Execute the task thoroughly and report all outputs."""


class ExecutorAgent(BaseAgent):
    """Gemini-backed executor. Runs tasks in Harbor sandbox."""

    @property
    def role(self) -> str:
        return "executor"

    def __init__(
        self,
        llm_client: LLMClient,
        sandbox_config: dict | None = None,
    ) -> None:
        super().__init__("executor", llm_client)
        self._sandbox_config = sandbox_config or {}

    def construct_messages(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        ]

        if skills := context.get("skills"):
            messages.append({"role": "system", "content": (
                "# Active Skills\n\n"
                "The following skills provide **verified procedures** for "
                "this task. Follow step-by-step instructions where given.\n\n"
                + "\n\n---\n\n".join(skills)
            )})

        messages.append({"role": "user", "content": context.get("instruction", "")})
        return messages

    async def process(self, context: dict[str, Any]) -> dict[str, Any]:
        messages = self.construct_messages(context)
        response = await self.get_llm_response(messages)
        self.increment_step()
        return {
            "content": response["content"],
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
        }

    async def execute_task(
        self,
        task_spec: TaskSpec,
        skills: list[str],
    ) -> ExecutionTrace:
        """Execute a task and return an ExecutionTrace.

        In the full system, this submits the task to a Harbor sandbox
        and parses result.json. For now, it calls the Gemini LLM
        directly and wraps the output as a trace.
        """
        from mediated_coevo.models.trace import ExecutionTrace, TokenUsage

        start = time.time()
        context = {
            "instruction": task_spec.instruction,
            "skills": skills,
        }
        result = await self.process(context)
        duration = time.time() - start

        # TODO: When Harbor integration is ready, replace this with
        # actual subprocess execution + result.json parsing.
        # For now, the LLM response simulates execution output.
        content = result["content"]

        return ExecutionTrace(
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            stdout=content,
            stderr="",
            exit_code=0,
            test_results=None,
            reward=0.0,  # Placeholder — real reward from test results
            token_usage=TokenUsage(
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
            ),
            duration_sec=duration,
        )
