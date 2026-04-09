"""Condition 4: Static mediator — fixed heuristic rules, no LLM call."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediated_coevo.conditions import FeedbackCondition

_ERROR_CONTEXT_LINES = 20

if TYPE_CHECKING:
    from mediated_coevo.agents.mediator import MediatorAgent
    from mediated_coevo.models.task import TaskSpec
    from mediated_coevo.models.trace import ExecutionTrace
    from mediated_coevo.stores.artifact_store import ArtifactStore


class StaticMediatorCondition(FeedbackCondition):
    """Fixed rules: always send last error summary to Planner.

    No LLM call — deterministic string extraction. Serves as a
    baseline for whether a simple heuristic can match a learned Mediator.
    """

    @property
    def name(self) -> str:
        return "static_mediator"

    async def produce_feedback(
        self,
        trace: ExecutionTrace,
        task_context: TaskSpec,
        artifact_store: ArtifactStore,
        mediator: MediatorAgent | None,
    ) -> str | None:
        parts: list[str] = [f"reward: {trace.reward:.2f}"]

        if trace.stderr:
            # Extract last error block
            lines = trace.stderr.strip().split("\n")
            last_error = "\n".join(lines[-_ERROR_CONTEXT_LINES:])
            parts.append(f"Last error:\n{last_error}")

        if trace.test_results:
            parts.append(f"Test results: {trace.test_results}")

        return "\n".join(parts)

    def supports_coevolution(self) -> bool:
        return False
