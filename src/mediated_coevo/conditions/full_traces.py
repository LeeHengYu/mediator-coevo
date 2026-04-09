"""Condition 2: Full traces — Planner sees all raw execution output (context bloat)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediated_coevo.conditions import FeedbackCondition

if TYPE_CHECKING:
    from mediated_coevo.agents.mediator import MediatorAgent
    from mediated_coevo.models.task import TaskSpec
    from mediated_coevo.models.trace import ExecutionTrace
    from mediated_coevo.stores.artifact_store import ArtifactStore


class FullTracesCondition(FeedbackCondition):
    @property
    def name(self) -> str:
        return "full_traces"

    async def produce_feedback(
        self,
        trace: ExecutionTrace,
        task_context: TaskSpec,
        artifact_store: ArtifactStore,
        mediator: MediatorAgent | None,
    ) -> str | None:
        parts: list[str] = []
        if trace.stdout:
            parts.append(f"## stdout\n{trace.stdout}")
        if trace.stderr:
            parts.append(f"## stderr\n{trace.stderr}")
        if trace.test_results:
            parts.append(f"## test_results\n{trace.test_results}")
        parts.append(f"## reward: {trace.reward}")
        return "\n\n".join(parts) if parts else None

    def supports_coevolution(self) -> bool:
        return False
