"""Condition 1: No feedback — Planner updates skills blind."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediated_coevo.conditions import FeedbackCondition

if TYPE_CHECKING:
    from mediated_coevo.agents.mediator import MediatorAgent
    from mediated_coevo.models.task import TaskSpec
    from mediated_coevo.models.trace import ExecutionTrace
    from mediated_coevo.stores.artifact_store import ArtifactStore


class NoFeedbackCondition(FeedbackCondition):
    @property
    def name(self) -> str:
        return "no_feedback"

    async def produce_feedback(
        self,
        trace: ExecutionTrace,
        task_context: TaskSpec,
        artifact_store: ArtifactStore,
        mediator: MediatorAgent | None,
    ) -> str | None:
        return None

    def supports_coevolution(self) -> bool:
        return False
