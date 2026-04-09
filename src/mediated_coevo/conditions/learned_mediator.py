"""Condition 5: Learned mediator — full GPT-5.4 mediation pipeline + co-evolution.

This is the core research contribution. The Mediator agent:
1. Observes the Executor's raw output
2. Filters, compresses, and selects what to expose to the Planner
3. Co-evolves its mediation skill over time
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediated_coevo.conditions import FeedbackCondition

if TYPE_CHECKING:
    from mediated_coevo.agents.mediator import MediatorAgent
    from mediated_coevo.models.task import TaskSpec
    from mediated_coevo.models.trace import ExecutionTrace
    from mediated_coevo.stores.artifact_store import ArtifactStore


class LearnedMediatorCondition(FeedbackCondition):
    """Full mediation pipeline with co-evolution.

    Requires a MediatorAgent to be instantiated and passed
    to produce_feedback(). This is the only condition that
    makes an LLM call to a third model (GPT-5.4).
    """

    @property
    def name(self) -> str:
        return "learned_mediator"

    async def produce_feedback(
        self,
        trace: ExecutionTrace,
        task_context: TaskSpec,
        artifact_store: ArtifactStore,
        mediator: MediatorAgent | None,
    ) -> str | None:
        if mediator is None:
            raise ValueError(
                "LearnedMediatorCondition requires a MediatorAgent, but None was passed."
            )

        report = await mediator.process_trace(trace, task_context)

        # Store the report
        artifact_store.store_report(report)

        if report.withheld:
            return None

        return report.content

    def supports_coevolution(self) -> bool:
        return True

    @property
    def uses_mediator_reports(self) -> bool:
        return True
