"""Condition 3: Shared notes — Planner reads Executor's accumulated notes file."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediated_coevo.conditions import FeedbackCondition

if TYPE_CHECKING:
    from mediated_coevo.agents.mediator import MediatorAgent
    from mediated_coevo.models.task import TaskSpec
    from mediated_coevo.models.trace import ExecutionTrace
    from mediated_coevo.stores.artifact_store import ArtifactStore


class SharedNotesCondition(FeedbackCondition):
    """Planner reads a notes.md file that the Executor appends to during execution.

    This simulates the "shared repository" approach (e.g., OpenSpace, Spark)
    where agents passively dump knowledge into a shared context.
    """

    def __init__(self, notes_content: str = "") -> None:
        self._notes = notes_content

    @property
    def name(self) -> str:
        return "shared_notes"

    def append_notes(self, content: str) -> None:
        """Called after execution to simulate Executor writing notes."""
        self._notes += f"\n{content}"

    async def produce_feedback(
        self,
        trace: ExecutionTrace,
        task_context: TaskSpec,
        artifact_store: ArtifactStore,
        mediator: MediatorAgent | None,
    ) -> str | None:
        # Append this execution's result to notes
        note = (
            f"## Iteration {trace.iteration}\n"
            f"reward={trace.reward:.2f} exit_code={trace.exit_code}\n"
        )
        if trace.stderr:
            note += f"errors: {trace.stderr[:500]}\n"
        self.append_notes(note)
        return self._notes if self._notes.strip() else None

    def supports_coevolution(self) -> bool:
        return False
