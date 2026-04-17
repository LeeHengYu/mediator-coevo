"""Experimental conditions — strategy pattern.

Each condition defines how execution feedback reaches the Planner.
The Orchestrator is condition-agnostic; it delegates to whichever
FeedbackCondition is configured.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mediated_coevo.agents.mediator import MediatorAgent
    from mediated_coevo.models.task import TaskSpec
    from mediated_coevo.models.trace import ExecutionTrace
    from mediated_coevo.stores.artifact_store import ArtifactStore


class FeedbackCondition(ABC):
    """Base class for experimental conditions."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def produce_feedback(
        self,
        trace: ExecutionTrace,
        task_context: TaskSpec,
        artifact_store: ArtifactStore,
        mediator: MediatorAgent | None,
    ) -> str | None:
        """Return feedback string for the Planner, or None."""
        ...

    @abstractmethod
    def supports_coevolution(self) -> bool:
        """True iff the condition runs a MediatorAgent.

        A mediator is what produces `MediatorReport` artifacts and what
        co-evolves (via Reflector) against its own reporting history, so
        "has a mediator in the loop" and "supports co-evolution" are the
        same predicate. The orchestrator uses this flag both to decide
        whether to carry a prior `MediatorReport` into the next plan and
        to decide whether to run co-evolution at checkpoints.
        """
        ...


# Import conditions after base class is defined
from .no_feedback import NoFeedbackCondition
from .full_traces import FullTracesCondition
from .shared_notes import SharedNotesCondition
from .static_mediator import StaticMediatorCondition
from .learned_mediator import LearnedMediatorCondition

REGISTRY: dict[str, type[FeedbackCondition]] = {
    "no_feedback": NoFeedbackCondition,
    "full_traces": FullTracesCondition,
    "shared_notes": SharedNotesCondition,
    "static_mediator": StaticMediatorCondition,
    "learned_mediator": LearnedMediatorCondition,
}


def create_condition(name: str, **kwargs) -> FeedbackCondition:
    """Factory: create a condition by name."""
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown condition: {name!r}. Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[name](**kwargs)
