"""Experiment condition definitions and prior-context routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mediated_coevo.models.report import MediatorReport
    from mediated_coevo.stores.artifact_store import ArtifactStore

ConditionName = Literal[
    "no_feedback",
    "full_traces",
    "shared_notes",
    "static_mediator",
    "learned_mediator",
]

MEDIATOR_CONDITIONS: frozenset[ConditionName] = frozenset({"static_mediator", "learned_mediator"})
MEDIATOR_EVOLVE_CONDITIONS: frozenset[ConditionName] = frozenset({"learned_mediator"})


def get_prior_context(
    condition: ConditionName,
    task_id: str,
    artifact_store: ArtifactStore,
    previous_report: MediatorReport | None,
    shared_notes: str | None,
) -> str | None:
    """Return the prior-context string the planner should receive, or None."""
    if condition == "no_feedback":
        return None
    elif condition == "full_traces":
        summaries = artifact_store.query_summaries(task_id=task_id, recent=3)
        return "\n".join(summaries) if summaries else None
    elif condition == "shared_notes":
        return shared_notes
    else:  # static_mediator, learned_mediator
        if previous_report and not previous_report.withheld:
            return previous_report.content
        return None
