"""Curated benchmark task sets."""

from __future__ import annotations


class TaskSetError(ValueError):
    """Raised when a named task set or task selection is invalid."""


SKILLSBENCH_10_TASK_IDS: tuple[str, ...] = (
    "fix-build-google-auto",
    "fix-build-agentops",
    "adaptive-cruise-control",
    "azure-bgp-oscillation-route-leak",
    "bike-rebalance",
    "citation-check",
    "court-form-filling",
    "crystallographic-wyckoff-position-analysis",
    "data-to-d3",
    "dialogue-parser",
)

TASK_SETS: dict[str, tuple[str, ...]] = {
    "skillsbench-10": SKILLSBENCH_10_TASK_IDS,
}


def parse_task_ids(raw_tasks: str) -> list[str]:
    """Parse a comma-separated task ID list."""
    task_ids = [task.strip() for task in raw_tasks.split(",") if task.strip()]
    if not task_ids:
        raise TaskSetError("at least one task ID is required")
    return task_ids


def resolve_task_selection(
    *,
    tasks: str | None,
    task_set: str | None,
    default_tasks: str,
) -> list[str]:
    """Resolve CLI task selection, with explicit tasks overriding task sets."""
    if tasks is not None:
        return parse_task_ids(tasks)
    if task_set is not None:
        name = task_set.strip()
        if not name:
            raise TaskSetError("task set name cannot be empty")
        if name not in TASK_SETS:
            allowed = ", ".join(sorted(TASK_SETS))
            raise TaskSetError(
                f"unknown task set {name!r}; expected one of: {allowed}"
            )
        return list(TASK_SETS[name])
    return parse_task_ids(default_tasks)
