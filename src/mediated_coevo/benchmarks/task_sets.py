"""Curated benchmark task sets."""

from __future__ import annotations

from dataclasses import dataclass


class TaskSetError(ValueError):
    """Raised when a named task set or task selection is invalid."""


@dataclass(frozen=True, slots=True)
class TaskMetadata:
    """Experiment-facing metadata for a curated benchmark task."""

    category: str
    difficulty: str
    expected_reward_range: tuple[float, float]
    verifier_type: str


SKILLSBENCH_EXPECTED_REWARD_RANGE: tuple[float, float] = (0.0, 1.0)
SKILLSBENCH_VERIFIER_TYPE = "skillsbench_pytest"

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

SKILLSBENCH_10_TASK_METADATA: dict[str, TaskMetadata] = {
    "fix-build-google-auto": TaskMetadata(
        category="Compilation & Build",
        difficulty="easy",
        expected_reward_range=SKILLSBENCH_EXPECTED_REWARD_RANGE,
        verifier_type=SKILLSBENCH_VERIFIER_TYPE,
    ),
    "fix-build-agentops": TaskMetadata(
        category="Compilation & Build",
        difficulty="easy",
        expected_reward_range=SKILLSBENCH_EXPECTED_REWARD_RANGE,
        verifier_type=SKILLSBENCH_VERIFIER_TYPE,
    ),
    "adaptive-cruise-control": TaskMetadata(
        category="control-systems",
        difficulty="medium",
        expected_reward_range=SKILLSBENCH_EXPECTED_REWARD_RANGE,
        verifier_type=SKILLSBENCH_VERIFIER_TYPE,
    ),
    "azure-bgp-oscillation-route-leak": TaskMetadata(
        category="bgp-route",
        difficulty="medium",
        expected_reward_range=SKILLSBENCH_EXPECTED_REWARD_RANGE,
        verifier_type=SKILLSBENCH_VERIFIER_TYPE,
    ),
    "bike-rebalance": TaskMetadata(
        category="transportation-logistics",
        difficulty="medium",
        expected_reward_range=SKILLSBENCH_EXPECTED_REWARD_RANGE,
        verifier_type=SKILLSBENCH_VERIFIER_TYPE,
    ),
    "citation-check": TaskMetadata(
        category="research",
        difficulty="medium",
        expected_reward_range=SKILLSBENCH_EXPECTED_REWARD_RANGE,
        verifier_type=SKILLSBENCH_VERIFIER_TYPE,
    ),
    "court-form-filling": TaskMetadata(
        category="document-processing",
        difficulty="easy",
        expected_reward_range=SKILLSBENCH_EXPECTED_REWARD_RANGE,
        verifier_type=SKILLSBENCH_VERIFIER_TYPE,
    ),
    "crystallographic-wyckoff-position-analysis": TaskMetadata(
        category="materials_science",
        difficulty="medium",
        expected_reward_range=SKILLSBENCH_EXPECTED_REWARD_RANGE,
        verifier_type=SKILLSBENCH_VERIFIER_TYPE,
    ),
    "data-to-d3": TaskMetadata(
        category="Data Visualization",
        difficulty="medium",
        expected_reward_range=SKILLSBENCH_EXPECTED_REWARD_RANGE,
        verifier_type=SKILLSBENCH_VERIFIER_TYPE,
    ),
    "dialogue-parser": TaskMetadata(
        category="game",
        difficulty="easy",
        expected_reward_range=SKILLSBENCH_EXPECTED_REWARD_RANGE,
        verifier_type=SKILLSBENCH_VERIFIER_TYPE,
    ),
}

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
