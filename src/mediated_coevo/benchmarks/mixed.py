"""Routing helpers for mixed benchmark experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from mediated_coevo.benchmarks.skillsbench import SkillsBenchRepository
from mediated_coevo.benchmarks import swebench as swebench_helpers

BenchmarkKind = Literal["skillsbench", "swebench"]


@dataclass(frozen=True, slots=True)
class BenchmarkTaskSelection:
    """Explicit benchmark routing for one unified experiment run."""

    task_ids: list[str]
    benchmark_by_task_id: dict[str, BenchmarkKind]
    skillsbench_task_ids: list[str]
    swebench_instance_ids: list[str]

    @property
    def has_skillsbench(self) -> bool:
        return bool(self.skillsbench_task_ids)

    @property
    def has_swebench(self) -> bool:
        return bool(self.swebench_instance_ids)

    @property
    def backend_name(self) -> str:
        if self.has_skillsbench and self.has_swebench:
            return "mixed"
        if self.has_swebench:
            return "swebench"
        return "skillsbench"


class MixedBenchmarkRepository:
    """Resolve selected task IDs through their benchmark-specific repository."""

    def __init__(
        self,
        *,
        benchmark_by_task_id: dict[str, BenchmarkKind],
        skillsbench_repo: SkillsBenchRepository | None = None,
        swebench_repo: swebench_helpers.SWEbenchRepository | None = None,
    ) -> None:
        self.benchmark_by_task_id = dict(benchmark_by_task_id)
        self.skillsbench_repo = skillsbench_repo
        self.swebench_repo = swebench_repo

    def resolve(self, task_id: str) -> Any:
        benchmark = self.benchmark_by_task_id.get(task_id)
        if benchmark == "skillsbench":
            if self.skillsbench_repo is None:
                raise FileNotFoundError(
                    f"SkillsBench backend is not configured for task {task_id!r}"
                )
            return self.skillsbench_repo.resolve(task_id)
        if benchmark == "swebench":
            if self.swebench_repo is None:
                raise FileNotFoundError(
                    f"SWE-bench backend is not configured for task {task_id!r}"
                )
            return self.swebench_repo.resolve(task_id)
        raise FileNotFoundError(f"Task {task_id!r} is not selected for this run")


def build_benchmark_task_selection(
    *,
    skillsbench_task_ids: list[str],
    swebench_instance_ids: list[str],
) -> BenchmarkTaskSelection:
    """Build explicit routing, rejecting duplicate task IDs across benchmarks."""
    skillsbench_ids = _dedupe(skillsbench_task_ids)
    swebench_ids = _dedupe(swebench_instance_ids)
    overlap = sorted(set(skillsbench_ids) & set(swebench_ids))
    if overlap:
        raise ValueError(
            "task IDs cannot be selected for both SkillsBench and SWE-bench: "
            f"{overlap}"
        )

    benchmark_by_task_id: dict[str, BenchmarkKind] = {}
    for task_id in skillsbench_ids:
        benchmark_by_task_id[task_id] = "skillsbench"
    for task_id in swebench_ids:
        benchmark_by_task_id[task_id] = "swebench"

    return BenchmarkTaskSelection(
        task_ids=[*skillsbench_ids, *swebench_ids],
        benchmark_by_task_id=benchmark_by_task_id,
        skillsbench_task_ids=skillsbench_ids,
        swebench_instance_ids=swebench_ids,
    )


def _dedupe(task_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_ids: list[str] = []
    for task_id in task_ids:
        if task_id not in seen:
            unique_ids.append(task_id)
            seen.add(task_id)
    return unique_ids
