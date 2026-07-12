from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mediated_coevo.models.iteration import IterationRecord

ENV_FAILURE_STATUSES = {"env_failure", "parse_error", "harbor_failed"}
COEVOLUTION_TASK_ID = "__coevolution__"
DEFAULT_BOOTSTRAP_SAMPLES = 10**4
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_DOMINANCE_THRESHOLD = 0.50


class BootstrapConfidenceInterval(BaseModel):
    """Bootstrap interval for a mean score."""

    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    lower: float | None = None
    upper: float | None = None
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    seed: int = DEFAULT_BOOTSTRAP_SEED


class TaskScoreSummary(BaseModel):
    """Score summary for one benchmark task."""

    task_id: str
    total_runs: int
    scored_count: int
    unscored_count: int
    env_failure_count: int
    mean_reward: float | None = None
    median_reward: float | None = None
    bootstrap_ci: BootstrapConfidenceInterval = Field(
        default_factory=BootstrapConfidenceInterval
    )
    scored_share: float = 0.0

    task_category: str | None = None
    task_difficulty: str | None = None
    expected_reward_range: tuple[float, float] | None = None
    verifier_type: str | None = None


class RewardScoreSummary(BaseModel):
    """Score summary for one reward source."""

    total_runs: int
    scored_count: int
    unscored_count: int
    env_failure_count: int
    mean_reward: float | None = None
    median_reward: float | None = None
    macro_mean_reward: float | None = None
    bootstrap_ci: BootstrapConfidenceInterval = Field(
        default_factory=BootstrapConfidenceInterval
    )
    per_task: list[TaskScoreSummary] = Field(default_factory=list)
    total_tokens: int = 0

    dominance_threshold: float = DEFAULT_DOMINANCE_THRESHOLD
    dominant_task_id: str | None = None
    max_task_scored_share: float = 0.0
    dominance_warning: bool = False


class ExperimentScoreSummary(RewardScoreSummary):
    """Score summary for an experiment row."""

    judge_reward_summary: RewardScoreSummary | None = None


def build_score_summary(
    records: list[IterationRecord],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    dominance_threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
) -> ExperimentScoreSummary:
    """Build aggregate and per-task score summaries."""
    rows = _record_rows(records)
    scored_rewards = _scored_rewards(rows)
    total_runs = len(rows)
    total_scored = len(scored_rewards)

    per_task = _build_task_summaries(
        rows=rows,
        total_scored=total_scored,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    task_means = [task.mean_reward for task in per_task if task.mean_reward is not None]
    dominant_task = None
    if total_scored:
        dominant_task = max(per_task, key=lambda task: task.scored_share, default=None)
    max_task_scored_share = dominant_task.scored_share if dominant_task else 0.0
    scored_task_count = sum(1 for task in per_task if task.scored_count > 0)
    dominance_warning = (
        scored_task_count > 1 and max_task_scored_share > dominance_threshold
    )

    return ExperimentScoreSummary(
        total_runs=total_runs,
        scored_count=total_scored,
        unscored_count=total_runs - total_scored,
        env_failure_count=sum(1 for row in rows if row["is_env_failure"]),
        mean_reward=_mean(scored_rewards),
        median_reward=_median(scored_rewards),
        macro_mean_reward=_mean(task_means),
        bootstrap_ci=_bootstrap_mean_ci(
            scored_rewards,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
        ),
        per_task=per_task,
        total_tokens=sum(int(row["total_tokens"]) for row in rows),
        dominance_threshold=dominance_threshold,
        dominant_task_id=dominant_task.task_id if dominant_task else None,
        max_task_scored_share=max_task_scored_share,
        dominance_warning=dominance_warning,
    )


def write_score_summary(summary: ExperimentScoreSummary, path: Path) -> None:
    """Write a score summary as stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary.model_dump(mode="json"), f, indent=2, sort_keys=True)
        f.write("\n")


def _build_task_summaries(
    *,
    rows: list[dict[str, Any]],
    total_scored: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> list[TaskScoreSummary]:
    if not rows:
        return []

    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_task[str(row["task_id"])].append(row)

    summaries = []
    for offset, task_id in enumerate(sorted(rows_by_task)):
        task_rows = rows_by_task[task_id]
        rewards = _scored_rewards(task_rows)
        scored_count = len(rewards)
        total_runs = len(task_rows)
        scored_share = scored_count / total_scored if total_scored else 0.0
        expected_reward_range = _first_present(task_rows, "expected_reward_range")

        summaries.append(
            TaskScoreSummary(
                task_id=task_id,
                total_runs=total_runs,
                scored_count=scored_count,
                unscored_count=total_runs - scored_count,
                env_failure_count=sum(1 for row in task_rows if row["is_env_failure"]),
                mean_reward=_mean(rewards),
                median_reward=_median(rewards),
                bootstrap_ci=_bootstrap_mean_ci(
                    rewards,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + offset + 1,
                    confidence_level=confidence_level,
                ),
                scored_share=scored_share,
                task_category=_first_string(task_rows, "task_category"),
                task_difficulty=_first_string(task_rows, "task_difficulty"),
                expected_reward_range=(
                    expected_reward_range
                    if isinstance(expected_reward_range, tuple)
                    else None
                ),
                verifier_type=_first_string(task_rows, "verifier_type"),
            )
        )
    return summaries


def _record_rows(records: list[IterationRecord]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if record.task_id == COEVOLUTION_TASK_ID:
            continue
        status = (
            record.execution_trace.status
            if record.execution_trace is not None
            else None
        )
        reward = record.reward
        is_env_failure = status in ENV_FAILURE_STATUSES
        if reward is not None and not is_env_failure:
            is_scored = True
            scored_reward = float(reward)
        else:
            is_scored = False
            scored_reward = None
        rows.append(
            {
                "task_id": record.task_id,
                "reward": record.reward,
                "status": status,
                "is_env_failure": is_env_failure,
                "is_scored": is_scored,
                "scored_reward": scored_reward,
                "total_tokens": _numeric_or_zero(record.total_tokens),
                "task_category": record.task_category,
                "task_difficulty": record.task_difficulty,
                "expected_reward_range": record.expected_reward_range,
                "verifier_type": record.verifier_type,
            }
        )
    return rows


def _scored_rewards(rows: list[dict[str, Any]]) -> list[float]:
    return [
        float(row["scored_reward"])
        for row in rows
        if row["is_scored"] and row["scored_reward"] is not None
    ]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> BootstrapConfidenceInterval:
    values = [value for value in values if not math.isnan(value)]
    if not values or samples <= 0:
        return BootstrapConfidenceInterval(
            confidence_level=confidence_level,
            samples=samples,
            seed=seed,
        )

    sample_size = len(values)
    bootstrap_means = [
        statistics.fmean(
            random.Random(seed + sample_index).choices(values, k=sample_size)
        )
        for sample_index in range(samples)
    ]
    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile

    return BootstrapConfidenceInterval(
        confidence_level=confidence_level,
        lower=float(_quantile(bootstrap_means, lower_percentile)),
        upper=float(_quantile(bootstrap_means, upper_percentile)),
        samples=samples,
        seed=seed,
    )


def _quantile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return lower + (upper - lower) * (position - lower_index)


def _first_string(rows: list[dict[str, Any]], column: str) -> str | None:
    value = _first_present(rows, column)
    if isinstance(value, str):
        return value
    return None


def _first_present(rows: list[dict[str, Any]], column: str) -> Any:
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        return value
    return None


def _numeric_or_zero(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and not math.isnan(value):
        return int(value)
    return 0
