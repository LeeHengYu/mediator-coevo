from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from pydantic import BaseModel, Field

from mediated_coevo.core.utils import as_optional_float
from mediated_coevo.models.iteration import IterationRecord

ENV_FAILURE_STATUSES = {"env_failure", "parse_error", "harbor_failed"}
COEVOLUTION_TASK_ID = "__coevolution__"
DEFAULT_BOOTSTRAP_SAMPLES = 10**4
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_DOMINANCE_THRESHOLD = 0.50
RECORD_COLUMNS = [
    "task_id",
    "reward",
    "status",
    "is_env_failure",
    "is_scored",
    "scored_reward",
    "total_tokens",
    "task_category",
    "task_difficulty",
    "expected_reward_range",
    "verifier_type",
]


class _TaskMetadata(TypedDict):
    task_category: str | None
    task_difficulty: str | None
    expected_reward_range: tuple[float, float] | None
    verifier_type: str | None


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
    frame = _records_frame(records)
    scored_rewards = _scored_rewards(frame)
    total_runs = len(frame)
    total_scored = len(scored_rewards)

    per_task = _build_task_summaries(
        frame=frame,
        total_scored=total_scored,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    task_means = pd.Series(
        [task.mean_reward for task in per_task if task.mean_reward is not None],
        dtype="float64",
    )
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
        env_failure_count=_env_failure_count(frame),
        mean_reward=_series_mean(scored_rewards),
        median_reward=_series_median(scored_rewards),
        macro_mean_reward=_series_mean(task_means),
        bootstrap_ci=_bootstrap_mean_ci(
            scored_rewards,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
        ),
        per_task=per_task,
        total_tokens=int(frame["total_tokens"].sum()) if total_runs else 0,
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
    frame: pd.DataFrame,
    total_scored: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> list[TaskScoreSummary]:
    if frame.empty:
        return []

    base_stats = frame.groupby("task_id", sort=True).agg(
        total_runs=("task_id", "size"),
        env_failure_count=("is_env_failure", "sum"),
    )
    reward_stats = (
        frame[frame["is_scored"]]
        .groupby("task_id")["scored_reward"]
        .agg(
            scored_count="count",
            mean_reward="mean",
            median_reward="median",
        )
    )
    task_stats = base_stats.join(reward_stats, how="left")
    task_stats["scored_count"] = task_stats["scored_count"].fillna(0).astype(int)

    summaries = []
    for offset, (task_id, stats) in enumerate(task_stats.iterrows()):
        task_frame = frame[frame["task_id"] == task_id]
        rewards = _scored_rewards(task_frame)
        scored_count = int(stats["scored_count"])
        total_runs = int(stats["total_runs"])
        scored_share = scored_count / total_scored if total_scored else 0.0

        summaries.append(
            TaskScoreSummary(
                task_id=str(task_id),
                total_runs=total_runs,
                scored_count=scored_count,
                unscored_count=total_runs - scored_count,
                env_failure_count=int(stats["env_failure_count"]),
                mean_reward=as_optional_float(stats["mean_reward"]),
                median_reward=as_optional_float(stats["median_reward"]),
                bootstrap_ci=_bootstrap_mean_ci(
                    rewards,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + offset + 1,
                    confidence_level=confidence_level,
                ),
                scored_share=scored_share,
                **_first_task_metadata(task_frame),
            )
        )
    return summaries


def _records_frame(records: list[IterationRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        if record.task_id == COEVOLUTION_TASK_ID:
            continue
        status = _record_status(record)
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
                "total_tokens": record.total_tokens,
                "task_category": record.task_category,
                "task_difficulty": record.task_difficulty,
                "expected_reward_range": record.expected_reward_range,
                "verifier_type": record.verifier_type,
            }
        )

    frame = pd.DataFrame(rows, columns=RECORD_COLUMNS)
    frame["scored_reward"] = pd.to_numeric(frame["scored_reward"], errors="coerce")
    frame["total_tokens"] = pd.to_numeric(
        frame["total_tokens"], errors="coerce"
    ).fillna(0)
    return frame


def _scored_rewards(frame: pd.DataFrame) -> pd.Series:
    return frame.loc[frame["is_scored"], "scored_reward"].dropna().astype(float)


def _env_failure_count(frame: pd.DataFrame) -> int:
    return int(frame["is_env_failure"].sum()) if not frame.empty else 0


def _record_status(record: IterationRecord) -> str | None:
    if record.execution_trace is None:
        return None
    return record.execution_trace.status


def _series_mean(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float(values.mean())


def _series_median(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float(values.median())


def _bootstrap_mean_ci(
    values: pd.Series,
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> BootstrapConfidenceInterval:
    values = values.dropna().astype(float).reset_index(drop=True)
    if values.empty:
        return BootstrapConfidenceInterval(
            confidence_level=confidence_level,
            samples=samples,
            seed=seed,
        )

    sample_size = len(values)
    bootstrap_means = pd.Series(
        [
            float(
                values.sample(
                    n=sample_size,
                    replace=True,
                    random_state=seed + sample_index,
                ).mean()
            )
            for sample_index in range(samples)
        ],
        dtype="float64",
    )
    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile

    return BootstrapConfidenceInterval(
        confidence_level=confidence_level,
        lower=float(bootstrap_means.quantile(lower_percentile)),
        upper=float(bootstrap_means.quantile(upper_percentile)),
        samples=samples,
        seed=seed,
    )


def _first_task_metadata(frame: pd.DataFrame) -> _TaskMetadata:
    return {
        "task_category": _first_string(frame, "task_category"),
        "task_difficulty": _first_string(frame, "task_difficulty"),
        "expected_reward_range": _first_reward_range(frame),
        "verifier_type": _first_string(frame, "verifier_type"),
    }


def _first_string(frame: pd.DataFrame, column: str) -> str | None:
    value = _first_present(frame, column)
    if isinstance(value, str):
        return value
    return None


def _first_reward_range(frame: pd.DataFrame) -> tuple[float, float] | None:
    value = _first_present(frame, "expected_reward_range")
    if isinstance(value, tuple):
        return value
    return None


def _first_present(frame: pd.DataFrame, column: str) -> Any:
    values = frame[column].dropna()
    if values.empty:
        return None
    return values.iloc[0]
