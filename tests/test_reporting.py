from __future__ import annotations

import json

import pytest

from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.trace import ExecutionTrace, TraceStatus
from mediated_coevo.reporting import (
    COEVOLUTION_TASK_ID,
    build_score_summary,
    write_score_summary,
)


def _record(
    task_id: str,
    reward: float | None,
    *,
    iteration: int = 0,
    status: TraceStatus = "ok",
    total_tokens: int = 10,
) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        task_id=task_id,
        reward=reward,
        total_tokens=total_tokens,
        execution_trace=ExecutionTrace(
            task_id=task_id,
            iteration=iteration,
            reward=reward,
            status=status,
        ),
        task_category=f"category-{task_id}",
        task_difficulty="medium",
        expected_reward_range=(0.0, 1.0),
        verifier_type="pytest",
    )


def test_score_summary_reports_overall_and_per_task_metrics():
    records = [
        _record("build-fix", 1.0, iteration=0),
        _record("build-fix", 0.0, iteration=1),
        _record("build-fix", None, iteration=2, status="env_failure"),
        _record("api-misuse", 0.5, iteration=0),
        _record("api-misuse", 1.0, iteration=1),
        _record("api-misuse", 0.5, iteration=2),
        _record("parser", 0.0, iteration=0),
        _record("parser", None, iteration=1, status="env_failure"),
        _record("parser", 1.0, iteration=2),
    ]

    summary = build_score_summary(
        records,
        bootstrap_samples=200,
        bootstrap_seed=7,
        dominance_threshold=0.4,
    )

    assert summary.total_runs == 9
    assert summary.scored_count == 7
    assert summary.unscored_count == 2
    assert summary.env_failure_count == 2
    expected_macro_mean = (0.5 + (2.0 / 3.0) + 0.5) / 3.0
    assert summary.mean_reward == pytest.approx(4.0 / 7.0)
    assert summary.median_reward == pytest.approx(0.5)
    assert summary.macro_mean_reward == pytest.approx(expected_macro_mean)
    assert summary.total_tokens == 90
    assert summary.dominant_task_id == "api-misuse"
    assert summary.max_task_scored_share == pytest.approx(3.0 / 7.0)
    assert summary.dominance_warning is True
    assert summary.bootstrap_ci.lower is not None
    assert summary.bootstrap_ci.upper is not None
    assert summary.bootstrap_ci.lower <= summary.mean_reward <= summary.bootstrap_ci.upper

    by_task = {task.task_id: task for task in summary.per_task}
    assert list(by_task) == ["api-misuse", "build-fix", "parser"]
    assert by_task["build-fix"].total_runs == 3
    assert by_task["build-fix"].scored_count == 2
    assert by_task["build-fix"].env_failure_count == 1
    assert by_task["build-fix"].mean_reward == pytest.approx(0.5)
    assert by_task["build-fix"].median_reward == pytest.approx(0.5)
    assert by_task["api-misuse"].mean_reward == pytest.approx(2.0 / 3.0)
    assert by_task["api-misuse"].median_reward == pytest.approx(0.5)
    assert by_task["api-misuse"].task_category == "category-api-misuse"
    assert by_task["api-misuse"].expected_reward_range == (0.0, 1.0)


def test_score_summary_excludes_coevolution_records_and_writes_json(tmp_path):
    records = [
        _record("task-a", 1.0, iteration=0, total_tokens=5),
        _record("task-a", 0.0, iteration=1, total_tokens=7),
        IterationRecord(
            iteration=1,
            task_id=COEVOLUTION_TASK_ID,
            reward=1.0,
            total_tokens=999,
        ),
    ]

    summary = build_score_summary(
        records,
        bootstrap_samples=50,
        bootstrap_seed=123,
    )
    repeat = build_score_summary(
        records,
        bootstrap_samples=50,
        bootstrap_seed=123,
    )

    assert summary.model_dump() == repeat.model_dump()
    assert summary.total_runs == 2
    assert summary.scored_count == 2
    assert summary.total_tokens == 12

    summary_path = tmp_path / "summary.json"
    write_score_summary(summary, summary_path)
    loaded = json.loads(summary_path.read_text())

    assert loaded["total_runs"] == 2
    assert loaded["scored_count"] == 2
    assert loaded["mean_reward"] == pytest.approx(0.5)
    assert loaded["per_task"][0]["task_id"] == "task-a"
    assert loaded["per_task"][0]["expected_reward_range"] == [0.0, 1.0]


def test_environment_failures_are_not_scored_even_with_reward():
    records = [
        _record("task-a", 1.0, status="ok"),
        _record("task-a", 1.0, status="harbor_failed"),
    ]

    summary = build_score_summary(records, bootstrap_samples=50)

    assert summary.total_runs == 2
    assert summary.scored_count == 1
    assert summary.env_failure_count == 1
    assert summary.mean_reward == pytest.approx(1.0)


def test_all_unscored_summary_has_no_dominant_task():
    summary = build_score_summary(
        [_record("task-a", None, status="env_failure")],
        bootstrap_samples=50,
    )

    assert summary.scored_count == 0
    assert summary.mean_reward is None
    assert summary.bootstrap_ci.lower is None
    assert summary.dominant_task_id is None
    assert summary.dominance_warning is False
