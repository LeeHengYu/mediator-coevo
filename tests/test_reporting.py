from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from mediated_coevo.diffusion import (
    DiffusedRecord,
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    DiffusionStore,
    TaskGraphSnapshot,
)
from mediated_coevo.analysis.metrics import metric_row
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.trace import ExecutionTrace, TraceStatus
from mediated_coevo.analysis.reporting import (
    COEVOLUTION_TASK_ID,
    build_score_summary,
    write_score_summary,
)
from mediated_coevo.main import (
    app,
    _inspection_payload,
    _latest_experiment_dir,
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


def test_metric_row_includes_diffusion_process_and_transfer_fields():
    record = _record("task-a", 0.25)
    record.diffusion_enabled = True
    record.diffusion_policy = "top_k_similarity"
    record.diffusion_graph = "precomputed_similarity"
    record.graph_snapshot_id = "snapshot-1"
    record.diffusion_artifacts_eligible = 4
    record.diffusion_artifacts_selected = 2
    record.diffusion_artifacts_rendered = 1
    record.diffusion_context_tokens = 37
    record.source_task_ids = ["task-b"]
    record.reward_after_diffusion_context = 0.25
    record.regression_after_diffusion_context = True

    row = metric_row(record)

    assert row["diffusion_artifacts_eligible"] == 4
    assert row["diffusion_artifacts_selected"] == 2
    assert row["diffusion_artifacts_rendered"] == 1
    assert row["diffusion_context_tokens"] == 37
    assert row["source_task_ids"] == ["task-b"]
    assert row["reward_after_diffusion_context"] == 0.25
    assert row["regression_after_diffusion_context"] is True


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
    assert loaded["judge_reward_summary"] is None


def test_score_summary_can_include_judge_reward_summary(tmp_path):
    verifier_summary = build_score_summary(
        [_record("task-a", 0.5)],
        bootstrap_samples=50,
    )
    verifier_summary.judge_reward_summary = build_score_summary(
        [_record("task-a", 0.75)],
        bootstrap_samples=50,
    )

    summary_path = tmp_path / "summary.json"
    write_score_summary(verifier_summary, summary_path)
    loaded = json.loads(summary_path.read_text())

    assert loaded["mean_reward"] == pytest.approx(0.5)
    assert loaded["judge_reward_summary"]["mean_reward"] == pytest.approx(0.75)


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


def test_single_scored_task_does_not_warn_about_dominance():
    records = [
        _record("task-a", 1.0, iteration=0),
        _record("task-a", 0.0, iteration=1),
    ]

    summary = build_score_summary(records, bootstrap_samples=50)

    assert summary.scored_count == 2
    assert summary.dominant_task_id == "task-a"
    assert summary.max_task_scored_share == pytest.approx(1.0)
    assert summary.dominance_warning is False


def test_balanced_scored_tasks_do_not_warn_about_dominance():
    records = [
        _record("task-a", 1.0, iteration=0),
        _record("task-b", 0.0, iteration=0),
    ]

    summary = build_score_summary(records, bootstrap_samples=50)

    assert summary.scored_count == 2
    assert summary.max_task_scored_share == pytest.approx(0.5)
    assert summary.dominance_warning is False


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


def test_latest_experiment_dir_uses_newest_named_child(tmp_path):
    experiments_root = tmp_path / "experiments"
    older = experiments_root / "20260501-000000-42-no_feedback"
    newer = experiments_root / "20260502-000000-42-no_feedback"
    older.mkdir(parents=True)
    newer.mkdir()

    assert _latest_experiment_dir(experiments_root) == newer


def test_latest_experiment_dir_rejects_empty_root(tmp_path):
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    with pytest.raises(typer.BadParameter, match="no experiment outputs"):
        _latest_experiment_dir(experiments_root)


def test_single_run_inspection_payload_reports_summary_and_paths(tmp_path):
    run_dir = tmp_path / "20260502-000000-42-learned_mediator"
    artifact_dir = run_dir / "artifacts" / "traces"
    artifact_dir.mkdir(parents=True)
    records = [_record("task-a", 1.0)]
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    write_score_summary(build_score_summary(records, bootstrap_samples=50), summary_path)
    metrics_path.write_text('{"task_id": "task-a"}\n')

    payload = _inspection_payload(run_dir)

    assert payload["kind"] == "single"
    assert payload["summary_path"] == str(summary_path)
    assert payload["metrics_path"] == str(metrics_path)
    assert payload["artifact_dirs"] == [str(artifact_dir)]
    assert payload["summary"]["total_runs"] == 1
    assert payload["summary"]["scored_count"] == 1


def test_single_run_inspection_payload_reports_diffusion_bundle(tmp_path):
    run_dir = tmp_path / "20260502-000000-42-learned_mediator"
    run_dir.mkdir()
    write_score_summary(
        build_score_summary([_record("task-a", 1.0)], bootstrap_samples=50),
        run_dir / "summary.json",
    )
    store = DiffusionStore(run_dir / "diffusion")
    store.store_artifact(
        DiffusionArtifact(
            artifact_id="artifact-1",
            source_task_id="task-b",
            source_iteration=0,
            artifact_type=DiffusionArtifactType.DEBUG_HINT,
            risk_level=DiffusionRiskLevel.LOW,
            content="hint",
        )
    )
    store.store_graph_snapshot(
        TaskGraphSnapshot(
            snapshot_id="snapshot-1",
            run_id=run_dir.name,
            iteration=1,
            task_ids=["task-a", "task-b"],
            graph_policy="precomputed_similarity",
        )
    )
    store.append_diffused_record(
        DiffusedRecord(
            artifact_id="artifact-1",
            source_task_id="task-b",
            source_iteration=0,
            target_task_id="task-a",
            target_iteration=1,
            snapshot_id="snapshot-1",
            policy_name="capped_broadcast",
            selected=True,
            rendered=True,
        )
    )

    payload = _inspection_payload(run_dir)

    assert payload["diffusion"]["artifact_count"] == 1
    assert payload["diffusion"]["graph_snapshot_count"] == 1
    assert payload["diffusion"]["diffused_record_count"] == 1
    assert payload["diffusion"]["rendered_record_count"] == 1
    assert payload["diffusion"]["source_task_ids"] == ["task-b"]
    assert payload["diffusion"]["graph_snapshot_ids"] == ["snapshot-1"]


def test_single_run_inspection_payload_warns_when_summary_missing(tmp_path):
    run_dir = tmp_path / "20260502-000000-42-learned_mediator"
    run_dir.mkdir()
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text('{"task_id": "task-a"}\n')

    payload = _inspection_payload(run_dir)

    assert payload["kind"] == "single"
    assert payload["summary_path"] is None
    assert payload["metrics_path"] == str(metrics_path)
    assert "summary.json is missing" in payload["warning"]


def test_matrix_inspection_payload_reports_row_summaries(tmp_path):
    matrix_dir = tmp_path / "20260502-000000-42-baseline-matrix"
    row_a = matrix_dir / "no_feedback"
    row_b = matrix_dir / "full_coevolution"
    for row_dir, reward in ((row_a, 0.0), (row_b, 1.0)):
        row_dir.mkdir(parents=True)
        write_score_summary(
            build_score_summary([_record("task-a", reward)], bootstrap_samples=50),
            row_dir / "summary.json",
        )
        (row_dir / "metrics.jsonl").write_text('{"task_id": "task-a"}\n')

    payload = _inspection_payload(matrix_dir)

    assert payload["kind"] == "matrix"
    rows = {row["row"]: row for row in payload["rows"]}
    assert set(rows) == {"full_coevolution", "no_feedback"}
    assert rows["full_coevolution"]["summary"]["mean_reward"] == pytest.approx(1.0)
    assert rows["no_feedback"]["summary"]["mean_reward"] == pytest.approx(0.0)


def test_inspect_command_emits_json_for_single_run(tmp_path):
    run_dir = tmp_path / "20260502-000000-42-learned_mediator"
    run_dir.mkdir()
    write_score_summary(
        build_score_summary([_record("task-a", 1.0)], bootstrap_samples=50),
        run_dir / "summary.json",
    )
    (run_dir / "metrics.jsonl").write_text('{"task_id": "task-a"}\n')

    result = CliRunner().invoke(app, ["inspect", str(run_dir), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["kind"] == "single"
    assert payload["summary"]["total_runs"] == 1
