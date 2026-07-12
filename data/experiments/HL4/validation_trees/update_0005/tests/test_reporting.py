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
from mediated_coevo.analysis.inspection import _inspection_payload
from mediated_coevo.analysis.metrics import metric_row
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.trace import ExecutionTrace, TokenUsage, TraceStatus
from mediated_coevo.analysis.reporting import (
    COEVOLUTION_TASK_ID,
    build_score_summary,
    write_score_summary,
)
from mediated_coevo.cli.inspect import latest_experiment_dir
from mediated_coevo.main import app


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
    record.diffusion_artifact_store_path = "data/artifact-stores/warmup"
    record.diffusion_artifact_store_count = 12
    record.diffusion_artifact_store_frozen = True
    record.transfer_context_kind = "diffusion"
    record.transfer_context_tokens = 37
    record.source_task_ids = ["task-b"]
    record.reward_after_diffusion_context = 0.25
    record.regression_after_diffusion_context = True

    row = metric_row(record)

    assert row["diffusion_artifacts_eligible"] == 4
    assert row["diffusion_artifacts_selected"] == 2
    assert row["diffusion_artifacts_rendered"] == 1
    assert row["diffusion_artifact_store_path"] == "data/artifact-stores/warmup"
    assert row["diffusion_artifact_store_count"] == 12
    assert row["diffusion_artifact_store_frozen"] is True
    assert row["transfer_context_kind"] == "diffusion"
    assert row["transfer_context_tokens"] == 37
    assert row["source_task_ids"] == ["task-b"]
    assert row["reward_after_diffusion_context"] == 0.25
    assert row["regression_after_diffusion_context"] is True


def test_metric_row_reports_executor_cache_read_as_audit_only():
    record = _record("task-a", 0.25)
    record.total_tokens = 123
    assert record.execution_trace is not None
    record.execution_trace.token_usage = TokenUsage(
        input_tokens=7,
        output_tokens=3,
    )
    record.execution_trace.harbor_metadata = {
        "executor_token_source": "hermes_session",
        "executor_session_cache_read_tokens": "303",
        "agent_result.cost_usd": "0.123",
        "executor_reported_cost_source": "harbor_agent_result",
    }

    row = metric_row(record)

    assert row["prompt_tokens_by_agent"]["executor"] == 7
    assert row["completion_tokens_by_agent"]["executor"] == 3
    assert row["total_tokens_by_agent"]["executor"] == 10
    assert row["total_tokens"] == 123
    assert row["executor_token_source"] == "hermes_session"
    assert row["executor_cache_read_tokens"] == "303"
    assert row["executor_reported_cost_usd"] == "0.123"
    assert row["executor_reported_cost_source"] == "harbor_agent_result"


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

    assert latest_experiment_dir(experiments_root) == newer


def test_latest_experiment_dir_rejects_empty_root(tmp_path):
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    with pytest.raises(typer.BadParameter, match="no experiment outputs"):
        latest_experiment_dir(experiments_root)


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
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "task_id": "task-a",
                    "diffusion_enabled": True,
                    "diffusion_policy": "top_k_similarity",
                    "diffusion_graph": "precomputed_similarity",
                    "diffusion_artifacts_rendered": 1,
                    "transfer_context_kind": "diffusion",
                    "transfer_context_tokens": 20,
                    "source_task_ids": ["task-b"],
                    "reward_after_diffusion_context": 0.5,
                    "regression_after_diffusion_context": False,
                },
                {
                    "task_id": "task-c",
                    "diffusion_enabled": True,
                    "diffusion_policy": "top_k_similarity",
                    "diffusion_graph": "precomputed_similarity",
                    "diffusion_artifacts_rendered": 1,
                    "transfer_context_kind": "diffusion",
                    "transfer_context_tokens": 10,
                    "source_task_ids": ["task-b", "task-d"],
                    "reward_after_diffusion_context": 0.0,
                    "regression_after_diffusion_context": True,
                },
            ]
        )
        + "\n"
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
    store.store_artifact(
        DiffusionArtifact(
            artifact_id="artifact-2",
            source_task_id="task-d",
            source_iteration=0,
            artifact_type=DiffusionArtifactType.DEBUG_HINT,
            risk_level=DiffusionRiskLevel.LOW,
            content="another hint",
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
    store.append_diffused_record(
        DiffusedRecord(
            artifact_id="artifact-2",
            source_task_id="task-d",
            source_iteration=0,
            target_task_id="task-c",
            target_iteration=1,
            snapshot_id="snapshot-1",
            policy_name="top_k_similarity",
            selected=True,
            rendered=True,
        )
    )

    payload = _inspection_payload(run_dir)

    assert payload["diffusion"]["artifact_count"] == 2
    assert payload["diffusion"]["graph_snapshot_count"] == 1
    assert payload["diffusion"]["diffused_record_count"] == 2
    assert payload["diffusion"]["rendered_record_count"] == 2
    assert payload["diffusion"]["source_task_ids"] == ["task-b", "task-d"]
    assert payload["diffusion"]["graph_snapshot_ids"] == ["snapshot-1"]
    assert payload["diffusion"]["paths"]["metrics"] == str(metrics_path)
    assert payload["diffusion"]["paths"]["diffused_records"] == str(
        run_dir / "diffusion" / "diffused_records.jsonl"
    )
    assert payload["diffusion"]["records"] == {
        "eligible_count": 2,
        "selected_count": 2,
        "rendered_count": 2,
    }
    metrics = payload["diffusion"]["metrics"]
    assert metrics["diffusion_enabled"] is True
    assert metrics["diffusion_policy"] == "top_k_similarity"
    assert metrics["diffusion_graph"] == "precomputed_similarity"
    context = metrics["context"]
    assert context["rows_with_rendered_context"] == 2
    assert context["transfer_context_tokens"]["total"] == pytest.approx(30.0)
    assert context["transfer_context_tokens"]["mean"] == pytest.approx(15.0)
    assert context["transfer_context_tokens"]["max"] == pytest.approx(20.0)
    assert context["source_task_count"] == 2
    assert context["source_task_ids"] == ["task-b", "task-d"]
    assert context["reward_after_diffusion_context"]["count"] == 2
    assert context["reward_after_diffusion_context"]["mean"] == pytest.approx(0.25)
    assert context["regression_after_diffusion_context"]["count"] == 1
    assert context["regression_after_diffusion_context"]["rate"] == pytest.approx(0.5)


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


def test_inspect_command_prints_diffusion_context_summary(tmp_path):
    run_dir = tmp_path / "20260502-000000-42-learned_mediator"
    run_dir.mkdir()
    write_score_summary(
        build_score_summary([_record("task-a", 1.0)], bootstrap_samples=50),
        run_dir / "summary.json",
    )
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "task_id": "task-a",
                "diffusion_enabled": True,
                "diffusion_policy": "top_k_similarity",
                "diffusion_graph": "precomputed_similarity",
                "diffusion_artifacts_rendered": 1,
                "transfer_context_kind": "diffusion",
                "transfer_context_tokens": 12,
                "source_task_ids": ["task-b"],
                "reward_after_diffusion_context": 0.0,
                "regression_after_diffusion_context": True,
            }
        )
        + "\n"
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
    store.append_diffused_record(
        DiffusedRecord(
            artifact_id="artifact-1",
            source_task_id="task-b",
            source_iteration=0,
            target_task_id="task-a",
            target_iteration=1,
            snapshot_id="snapshot-1",
            policy_name="top_k_similarity",
            selected=True,
            rendered=True,
        )
    )

    payload = _inspection_payload(run_dir)

    diffusion = payload["diffusion"]
    metrics = diffusion["metrics"]
    context = metrics["context"]
    assert metrics["diffusion_enabled"] is True
    assert metrics["diffusion_policy"] == "top_k_similarity"
    assert context["rows_with_rendered_context"] == 1
    assert context["transfer_context_tokens"]["total"] == 12
    assert context["transfer_context_tokens"]["mean"] == 12
    assert context["transfer_context_tokens"]["max"] == 12
    assert context["source_task_ids"] == ["task-b"]
    assert context["reward_after_diffusion_context"]["count"] == 1
    assert context["regression_after_diffusion_context"]["count"] == 1


def test_matrix_inspection_payload_reports_row_summaries(tmp_path):
    matrix_dir = tmp_path / "20260502-000000-42-baseline-matrix"
    row_a = matrix_dir / "skill_none_diffusion_none"
    row_b = matrix_dir / "skill_all_top_k_similarity"
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
    assert set(rows) == {
        "skill_all_top_k_similarity",
        "skill_none_diffusion_none",
    }
    assert rows["skill_all_top_k_similarity"]["summary"]["mean_reward"] == (
        pytest.approx(1.0)
    )
    assert rows["skill_none_diffusion_none"]["summary"]["mean_reward"] == pytest.approx(
        0.0
    )


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
