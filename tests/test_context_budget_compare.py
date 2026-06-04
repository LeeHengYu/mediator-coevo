from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mediated_coevo.analysis.context_budget_compare import (
    compare_context_budget_runs,
)
from mediated_coevo.diffusion import DiffusedRecord
from mediated_coevo.main import app


def test_compare_context_budget_runs_reports_budget_only_warning(tmp_path: Path):
    run_a = _write_run(
        tmp_path / "run-a",
        max_diffusion_context_tokens=4000,
        total_planner_prior_context_tokens=100,
    )
    run_b = _write_run(
        tmp_path / "run-b",
        max_diffusion_context_tokens=2000,
        total_planner_prior_context_tokens=150,
    )

    comparison = compare_context_budget_runs(run_a, run_b)

    assert comparison.comparability_status == "warning"
    assert [difference.path for difference in comparison.setup_mismatches] == []
    assert [difference.path for difference in comparison.budget_differences] == [
        "budgets.max_diffusion_context_tokens"
    ]
    assert comparison.token_delta_percent[
        "total_planner_prior_context_tokens"
    ] == pytest.approx(0.5)
    assert comparison.artifact_validity_failures == []


def test_compare_context_budget_runs_flags_invalid_diffusion_records(
    tmp_path: Path,
):
    run_a = _write_run(tmp_path / "run-a")
    run_b = _write_run(
        tmp_path / "run-b",
        record=DiffusedRecord(
            artifact_id="artifact-1",
            source_task_id="task-a",
            source_iteration=0,
            target_task_id="task-a",
            target_iteration=1,
            policy_name="capped_broadcast",
            relation="broadcast",
            reason="selected_by_test",
            selected=True,
            rendered=True,
            citation_text="artifact_id=artifact-1",
        ),
    )

    comparison = compare_context_budget_runs(run_a, run_b)

    assert comparison.comparability_status == "fail"
    assert len(comparison.artifact_validity_failures) == 1
    assert (
        comparison.artifact_validity_failures[0].description
        == "diffusion record leaks same-task context"
    )


def test_compare_context_budgets_cli_emits_json(tmp_path: Path):
    run_a = _write_run(tmp_path / "run-a")
    run_b = _write_run(tmp_path / "run-b")

    result = CliRunner().invoke(
        app,
        ["compare-context-budgets", str(run_a), str(run_b), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["comparability_status"] == "pass"


def _write_run(
    run_dir: Path,
    *,
    max_diffusion_context_tokens: int = 4000,
    total_planner_prior_context_tokens: int = 100,
    record: DiffusedRecord | None = None,
) -> Path:
    run_dir.mkdir()
    (run_dir / "config.toml").write_text(
        _config_toml(max_diffusion_context_tokens=max_diffusion_context_tokens)
    )
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "iteration": 1,
                "task_id": "task-a",
                "same_task_prior_tokens": 30,
                "cross_task_prior_tokens": 20,
                "diffusion_context_tokens": 50,
                "total_planner_prior_context_tokens": (
                    total_planner_prior_context_tokens
                ),
                "context_budget_violation": False,
                "compacted_diffusion_artifact_ids": [],
                "dropped_for_budget_artifact_ids": [],
            }
        )
        + "\n"
    )
    diffusion_dir = run_dir / "diffusion"
    artifacts_dir = diffusion_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "artifact-1.json").write_text("{}")
    diffused_record = record or DiffusedRecord(
        artifact_id="artifact-1",
        source_task_id="task-b",
        source_iteration=0,
        target_task_id="task-a",
        target_iteration=1,
        policy_name="capped_broadcast",
        relation="broadcast",
        reason="selected_by_test",
        selected=True,
        rendered=True,
        citation_text="artifact_id=artifact-1\ncontent=hint",
    )
    (diffusion_dir / "diffused_records.jsonl").write_text(
        diffused_record.model_dump_json() + "\n"
    )
    return run_dir


def _config_toml(*, max_diffusion_context_tokens: int) -> str:
    return f"""
[models]
planner = "test-planner"
executor = "test-executor"
mediator = "test-mediator"
judge = "test-judge"

[budgets]
max_skill_tokens = 4000
max_diffusion_context_tokens = {max_diffusion_context_tokens}
trace_excerpt_tokens = 6000
historical_summary_tokens = 3000
mediator_report_tokens = 4000
planner_context_tokens = 24000
skill_update_diff_tokens = 6000
mediator_prompt_tokens = 16000
advisor_prompt_tokens = 12000
reflector_prompt_tokens = 16000
judge_prompt_tokens = 16000
planner_completion_tokens = 4096
mediator_completion_tokens = 2048
advisor_completion_tokens = 512
reflector_completion_tokens = 4096
judge_completion_tokens = 2048

[experiment]
num_iterations = 2
coevo_interval = 2
advisor_buffer_max = 2
seed = 42
condition_name = "learned_mediator"

[experiment.benchmark_selection]
tasks = ["task-a", "task-b"]

[experiment.skill_updates]
executor = true
planner = true
mediator = true

[diffusion]
enabled = true
policy = "capped_broadcast"
graph = "none"
max_artifacts = 3
top_k_neighbors = 3
"""
