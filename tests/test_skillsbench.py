from __future__ import annotations

import json

from mediated_coevo.analysis.reporting import build_score_summary
from mediated_coevo.benchmarks.skillsbench import (
    HarborRunResult,
    parse_execution_trace,
)
from mediated_coevo.models.iteration import IterationRecord


def test_harbor_exception_with_reward_counts_as_env_failure(tmp_path):
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "run-agentops__abc123"
    trial_dir.mkdir(parents=True)

    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "job-123",
                "stats": {
                    "evals": {
                        "opencode__model__adhoc": {
                            "metrics": [{"mean": 0.0}],
                        }
                    }
                },
            }
        )
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "trial-123",
                "trial_name": "run-agentops__abc123",
                "agent_result": {
                    "n_input_tokens": 0,
                    "n_output_tokens": 0,
                },
                "exception_info": {
                    "exception_type": "NonZeroAgentExitCodeError",
                    "exception_message": "opencode: command not found",
                },
            }
        )
    )

    trace = parse_execution_trace(
        HarborRunResult(
            job_dir=job_dir,
            trial_dir=trial_dir,
            returncode=0,
            stdout="",
            stderr="",
        ),
        task_id="fix-build-agentops",
        iteration=0,
        duration_sec=1.0,
    )

    assert trace.status == "env_failure"
    assert trace.error_kind == "harbor_exception"
    assert trace.reward == 0.0

    summary = build_score_summary(
        [
            IterationRecord(
                iteration=0,
                task_id="fix-build-agentops",
                reward=trace.reward,
                execution_trace=trace,
            )
        ],
        bootstrap_samples=50,
    )

    assert summary.total_runs == 1
    assert summary.env_failure_count == 1
    assert summary.scored_count == 0
    assert summary.mean_reward is None
