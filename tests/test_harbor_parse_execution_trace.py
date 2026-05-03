"""Unit tests for parse_execution_trace.

These pin down the contract from P0 #5: every Harbor-side failure mode
must surface as an explicitly classified ExecutionTrace, and the Harbor
job aggregate mean reward is the canonical score for an iteration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mediated_coevo.benchmarks.skillsbench import (
    HarborRunResult,
    parse_execution_trace,
)


_MISSING_JOB_RESULT = object()


def _job_result(mean: Any = 0.75) -> dict[str, Any]:
    return {
        "stats": {
            "evals": {
                "opencode__model__adhoc": {
                    "metrics": [{"mean": mean}],
                }
            }
        }
    }


def _make_trial(
    tmp_path: Path,
    *,
    trial_result_json: dict[str, Any] | str | None = None,
    job_result_json: Any = _MISSING_JOB_RESULT,
    ctrf_json: dict[str, Any] | str | None = None,
    agent_summary: str | None = None,
) -> Path:
    """Build a minimal trial directory; pass None to omit a given file."""
    job_dir = tmp_path / "job-001"
    trial_dir = job_dir / "trial-001"
    trial_dir.mkdir(parents=True)

    if job_result_json is _MISSING_JOB_RESULT:
        job_result_json = _job_result()
    if job_result_json is not None:
        text = (
            job_result_json
            if isinstance(job_result_json, str)
            else json.dumps(job_result_json)
        )
        (job_dir / "result.json").write_text(text)

    if trial_result_json is not None:
        text = (
            trial_result_json
            if isinstance(trial_result_json, str)
            else json.dumps(trial_result_json)
        )
        (trial_dir / "result.json").write_text(text)

    if ctrf_json is not None:
        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir()
        text = ctrf_json if isinstance(ctrf_json, str) else json.dumps(ctrf_json)
        (verifier_dir / "ctrf.json").write_text(text)

    if agent_summary is not None:
        agent_dir = trial_dir / "agent"
        agent_dir.mkdir()
        (agent_dir / "summary.txt").write_text(agent_summary)

    return trial_dir


def _run_result(
    trial_dir: Path | None,
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> HarborRunResult:
    return HarborRunResult(
        job_dir=trial_dir.parent if trial_dir else None,
        trial_dir=trial_dir,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


# ── Happy path ──────────────────────────────────────────────────────────


def test_happy_path_reward_from_job_result_json(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json={
            "agent_result": {"n_input_tokens": 100, "n_output_tokens": 50},
        },
        job_result_json=_job_result(0.75),
        ctrf_json={
            "results": {
                "summary": {"passed": 3, "failed": 0},
                "tests": [{"name": "t1", "status": "passed"}],
            }
        },
        agent_summary="Did the thing.",
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.5)

    assert trace.status == "ok"
    assert trace.error_kind is None
    assert trace.reward == pytest.approx(0.75)
    assert trace.token_usage.input_tokens == 100
    assert trace.token_usage.output_tokens == 50
    assert trace.test_results is not None
    assert trace.test_results["summary"]["passed"] == 3
    assert "Did the thing." in trace.stdout


def test_verifier_dir_is_not_required_for_reward(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json={
            "agent_result": {},
        },
        job_result_json=_job_result(0.6),
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "ok"
    assert trace.reward == pytest.approx(0.6)
    assert trace.test_results is None


# ── Regression: legitimate 0.0 must NOT be silently overridden ─


def test_legitimate_zero_reward_is_preserved(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json={
            "verifier_result": {"rewards": {"reward": 1.0}},
        },
        job_result_json=_job_result(0.0),
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "ok"
    assert trace.reward == pytest.approx(0.0)
    assert trace.error_kind is None


def test_trial_reward_cannot_override_job_result_json(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json={
            "agent_result": {},
            "verifier_result": {"rewards": {"reward": 1.0}},
        },
        job_result_json=_job_result(0.25),
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "ok"
    assert trace.reward == pytest.approx(0.25)


# ── Failure modes ───────────────────────────────────────────────────────


def test_no_trial_dir_yields_env_failure(tmp_path):
    trace = parse_execution_trace(
        _run_result(None, returncode=1, stderr="harbor died"),
        "task-A",
        0,
        0.1,
    )

    assert trace.status == "env_failure"
    assert trace.error_kind == "missing_trial_dir"
    assert trace.reward is None
    assert "harbor died" in (trace.error_detail or "")


def test_missing_trial_result_json_yields_env_failure(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json=None,
        job_result_json=_job_result(1.0),
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "env_failure"
    assert trace.error_kind == "missing_result_json"
    assert trace.reward is None


def test_malformed_trial_result_json_yields_env_failure(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json="{not json",
        job_result_json=_job_result(1.0),
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "env_failure"
    assert trace.error_kind == "malformed_result_json"
    assert trace.reward is None


def test_missing_job_result_json_yields_env_failure(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json={"agent_result": {}},
        job_result_json=None,
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "env_failure"
    assert trace.error_kind == "missing_job_result_json"
    assert trace.reward is None


def test_malformed_job_result_json_yields_env_failure(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json={"agent_result": {}},
        job_result_json="{not json",
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "env_failure"
    assert trace.error_kind == "malformed_job_result_json"
    assert trace.reward is None


def test_missing_job_reward_yields_env_failure(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json={"agent_result": {}},
        job_result_json={"stats": {"evals": {"eval": {"metrics": [{}]}}}},
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "env_failure"
    assert trace.error_kind == "missing_job_reward"
    assert trace.reward is None


def test_malformed_job_reward_yields_env_failure(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json={"agent_result": {}},
        job_result_json=_job_result("not-a-number"),
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "env_failure"
    assert trace.error_kind == "malformed_job_reward"
    assert trace.reward is None


def test_ambiguous_job_reward_yields_env_failure(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json={"agent_result": {}},
        job_result_json={
            "stats": {
                "evals": {
                    "eval-a": {"metrics": [{"mean": 0.2}]},
                    "eval-b": {"metrics": [{"mean": 0.8}]},
                }
            }
        },
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "env_failure"
    assert trace.error_kind == "ambiguous_job_reward"
    assert trace.reward is None


def test_harbor_nonzero_with_reward_keeps_reward_but_marks_harbor_failed(
    tmp_path,
):
    trial = _make_trial(
        tmp_path,
        trial_result_json={"agent_result": {}},
        job_result_json=_job_result(0.4),
    )
    trace = parse_execution_trace(
        _run_result(trial, returncode=1, stderr="warning: weird"),
        "task-A",
        0,
        1.0,
    )

    assert trace.status == "harbor_failed"
    assert trace.error_kind == "harbor_nonzero"
    assert trace.reward == pytest.approx(0.4)


# ── Warning fields (status stays ok) ────────────────────────────────────


def test_exception_info_propagates_into_error_detail(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json={
            "agent_result": {},
            "exception_info": {"type": "RuntimeError", "msg": "boom"},
        },
        job_result_json=_job_result(0.5),
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "ok"
    assert trace.error_kind == "harbor_exception"
    assert trace.error_detail == {"type": "RuntimeError", "msg": "boom"}
    # Exit code is bumped from 0→1 to mark the exception in legacy fields.
    assert trace.exit_code == 1


def test_ctrf_diagnostics_are_copied_without_affecting_reward(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json={"agent_result": {}},
        job_result_json=_job_result(1.0),
        ctrf_json={
            "results": {
                "summary": {"passed": 1, "failed": 1},
                "tests": [
                    {"name": "t1", "status": "passed"},
                    {"name": "t2", "status": "failed", "message": "oops"},
                ],
            }
        },
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "ok"
    assert trace.error_kind is None
    assert trace.reward == pytest.approx(1.0)
    assert trace.test_results is not None
    assert any(
        t["name"] == "t2" for t in trace.test_results["failed_tests"]
    )


def test_ctrf_diagnostics_do_not_override_zero_reward(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json={"agent_result": {}},
        job_result_json=_job_result(0.0),
        ctrf_json={
            "results": {
                "summary": {"passed": 3, "failed": 0},
                "tests": [{"name": "t1", "status": "passed"}],
            }
        },
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "ok"
    assert trace.error_kind is None
    assert trace.reward == pytest.approx(0.0)
    assert trace.test_results is not None
    assert trace.test_results["summary"]["passed"] == 3


def test_malformed_token_counts_do_not_escape(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json={
            "agent_result": {
                "n_input_tokens": "not-an-int",
                "n_output_tokens": "also-not-an-int",
            }
        },
        job_result_json=_job_result(0.5),
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "ok"
    assert trace.reward == pytest.approx(0.5)
    assert trace.token_usage.input_tokens == 0
    assert trace.token_usage.output_tokens == 0


def test_malformed_ctrf_summary_counts_do_not_escape(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json={"agent_result": {}},
        job_result_json=_job_result(1.0),
        ctrf_json={
            "results": {
                "summary": {"passed": "several", "failed": "many"},
                "tests": [{"name": "t1", "status": "passed"}],
            }
        },
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "ok"
    assert trace.reward == pytest.approx(1.0)
    assert trace.error_kind is None


def test_malformed_ctrf_json_is_ignored_for_score(tmp_path):
    trial = _make_trial(
        tmp_path,
        trial_result_json={"agent_result": {}},
        job_result_json=_job_result(0.8),
        ctrf_json="{not json",
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "ok"
    assert trace.reward == pytest.approx(0.8)
    assert trace.error_kind is None
    assert trace.test_results is None
