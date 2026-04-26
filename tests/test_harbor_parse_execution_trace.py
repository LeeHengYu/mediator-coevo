"""Unit tests for parse_execution_trace.

These pin down the contract from P0 #5: every Harbor-side failure mode
must surface as an explicitly classified ExecutionTrace, and a legitimate
reward of 0.0 must never be silently overridden.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mediated_coevo.benchmarks.skillsbench import (
    HarborRunResult,
    parse_execution_trace,
)


def _make_trial(
    tmp_path: Path,
    *,
    result_json: dict | str | None = None,
    reward_txt: str | None = None,
    ctrf_json: dict | str | None = None,
    agent_summary: str | None = None,
) -> Path:
    """Build a minimal trial directory; pass None to omit a given file."""
    trial_dir = tmp_path / "job-001" / "trial-001"
    trial_dir.mkdir(parents=True)

    if result_json is not None:
        text = (
            result_json if isinstance(result_json, str) else json.dumps(result_json)
        )
        (trial_dir / "result.json").write_text(text)

    if reward_txt is not None or ctrf_json is not None:
        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir()
        if reward_txt is not None:
            (verifier_dir / "reward.txt").write_text(reward_txt)
        if ctrf_json is not None:
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


def test_happy_path_reward_from_reward_txt(tmp_path):
    trial = _make_trial(
        tmp_path,
        result_json={
            "agent_result": {"n_input_tokens": 100, "n_output_tokens": 50},
        },
        reward_txt="0.75",
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


def test_reward_falls_back_to_result_json_when_reward_txt_missing(tmp_path):
    trial = _make_trial(
        tmp_path,
        result_json={
            "verifier_result": {"rewards": {"reward": 0.6}},
        },
        reward_txt=None,
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "ok"
    assert trace.reward == pytest.approx(0.6)


# ── Regression: legitimate 0.0 must NOT trigger the result.json fallback ─


def test_legitimate_zero_reward_is_preserved(tmp_path):
    """The old code overwrote a real 0.0 from reward.txt with whatever was
    nested in result.json. This regression test pins the fix.
    """
    trial = _make_trial(
        tmp_path,
        result_json={
            "verifier_result": {"rewards": {"reward": 1.0}},  # would-be override
        },
        reward_txt="0.0",
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "ok"
    assert trace.reward == pytest.approx(0.0)
    assert trace.error_kind is None


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


def test_missing_result_json_yields_env_failure(tmp_path):
    trial = _make_trial(tmp_path, result_json=None, reward_txt="1.0")
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "env_failure"
    assert trace.error_kind == "missing_result_json"
    assert trace.reward is None


def test_malformed_result_json_yields_env_failure(tmp_path):
    trial = _make_trial(tmp_path, result_json="{not json", reward_txt="1.0")
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "env_failure"
    assert trace.error_kind == "malformed_result_json"
    assert trace.reward is None


def test_malformed_reward_txt_yields_parse_error(tmp_path):
    trial = _make_trial(
        tmp_path,
        result_json={"agent_result": {}},
        reward_txt="not-a-number",
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "parse_error"
    assert trace.error_kind == "malformed_reward"
    assert trace.reward is None
    # The bare ValueError must NOT escape.


def test_no_reward_source_yields_env_failure(tmp_path):
    trial = _make_trial(
        tmp_path,
        result_json={"agent_result": {}},
        reward_txt=None,
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "env_failure"
    assert trace.error_kind == "no_reward"
    assert trace.reward is None


def test_harbor_nonzero_with_no_reward_yields_harbor_failed(tmp_path):
    trial = _make_trial(
        tmp_path,
        result_json={"agent_result": {}},
        reward_txt=None,
    )
    trace = parse_execution_trace(
        _run_result(trial, returncode=2, stderr="harbor crashed"),
        "task-A",
        0,
        1.0,
    )

    assert trace.status == "harbor_failed"
    assert trace.error_kind == "harbor_nonzero_no_reward"
    assert trace.reward is None
    assert trace.exit_code == 2


def test_harbor_nonzero_with_reward_keeps_reward_but_marks_harbor_failed(
    tmp_path,
):
    trial = _make_trial(
        tmp_path,
        result_json={"agent_result": {}},
        reward_txt="0.4",
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
        result_json={
            "agent_result": {},
            "exception_info": {"type": "RuntimeError", "msg": "boom"},
        },
        reward_txt="0.5",
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "ok"
    assert trace.error_kind == "harbor_exception"
    assert trace.error_detail == {"type": "RuntimeError", "msg": "boom"}
    # Exit code is bumped from 0→1 to mark the exception in legacy fields.
    assert trace.exit_code == 1


def test_ctrf_failed_tests_with_perfect_reward_flags_mismatch(tmp_path):
    trial = _make_trial(
        tmp_path,
        result_json={"agent_result": {}},
        reward_txt="1.0",
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

    assert trace.status == "ok"  # sanity flag, not a failure
    assert trace.error_kind == "reward_ctrf_mismatch"
    assert trace.test_results is not None
    assert any(
        t["name"] == "t2" for t in trace.test_results["failed_tests"]
    )


def test_ctrf_all_passed_with_zero_reward_flags_mismatch(tmp_path):
    trial = _make_trial(
        tmp_path,
        result_json={"agent_result": {}},
        reward_txt="0.0",
        ctrf_json={
            "results": {
                "summary": {"passed": 3, "failed": 0},
                "tests": [{"name": "t1", "status": "passed"}],
            }
        },
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "ok"
    assert trace.error_kind == "reward_ctrf_mismatch"
    assert trace.reward == pytest.approx(0.0)


def test_malformed_token_counts_do_not_escape(tmp_path):
    trial = _make_trial(
        tmp_path,
        result_json={
            "agent_result": {
                "n_input_tokens": "not-an-int",
                "n_output_tokens": "also-not-an-int",
            }
        },
        reward_txt="0.5",
    )
    trace = parse_execution_trace(_run_result(trial), "task-A", 0, 1.0)

    assert trace.status == "ok"
    assert trace.reward == pytest.approx(0.5)
    assert trace.token_usage.input_tokens == 0
    assert trace.token_usage.output_tokens == 0


def test_malformed_ctrf_summary_counts_do_not_escape(tmp_path):
    trial = _make_trial(
        tmp_path,
        result_json={"agent_result": {}},
        reward_txt="1.0",
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
