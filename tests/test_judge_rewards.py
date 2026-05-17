from __future__ import annotations

import json

import pytest

from mediated_coevo.analysis.judge_rewards import (
    annotate_judge_rewards,
    compute_judge_reward,
)
from mediated_coevo.analysis.reporting import build_score_summary, write_score_summary
from mediated_coevo.core.config import Config
from mediated_coevo.models.history_signals import MediatorSignal
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.judge import JudgeAxisScores, JudgeCapFlags, JudgeLLMResponse
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.stores.history_store import HistoryEntry, HistoryStore


class _FakeJudgeClient:
    model = "openrouter/test/judge"

    def __init__(self, content: str | None = None) -> None:
        self.content = content or json.dumps(
            {
                "axis_scores": {
                    "task_outcome": 0.8,
                    "evidence_quality": 0.6,
                    "skill_update_usefulness": 0.4,
                    "token_efficiency": 0.5,
                    "reflection_depth": 0.2,
                },
                "flags": {
                    "benchmark_gaming_or_obscured_failure": False,
                    "no_meaningful_progress": False,
                    "brittle_or_one_off_patch": False,
                    "unverifiable_outcome": False,
                },
                "confidence": 0.9,
                "rationale": "Trace shows verified partial progress.",
                "flag_evidence": {},
            }
        )
        self.calls = 0

    async def complete(self, *args, **kwargs) -> dict:
        self.calls += 1
        return {
            "content": self.content,
            "input_tokens": 10,
            "output_tokens": 5,
            "model": self.model,
            "raw": {},
        }


def _config() -> Config:
    return Config(
        models={
            "planner": "openrouter/test/planner",
            "executor": "openrouter/test/executor",
            "mediator": "openrouter/test/mediator",
            "judge": "openrouter/test/judge",
        }
    )


def _trace(task_id: str = "task-a", iteration: int = 1) -> ExecutionTrace:
    return ExecutionTrace(
        task_id=task_id,
        iteration=iteration,
        reward=0.5,
        status="ok",
        stdout="passed two checks",
        stderr="",
    )


def test_compute_judge_reward_applies_weights_exactly():
    response = JudgeLLMResponse(
        axis_scores=JudgeAxisScores(
            task_outcome=0.8,
            evidence_quality=0.6,
            skill_update_usefulness=0.4,
            token_efficiency=0.5,
            reflection_depth=0.2,
        ),
        flags=JudgeCapFlags(),
        confidence=0.9,
        rationale="Evidence is inspectable.",
    )

    reward, base_reward, applied_cap = compute_judge_reward(response)

    assert base_reward == pytest.approx(0.63)
    assert reward == pytest.approx(0.63)
    assert applied_cap is None


def test_compute_judge_reward_applies_caps():
    response = JudgeLLMResponse(
        axis_scores=JudgeAxisScores(
            task_outcome=1.0,
            evidence_quality=1.0,
            skill_update_usefulness=1.0,
            token_efficiency=1.0,
            reflection_depth=1.0,
        ),
        flags=JudgeCapFlags(no_meaningful_progress=True),
        confidence=0.8,
        rationale="No real task movement.",
        flag_evidence={"no_meaningful_progress": "Only listed files."},
    )

    reward, base_reward, applied_cap = compute_judge_reward(response)

    assert base_reward == pytest.approx(1.0)
    assert reward == pytest.approx(0.2)
    assert applied_cap == "no_meaningful_progress"


def test_true_flag_requires_evidence():
    with pytest.raises(ValueError, match="require evidence"):
        JudgeLLMResponse(
            axis_scores=JudgeAxisScores(
                task_outcome=0.0,
                evidence_quality=0.0,
                skill_update_usefulness=0.0,
                token_efficiency=0.0,
                reflection_depth=0.0,
            ),
            flags=JudgeCapFlags(unverifiable_outcome=True),
            confidence=0.5,
            rationale="Claim lacks verifier support.",
        )


@pytest.mark.asyncio
async def test_annotate_judge_rewards_writes_sidecar_summary_and_history(tmp_path):
    run_dir = tmp_path / "run"
    trace_dir = run_dir / "artifacts" / "traces"
    trace_dir.mkdir(parents=True)
    trace = _trace()
    trace_path = trace_dir / "task-a_iter0001.json"
    trace_path.write_text(trace.model_dump_json())
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps(
            {
                "iteration": 1,
                "task_id": "task-a",
                "run_id": "run-1",
                "reward": 0.5,
                "verifier_status": "ok",
                "total_tokens": 12,
                "task_category": "skillsbench",
                "task_difficulty": "medium",
                "expected_reward_range": [0.0, 1.0],
                "verifier_type": "pytest",
                "trace_artifact_path": "artifacts/traces/task-a_iter0001.json",
            }
        )
        + "\n"
    )
    write_score_summary(
        build_score_summary(
            [
                IterationRecord(
                    iteration=1,
                    task_id="task-a",
                    reward=0.5,
                    total_tokens=12,
                    execution_trace=trace,
                )
            ],
            bootstrap_samples=50,
        ),
        run_dir / "summary.json",
    )
    history = HistoryStore(run_dir / "history")
    source_entry_id = history.add(
        HistoryEntry(
            iteration=0,
            agent_role="mediator",
            payload=MediatorSignal(headline="useful feedback"),
            metadata={"task_id": "task-a"},
        )
    )
    history.tag_outcome_by_id(
        source_entry_id,
        reward=0.5,
        metadata={
            "verifier_reward": 0.5,
            "outcome_task_id": "task-a",
            "outcome_iteration": 1,
            "trace_status": "ok",
        },
    )

    records = await annotate_judge_rewards(
        data_dir=run_dir,
        config=_config(),
        history_store=history,
        llm_client=_FakeJudgeClient(),
    )

    assert len(records) == 1
    assert records[0].judge_reward == pytest.approx(0.63)
    sidecar_rows = [
        json.loads(line)
        for line in (run_dir / "artifacts" / "judge_rewards.jsonl")
        .read_text()
        .splitlines()
    ]
    assert sidecar_rows[0]["raw_reward"] == pytest.approx(0.5)
    assert sidecar_rows[0]["judge_reward"] == pytest.approx(0.63)

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["mean_reward"] == pytest.approx(0.5)
    assert summary["judge_reward_summary"]["mean_reward"] == pytest.approx(0.63)

    entries = history.query(recent=10)
    assert len(entries) == 1
    assert entries[0].entry_id == source_entry_id
    assert entries[0].reward == pytest.approx(0.5)
    assert entries[0].metadata["judge_reward"] == pytest.approx(0.63)
    assert entries[0].metadata["judge_reward_record_id"] == records[0].record_id
    assert entries[0].metadata["judge_rubric_version"] == "rar-v1"
    assert entries[0].metadata["judge_confidence"] == pytest.approx(0.9)
    assert entries[0].metadata["judge_applied_cap"] is None
    assert "reward_source" not in entries[0].metadata


@pytest.mark.asyncio
async def test_annotate_judge_rewards_processes_frozen_eval_traces(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    trace = _trace(task_id="django__django-11099", iteration=0)
    (eval_dir / "traces.jsonl").write_text(trace.model_dump_json() + "\n")
    write_score_summary(
        build_score_summary(
            [
                IterationRecord(
                    iteration=0,
                    task_id=trace.task_id,
                    reward=trace.reward,
                    execution_trace=trace,
                )
            ],
            bootstrap_samples=50,
        ),
        eval_dir / "summary.json",
    )

    records = await annotate_judge_rewards(
        data_dir=eval_dir,
        config=_config(),
        llm_client=_FakeJudgeClient(),
    )

    assert len(records) == 1
    assert (eval_dir / "artifacts" / "judge_rewards.jsonl").exists()
    summary = json.loads((eval_dir / "summary.json").read_text())
    assert summary["judge_reward_summary"]["scored_count"] == 1
