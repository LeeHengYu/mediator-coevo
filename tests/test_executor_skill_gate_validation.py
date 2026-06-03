from __future__ import annotations

from typing import Any

import pytest

from mediated_coevo.core.config import Config, ModelsConfig
from mediated_coevo.evolution.executor_skill_gate import ExecutorSkillGate
from mediated_coevo.models.skill import SkillValidationTaskResult
from tests.config_helpers import diffusion_config, experiment_config


def _validation_gate(*, min_mean_delta: float) -> ExecutorSkillGate:
    config = Config(
        models=ModelsConfig(
            planner="test-planner",
            executor="test-executor",
            mediator="test-mediator",
            judge="test-judge",
        ),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )
    config.experiment.skill_validation.min_mean_delta = min_mean_delta
    unused: Any = object()
    return ExecutorSkillGate(
        config=config,
        skill_store=unused,
        history_store=unused,
        planner=unused,
        skill_advisor=unused,
        executor=unused,
        benchmark_repo=unused,
        artifact_store=unused,
    )


def _task_result(
    task_id: str,
    *,
    current_reward: float,
    candidate_reward: float,
    regressed: bool = False,
) -> SkillValidationTaskResult:
    return SkillValidationTaskResult(
        task_id=task_id,
        current_reward=current_reward,
        candidate_reward=candidate_reward,
        current_status="ok",
        candidate_status="ok",
        usable=True,
        regressed=regressed,
    )


def test_validation_accepts_aggregate_improvement_with_regressed_task() -> None:
    gate = _validation_gate(min_mean_delta=0.01)
    task_results = [
        _task_result(
            "task-A",
            current_reward=0.8,
            candidate_reward=0.7,
            regressed=True,
        ),
        _task_result("task-B", current_reward=0.2, candidate_reward=0.5),
        _task_result("task-C", current_reward=0.3, candidate_reward=0.6),
    ]

    result = gate._validation_decision(
        validation_id="validation-aggregate-improved",
        task_ids=["task-A", "task-B", "task-C"],
        task_results=task_results,
    )
    expected_current_mean = (0.8 + 0.2 + 0.3) / 3
    expected_candidate_mean = (0.7 + 0.5 + 0.6) / 3

    assert result.decision == "accepted"
    assert result.reason == "accepted"
    assert result.current_mean_reward == pytest.approx(expected_current_mean)
    assert result.candidate_mean_reward == pytest.approx(expected_candidate_mean)
    assert result.mean_delta == pytest.approx(
        expected_candidate_mean - expected_current_mean
    )
    assert result.task_results[0].regressed is True


def test_validation_rejects_sum_improvement_that_does_not_clear_threshold() -> None:
    gate = _validation_gate(min_mean_delta=0.25)
    task_results = [
        _task_result("task-A", current_reward=0.0, candidate_reward=0.25),
        _task_result("task-B", current_reward=0.0, candidate_reward=0.25),
    ]

    result = gate._validation_decision(
        validation_id="validation-threshold-not-cleared",
        task_ids=["task-A", "task-B"],
        task_results=task_results,
    )

    assert result.decision == "rejected"
    assert result.reason == "mean_not_improved"
    assert result.current_mean_reward == pytest.approx(0.0)
    assert result.candidate_mean_reward == pytest.approx(0.25)
    assert result.mean_delta == pytest.approx(0.25)
