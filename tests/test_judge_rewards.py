import pytest
from pydantic import ValidationError

from mediated_coevo.analysis.judge_rewards import (
    compute_judge_reward,
    judge_reward_for_trace,
)
from mediated_coevo.core.config import Config
from mediated_coevo.models.judge import (
    JudgeAxisScores,
    JudgeCapFlags,
    JudgeLLMResponse,
)
from mediated_coevo.models.trace import ExecutionTrace
from tests.config_helpers import (
    budgets_config,
    diffusion_config,
    experiment_config,
    models_config,
)


def _config() -> Config:
    return Config(
        models=models_config(),
        budgets=budgets_config(),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )


def test_compute_judge_reward_uses_fixed_skill_axes():
    response = JudgeLLMResponse(
        axis_scores=JudgeAxisScores(
            task_outcome=0.9,
            evidence_quality=0.8,
            token_efficiency=0.7,
        ),
        confidence=0.8,
        rationale="Supported by the trace.",
    )

    reward, base_reward, cap = compute_judge_reward(response)

    expected = (2 / 3) * 0.9 + (1 / 5) * 0.8 + (2 / 15) * 0.7
    assert reward == pytest.approx(expected)
    assert base_reward == pytest.approx(expected)
    assert cap is None


def test_compute_judge_reward_applies_caps():
    response = JudgeLLMResponse(
        axis_scores=JudgeAxisScores(
            task_outcome=1.0,
            evidence_quality=1.0,
            token_efficiency=1.0,
        ),
        flags=JudgeCapFlags(no_meaningful_progress=True),
        confidence=0.8,
        rationale="No meaningful progress.",
        flag_evidence={"no_meaningful_progress": "No output changed."},
    )

    reward, base_reward, cap = compute_judge_reward(response)

    assert base_reward == pytest.approx(1.0)
    assert reward == pytest.approx(0.2)
    assert cap == "no_meaningful_progress"


def test_true_flag_requires_evidence():
    with pytest.raises(ValidationError):
        JudgeLLMResponse(
            axis_scores=JudgeAxisScores(
                task_outcome=0.5,
                evidence_quality=0.5,
                token_efficiency=0.5,
            ),
            flags=JudgeCapFlags(unverifiable_outcome=True),
            confidence=0.5,
            rationale="Missing evidence.",
        )


@pytest.mark.asyncio
async def test_live_judge_reward_falls_back_without_client():
    trace = ExecutionTrace(
        task_id="task-a",
        iteration=1,
        status="ok",
        reward=0.75,
    )

    record = await judge_reward_for_trace(
        trace=trace,
        config=_config(),
        llm_client=None,
    )

    assert record is not None
    assert record.judge_reward == pytest.approx(0.75)
    assert record.metadata["judge_reward_fallback"] is True
    assert record.axis_scores.model_dump() == {
        "task_outcome": 0.75,
        "evidence_quality": 0.0,
        "token_efficiency": 0.0,
    }
