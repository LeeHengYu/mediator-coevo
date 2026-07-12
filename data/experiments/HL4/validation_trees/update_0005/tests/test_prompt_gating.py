from __future__ import annotations

import pytest

from mediated_coevo.agents.mediator import MediatorAgent
from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.agents.prompt_context import PromptSection
from mediated_coevo.core.config import SkillUpdateConfig
from mediated_coevo.llm.client import LLMClient
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.prompt_text import PromptText
from tests.prompt_helpers import assert_contains_all, assert_omits_all, message_text


PLANNER_SKILL = """\
## Task Planning Guidelines

Plan concrete task instructions.

## Guidelines for Updating Executor Skills

Update Executor skills from reusable evidence.

## Executor Skill Update Criteria

Only update the Executor skill when the lesson generalizes.

## Artifact-Contract Thinking

Treat artifacts as interfaces with contracts.

## Planner Self-Evolution Guidelines

Revise this Planner skill from contrastive pairs.

## Self-Evolution Edit Style

Make the smallest integrated change.
"""


MEDIATOR_PROTOCOL = """\
---
name: mediator
description: Used to help the Planner make better skill-update decisions.
---

## Abstraction Levels

Choose trace, reflection, or pattern.

## When to Withhold

Withhold when the trace is duplicate or noisy.

Do not withhold when the run exposes a skill-evolution hazard, such as a
candidate that appears broadly helpful but regresses a held-out validation task.

## Reporting Skill-Evolution Direction

When an update is rejected by validation, report the validation evidence before
new fix ideas.

## Output Format

Respond with JSON.
"""


def _skill_updates(
    *,
    executor: bool,
) -> SkillUpdateConfig:
    return SkillUpdateConfig(
        executor=executor,
        planner=False,
        mediator=False,
    )


def _planner_content(*, executor_updates_enabled: bool) -> str:
    planner = PlannerAgent(LLMClient(model="test-model"))
    planner.configure_skill_updates(_skill_updates(executor=executor_updates_enabled))
    planner.set_skill_context(
        executor_skills="# Executor Skill",
        skill_refiner=PLANNER_SKILL,
    )
    return _plan_content(planner)


def _plan_content(planner: PlannerAgent) -> str:
    return message_text(
        planner.construct_messages(
            {
                "action": "plan_task",
                "task_id": "task-A",
                "base_instruction": "Fix the build.",
            }
        )
    )


@pytest.mark.parametrize(
    ("executor_updates_enabled", "expected", "unexpected"),
    [
        pytest.param(
            False,
            [
                PromptText.SKILL_REFINER_PLANNING_HEADING,
                "Task Planning Guidelines",
                "Artifact-Contract Thinking",
                "Skill updates are disabled for this run",
            ],
            [
                PromptText.SKILL_REFINER_UPDATE_HEADING,
                "Guidelines for Updating Executor Skills",
                "Executor Skill Update Criteria",
                "Planner Self-Evolution Guidelines",
                "When updating skills, edit this content.",
            ],
            id="disabled",
        ),
        pytest.param(
            True,
            [
                PromptText.SKILL_REFINER_PLANNING_HEADING,
                "Guidelines for Updating Executor Skills",
                "Executor Skill Update Criteria",
                "When updating skills, edit this content.",
            ],
            [
                PromptText.SKILL_REFINER_UPDATE_HEADING,
                "Planner Self-Evolution Guidelines",
            ],
            id="enabled",
        ),
    ],
)
def test_planner_executor_update_sections_follow_runtime_gate(
    executor_updates_enabled,
    expected,
    unexpected,
):
    content = _planner_content(executor_updates_enabled=executor_updates_enabled)

    assert_contains_all(content, expected)
    assert_omits_all(content, unexpected)


def test_planner_set_skill_context_clears_stale_refiner():
    planner = PlannerAgent(LLMClient(model="test-model"))
    planner.configure_skill_updates(_skill_updates(executor=True))
    planner.set_skill_context(
        executor_skills="# Executor Skill",
        skill_refiner=PLANNER_SKILL,
    )
    assert_contains_all(_plan_content(planner), ["Task Planning Guidelines"])

    planner.set_skill_context(executor_skills="# Executor Skill", skill_refiner=None)

    content = _plan_content(planner)
    assert_omits_all(
        content,
        ["Task Planning Guidelines", PromptText.SKILL_REFINER_PLANNING_HEADING],
    )


def test_planner_plan_prompt_accepts_explicit_prior_context_sections():
    planner = PlannerAgent(LLMClient(model="test-model"))

    messages = planner.construct_messages(
        {
            "action": "plan_task",
            "task_id": "task-A",
            "base_instruction": "Fix the build.",
            "prior_context_sections": [
                PromptSection(
                    "same_task_prior",
                    "same_task_prior",
                    "## Same-Task Prior\nsame task report",
                ),
                PromptSection(
                    "diffusion_context",
                    "diffusion_context",
                    "## Diffused Cross-Task Context\ndiffusion hint",
                ),
            ],
            "prior_context": "flat prior should not appear",
        }
    )

    content = message_text(messages)

    assert_contains_all(
        content,
        [
            "## Same-Task Prior",
            "same task report",
            "## Diffused Cross-Task Context",
            "diffusion hint",
        ],
    )
    assert_omits_all(content, ["flat prior should not appear"])


def _mediator_messages(mediator: MediatorAgent) -> list[dict]:
    return mediator.construct_messages(
        {
            "trace": ExecutionTrace(
                task_id="task-A",
                iteration=1,
                status="ok",
                reward=0.0,
            ),
            "task_context": TaskSpec(
                task_id="task-A",
                instruction="Fix the build.",
                iteration=1,
            ),
        }
    )


@pytest.mark.parametrize(
    ("executor_updates_enabled", "expected", "unexpected"),
    [
        pytest.param(
            False,
            [
                "Abstraction Levels",
                "When to Withhold",
                "Withhold when the trace is duplicate or noisy.",
                "Output Format",
            ],
            [
                "skill-update decisions",
                "skill-evolution hazard",
                "regresses a held-out validation task",
                "Reporting Skill-Evolution Direction",
                "validation evidence",
            ],
            id="disabled",
        ),
        pytest.param(
            True,
            [
                "skill-update decisions",
                "skill-evolution hazard",
                "Reporting Skill-Evolution Direction",
            ],
            [],
            id="enabled",
        ),
    ],
)
def test_mediator_protocol_sections_follow_executor_update_gate(
    executor_updates_enabled,
    expected,
    unexpected,
):
    mediator = MediatorAgent(LLMClient(model="test-model"))
    mediator.configure_skill_updates(_skill_updates(executor=executor_updates_enabled))
    mediator.load_protocol(MEDIATOR_PROTOCOL)

    system_prompt = str(_mediator_messages(mediator)[0]["content"])

    assert_contains_all(system_prompt, expected)
    assert_omits_all(system_prompt, unexpected)
