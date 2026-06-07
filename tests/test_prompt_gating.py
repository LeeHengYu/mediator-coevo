from __future__ import annotations

from mediated_coevo.agents.mediator import MediatorAgent
from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.core.config import SkillUpdateConfig
from mediated_coevo.llm.client import LLMClient
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace


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


def _all_message_content(messages: list[dict]) -> str:
    return "\n\n".join(str(message["content"]) for message in messages)


def _skill_updates(
    *,
    executor: bool,
    planner: bool = False,
    mediator: bool = False,
) -> SkillUpdateConfig:
    return SkillUpdateConfig(
        executor=executor,
        planner=planner,
        mediator=mediator,
    )


def _plan_messages(planner: PlannerAgent) -> list[dict]:
    return planner.construct_messages(
        {
            "action": "plan_task",
            "task_id": "task-A",
            "base_instruction": "Fix the build.",
        }
    )


def test_planner_omits_executor_update_sections_when_updates_disabled():
    planner = PlannerAgent(LLMClient(model="test-model"))
    planner.configure_skill_updates(_skill_updates(executor=False))
    planner.set_skill_context(
        executor_skills="# Executor Skill",
        skill_refiner=PLANNER_SKILL,
    )

    content = _all_message_content(_plan_messages(planner))

    assert "Your Planning Guidelines" in content
    assert "Task Planning Guidelines" in content
    assert "Artifact-Contract Thinking" in content
    assert "Your Skill-Refinement Guidelines" not in content
    assert "Guidelines for Updating Executor Skills" not in content
    assert "Executor Skill Update Criteria" not in content
    assert "Planner Self-Evolution Guidelines" not in content
    assert "Skill updates are disabled for this run" in content
    assert "When updating skills, edit this content." not in content


def test_planner_includes_executor_update_sections_when_updates_enabled():
    planner = PlannerAgent(LLMClient(model="test-model"))
    planner.configure_skill_updates(_skill_updates(executor=True))
    planner.set_skill_context(
        executor_skills="# Executor Skill",
        skill_refiner=PLANNER_SKILL,
    )

    content = _all_message_content(_plan_messages(planner))

    assert "Your Planning Guidelines" in content
    assert "Guidelines for Updating Executor Skills" in content
    assert "Executor Skill Update Criteria" in content
    assert "Planner Self-Evolution Guidelines" not in content
    assert "When updating skills, edit this content." in content


def test_planner_set_skill_context_clears_stale_refiner():
    planner = PlannerAgent(LLMClient(model="test-model"))
    planner.configure_skill_updates(_skill_updates(executor=True))
    planner.set_skill_context(
        executor_skills="# Executor Skill",
        skill_refiner=PLANNER_SKILL,
    )
    assert "Task Planning Guidelines" in _all_message_content(_plan_messages(planner))

    planner.set_skill_context(executor_skills="# Executor Skill", skill_refiner=None)

    content = _all_message_content(_plan_messages(planner))
    assert "Task Planning Guidelines" not in content
    assert "Your Planning Guidelines" not in content


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


def test_mediator_omits_skill_evolution_sections_when_executor_updates_disabled():
    mediator = MediatorAgent(LLMClient(model="test-model"))
    mediator.configure_skill_updates(_skill_updates(executor=False))
    mediator.load_protocol(MEDIATOR_PROTOCOL)

    system_prompt = str(_mediator_messages(mediator)[0]["content"])

    assert "Abstraction Levels" in system_prompt
    assert "When to Withhold" in system_prompt
    assert "Withhold when the trace is duplicate or noisy." in system_prompt
    assert "Output Format" in system_prompt
    assert "skill-update decisions" not in system_prompt
    assert "skill-evolution hazard" not in system_prompt
    assert "regresses a held-out validation task" not in system_prompt
    assert "Reporting Skill-Evolution Direction" not in system_prompt
    assert "validation evidence" not in system_prompt


def test_mediator_keeps_skill_evolution_sections_when_executor_updates_enabled():
    mediator = MediatorAgent(LLMClient(model="test-model"))
    mediator.configure_skill_updates(_skill_updates(executor=True))
    mediator.load_protocol(MEDIATOR_PROTOCOL)

    system_prompt = str(_mediator_messages(mediator)[0]["content"])

    assert "skill-update decisions" in system_prompt
    assert "skill-evolution hazard" in system_prompt
    assert "Reporting Skill-Evolution Direction" in system_prompt
