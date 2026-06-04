from __future__ import annotations

from mediated_coevo.agents.planner import PlannerAgent, UPDATE_BATCH_RESPONSE_SCHEMA
from mediated_coevo.evolution.reflector import Reflector


def test_executor_candidate_prompt_requires_integrated_rewrite():
    assert "complete, semantically integrated rewrite" in UPDATE_BATCH_RESPONSE_SCHEMA
    assert "do not append an addendum" in UPDATE_BATCH_RESPONSE_SCHEMA


def test_reflector_prompts_require_integrated_rewrites():
    mediator_system = Reflector._build_mediator_prompt("", [])[0]["content"]
    planner_system = Reflector._build_planner_prompt("", [])[0]["content"]

    assert "integrate changes into existing sections" in mediator_system
    assert "avoid appended addenda" in mediator_system
    assert "integrate changes into existing sections" in planner_system
    assert "avoid appended addenda" in planner_system


def test_reflector_prompts_describe_same_iteration_outcome_evidence():
    mediator_messages = Reflector._build_mediator_prompt("", [])
    planner_messages = Reflector._build_planner_prompt("", [])
    prompt_text = "\n".join(
        message["content"] for message in mediator_messages + planner_messages
    )

    assert "same-iteration same-task outcome reward" in prompt_text
    assert "associated with better vs. worse same-iteration same-task" in prompt_text
    assert "led to better" not in prompt_text
    assert "downstream" not in prompt_text


def test_planner_batch_prompt_includes_rejected_update_history():
    prompt = PlannerAgent._build_update_prompt(
        {
            "current_skill": "# Executor\n",
            "feedback": "add a broader parsing rule",
            "task_ids": ["task-A"],
            "rejected_update_history": [
                {
                    "batch_id": "batch-old",
                    "reason": "validation: task_regression",
                    "validation": {
                        "reason": "task_regression",
                        "regressed_task_ids": ["heldout-A"],
                    },
                }
            ],
        },
        response_schema=UPDATE_BATCH_RESPONSE_SCHEMA,
        batch_mode=True,
    )

    assert "Recently Rejected Skill Updates" in prompt
    assert "Treat these as negative evidence" in prompt
    assert "task_regression" in prompt
    assert "heldout-A" in prompt
