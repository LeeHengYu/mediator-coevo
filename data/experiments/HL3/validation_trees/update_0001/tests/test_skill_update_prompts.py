from __future__ import annotations

import pytest

from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.evolution.reflector import Reflector
from mediated_coevo.prompt_text import PromptText
from tests.prompt_helpers import assert_contains_all


@pytest.mark.parametrize(
    ("prompt_text", "expected"),
    [
        pytest.param(
            PromptText.UPDATE_BATCH_RESPONSE_SCHEMA,
            [
                "complete, semantically integrated rewrite",
                "do not append an addendum",
            ],
            id="executor_batch",
        ),
        pytest.param(
            Reflector._build_mediator_prompt("", [])[0]["content"],
            [
                "integrate changes into existing sections",
                "avoid appended addenda",
            ],
            id="mediator_reflection",
        ),
        pytest.param(
            Reflector._build_planner_prompt("", [])[0]["content"],
            [
                "integrate changes into existing sections",
                "avoid appended addenda",
            ],
            id="planner_reflection",
        ),
    ],
)
def test_skill_update_prompts_require_integrated_rewrites(prompt_text, expected):
    assert_contains_all(prompt_text, expected)


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
        response_schema=PromptText.UPDATE_BATCH_RESPONSE_SCHEMA,
        batch_mode=True,
    )

    assert_contains_all(
        prompt,
        [
            "Recently Rejected Skill Updates",
            "Treat these as negative evidence",
            "task_regression",
            "heldout-A",
        ],
    )
