from mediated_coevo.agents.mediator import MediatorAgent
from mediated_coevo.agents.planner import PlannerAgent


class _LLM:
    model = "test-model"


def test_planner_injects_fixed_skills_as_system_prompts():
    planner = PlannerAgent(_LLM())
    planner.set_skill_context(
        executor_skills="# Executor\n\nFixed executor policy.",
        planner_skill="# Planner\n\nFixed planning policy.",
    )

    messages = planner.construct_messages(
        {
            "action": "plan_task",
            "task_id": "task-a",
            "base_instruction": "Do the task.",
        }
    )

    system_text = "\n".join(
        message["content"] for message in messages if message["role"] == "system"
    )
    assert "Fixed planning policy." in system_text
    assert "Fixed executor policy." in system_text
    assert "immutable skill" in system_text
    assert "modify any agent skill" in system_text


def test_mediator_uses_fixed_skill_as_its_system_prompt():
    mediator = MediatorAgent(_LLM())
    mediator.load_protocol("# Mediator\n\nFixed mediation policy.")

    messages = mediator.construct_messages({})

    assert messages[0] == {
        "role": "system",
        "content": "# Mediator\n\nFixed mediation policy.",
    }
