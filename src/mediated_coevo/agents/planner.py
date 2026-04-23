"""Planner agent — Claude.

Plans tasks for the Executor and decides skill updates based on
feedback from the Mediator (or raw traces, depending on condition).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base import BaseAgent

if TYPE_CHECKING:
    from mediated_coevo.llm.client import LLMClient
    from mediated_coevo.models.skill import SkillProposal
    from mediated_coevo.models.task import TaskSpec
    from mediated_coevo.stores.history_store import HistoryEntry

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are the Planner in a multi-agent skill co-evolution system.

Your responsibilities:
1. Plan tasks for the Executor agent to carry out.
2. Read feedback reports (from the Mediator or raw traces) about past executions.
3. Decide whether and how to update the Executor's skills based on that feedback.

You do NOT execute tasks yourself. You plan and refine skills."""


class PlannerAgent(BaseAgent):
    """Claude-backed planner. Plans tasks and decides skill updates."""

    @property
    def role(self) -> str:
        return "planner"

    def __init__(
        self,
        llm_client: LLMClient,
    ) -> None:
        super().__init__("planner", llm_client)
        self._skill_context: str | None = None
        self._skill_refiner: str | None = None

    def set_skill_context(self, executor_skills: str, skill_refiner: str | None = None) -> None:
        """Inject executor skills and planner's own skill-refiner guidance.

        Called by the orchestrator before process().
        """
        self._skill_context = executor_skills or None
        if skill_refiner is not None:
            self._skill_refiner = skill_refiner

    def construct_messages(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        ]

        # Skill injection — separate system messages like OpenSpace
        if self._skill_refiner:
            messages.append({"role": "system", "content": (
                "# Your Skill-Refinement Guidelines\n\n"
                "The following skill provides **procedures for updating the "
                "Executor's skills**. Follow these when deciding skill edits.\n\n"
                f"{self._skill_refiner}"
            )})

        if self._skill_context:
            messages.append({"role": "system", "content": (
                "# Executor's Active Skills\n\n"
                "The following skills are currently loaded into the Executor. "
                "When planning tasks, reference these capabilities. When "
                "updating skills, edit this content.\n\n"
                f"{self._skill_context}"
            )})

        action = context.get("action", "plan_task")
        if action == "plan_task":
            user_content = self._build_plan_prompt(context)
        elif action == "update_skill":
            user_content = self._build_update_prompt(context)
        else:
            user_content = context.get("instruction", "")

        messages.append({"role": "user", "content": user_content})
        return messages

    async def process(self, context: dict[str, Any]) -> dict[str, Any]:
        messages = self.construct_messages(context)
        response = await self.get_llm_response(messages)
        self.increment_step()
        return {
            "content": response["content"],
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
            "parsed": self.response_to_dict(response["content"]),
        }

    # ── Convenience wrappers ──

    async def plan_task(
        self,
        task_id: str,
        base_instruction: str,
        prior_context: str | None,
        current_skills: list[str],
    ) -> TaskSpec:
        from mediated_coevo.models.task import TaskSpec

        context: dict[str, Any] = {
            "action": "plan_task",
            "task_id": task_id,
            "base_instruction": base_instruction,
            "current_skills": current_skills,
        }
        if prior_context:
            context["mediator_report"] = prior_context

        result = await self.process(context)
        parsed = result["parsed"]
        return TaskSpec(
            task_id=task_id,
            instruction=parsed.get("instruction", base_instruction),
            skills_context=current_skills,
            planner_reasoning=parsed.get("reasoning"),
            iteration=self.step,
        )

    async def propose_skill_update(
        self,
        current_skill_content: str,
        feedback: str | None,
        edit_history: list[HistoryEntry],
        task_id: str = "",
        iteration: int = 0,
    ) -> SkillProposal | None:
        """Propose a skill update without writing to disk.

        Returns a SkillProposal for the advisor buffer, or None if the
        Planner decides no change is needed.
        """
        from mediated_coevo.models.skill import SkillProposal

        context: dict[str, Any] = {
            "action": "update_skill",
            "current_skill": current_skill_content,
            "feedback": feedback,
            "edit_history": [
                {
                    "iteration": e.iteration,
                    "reasoning": getattr(e.payload, "reasoning", ""),
                    "reward": e.reward,
                }
                for e in edit_history[-5:]
            ],
        }
        result = await self.process(context)
        parsed = result["parsed"]

        if parsed.get("no_update") or parsed.get("error"):
            return None

        new_content = parsed.get("new_content", "")
        if not new_content:
            return None

        return SkillProposal(
            iteration=iteration,
            task_id=task_id,
            old_content=current_skill_content,
            new_content=new_content,
            reasoning=parsed.get("reasoning", ""),
        )

    # ── Prompt builders ──

    @staticmethod
    def _build_plan_prompt(context: dict[str, Any]) -> str:
        parts = [f"Plan a task for task_id: {context.get('task_id', 'unknown')}"]
        if instruction := context.get("base_instruction"):
            parts.append(
                "\n## Benchmark Instruction\n"
                "Use the following as the base task instruction. You may clarify or "
                "restructure it for the Executor, but do not change the task goal.\n\n"
                f"{instruction}"
            )
        if report := context.get("mediator_report"):
            parts.append(f"\n## Feedback from previous execution\n{report}")
        parts.append(
            "\nRespond with JSON: "
            '{"instruction": "...", "reasoning": "..."}'
        )
        return "\n".join(parts)

    @staticmethod
    def _build_update_prompt(context: dict[str, Any]) -> str:
        parts = [
            "## Current Skill Content",
            context.get("current_skill", "(none)"),
        ]
        if feedback := context.get("feedback"):
            parts.append(f"\n## Execution Feedback\n{feedback}")
        if history := context.get("edit_history"):
            parts.append(f"\n## Recent Edit History\n{history}")
        parts.append(
            "\nDecide whether to update the skill. Respond with JSON:\n"
            '{"no_update": true} if no change needed, or\n'
            '{"new_content": "...", "reasoning": "..."}'
        )
        return "\n".join(parts)
