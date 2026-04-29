"""Planner agent — Claude.

Plans tasks for the Executor and decides skill updates based on
feedback from the Mediator (or raw traces, depending on condition).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base import BaseAgent

if TYPE_CHECKING:
    from mediated_coevo.config import BudgetsConfig
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
        self._budgets: BudgetsConfig | None = None
        self._condition_name: str | None = None

    def configure_token_budget(
        self,
        budgets: BudgetsConfig,
        *,
        condition_name: str | None = None,
    ) -> None:
        self._budgets = budgets
        self._condition_name = condition_name

    def set_skill_context(self, executor_skills: str, skill_refiner: str | None = None) -> None:
        """Inject executor skills and planner's own skill-refiner guidance.

        Called by the orchestrator before process().
        """
        self._skill_context = executor_skills or None
        if skill_refiner is not None:
            self._skill_refiner = skill_refiner

    def _append_budgeted_system_context(
        self,
        messages: list[dict[str, Any]],
        *,
        heading: str,
        description: str,
        content: str,
    ) -> None:
        """Append optional skill context, respecting the configured token cap."""
        if self._budgets:
            from mediated_coevo.token_budget import fit_text_to_tokens

            content = fit_text_to_tokens(
                self.llm_client.model,
                content,
                self._budgets.max_skill_tokens,
            )
        messages.append({
            "role": "system",
            "content": f"# {heading}\n\n{description}\n\n{content}",
        })

    def construct_messages(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        ]

        from mediated_coevo.token_budget import count_message_tokens

        model = self.llm_client.model

        # Skill injection — separate system messages like OpenSpace
        if self._skill_refiner:
            self._append_budgeted_system_context(
                messages,
                heading="Your Skill-Refinement Guidelines",
                description=(
                    "The following skill provides **procedures for updating the "
                    "Executor's skills**. Follow these when deciding skill edits."
                ),
                content=self._skill_refiner,
            )

        if self._skill_context:
            self._append_budgeted_system_context(
                messages,
                heading="Executor's Active Skills",
                description=(
                    "The following skills are currently loaded into the Executor. "
                    "When planning tasks, reference these capabilities. When "
                    "updating skills, edit this content."
                ),
                content=self._skill_context,
            )

        action = context.get("action", "plan_task")
        user_budget = None
        if self._budgets:
            system_tokens = count_message_tokens(model, messages)
            user_budget = max(1, self._budgets.planner_context_tokens - system_tokens)
        if action == "plan_task":
            user_content = self._build_plan_prompt(context, model=model, budgets=self._budgets, budget=user_budget)
        elif action == "update_skill":
            user_content = self._build_update_prompt(context, model=model, budgets=self._budgets, budget=user_budget)
        else:
            user_content = context.get("instruction", "")

        messages.append({"role": "user", "content": user_content})
        return messages

    async def process(self, context: dict[str, Any]) -> dict[str, Any]:
        messages = self.construct_messages(context)
        action = context.get("action", "process")
        kwargs: dict[str, Any] = {}
        if self._budgets:
            kwargs = {
                "max_tokens": self._budgets.planner_completion_tokens,
                "prompt_budget": self._budgets.planner_context_tokens,
                "budget_label": f"planner.{action}",
                "budget_overflow_strategy": "section_pack",
                "condition_name": self._condition_name,
            }
        response = await self.get_llm_response(messages, **kwargs)
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
        iteration: int,
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
            iteration=iteration,
        )

    async def register_skill_update(
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
        from mediated_coevo.models.history_signals import PlannerSignal
        from mediated_coevo.models.skill import SkillProposal

        context: dict[str, Any] = {
            "action": "update_skill",
            "current_skill": current_skill_content,
            "feedback": feedback,
            "edit_history": [
                {
                    "iteration": e.iteration,
                    "reasoning": (
                        e.payload.reasoning
                        if isinstance(e.payload, PlannerSignal)
                        else ""
                    ),
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
    def _build_plan_prompt(
        context: dict[str, Any],
        *,
        model: str = "",
        budgets: BudgetsConfig | None = None,
        budget: int | None = None,
    ) -> str:
        if budgets and budget:
            from mediated_coevo.token_budget import BudgetSection, pack_sections

            sections = [
                BudgetSection(
                    "task_header",
                    f"Plan a task for task_id: {context.get('task_id', 'unknown')}",
                    required=True,
                )
            ]
            if instruction := context.get("base_instruction"):
                sections.append(BudgetSection(
                    "benchmark_instruction",
                    (
                        "## Benchmark Instruction\n"
                        "Use the following as the base task instruction. You may clarify or "
                        "restructure it for the Executor, but do not change the task goal.\n\n"
                        f"{instruction}"
                    ),
                    required=True,
                ))
            if report := context.get("mediator_report"):
                sections.append(BudgetSection(
                    "prior_context",
                    f"## Feedback from previous execution\n{report}",
                    max_tokens=budgets.mediator_report_tokens,
                ))
            sections.append(BudgetSection(
                "response_schema",
                'Respond with JSON: {"instruction": "...", "reasoning": "..."}',
                required=True,
            ))
            return pack_sections(model, sections, budget)

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
    def _build_update_prompt(
        context: dict[str, Any],
        *,
        model: str = "",
        budgets: BudgetsConfig | None = None,
        budget: int | None = None,
    ) -> str:
        if budgets and budget:
            from mediated_coevo.token_budget import BudgetSection, pack_sections

            sections = [
                BudgetSection(
                    "current_skill",
                    "## Current Skill Content\n"
                    f"{context.get('current_skill', '(none)')}",
                    required=True,
                    max_tokens=budgets.max_skill_tokens,
                )
            ]
            if feedback := context.get("feedback"):
                sections.append(BudgetSection(
                    "execution_feedback",
                    f"## Execution Feedback\n{feedback}",
                    max_tokens=budgets.mediator_report_tokens,
                ))
            if history := context.get("edit_history"):
                sections.append(BudgetSection(
                    "recent_edit_history",
                    f"## Recent Edit History\n{history}",
                    max_tokens=budgets.historical_summary_tokens,
                ))
            sections.append(BudgetSection(
                "response_schema",
                (
                    "Decide whether to update the skill. Respond with JSON:\n"
                    '{"no_update": true} if no change needed, or\n'
                    '{"new_content": "...", "reasoning": "..."}'
                ),
                required=True,
            ))
            return pack_sections(model, sections, budget)

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
