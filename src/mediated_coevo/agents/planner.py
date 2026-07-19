"""Planner agent with a fixed prompt-injected planning skill."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mediated_coevo.prompt_text import PromptText

from .base import BaseAgent
from .prompt_context import PromptSection

if TYPE_CHECKING:
    from mediated_coevo.core.config import BudgetsConfig
    from mediated_coevo.llm.client import LLMClient
    from mediated_coevo.models.task import TaskSpec

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = PromptText.PLANNER_SYSTEM
PLAN_RESPONSE_SCHEMA = PromptText.PLAN_RESPONSE_SCHEMA


class PlannerAgent(BaseAgent):
    """Claude-backed planner using immutable prompt-injected skills."""

    def __init__(
        self,
        llm_client: LLMClient,
    ) -> None:
        super().__init__("planner", llm_client)
        self._skill_context: str | None = None
        self._planner_skill: str | None = None
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

    def set_skill_context(
        self, executor_skills: str, planner_skill: str | None = None
    ) -> None:
        """Inject the fixed Planner and Executor skills as prompt context."""
        self._skill_context = executor_skills or None
        self._planner_skill = planner_skill or None

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
            from mediated_coevo.runtime.token_budget import fit_text_to_tokens

            content = fit_text_to_tokens(
                self.llm_client.model,
                content,
                self._budgets.max_skill_tokens,
            )
        section = PromptSection(
            name=heading.lower().replace(" ", "_"),
            kind="scaffold",
            content=PromptText.system_context(heading, description, content),
        )
        messages.append(
            {
                "role": "system",
                "content": section.content,
            }
        )

    def construct_messages(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": PromptText.PLANNER_SYSTEM},
        ]

        model = self.llm_client.model
        if self._planner_skill:
            self._append_budgeted_system_context(
                messages,
                heading=PromptText.PLANNER_SKILL_HEADING,
                description=PromptText.PLANNER_SKILL_DESCRIPTION,
                content=self._planner_skill,
            )

        if self._skill_context:
            self._append_budgeted_system_context(
                messages,
                heading=PromptText.EXECUTOR_ACTIVE_SKILLS_HEADING,
                description=PromptText.EXECUTOR_SKILL_READ_ONLY_DESCRIPTION,
                content=self._skill_context,
            )

        user_budget = self._user_prompt_budget(model, messages)
        user_content = self._build_user_prompt(context, model=model, budget=user_budget)
        messages.append({"role": "user", "content": user_content})
        return messages

    def _user_prompt_budget(
        self,
        model: str,
        messages: list[dict[str, Any]],
    ) -> int | None:
        if not self._budgets:
            return None

        from mediated_coevo.runtime.token_budget import count_message_tokens

        system_tokens = count_message_tokens(model, messages)
        with_empty_user_tokens = count_message_tokens(
            model,
            [*messages, {"role": "user", "content": ""}],
        )
        user_message_overhead = max(0, with_empty_user_tokens - system_tokens)
        return max(
            1,
            self._budgets.planner_context_tokens
            - system_tokens
            - user_message_overhead,
        )

    def _build_user_prompt(
        self,
        context: dict[str, Any],
        *,
        model: str,
        budget: int | None,
    ) -> str:
        action = context.get("action", "plan_task")
        if action == "plan_task":
            return self._build_plan_prompt(
                context,
                model=model,
                budgets=self._budgets,
                budget=budget,
            )
        return context.get("instruction", "")

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

        instruction = await self._compact_plan_instruction_if_needed(
            task_id=task_id,
            base_instruction=base_instruction,
        )
        context: dict[str, Any] = {
            "action": "plan_task",
            "task_id": task_id,
            "base_instruction": instruction,
            "current_skills": current_skills,
        }
        if prior_context:
            context["prior_context"] = prior_context

        result = await self.process(context)
        parsed = result["parsed"]
        return TaskSpec(
            task_id=task_id,
            instruction=parsed.get("instruction", base_instruction),
            skills_context=current_skills,
            planner_reasoning=parsed.get("reasoning"),
            iteration=iteration,
        )

    async def _compact_plan_instruction_if_needed(
        self,
        *,
        task_id: str,
        base_instruction: str,
    ) -> str:
        if not self._budgets:
            return base_instruction

        from mediated_coevo.runtime.context_compactor import compact_text_for_context
        from mediated_coevo.runtime.token_budget import count_text_tokens

        model = self.llm_client.model
        instruction_tokens = count_text_tokens(model, base_instruction)
        budget_tokens = max(
            self._budgets.trace_excerpt_tokens,
            self._budgets.planner_context_tokens // 2,
        )
        if instruction_tokens <= budget_tokens:
            return base_instruction

        compacted = await compact_text_for_context(
            base_instruction,
            llm_client=self.llm_client,
            label=f"benchmark instruction for {task_id}",
            model=model,
            budget_tokens=budget_tokens,
            completion_tokens=min(1024, self._budgets.planner_completion_tokens),
            condition_name=self._condition_name,
        )
        return PromptText.compacted_benchmark_instruction(compacted)

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
            from mediated_coevo.runtime.token_budget import (
                count_text_tokens,
                pack_sections,
            )

            task_header = PromptText.task_header(context.get("task_id", "unknown"))
            response_schema = PromptText.PLAN_RESPONSE_SCHEMA
            fixed_required_tokens = count_text_tokens(
                model,
                f"{task_header}\n\n{response_schema}",
            )
            instruction_budget = max(1, budget - fixed_required_tokens)

            sections = [
                PromptSection(
                    "task_header",
                    "scaffold",
                    task_header,
                    required=True,
                ).to_budget_section(overflow_strategy="section_pack")
            ]
            if instruction := context.get("base_instruction"):
                sections.append(
                    PromptSection(
                        "benchmark_instruction",
                        "scaffold",
                        PromptText.benchmark_instruction(instruction),
                        required=True,
                        max_tokens=instruction_budget,
                    ).to_budget_section(overflow_strategy="section_pack")
                )
            sections.extend(
                section.to_budget_section(overflow_strategy="section_pack")
                for section in PlannerAgent._prior_context_sections(context)
            )
            sections.append(
                PromptSection(
                    "response_schema",
                    "response_schema",
                    response_schema,
                    required=True,
                ).to_budget_section(overflow_strategy="section_pack")
            )
            return pack_sections(model, sections, budget)

        parts = [PromptText.task_header(context.get("task_id", "unknown"))]
        if instruction := context.get("base_instruction"):
            parts.append(f"\n{PromptText.benchmark_instruction(instruction)}")
        parts.extend(
            f"\n{section.content}"
            for section in PlannerAgent._prior_context_sections(context)
        )
        parts.append(f"\n{PromptText.PLAN_RESPONSE_SCHEMA}")
        return "\n".join(parts)

    @staticmethod
    def _prior_context_sections(
        context: dict[str, Any],
    ) -> tuple[PromptSection, ...]:
        explicit_sections = context.get("prior_context_sections")
        if explicit_sections:
            return tuple(explicit_sections)

        prior_context = context.get("prior_context")
        if not prior_context:
            return ()
        return (
            PromptSection(
                "prior_context",
                "same_task_prior",
                PromptText.prior_context(prior_context),
                required=True,
            ),
        )
