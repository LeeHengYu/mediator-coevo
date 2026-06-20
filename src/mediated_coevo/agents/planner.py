"""Planner agent — Claude.

Plans tasks for the Executor and decides skill updates based on
feedback from the Mediator (or raw traces, depending on condition).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mediated_coevo.evolution.candidates import candidate_from_mapping
from mediated_coevo.prompt_text import PromptText

from .base import BaseAgent
from .prompt_context import PromptSection
from .prompt_sections import select_markdown_sections

if TYPE_CHECKING:
    from mediated_coevo.core.config import BudgetsConfig, SkillUpdateConfig
    from mediated_coevo.llm.client import LLMClient
    from mediated_coevo.models.skill import SkillProposal, SkillUpdateCandidate
    from mediated_coevo.models.task import TaskSpec
    from mediated_coevo.stores.history_store import HistoryEntry

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = PromptText.PLANNER_SYSTEM
PLAN_RESPONSE_SCHEMA = PromptText.PLAN_RESPONSE_SCHEMA
UPDATE_RESPONSE_SCHEMA = PromptText.UPDATE_RESPONSE_SCHEMA
UPDATE_BATCH_RESPONSE_SCHEMA = PromptText.UPDATE_BATCH_RESPONSE_SCHEMA


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
        self._skill_updates: SkillUpdateConfig | None = None

    def configure_token_budget(
        self,
        budgets: BudgetsConfig,
        *,
        condition_name: str | None = None,
    ) -> None:
        self._budgets = budgets
        self._condition_name = condition_name

    def set_skill_context(
        self, executor_skills: str, skill_refiner: str | None = None
    ) -> None:
        """Inject executor skills and planner's own skill-refiner guidance.

        Called by the orchestrator before process().
        """
        self._skill_context = executor_skills or None
        self._skill_refiner = skill_refiner or None

    def configure_skill_updates(self, skill_updates: SkillUpdateConfig) -> None:
        """Configure which update-specific prompt sections are relevant."""
        self._skill_updates = skill_updates

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
        action = str(context.get("action", "plan_task"))

        # Skill injection — separate system messages like OpenSpace
        skill_refiner_prompt = self._skill_refiner_prompt(action)
        if skill_refiner_prompt:
            self._append_budgeted_system_context(
                messages,
                heading=skill_refiner_prompt["heading"],
                description=skill_refiner_prompt["description"],
                content=skill_refiner_prompt["content"],
            )

        if self._skill_context:
            skill_context_description = (
                PromptText.EXECUTOR_SKILL_EDITABLE_DESCRIPTION
                if self._executor_skill_updates_enabled()
                else PromptText.EXECUTOR_SKILL_READ_ONLY_DESCRIPTION
            )
            self._append_budgeted_system_context(
                messages,
                heading=PromptText.EXECUTOR_ACTIVE_SKILLS_HEADING,
                description=skill_context_description,
                content=self._skill_context,
            )

        user_budget = self._user_prompt_budget(model, messages)
        user_content = self._build_user_prompt(context, model=model, budget=user_budget)
        messages.append({"role": "user", "content": user_content})
        return messages

    def _skill_refiner_prompt(self, action: str) -> dict[str, str] | None:
        if not self._skill_refiner:
            return None

        content = self._skill_refiner_content_for_action(action)
        if not content:
            return None

        if action in {"update_skill", "update_skill_batch"}:
            return {
                "heading": PromptText.SKILL_REFINER_UPDATE_HEADING,
                "description": PromptText.SKILL_REFINER_UPDATE_DESCRIPTION,
                "content": content,
            }

        if self._executor_skill_updates_enabled():
            description = PromptText.SKILL_REFINER_PLANNING_WITH_UPDATES_DESCRIPTION
        else:
            description = PromptText.SKILL_REFINER_PLANNING_READ_ONLY_DESCRIPTION
        return {
            "heading": PromptText.SKILL_REFINER_PLANNING_HEADING,
            "description": description,
            "content": content,
        }

    def _skill_refiner_content_for_action(self, action: str) -> str | None:
        if not self._skill_refiner:
            return None
        if action in {"update_skill", "update_skill_batch"}:
            if not self._executor_skill_updates_enabled():
                return None
            return select_markdown_sections(
                self._skill_refiner,
                [
                    "Guidelines for Updating Executor Skills",
                    "Executor Skill Update Criteria",
                    "Artifact-Contract Thinking",
                ],
            )

        plan_sections = [
            "Task Planning Guidelines",
            "Artifact-Contract Thinking",
        ]
        if self._executor_skill_updates_enabled():
            plan_sections.extend(
                [
                    "Guidelines for Updating Executor Skills",
                    "Executor Skill Update Criteria",
                ]
            )
        return select_markdown_sections(self._skill_refiner, plan_sections)

    def _executor_skill_updates_enabled(self) -> bool:
        return self._skill_updates is None or self._skill_updates.executor

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
        if action == "update_skill":
            return self._build_update_prompt(
                context,
                response_schema=PromptText.UPDATE_RESPONSE_SCHEMA,
                batch_mode=False,
                model=model,
                budgets=self._budgets,
                budget=budget,
            )
        if action == "update_skill_batch":
            return self._build_update_prompt(
                context,
                response_schema=PromptText.UPDATE_BATCH_RESPONSE_SCHEMA,
                batch_mode=True,
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

        from mediated_coevo.evolution.compactor import compact_text_for_context
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

    async def suggest_skill_revision(
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

    async def suggest_skill_revision_batch(
        self,
        current_skill_content: str,
        feedback: str | None,
        edit_history: list[HistoryEntry],
        rejected_update_history: list[dict[str, Any]] | None = None,
        *,
        skill_id: str,
        task_ids: list[str],
        iteration: int = 0,
    ) -> list[SkillUpdateCandidate]:
        """Propose multiple candidate skill updates without writing to disk."""
        from mediated_coevo.models.history_signals import PlannerSignal

        context: dict[str, Any] = {
            "action": "update_skill_batch",
            "current_skill": current_skill_content,
            "feedback": feedback,
            "task_ids": task_ids,
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
        if rejected_update_history:
            context["rejected_update_history"] = rejected_update_history[-5:]
        result = await self.process(context)
        parsed = result["parsed"]
        raw_candidates = parsed.get("candidates")
        if not isinstance(raw_candidates, list):
            return []

        candidates: list[SkillUpdateCandidate] = []
        for raw_candidate in raw_candidates:
            candidate = candidate_from_mapping(
                raw_candidate,
                skill_id=skill_id,
                current_skill=current_skill_content,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

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

    @staticmethod
    def _build_update_prompt(
        context: dict[str, Any],
        *,
        response_schema: str,
        batch_mode: bool,
        model: str = "",
        budgets: BudgetsConfig | None = None,
        budget: int | None = None,
    ) -> str:
        if budgets and budget:
            from mediated_coevo.runtime.token_budget import BudgetSection, pack_sections

            sections = [
                BudgetSection(
                    "current_skill",
                    PromptText.current_skill(context.get("current_skill", "(none)")),
                    required=True,
                    max_tokens=budgets.max_skill_tokens,
                    overflow_strategy="section_pack",
                )
            ]
            if feedback := context.get("feedback"):
                sections.append(
                    BudgetSection(
                        "execution_feedback",
                        PromptText.execution_feedback(feedback),
                        max_tokens=budgets.mediator_report_tokens,
                        overflow_strategy="section_pack",
                    )
                )
            if batch_mode and (task_ids := context.get("task_ids")):
                sections.append(
                    BudgetSection(
                        "candidate_scope",
                        PromptText.candidate_scope(task_ids),
                        overflow_strategy="section_pack",
                    )
                )
            if history := context.get("edit_history"):
                sections.append(
                    BudgetSection(
                        "recent_edit_history",
                        PromptText.recent_edit_history(history),
                        max_tokens=budgets.historical_summary_tokens,
                        overflow_strategy="section_pack",
                    )
                )
            if rejected_history := context.get("rejected_update_history"):
                sections.append(
                    BudgetSection(
                        "rejected_update_history",
                        PromptText.rejected_skill_updates(rejected_history),
                        max_tokens=budgets.historical_summary_tokens,
                        overflow_strategy="section_pack",
                    )
                )
            sections.append(
                BudgetSection(
                    "response_schema",
                    response_schema,
                    required=True,
                    overflow_strategy="section_pack",
                )
            )
            return pack_sections(model, sections, budget)

        parts = [
            PromptText.current_skill(context.get("current_skill", "(none)")),
        ]
        if feedback := context.get("feedback"):
            parts.append(f"\n{PromptText.execution_feedback(feedback)}")
        if batch_mode and (task_ids := context.get("task_ids")):
            parts.append(f"\n{PromptText.candidate_scope(task_ids)}")
        if history := context.get("edit_history"):
            parts.append(f"\n{PromptText.recent_edit_history(history)}")
        if rejected_history := context.get("rejected_update_history"):
            parts.append(f"\n{PromptText.rejected_skill_updates(rejected_history)}")
        parts.append(f"\n{response_schema}")
        return "\n".join(parts)
