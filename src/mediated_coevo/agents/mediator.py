"""Mediator agent with a fixed prompt-injected coordination skill.

Architecturally distinct from the Planner and Executor. The Mediator
does NOT plan tasks or execute them. It observes Gemini's execution
outputs, filters/compresses them, and produces curated reports for
Claude. Its system prompt is the fixed mediator SKILL.md content.

The Mediator actions:
  1. FILTER   — select relevant artifacts based on task context
  2. COMPRESS — distill raw traces into concise reports within token budget
  3. DECIDE   — expose or withhold (sometimes surfacing nothing is best)
  4. TAG      — annotate reports with task outcome rewards

Trace and report persistence is owned by the Orchestrator.
The Mediator has read-only access to the artifact store (query_summaries).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mediated_coevo.experiment.conditions import MEDIATOR_CONDITIONS, ConditionName
from mediated_coevo.prompt_text import PromptText

from .base import BaseAgent
from .prompt_context import PromptSection

if TYPE_CHECKING:
    from mediated_coevo.core.config import BudgetsConfig
    from mediated_coevo.llm.client import LLMClient
    from mediated_coevo.models.report import MediatorReport
    from mediated_coevo.models.task import TaskSpec
    from mediated_coevo.models.trace import ExecutionTrace
    from mediated_coevo.stores.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)


class MediatorAgent(BaseAgent):
    """GPT-5.4-backed mediator. Curates execution knowledge for the Planner.

    Unlike the Planner and Executor, the Mediator:
    - Does NOT plan or submit tasks
    - Does NOT write or modify skills
    - Controls only the Planner's *information diet*
    - Uses an immutable prompt-injected coordination skill
    """

    def __init__(
        self,
        llm_client: LLMClient,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        super().__init__("mediator", llm_client)
        self._artifact_store = artifact_store
        self._protocol_skill: str = ""
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

    def load_protocol(self, skill_content: str) -> None:
        """Load the fixed mediator skill as the system prompt."""
        self._protocol_skill = skill_content
        logger.info("Mediator protocol loaded (%d chars)", len(skill_content))

    def construct_messages(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        from mediated_coevo.runtime.token_budget import (
            count_message_tokens,
            fit_text_to_tokens,
            pack_sections,
        )

        model = self.llm_client.model
        protocol_skill = self._protocol_skill
        if self._budgets:
            protocol_skill = fit_text_to_tokens(
                model,
                protocol_skill,
                self._budgets.max_skill_tokens,
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": protocol_skill},
        ]

        # Prior trace summaries as separate system context (if available)
        if history := context.get("history"):
            messages.append(self._history_system_message(history, model))

        user_budget = None
        if self._budgets:
            system_tokens = count_message_tokens(model, messages)
            user_budget = max(1, self._budgets.mediator_prompt_tokens - system_tokens)

        sections: list[PromptSection] = []
        if trace := context.get("trace"):
            sections.append(self._trace_section(trace))

        if task_context := context.get("task_context"):
            sections.append(
                PromptSection(
                    "task_context",
                    PromptText.mediator_task_context(task_context.instruction),
                    required=True,
                )
            )

        if self._budgets and user_budget:
            user_content = pack_sections(
                model,
                [
                    section.to_budget_section(overflow_strategy="section_pack")
                    for section in sections
                ],
                user_budget,
            )
        else:
            user_content = "\n\n".join(section.content for section in sections)
        messages.append({"role": "user", "content": user_content})
        return messages

    def _history_system_message(
        self,
        history: list[str],
        model: str,
    ) -> dict[str, Any]:
        from mediated_coevo.runtime.token_budget import fit_text_to_tokens

        history_lines = "\n".join(f"- {item}" for item in history[:5])
        if self._budgets:
            history_lines = fit_text_to_tokens(
                model,
                history_lines,
                self._budgets.historical_summary_tokens,
            )
        return {
            "role": "system",
            "content": PromptSection(
                "relevant_history",
                PromptText.mediator_history(history_lines),
            ).content,
        }

    def _trace_section(self, trace: ExecutionTrace) -> PromptSection:
        return PromptSection(
            "execution_trace",
            PromptText.mediator_execution_trace(
                stdout=trace.stdout,
                stderr=trace.stderr,
                test_results=trace.test_results,
                reward=trace.reward,
            ),
            max_tokens=self._budgets.trace_excerpt_tokens if self._budgets else None,
        )

    async def process(self, context: dict[str, Any]) -> dict[str, Any]:
        messages = self.construct_messages(context)
        kwargs: dict[str, Any] = {}
        if self._budgets:
            kwargs = {
                "max_tokens": self._budgets.mediator_completion_tokens,
                "prompt_budget": self._budgets.mediator_prompt_tokens,
                "budget_label": "mediator.process_trace",
                "budget_overflow_strategy": "section_pack",
                "condition_name": self._condition_name,
            }
        response = await self.get_llm_response(messages, **kwargs)
        self.increment_step()

        parsed = self.response_to_dict(response["content"])
        return {
            "abstraction_level": parsed.get("abstraction_level", "reflection"),
            "content": parsed.get("content", ""),
            "withheld": parsed.get("withheld", False),
            "reasoning": parsed.get("reasoning", ""),
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
        }

    # ── Convenience wrappers ──

    async def mediate_trace(
        self,
        condition: ConditionName,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport | None:
        """Run mediation when the experiment condition and trace allow it."""
        if condition not in MEDIATOR_CONDITIONS:
            logger.info(
                "Step 3: Skipped (condition=%s does not use mediator).",
                condition,
            )
            return None

        if not trace.is_usable_feedback_signal:
            logger.warning(
                "Step 3: Skipped — trace unusable (status=%s error_kind=%s reward=%s)",
                trace.status,
                trace.error_kind,
                trace.reward,
            )
            return None

        logger.info("Step 3: Mediator processing trace...")
        return await self.process_trace(trace, task_context)

    async def process_trace(
        self,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport:
        """Full mediation pipeline: filter → compress → decide.

        Reads recent trace summaries from the artifact store (read-only).
        Persistence of the returned MediatorReport is the Orchestrator's
        responsibility.
        """
        from mediated_coevo.models.report import AbstractionLevel, MediatorReport

        # 1. FILTER — query relevant history
        history: list[str] = []
        if self._artifact_store:
            history = self._artifact_store.query_summaries(
                task_id=trace.task_id,
                recent=5,
                before_iteration=trace.iteration,
            )

        # 2-3. COMPRESS + DECIDE via LLM call
        context = {
            "trace": trace,
            "history": history,
            "task_context": task_context,
        }
        result = await self.process(context)

        # Parse abstraction level
        try:
            level = AbstractionLevel(result["abstraction_level"])
        except ValueError:
            level = AbstractionLevel.REFLECTION

        return MediatorReport(
            task_id=trace.task_id,
            iteration=trace.iteration,
            abstraction_level=level,
            content=result["content"],
            token_count=result["output_tokens"],
            withheld=result["withheld"],
            reasoning=result["reasoning"],
        )
