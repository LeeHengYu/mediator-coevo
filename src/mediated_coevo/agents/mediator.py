"""Mediator agent — GPT-5.4.

Architecturally distinct from the Planner and Executor. The Mediator
does NOT plan tasks or execute them. It observes Gemini's execution
outputs, filters/compresses them, and produces curated reports for
Claude. Its system prompt is the coordination-protocol.md skill, which
evolves over time through the co-evolution loop.

The Mediator actions:
  1. FILTER   — select relevant artifacts based on task context
  2. COMPRESS — distill raw traces into concise reports within token budget
  3. DECIDE   — expose or withhold (sometimes surfacing nothing is best)
  4. TAG      — annotate past reports with downstream results (deferred)

Trace and report persistence is owned by the Orchestrator.
The Mediator has read-only access to the artifact store (query_summaries).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mediated_coevo.conditions import ConditionName, MEDIATOR_CONDITIONS

from .base import BaseAgent

if TYPE_CHECKING:
    from mediated_coevo.config import BudgetsConfig
    from mediated_coevo.llm.client import LLMClient
    from mediated_coevo.models.history_signals import MediatorSignal
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
    - Has a self-evolving system prompt (coordination-protocol.md)
    """

    @property
    def role(self) -> str:
        return "mediator"

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
        """Load coordination-protocol.md as the system prompt.

        This skill evolves over time — the Mediator learns what
        reporting style leads to better skill updates from the Planner.
        """
        self._protocol_skill = skill_content
        logger.info("Mediator protocol loaded (%d chars)", len(skill_content))

    def construct_messages(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        from mediated_coevo.token_budget import (
            BudgetSection,
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
            history_lines = "\n".join(f"- {item}" for item in history[:5])
            if self._budgets:
                history_lines = fit_text_to_tokens(
                    model,
                    history_lines,
                    self._budgets.historical_summary_tokens,
                )
            messages.append({"role": "system", "content": (
                "# Relevant History\n\n"
                "Previous execution trace summaries for this task:\n\n"
                f"{history_lines}"
            )})

        user_budget = None
        if self._budgets:
            system_tokens = count_message_tokens(model, messages)
            user_budget = max(1, self._budgets.mediator_prompt_tokens - system_tokens)

        sections: list[BudgetSection] = []
        if trace := context.get("trace"):
            trace_parts = ["## Execution Trace"]
            if trace.stdout:
                trace_parts.append(f"### stdout\n{trace.stdout}")
            if trace.stderr:
                trace_parts.append(f"### stderr\n{trace.stderr}")
            if trace.test_results:
                trace_parts.append(f"### test_results\n{trace.test_results}")
            trace_parts.append(f"### reward: {trace.reward}")
            sections.append(BudgetSection(
                "execution_trace",
                "\n\n".join(trace_parts),
                max_tokens=self._budgets.trace_excerpt_tokens if self._budgets else None,
            ))

        if task_context := context.get("task_context"):
            sections.append(BudgetSection(
                "task_context",
                f"## Task Context\n{task_context.instruction}",
                required=True,
            ))

        if self._budgets and user_budget:
            user_content = pack_sections(model, sections, user_budget)
        else:
            user_content = "\n\n".join(section.content for section in sections)
        messages.append({"role": "user", "content": user_content})
        return messages

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
                task_id=trace.task_id, recent=5
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

    async def compact_feedback(
        self,
        report: MediatorReport,
    ) -> "MediatorSignal":
        """Compact a mediator report into a structured ``MediatorSignal``.

        Short feedback (``<= RAW_PASSTHROUGH_CHARS``) passes through the
        deterministic path with no LLM call. Long feedback triggers one
        extra call to **this mediator's own LLM client** — the same
        model that produced the content — to extract a headline and a
        diagnostic evidence excerpt. On any LLM failure we fall back to
        the deterministic path so the iteration loop never breaks.

        The signal returned by this method is stored in ``HistoryEntry``
        and consumed by the Reflector at co-evolution checkpoints, where
        the mediator reflects on its own coordination-protocol.md. The
        producer, compactor, and reflector are therefore all the same
        model — see the design notes in ``evolution/compactor.py``.
        """
        from mediated_coevo.evolution.compactor import (
            COMPACTOR_SYSTEM_PROMPT,
            RAW_PASSTHROUGH_CHARS,
            TARGET_EVIDENCE_CHARS,
            TARGET_HEADLINE_CHARS,
            abstraction_level_str,
            deterministic_mediator_signal,
            first_sentence,
            head_tail_text,
        )
        from mediated_coevo.token_budget import BudgetSection, fit_text_to_tokens, pack_sections
        from mediated_coevo.models.history_signals import MediatorSignal
        from mediated_coevo.utils import parse_json_object

        feedback = report.exposed_content or ""
        raw_length = len(feedback)
        if raw_length <= RAW_PASSTHROUGH_CHARS:
            return deterministic_mediator_signal(report)

        try:
            prompt_feedback = feedback
            prompt_budget = None
            max_tokens = 600
            if self._budgets:
                prompt_budget = self._budgets.mediator_prompt_tokens
                max_tokens = min(600, self._budgets.mediator_completion_tokens)
                prompt_feedback = pack_sections(
                    self.llm_client.model,
                    [
                        BudgetSection(
                            "feedback",
                            feedback,
                            required=True,
                            max_tokens=self._budgets.mediator_report_tokens,
                        )
                    ],
                    max(1, self._budgets.mediator_prompt_tokens - 500),
                )
            response = await self._llm_client.complete(
                messages=[
                    {"role": "system", "content": COMPACTOR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"## Mediator report ({raw_length} chars)\n\n"
                            f"{prompt_feedback}\n\n"
                            f"Return JSON with `headline` "
                            f"(≤{TARGET_HEADLINE_CHARS} chars) and `evidence` "
                            f"(≤{TARGET_EVIDENCE_CHARS} chars)."
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                budget_label="mediator.compact_feedback",
                prompt_budget=prompt_budget,
                budget_overflow_strategy="head_tail",
                condition_name=self._condition_name,
            )
            parsed = parse_json_object(response["content"])
            headline = str(parsed.get("headline", "")).strip() or first_sentence(
                feedback, TARGET_HEADLINE_CHARS
            )
            evidence = str(parsed.get("evidence", "")).strip() or head_tail_text(
                feedback, TARGET_EVIDENCE_CHARS
            )
        except Exception as e:
            logger.warning(
                "%s: compaction LLM call failed (%s) — falling back to head+tail",
                self.name,
                e,
            )
            return deterministic_mediator_signal(report)

        return MediatorSignal(
            headline=headline[: TARGET_HEADLINE_CHARS * 2],
            evidence=(
                fit_text_to_tokens(
                    self.llm_client.model,
                    evidence,
                    self._budgets.mediator_report_tokens,
                )
                if self._budgets
                else evidence[: TARGET_EVIDENCE_CHARS * 2]
            ),
            abstraction_level=abstraction_level_str(report),
            withheld=report.withheld,
            mediator_reasoning=report.reasoning,
            raw_length=raw_length,
        )
