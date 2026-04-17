"""Mediator agent — GPT-5.4.

Architecturally distinct from the Planner and Executor. The Mediator
does NOT plan tasks or execute them. It observes Gemini's execution
outputs, filters/compresses them, and produces curated reports for
Claude. Its system prompt is the coordination-protocol.md skill, which
evolves over time through the co-evolution loop.

The 5 Mediator actions:
  1. STORE   — extract and persist artifacts from Gemini's outputs
  2. FILTER  — select relevant artifacts based on task context
  3. COMPRESS — distill raw traces into concise reports within token budget
  4. DECIDE  — expose or withhold (sometimes surfacing nothing is best)
  5. TAG     — annotate past reports with downstream results (deferred)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base import BaseAgent

if TYPE_CHECKING:
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
        token_budget: int = 2000,
    ) -> None:
        super().__init__("mediator", llm_client)
        self._artifact_store = artifact_store
        self._token_budget = token_budget
        self._protocol_skill: str = ""

    def load_protocol(self, skill_content: str) -> None:
        """Load coordination-protocol.md as the system prompt.

        This skill evolves over time — the Mediator learns what
        reporting style leads to better skill updates from the Planner.
        """
        self._protocol_skill = skill_content
        logger.info("Mediator protocol loaded (%d chars)", len(skill_content))

    def construct_messages(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._protocol_skill},
        ]

        # History as separate system context (if available)
        if history := context.get("history"):
            history_lines = "\n".join(f"- {item}" for item in history[:5])
            messages.append({"role": "system", "content": (
                "# Relevant History\n\n"
                "Previous mediation reports for this task:\n\n"
                f"{history_lines}"
            )})

        # User message: trace + task context
        parts: list[str] = []

        if trace := context.get("trace"):
            parts.append("## Execution Trace")
            if trace.stdout:
                parts.append(f"### stdout\n{trace.stdout[:8000]}")
            if trace.stderr:
                parts.append(f"### stderr\n{trace.stderr[:4000]}")
            if trace.test_results:
                parts.append(f"### test_results\n{trace.test_results}")
            parts.append(f"### reward: {trace.reward}")

        if task_context := context.get("task_context"):
            parts.append(f"\n## Task Context\n{task_context.instruction}")

        messages.append({"role": "user", "content": "\n\n".join(parts)})
        return messages

    async def process(self, context: dict[str, Any]) -> dict[str, Any]:
        messages = self.construct_messages(context)
        response = await self.get_llm_response(messages)
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

    async def process_trace(
        self,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport:
        """Full mediation pipeline: store → filter → compress → decide.

        This is the main entry point called by the LearnedMediatorCondition.
        """
        from mediated_coevo.models.report import AbstractionLevel, MediatorReport

        # 1. STORE artifacts
        if self._artifact_store:
            self._artifact_store.store_trace(trace)

        # 2. FILTER — query relevant history
        history: list[str] = []
        if self._artifact_store:
            history = self._artifact_store.query_summaries(
                task_id=trace.task_id, recent=5
            )

        # 3-4. COMPRESS + DECIDE via LLM call
        context = {
            "trace": trace,
            "history": history,
            "task_context": task_context,
            "token_budget": self._token_budget,
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
        feedback: str,
        report: MediatorReport | None = None,
    ) -> "MediatorSignal":
        """Compact a feedback event into a structured ``MediatorSignal``.

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
        from mediated_coevo.models.history_signals import MediatorSignal
        from mediated_coevo.utils import parse_json_object

        raw_length = len(feedback)
        if raw_length <= RAW_PASSTHROUGH_CHARS:
            return deterministic_mediator_signal(feedback, report)

        try:
            response = await self._llm_client.complete(
                messages=[
                    {"role": "system", "content": COMPACTOR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"## Mediator report ({raw_length} chars)\n\n"
                            f"{feedback}\n\n"
                            f"Return JSON with `headline` "
                            f"(≤{TARGET_HEADLINE_CHARS} chars) and `evidence` "
                            f"(≤{TARGET_EVIDENCE_CHARS} chars)."
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=600,
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
            return deterministic_mediator_signal(feedback, report)

        return MediatorSignal(
            headline=headline[: TARGET_HEADLINE_CHARS * 2],
            evidence=evidence[: TARGET_EVIDENCE_CHARS * 2],
            abstraction_level=abstraction_level_str(report),
            withheld=report.withheld if report else False,
            mediator_reasoning=report.reasoning if report else "",
            raw_length=raw_length,
        )
