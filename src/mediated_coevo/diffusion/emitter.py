"""Emit low-risk diffusion artifacts from task-local runtime evidence."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from mediated_coevo.diffusion.models import (
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
)
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.prompt_text import PromptText

if TYPE_CHECKING:
    from mediated_coevo.core.config import BudgetsConfig
    from mediated_coevo.llm.client import LLMClient

logger = logging.getLogger(__name__)

EMITTER_VERSION = "pr3-v1"


@dataclass(frozen=True)
class DiffusionEmitter:
    """Build low-risk source artifacts without changing target-side routing."""

    model: str
    llm_client: LLMClient | None = None
    budgets: BudgetsConfig | None = None
    condition_name: str | None = None
    emitter_version: str = EMITTER_VERSION

    async def emit(
        self,
        *,
        trace: ExecutionTrace,
        report: MediatorReport | None,
        record: IterationRecord,
        task_metadata: Mapping[str, Any] | None = None,
        judge_reward: float | None = None,
    ) -> list[DiffusionArtifact]:
        """Emit source-side diffusion artifacts for future task runs."""
        artifacts: list[DiffusionArtifact] = []
        metadata = self._common_metadata(trace, task_metadata)
        trace_refs = [_trace_ref(trace)]

        run_outcome = await self._run_outcome_content(
            trace,
            record=record,
            judge_reward=judge_reward,
        )
        if run_outcome:
            run_outcome_metadata: dict[str, Any] = {
                **metadata,
                "outcome_signal": _outcome_signal(trace, record),
            }
            if record.delta_reward is not None and record.delta_reward < 0:
                previous_reward = _previous_reward(record, trace)
                if previous_reward is not None and trace.reward is not None:
                    run_outcome_metadata.update(
                        {
                            "regression": True,
                            "delta_reward": record.delta_reward,
                            "previous_reward": previous_reward,
                            "current_reward": trace.reward,
                        }
                    )
            artifacts.append(
                self._build_artifact(
                    artifact_type=DiffusionArtifactType.RUN_OUTCOME,
                    trace=trace,
                    content=run_outcome,
                    evidence_trace_ids=trace_refs,
                    verifier_reward=trace.reward,
                    judge_reward=judge_reward,
                    metadata=run_outcome_metadata,
                )
            )

        report_summary = await self._report_summary_content(report)
        if report_summary and report is not None:
            report_metadata = {
                **metadata,
                "report_id": report.report_id,
                "report_artifact_path": (
                    f"artifacts/reports/{report.task_id}_iter"
                    f"{report.iteration:04d}_{report.report_id}.json"
                ),
            }
            artifacts.append(
                self._build_artifact(
                    artifact_type=DiffusionArtifactType.MEDIATOR_REPORT_SUMMARY,
                    trace=trace,
                    content=report_summary,
                    evidence_trace_ids=trace_refs,
                    evidence_report_ids=[report.report_id],
                    verifier_reward=trace.reward,
                    judge_reward=judge_reward,
                    metadata=report_metadata,
                )
            )
            debug_hint = self._debug_hint_content(report_summary)
            if debug_hint:
                artifacts.append(
                    self._build_artifact(
                        artifact_type=DiffusionArtifactType.DEBUG_HINT,
                        trace=trace,
                        content=debug_hint,
                        evidence_trace_ids=trace_refs,
                        evidence_report_ids=[report.report_id],
                        verifier_reward=trace.reward,
                        judge_reward=judge_reward,
                        metadata=report_metadata,
                    )
                )

        return artifacts

    async def _report_summary_content(
        self,
        report: MediatorReport | None,
    ) -> str | None:
        if report is None or not report.is_exposed:
            return None
        content = (report.exposed_content or "").strip()
        if not content:
            return None
        return await self._compact_text(
            content,
            label=f"mediator report for {report.task_id} iter {report.iteration}",
            budget_tokens=(
                self.budgets.mediator_report_tokens if self.budgets else None
            ),
        )

    async def _run_outcome_content(
        self,
        trace: ExecutionTrace,
        *,
        record: IterationRecord,
        judge_reward: float | None,
    ) -> str | None:
        if trace.status not in {"ok", "task_failed"} or trace.reward is None:
            return None

        evidence_parts = [
            f"task_id={trace.task_id}",
            f"iteration={trace.iteration}",
            f"status={trace.status}",
            f"verifier_reward={_reward_text(trace.reward)}",
            f"judge_reward={_reward_text(judge_reward)}",
            f"record_success={record.success}",
            f"record_verifier_status={record.verifier_status}",
        ]
        if record.delta_reward is not None:
            previous_reward = _previous_reward(record, trace)
            if previous_reward is not None:
                if record.delta_reward < 0:
                    regressed_reward_change = self._regressed_reward_change_content(
                        record,
                        trace,
                    )
                    if regressed_reward_change is not None:
                        evidence_parts.append(
                            f"reward_change={regressed_reward_change}"
                        )
                elif record.delta_reward > 0:
                    evidence_parts.append(
                        "reward_change=Same-task reward improved from "
                        f"{previous_reward:.2f} to {trace.reward:.2f} "
                        f"(delta={record.delta_reward:+.2f})."
                    )
                else:
                    evidence_parts.append(
                        "reward_change=Same-task reward stayed at "
                        f"{trace.reward:.2f} (delta={record.delta_reward:+.2f})."
                    )
        if trace.stdout:
            evidence_parts.append(f"stdout\n{trace.stdout}")
        if trace.stderr:
            evidence_parts.append(f"stderr\n{trace.stderr}")
        if trace.test_results:
            evidence_parts.append(
                "test_results\n"
                + json.dumps(trace.test_results, sort_keys=True, default=str)
            )
        if trace.error_detail is not None:
            evidence_parts.append(f"error_detail\n{trace.error_detail}")

        evidence = "\n\n".join(evidence_parts)
        signal = _outcome_signal(trace, record)
        content = PromptText.run_outcome_content(signal=signal, evidence=evidence)
        return await self._compact_text(
            content,
            label=f"run outcome for {trace.task_id} iter {trace.iteration}",
            budget_tokens=(
                self.budgets.historical_summary_tokens if self.budgets else None
            ),
        )

    async def _compact_text(
        self,
        text: str,
        *,
        label: str,
        budget_tokens: int | None,
    ) -> str:
        from mediated_coevo.runtime.context_compactor import compact_text_for_context

        compacted = await compact_text_for_context(
            text,
            llm_client=self.llm_client,
            label=label,
            model=self.model,
            budget_tokens=budget_tokens,
            completion_tokens=(
                self.budgets.mediator_completion_tokens if self.budgets else 600
            ),
            condition_name=self.condition_name,
        )
        return compacted.strip()

    def _common_metadata(
        self,
        trace: ExecutionTrace,
        task_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "emitter_version": self.emitter_version,
            "trace_ref": _trace_ref(trace),
            "trace_artifact_path": (
                f"artifacts/traces/{trace.task_id}_iter{trace.iteration:04d}.json"
            ),
            "verifier_status": trace.status,
        }
        if trace.error_kind:
            metadata["error_kind"] = trace.error_kind
        if task_metadata is None:
            return metadata
        for key in (
            "task_category",
            "task_difficulty",
            "expected_reward_range",
            "verifier_type",
        ):
            value = task_metadata.get(key)
            if value is not None:
                metadata[key] = value
        return metadata

    def _debug_hint_content(self, report_summary: str) -> str | None:
        from mediated_coevo.runtime.context_compactor import (
            TARGET_HEADLINE_CHARS,
            first_sentence,
        )

        headline = first_sentence(report_summary, TARGET_HEADLINE_CHARS).strip()
        if not headline:
            return None
        return headline

    def _regressed_reward_change_content(
        self,
        record: IterationRecord,
        trace: ExecutionTrace,
    ) -> str | None:
        delta = record.delta_reward
        if delta is None or delta >= 0 or trace.reward is None:
            return None
        previous_reward = _previous_reward(record, trace)
        if previous_reward is None:
            return None
        return (
            "Same-task reward regressed from "
            f"{previous_reward:.2f} to {trace.reward:.2f} "
            f"(delta={delta:+.2f}). Treat recent context as a suspect hypothesis "
            "until it is revalidated."
        )

    def _build_artifact(
        self,
        *,
        artifact_type: DiffusionArtifactType,
        trace: ExecutionTrace,
        content: str,
        evidence_trace_ids: list[str],
        evidence_report_ids: list[str] | None = None,
        verifier_reward: float | None,
        judge_reward: float | None,
        metadata: dict[str, Any],
    ) -> DiffusionArtifact:
        task_slug = re.sub(r"[^A-Za-z0-9]+", "-", trace.task_id).strip("-").lower()
        type_slug = artifact_type.value.replace("_", "-")
        if content:
            from mediated_coevo.runtime.token_budget import count_text_tokens

            try:
                token_cost = count_text_tokens(self.model, content)
            except Exception:
                logger.debug(
                    "Token counting failed for diffusion artifact",
                    exc_info=True,
                )
                token_cost = 0
        else:
            token_cost = 0
        return DiffusionArtifact(
            artifact_id=(
                f"{task_slug or 'artifact'}-iter{trace.iteration:04d}-{type_slug}"
            ),
            source_task_id=trace.task_id,
            source_iteration=trace.iteration,
            source_run_id=trace.run_id,
            artifact_type=artifact_type,
            risk_level=DiffusionRiskLevel.LOW,
            content=content,
            evidence_trace_ids=evidence_trace_ids,
            evidence_report_ids=list(evidence_report_ids or []),
            verifier_reward=verifier_reward,
            judge_reward=judge_reward,
            token_cost=token_cost,
            metadata=metadata,
        )


async def emit_diffusion_artifacts(
    *,
    trace: ExecutionTrace,
    report: MediatorReport | None,
    record: IterationRecord,
    model: str,
    llm_client: LLMClient | None = None,
    budgets: BudgetsConfig | None = None,
    condition_name: str | None = None,
    task_metadata: Mapping[str, Any] | None = None,
    judge_reward: float | None = None,
) -> list[DiffusionArtifact]:
    """Convenience wrapper for one-shot artifact emission."""
    emitter = DiffusionEmitter(
        model=model,
        llm_client=llm_client,
        budgets=budgets,
        condition_name=condition_name,
    )
    return await emitter.emit(
        trace=trace,
        report=report,
        record=record,
        task_metadata=task_metadata,
        judge_reward=judge_reward,
    )


def _trace_ref(trace: ExecutionTrace) -> str:
    return f"{trace.task_id}:iter{trace.iteration:04d}"


def _outcome_signal(
    trace: ExecutionTrace,
    record: IterationRecord,
) -> Literal["success", "failure", "mixed"]:
    if record.success is True:
        return "success"
    if record.success is False or trace.status == "task_failed":
        return "failure"
    if trace.status == "ok" and trace.reward is not None and trace.reward <= 0.0:
        return "failure"
    return "mixed"


def _previous_reward(
    record: IterationRecord,
    trace: ExecutionTrace,
) -> float | None:
    if record.delta_reward is None or trace.reward is None:
        return None
    return trace.reward - record.delta_reward


def _reward_text(reward: float | None) -> str:
    if reward is None:
        return "n/a"
    return f"{reward:.2f}"
