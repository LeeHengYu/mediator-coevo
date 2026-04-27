"""Experiment condition definitions and prior-context routing."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mediated_coevo.config import BudgetsConfig
    from mediated_coevo.llm.client import LLMClient
    from mediated_coevo.models.report import MediatorReport
    from mediated_coevo.models.trace import ExecutionTrace
    from mediated_coevo.stores.artifact_store import ArtifactStore

ConditionName = Literal[
    "no_feedback",
    "full_traces",
    "shared_notes",
    "static_mediator",
    "learned_mediator",
]

MEDIATOR_CONDITIONS: frozenset[ConditionName] = frozenset({"static_mediator", "learned_mediator"})
MEDIATOR_EVOLVE_CONDITIONS: frozenset[ConditionName] = frozenset({"learned_mediator"})


async def get_prior_context(
    condition: ConditionName,
    task_id: str,
    artifact_store: ArtifactStore,
    previous_report: MediatorReport | None,
    shared_notes: str | None,
    *,
    model: str,
    llm_client: LLMClient | None = None,
    budgets: BudgetsConfig | None = None,
    condition_name: str | None = None,
) -> str | None:
    """Return the prior-context string the planner should receive, or None."""
    from mediated_coevo.token_budget import BudgetSection, fit_text_to_tokens, pack_sections

    if condition == "no_feedback":
        return None
    elif condition == "full_traces":
        summaries = await build_trace_summaries(
            artifact_store.query_traces(task_id=task_id, recent=3),
            llm_client=llm_client,
            model=model,
            budgets=budgets,
            condition_name=condition_name,
        )
        if not summaries:
            return None
        text = "\n".join(summaries)
        if budgets:
            return pack_sections(
                model,
                [BudgetSection("full_traces", text, max_tokens=budgets.historical_summary_tokens)],
                budgets.historical_summary_tokens,
            )
        return text
    elif condition == "shared_notes":
        if shared_notes and budgets:
            return fit_text_to_tokens(model, shared_notes, budgets.historical_summary_tokens)
        return shared_notes
    else:  # static_mediator, learned_mediator
        if previous_report and not previous_report.withheld:
            if budgets:
                return fit_text_to_tokens(
                    model,
                    previous_report.content,
                    budgets.mediator_report_tokens,
                )
            return previous_report.content
        return None


async def get_cross_task_prior_context(
    condition: ConditionName,
    task_id: str,
    artifact_store: ArtifactStore,
    previous_reports_by_task: dict[str, MediatorReport],
    *,
    model: str,
    recent: int = 3,
    llm_client: LLMClient | None = None,
    budgets: BudgetsConfig | None = None,
    condition_name: str | None = None,
) -> str | None:
    """Return explicitly cross-task prior context for opt-in experiments."""
    from mediated_coevo.token_budget import BudgetSection, pack_sections

    if condition == "full_traces":
        traces = [
            trace
            for trace in artifact_store.query_traces(task_id=None, recent=recent * 4)
            if trace.task_id != task_id
        ][:recent]
        if not traces:
            return None
        summaries = await build_trace_summaries(
            traces,
            include_source_task=True,
            llm_client=llm_client,
            model=model,
            budgets=budgets,
            condition_name=condition_name,
        )
        text = "\n".join(summaries)
        if budgets:
            return pack_sections(
                model,
                [BudgetSection("cross_task_full_traces", text, max_tokens=budgets.historical_summary_tokens)],
                budgets.historical_summary_tokens,
            )
        return text

    if condition in MEDIATOR_CONDITIONS:
        reports = [
            report
            for source_task, report in previous_reports_by_task.items()
            if source_task != task_id and not report.withheld
        ]
        if not reports:
            return None
        reports.sort(key=lambda report: (report.iteration, report.timestamp), reverse=True)
        text = "\n\n".join(
            f"source_task={report.task_id} iter={report.iteration}\n{report.content}"
            for report in reports[:recent]
        )
        if budgets:
            return pack_sections(
                model,
                [BudgetSection("cross_task_reports", text, max_tokens=budgets.mediator_report_tokens)],
                budgets.mediator_report_tokens,
            )
        return text

    return None


async def build_trace_summaries(
    traces: list[ExecutionTrace],
    *,
    model: str,
    include_source_task: bool = False,
    llm_client: LLMClient | None = None,
    budgets: BudgetsConfig | None = None,
    condition_name: str | None = None,
) -> list[str]:
    """Build compact trace summaries for planner context."""
    return list(
        await asyncio.gather(
            *(
                _trace_summary(
                    trace,
                    include_source_task=include_source_task,
                    llm_client=llm_client,
                    model=model,
                    budgets=budgets,
                    condition_name=condition_name,
                )
                for trace in traces
            )
        )
    )


async def _trace_summary(
    trace: ExecutionTrace,
    *,
    include_source_task: bool,
    llm_client: LLMClient | None,
    model: str,
    budgets: BudgetsConfig | None,
    condition_name: str | None,
) -> str:
    from mediated_coevo.evolution.compactor import (
        compact_text_for_context,
        trace_header_summary,
    )

    summary = trace_header_summary(trace, include_source_task=include_source_task)
    if trace.stderr:
        compact_stderr = await compact_text_for_context(
            trace.stderr,
            llm_client=llm_client,
            label=f"stderr for {trace.task_id} iter {trace.iteration}",
            model=model,
            budget_tokens=budgets.trace_excerpt_tokens if budgets else None,
            completion_tokens=budgets.mediator_completion_tokens if budgets else 600,
            condition_name=condition_name,
        )
        summary += f" stderr={compact_stderr}"
    return summary
