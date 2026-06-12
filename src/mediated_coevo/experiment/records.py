"""Builders for iteration records and their shared metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from mediated_coevo.analysis.metrics import (
    metric_success,
    metric_verifier_status,
    token_totals_by_agent,
)
from mediated_coevo.benchmarks import HERMES_AGENT_NAME, SKILLFLOW_VERIFIER_TYPE
from mediated_coevo.core.config import Config
from mediated_coevo.core.utils import as_mapping, as_nonempty_string
from mediated_coevo.experiment.conditions import ConditionName
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.skill import (
    AdvisorBatchProvenance,
    SkillUpdate,
    SkillUpdateLedgerEntry,
)
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.runtime.token_budget import TokenBudgetEvent
from mediated_coevo.stores.skill_store import SkillStore

if TYPE_CHECKING:
    from mediated_coevo.evolution.reflector import ReflectionResult


class ExperimentRecordFields(TypedDict):
    baseline_preset: str | None
    diffusion_enabled: bool
    diffusion_policy: str
    diffusion_graph: str | None
    skill_update_policy: dict[str, bool]


class TaskMetadataFields(TypedDict):
    task_category: str | None
    task_difficulty: str | None
    expected_reward_range: tuple[float, float] | None
    verifier_type: str | None


def build_missing_task_record(
    *,
    task_id: str,
    iteration: int,
    condition: ConditionName,
    duration_sec: float,
    trace: ExecutionTrace,
    llm_token_events: list[TokenBudgetEvent],
    config: Config,
    skill_hashes: dict[str, str],
) -> IterationRecord:
    """Build the record for a task that could not be resolved."""
    return IterationRecord(
        iteration=iteration,
        task_id=task_id,
        execution_trace=trace,
        duration_sec=duration_sec,
        llm_token_events=llm_token_events,
        total_tokens=sum(e.total_tokens for e in llm_token_events),
        condition_name=condition,
        **experiment_record_fields(config),
        **runtime_record_fields(config),
        **task_metadata_fields(task_id=task_id, task_config=None),
        skill_hashes=dict(skill_hashes),
        success=trace.is_usable_feedback_signal,
        verifier_status=trace.status,
        token_totals_by_agent=token_totals_by_agent(trace, llm_token_events),
    )


def build_iteration_record(
    *,
    task_id: str,
    iteration: int,
    condition: ConditionName,
    duration_sec: float,
    task_spec: TaskSpec,
    trace: ExecutionTrace,
    report: MediatorReport | None,
    skill_update: SkillUpdate | None,
    mediator_entry_id: str | None,
    planner_entry_id: str | None,
    skill_hashes: dict[str, str],
    task_metadata: TaskMetadataFields,
    llm_token_events: list[TokenBudgetEvent],
    config: Config,
    previous_reward_by_task: dict[str, float],
) -> IterationRecord:
    """Build a complete metric/history record for one normal task iteration."""
    executor_tokens = trace.token_usage.input_tokens + trace.token_usage.output_tokens
    total_tokens = executor_tokens + sum(e.total_tokens for e in llm_token_events)
    history_entry_ids: dict[str, str] = {}
    if mediator_entry_id:
        history_entry_ids["mediator"] = mediator_entry_id
    if planner_entry_id:
        history_entry_ids["planner"] = planner_entry_id

    advisor_provenance = (
        skill_update.provenance
        if skill_update
        and isinstance(skill_update.provenance, AdvisorBatchProvenance)
        else None
    )
    proposal_ids = (
        [ref.proposal_id for ref in advisor_provenance.proposal_refs]
        if advisor_provenance
        else []
    )
    advisor_decision = None
    advisor_reason = None
    if advisor_provenance:
        advisor_decision = advisor_provenance.decision
        advisor_reason = advisor_provenance.reason or None

    return IterationRecord(
        iteration=iteration,
        task_id=task_id,
        task_spec=task_spec,
        execution_trace=trace,
        mediator_report=report if report and report.is_exposed else None,
        skill_update=skill_update,
        reward=trace.reward,
        total_tokens=total_tokens,
        llm_token_events=llm_token_events,
        duration_sec=duration_sec,
        run_id=trace.run_id,
        mediator_history_entry_id=mediator_entry_id,
        planner_history_entry_id=planner_entry_id,
        history_entry_ids=history_entry_ids,
        mediator_report_id=report.report_id if report else None,
        proposal_ids=proposal_ids,
        condition_name=condition,
        **experiment_record_fields(config),
        **runtime_record_fields(config),
        **task_metadata,
        skill_hashes=dict(skill_hashes),
        success=metric_success(trace),
        verifier_status=metric_verifier_status(trace),
        delta_reward=delta_reward(previous_reward_by_task, task_id, trace),
        token_totals_by_agent=token_totals_by_agent(trace, llm_token_events),
        advisor_decision=advisor_decision,
        advisor_reason=advisor_reason,
    )


def build_coevolution_record(
    *,
    iteration: int,
    condition: ConditionName,
    duration_sec: float,
    llm_token_events: list[TokenBudgetEvent],
    skill_updates: list[SkillUpdate] | None,
    config: Config,
    skill_hashes: dict[str, str],
) -> IterationRecord:
    """Build a metrics-only row for co-evolution LLM telemetry."""
    return IterationRecord(
        iteration=iteration,
        task_id="__coevolution__",
        skill_updates=list(skill_updates or []),
        total_tokens=sum(e.total_tokens for e in llm_token_events),
        llm_token_events=llm_token_events,
        duration_sec=duration_sec,
        condition_name=condition,
        **experiment_record_fields(config),
        **runtime_record_fields(config),
        skill_hashes=dict(skill_hashes),
        token_totals_by_agent=token_totals_by_agent(None, llm_token_events),
    )


def experiment_record_fields(config: Config) -> ExperimentRecordFields:
    """Return shared experiment metadata for metrics records."""
    return {
        "baseline_preset": config.experiment.baseline_preset,
        "diffusion_enabled": config.diffusion.enabled,
        "diffusion_policy": config.diffusion.policy,
        "diffusion_graph": config.diffusion.graph if config.diffusion.enabled else None,
        "skill_update_policy": config.experiment.skill_updates.model_dump(),
    }


def runtime_record_fields(config: Config) -> dict[str, Any]:
    """Return compact runtime fields that should travel with every row."""
    return {
        "seed": config.experiment.seed,
        "models": config.models.model_dump(exclude_none=True),
        "executor_agent": HERMES_AGENT_NAME,
    }


def task_metadata_fields(
    *,
    task_id: str,
    task_config: dict[str, Any] | None,
) -> TaskMetadataFields:
    """Return flat task metadata fields for metrics rows."""
    metadata = as_mapping(task_config.get("metadata")) if task_config else {}
    verifier = as_mapping(task_config.get("verifier")) if task_config else {}
    category = as_nonempty_string(metadata.get("category"))
    difficulty = as_nonempty_string(metadata.get("difficulty"))
    raw_reward_range = metadata.get("expected_reward_range")
    expected_reward_range = None
    if isinstance(raw_reward_range, (list, tuple)) and len(raw_reward_range) == 2:
        try:
            expected_reward_range = (
                float(raw_reward_range[0]),
                float(raw_reward_range[1]),
            )
        except (TypeError, ValueError):
            expected_reward_range = None
    verifier_type = as_nonempty_string(verifier.get("type"))

    if task_config:
        expected_reward_range = expected_reward_range or (0.0, 1.0)
        verifier_type = verifier_type or SKILLFLOW_VERIFIER_TYPE

    return {
        "task_category": category,
        "task_difficulty": difficulty,
        "expected_reward_range": expected_reward_range,
        "verifier_type": verifier_type,
    }


def attach_skill_identity(
    record: IterationRecord,
    skill_hashes: dict[str, str],
    skill_version: str,
) -> None:
    """Attach snapshot identity to a metric record before serialization."""
    if not record.skill_hashes:
        record.skill_hashes = dict(skill_hashes)
    record.skill_version = skill_version
    skill_updates = list(record.skill_updates)
    if record.skill_update:
        skill_updates.append(record.skill_update)
    for skill_update in skill_updates:
        skill_update.skill_version = skill_version


def record_skill_updates(record: IterationRecord) -> list[SkillUpdate]:
    """Return all committed skill updates attached to a record."""
    updates = list(record.skill_updates)
    if record.skill_update:
        updates.append(record.skill_update)
    return updates


def build_skill_update_ledger_entry(
    *,
    record: IterationRecord,
    update: SkillUpdate,
    update_id: str,
    artifact_path: str | None = None,
    diff_path: str | None = None,
) -> SkillUpdateLedgerEntry:
    """Build a compact committed-update ledger entry."""
    provenance = update.provenance
    validation = provenance.validation if provenance is not None else None
    return SkillUpdateLedgerEntry(
        update_id=update_id,
        iteration=update.iteration,
        skill_id=update.skill_id,
        skill_version=update.skill_version,
        record_task_id=record.task_id,
        update_task_id=update.task_id,
        task_ids=list(provenance.task_ids if provenance is not None else []),
        old_skill_hash=update.old_skill_hash,
        new_skill_hash=update.new_skill_hash,
        provenance_kind=provenance.kind if provenance is not None else None,
        decision=provenance.decision if provenance is not None else None,
        reason=provenance.reason if provenance is not None else update.reasoning,
        rollback_snapshot=(
            provenance.rollback_snapshot if provenance is not None else None
        ),
        candidate_batch_id=(
            provenance.candidate_batch_id if provenance is not None else None
        ),
        candidate_batch_path=(
            provenance.candidate_batch_path if provenance is not None else None
        ),
        selected_candidate_id=(
            provenance.selected_candidate_id if provenance is not None else None
        ),
        selected_update_kind=(
            provenance.selected_update_kind if provenance is not None else None
        ),
        candidate_count=len(provenance.candidate_refs) if provenance is not None else 0,
        validation_decision=validation.decision if validation is not None else None,
        validation_reason=validation.reason if validation is not None else None,
        validation_current_mean_reward=(
            validation.current_mean_reward if validation is not None else None
        ),
        validation_candidate_mean_reward=(
            validation.candidate_mean_reward if validation is not None else None
        ),
        validation_mean_delta=validation.mean_delta if validation is not None else None,
        validation_task_ids=list(validation.task_ids if validation is not None else []),
        validation_task_count=(
            len(validation.task_results) if validation is not None else 0
        ),
        reward=record.reward,
        delta_reward=record.delta_reward,
        success=record.success,
        verifier_status=record.verifier_status,
        artifact_path=artifact_path,
        diff_path=diff_path,
    )


def skill_version(iteration: int) -> str:
    """Return the run-local skill snapshot label for an iteration."""
    return f"iter_{iteration:04d}"


def rollback_snapshot(iteration: int) -> str | None:
    """Return the prior snapshot label that can restore pre-update state."""
    if iteration <= 0:
        return None
    return skill_version(iteration - 1)


def skill_update_from_reflection(result: ReflectionResult) -> SkillUpdate:
    """Convert a committed reflection result into a metrics skill update."""
    new_skill_hash = SkillStore.content_hash(result.new_content)
    return SkillUpdate(
        skill_id=result.skill_id,
        task_id=",".join(result.provenance.task_ids),
        old_content=result.old_content,
        new_content=result.new_content,
        reasoning=result.provenance.reason,
        iteration=result.provenance.iteration,
        old_skill_hash=result.provenance.base_skill_hash,
        new_skill_hash=new_skill_hash,
        provenance=result.provenance,
    )


def delta_reward(
    previous_reward_by_task: dict[str, float],
    task_id: str,
    trace: ExecutionTrace,
) -> float | None:
    """Return reward delta against the previous scored run for this task."""
    if not trace.is_usable_feedback_signal or trace.reward is None:
        return None

    previous_reward = previous_reward_by_task.get(task_id)
    previous_reward_by_task[task_id] = trace.reward
    if previous_reward is None:
        return None
    return trace.reward - previous_reward

