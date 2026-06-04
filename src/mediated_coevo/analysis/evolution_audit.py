"""Audit helpers for skill-evolution metrics traces."""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

COEVOLUTION_TASK_ID = "__coevolution__"
ENV_FAILURE_STATUSES = {"env_failure", "parse_error", "harbor_failed"}


class IterationRewardAudit(BaseModel):
    """Reward summary for one experiment iteration."""

    iteration: int
    total_runs: int
    scored_count: int
    mean_reward: float | None = None
    task_mean_rewards: dict[str, float] = Field(default_factory=dict)


class CandidateRefAudit(BaseModel):
    """Compact selected-candidate provenance from a skill update."""

    candidate_id: str
    update_kind: str | None = None
    audit_score: float | None = None
    selected: bool = False
    rejection_reason: str | None = None


class SkillUpdateAudit(BaseModel):
    """Committed skill update plus validation provenance."""

    iteration: int
    task_id: str
    skill_id: str
    provenance_kind: str | None = None
    decision: str | None = None
    selected_candidate_id: str | None = None
    selected_update_kind: str | None = None
    candidate_batch_id: str | None = None
    has_validation: bool = False
    validation_decision: str | None = None
    validation_reason: str | None = None
    current_mean_reward: float | None = None
    candidate_mean_reward: float | None = None
    mean_delta: float | None = None
    validation_task_ids: list[str] = Field(default_factory=list)
    validation_task_count: int = 0
    candidate_refs: list[CandidateRefAudit] = Field(default_factory=list)


class SkillUpdateLedgerAudit(BaseModel):
    """Committed-update ledger row written by the experiment runtime."""

    update_id: str
    iteration: int
    skill_id: str
    skill_version: str | None = None
    record_task_id: str
    update_task_id: str = ""
    selected_candidate_id: str | None = None
    selected_update_kind: str | None = None
    validation_decision: str | None = None
    validation_reason: str | None = None
    validation_mean_delta: float | None = None
    reward: float | None = None
    delta_reward: float | None = None
    artifact_path: str | None = None
    diff_path: str | None = None


class TaskRewardEffectAudit(BaseModel):
    """Per-task adjacent reward movement around a committed skill update."""

    task_id: str
    pre_reward: float | None = None
    update_reward: float | None = None
    post_reward: float | None = None
    update_delta_from_pre: float | None = None
    post_delta_from_update: float | None = None
    post_delta_from_pre: float | None = None


class SkillUpdateEffectAudit(BaseModel):
    """Adjacent reward movement around one committed skill update."""

    update_id: str | None = None
    iteration: int
    skill_id: str
    task_id: str = ""
    selected_candidate_id: str | None = None
    selected_update_kind: str | None = None
    pre_iteration: int | None = None
    post_iteration: int | None = None
    pre_mean_reward: float | None = None
    update_mean_reward: float | None = None
    post_mean_reward: float | None = None
    update_delta_from_pre: float | None = None
    post_delta_from_update: float | None = None
    post_delta_from_pre: float | None = None
    task_effects: list[TaskRewardEffectAudit] = Field(default_factory=list)


class MediatorReportEffectAudit(BaseModel):
    """Same-task outcome movement associated with one Mediator report."""

    entry_id: str
    report_iteration: int
    task_id: str
    outcome_iteration: int
    outcome_reward: float
    previous_iteration: int | None = None
    previous_reward: float | None = None
    previous_reward_source: str | None = None
    delta_from_previous: float | None = None
    reward_source: str | None = None
    verifier_reward: float | None = None
    judge_reward: float | None = None
    headline: str = ""


class RejectedReflectionAudit(BaseModel):
    """Rejected planner/mediator reflection candidate kept as negative evidence."""

    rejection_id: str
    batch_id: str
    iteration: int
    skill_id: str
    agent_role: str
    task_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    selected_candidate_id: str | None = None
    selected_update_kind: str | None = None
    candidate_batch_id: str | None = None
    candidate_batch_path: str | None = None
    validation_decision: str | None = None
    validation_reason: str | None = None
    validation_mean_delta: float | None = None
    candidate_refs: list[CandidateRefAudit] = Field(default_factory=list)
    timestamp: str | None = None


class EvolutionAuditWarning(BaseModel):
    """One audit finding that needs manual inspection."""

    kind: str
    iteration: int
    message: str
    skill_id: str | None = None
    comparison_iteration: int | None = None
    current_mean_reward: float | None = None
    comparison_mean_reward: float | None = None
    delta: float | None = None


class EvolutionAudit(BaseModel):
    """Skill-evolution audit summary for one metrics file."""

    metrics_path: str
    experiment_dir: str | None = None
    iterations: list[IterationRewardAudit] = Field(default_factory=list)
    skill_updates: list[SkillUpdateAudit] = Field(default_factory=list)
    skill_update_ledger: list[SkillUpdateLedgerAudit] = Field(default_factory=list)
    skill_update_effects: list[SkillUpdateEffectAudit] = Field(default_factory=list)
    mediator_report_effects: list[MediatorReportEffectAudit] = Field(
        default_factory=list
    )
    rejected_reflections: list[RejectedReflectionAudit] = Field(default_factory=list)
    warnings: list[EvolutionAuditWarning] = Field(default_factory=list)


def read_metrics_rows(metrics_path: Path) -> list[dict[str, Any]]:
    """Read compact metrics JSONL rows from an experiment."""
    return _read_jsonl_objects(metrics_path)


def read_skill_update_ledger_rows(experiment_dir: Path) -> list[dict[str, Any]]:
    """Read committed skill-update ledger rows when the artifact exists."""
    ledger_path = (
        experiment_dir
        / "artifacts"
        / "skill_updates"
        / "skill_update_history.jsonl"
    )
    if not ledger_path.exists():
        return []
    return _read_jsonl_objects(ledger_path)


def read_rejected_reflection_rows(experiment_dir: Path) -> list[dict[str, Any]]:
    """Read rejected planner/mediator reflection rows when present."""
    rejected_path = experiment_dir / "history" / "rejected_reflections.jsonl"
    if not rejected_path.exists():
        return []
    return _read_jsonl_objects(rejected_path)


def read_history_rows(experiment_dir: Path) -> list[dict[str, Any]]:
    """Read outcome-tagged history rows when present."""
    history_path = experiment_dir / "history" / "history.jsonl"
    if not history_path.exists():
        return []
    return _read_jsonl_objects(history_path)


def read_judge_reward_rows(experiment_dir: Path) -> list[dict[str, Any]]:
    """Read judge reward rows when present."""
    judge_path = experiment_dir / "artifacts" / "judge_rewards.jsonl"
    if not judge_path.exists():
        return []
    return _read_jsonl_objects(judge_path)


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as metrics_file:
        for line_number, line in enumerate(metrics_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} expected a JSON object row")
            rows.append(row)
    return rows


def audit_metrics_file(
    metrics_path: Path,
    *,
    experiment_dir: Path | None = None,
    regression_tolerance: float = 1e-9,
) -> EvolutionAudit:
    """Build an evolution audit from an experiment metrics JSONL file."""
    resolved_experiment_dir = experiment_dir or metrics_path.parent
    return build_evolution_audit(
        read_metrics_rows(metrics_path),
        skill_update_ledger_rows=read_skill_update_ledger_rows(
            resolved_experiment_dir,
        ),
        rejected_reflection_rows=read_rejected_reflection_rows(
            resolved_experiment_dir,
        ),
        history_rows=read_history_rows(resolved_experiment_dir),
        judge_reward_rows=read_judge_reward_rows(resolved_experiment_dir),
        metrics_path=metrics_path,
        experiment_dir=resolved_experiment_dir,
        regression_tolerance=regression_tolerance,
    )


def audit_experiment(
    experiment_dir: Path,
    *,
    regression_tolerance: float = 1e-9,
) -> EvolutionAudit:
    """Build an evolution audit from an experiment directory."""
    return audit_metrics_file(
        experiment_dir / "metrics.jsonl",
        experiment_dir=experiment_dir,
        regression_tolerance=regression_tolerance,
    )


def write_evolution_audit(audit: EvolutionAudit, path: Path) -> None:
    """Write an evolution audit as stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as audit_file:
        write_evolution_audit_json(audit, audit_file)
        audit_file.write("\n")


def write_evolution_audit_json(audit: EvolutionAudit, output: Any) -> None:
    """Write an evolution audit JSON payload to a file-like object."""
    json.dump(audit.model_dump(mode="json"), output, indent=2, sort_keys=True)


def build_evolution_audit(
    rows: Iterable[Mapping[str, Any]],
    *,
    skill_update_ledger_rows: Iterable[Mapping[str, Any]] | None = None,
    rejected_reflection_rows: Iterable[Mapping[str, Any]] | None = None,
    history_rows: Iterable[Mapping[str, Any]] | None = None,
    judge_reward_rows: Iterable[Mapping[str, Any]] | None = None,
    metrics_path: Path | str = "",
    experiment_dir: Path | str | None = None,
    regression_tolerance: float = 1e-9,
) -> EvolutionAudit:
    """Summarize reward trend and skill-update provenance from metrics rows."""
    materialized_rows = list(rows)
    iterations = _iteration_reward_audits(materialized_rows)
    skill_updates = _skill_update_audits(materialized_rows)
    skill_update_ledger = _skill_update_ledger_audits(
        list(skill_update_ledger_rows or [])
    )
    skill_update_effects = _skill_update_effect_audits(
        iterations,
        skill_updates,
        skill_update_ledger,
    )
    mediator_report_effects = _mediator_report_effect_audits(
        iterations,
        list(history_rows or []),
        list(judge_reward_rows or []),
    )
    rejected_reflections = _rejected_reflection_audits(
        list(rejected_reflection_rows or [])
    )
    warnings = _evolution_warnings(
        iterations,
        skill_updates,
        regression_tolerance=regression_tolerance,
    )
    return EvolutionAudit(
        metrics_path=str(metrics_path),
        experiment_dir=str(experiment_dir) if experiment_dir is not None else None,
        iterations=iterations,
        skill_updates=skill_updates,
        skill_update_ledger=skill_update_ledger,
        skill_update_effects=skill_update_effects,
        mediator_report_effects=mediator_report_effects,
        rejected_reflections=rejected_reflections,
        warnings=warnings,
    )


def _iteration_reward_audits(
    rows: list[Mapping[str, Any]],
) -> list[IterationRewardAudit]:
    total_runs_by_iteration: dict[int, int] = {}
    rewards_by_iteration: dict[int, list[float]] = {}
    task_rewards_by_iteration: dict[int, dict[str, list[float]]] = {}

    for row in rows:
        iteration = _as_int(row.get("iteration"))
        task_id = _as_str(row.get("task_id"))
        if iteration is None or task_id == COEVOLUTION_TASK_ID:
            continue

        total_runs_by_iteration[iteration] = (
            total_runs_by_iteration.get(iteration, 0) + 1
        )
        reward = _scored_reward(row)
        if reward is None:
            continue

        rewards_by_iteration.setdefault(iteration, []).append(reward)
        task_rewards_by_iteration.setdefault(iteration, {}).setdefault(task_id, []).append(
            reward
        )

    summaries = []
    for iteration in sorted(total_runs_by_iteration):
        rewards = rewards_by_iteration.get(iteration, [])
        task_mean_rewards = {
            task_id: _required_mean(task_rewards)
            for task_id, task_rewards in sorted(
                task_rewards_by_iteration.get(iteration, {}).items()
            )
        }
        summaries.append(
            IterationRewardAudit(
                iteration=iteration,
                total_runs=total_runs_by_iteration[iteration],
                scored_count=len(rewards),
                mean_reward=_mean(rewards),
                task_mean_rewards=task_mean_rewards,
            )
        )
    return summaries


def _skill_update_audits(rows: list[Mapping[str, Any]]) -> list[SkillUpdateAudit]:
    updates = []
    for row in rows:
        row_iteration = _as_int(row.get("iteration"))
        row_task_id = _as_str(row.get("task_id"))
        raw_updates = row.get("skill_updates")
        if not isinstance(raw_updates, list):
            continue

        for raw_update in raw_updates:
            if not isinstance(raw_update, Mapping):
                continue
            update = _skill_update_audit(
                raw_update,
                row_iteration=row_iteration,
                row_task_id=row_task_id,
            )
            updates.append(update)
    return updates


def _skill_update_ledger_audits(
    rows: list[Mapping[str, Any]],
) -> list[SkillUpdateLedgerAudit]:
    ledger_entries = []
    for row in rows:
        update_id = _as_str_or_none(row.get("update_id"))
        iteration = _as_int(row.get("iteration"))
        skill_id = _as_str_or_none(row.get("skill_id"))
        record_task_id = _as_str_or_none(row.get("record_task_id"))
        if (
            update_id is None
            or iteration is None
            or skill_id is None
            or record_task_id is None
        ):
            continue
        ledger_entries.append(
            SkillUpdateLedgerAudit(
                update_id=update_id,
                iteration=iteration,
                skill_id=skill_id,
                skill_version=_as_str_or_none(row.get("skill_version")),
                record_task_id=record_task_id,
                update_task_id=_as_str(row.get("update_task_id")),
                selected_candidate_id=_as_str_or_none(
                    row.get("selected_candidate_id")
                ),
                selected_update_kind=_as_str_or_none(
                    row.get("selected_update_kind")
                ),
                validation_decision=_as_str_or_none(row.get("validation_decision")),
                validation_reason=_as_str_or_none(row.get("validation_reason")),
                validation_mean_delta=_as_float(row.get("validation_mean_delta")),
                reward=_as_float(row.get("reward")),
                delta_reward=_as_float(row.get("delta_reward")),
                artifact_path=_as_str_or_none(row.get("artifact_path")),
                diff_path=_as_str_or_none(row.get("diff_path")),
            )
        )
    return ledger_entries


def _skill_update_effect_audits(
    iterations: list[IterationRewardAudit],
    skill_updates: list[SkillUpdateAudit],
    skill_update_ledger: list[SkillUpdateLedgerAudit],
) -> list[SkillUpdateEffectAudit]:
    effects = []
    by_iteration = {summary.iteration: summary for summary in iterations}
    sorted_iterations = [summary.iteration for summary in iterations]
    seen_keys: set[tuple[int, str, str]] = set()

    for metrics_update in skill_updates:
        current = by_iteration.get(metrics_update.iteration)
        if current is None:
            continue

        seen_keys.add(
            (metrics_update.iteration, metrics_update.skill_id, metrics_update.task_id)
        )
        effects.append(
            _skill_update_effect_audit(
                update_id=None,
                iteration=metrics_update.iteration,
                skill_id=metrics_update.skill_id,
                task_id=metrics_update.task_id,
                selected_candidate_id=metrics_update.selected_candidate_id,
                selected_update_kind=metrics_update.selected_update_kind,
                current=current,
                by_iteration=by_iteration,
                sorted_iterations=sorted_iterations,
            )
        )

    for ledger_update in skill_update_ledger:
        task_id = ledger_update.update_task_id or ledger_update.record_task_id
        key = (ledger_update.iteration, ledger_update.skill_id, task_id)
        if key in seen_keys:
            continue

        current = by_iteration.get(ledger_update.iteration)
        if current is None:
            continue

        effects.append(
            _skill_update_effect_audit(
                update_id=ledger_update.update_id,
                iteration=ledger_update.iteration,
                skill_id=ledger_update.skill_id,
                task_id=task_id,
                selected_candidate_id=ledger_update.selected_candidate_id,
                selected_update_kind=ledger_update.selected_update_kind,
                current=current,
                by_iteration=by_iteration,
                sorted_iterations=sorted_iterations,
            )
        )
    return effects


def _skill_update_effect_audit(
    *,
    update_id: str | None,
    iteration: int,
    skill_id: str,
    task_id: str,
    selected_candidate_id: str | None,
    selected_update_kind: str | None,
    current: IterationRewardAudit,
    by_iteration: Mapping[int, IterationRewardAudit],
    sorted_iterations: list[int],
) -> SkillUpdateEffectAudit:
    pre_iteration = _neighbor_iteration(sorted_iterations, iteration, -1)
    post_iteration = _neighbor_iteration(sorted_iterations, iteration, 1)
    pre = by_iteration.get(pre_iteration) if pre_iteration is not None else None
    post = by_iteration.get(post_iteration) if post_iteration is not None else None

    return SkillUpdateEffectAudit(
        update_id=update_id,
        iteration=iteration,
        skill_id=skill_id,
        task_id=task_id,
        selected_candidate_id=selected_candidate_id,
        selected_update_kind=selected_update_kind,
        pre_iteration=pre_iteration,
        post_iteration=post_iteration,
        pre_mean_reward=pre.mean_reward if pre is not None else None,
        update_mean_reward=current.mean_reward,
        post_mean_reward=post.mean_reward if post is not None else None,
        update_delta_from_pre=_delta(
            current.mean_reward,
            pre.mean_reward if pre is not None else None,
        ),
        post_delta_from_update=_delta(
            post.mean_reward if post is not None else None,
            current.mean_reward,
        ),
        post_delta_from_pre=_delta(
            post.mean_reward if post is not None else None,
            pre.mean_reward if pre is not None else None,
        ),
        task_effects=_task_reward_effects(pre, current, post),
    )


def _task_reward_effects(
    pre: IterationRewardAudit | None,
    current: IterationRewardAudit,
    post: IterationRewardAudit | None,
) -> list[TaskRewardEffectAudit]:
    task_ids = set(current.task_mean_rewards)
    if pre is not None:
        task_ids.update(pre.task_mean_rewards)
    if post is not None:
        task_ids.update(post.task_mean_rewards)

    effects = []
    for task_id in sorted(task_ids):
        pre_reward = pre.task_mean_rewards.get(task_id) if pre is not None else None
        update_reward = current.task_mean_rewards.get(task_id)
        post_reward = post.task_mean_rewards.get(task_id) if post is not None else None
        effects.append(
            TaskRewardEffectAudit(
                task_id=task_id,
                pre_reward=pre_reward,
                update_reward=update_reward,
                post_reward=post_reward,
                update_delta_from_pre=_delta(update_reward, pre_reward),
                post_delta_from_update=_delta(post_reward, update_reward),
                post_delta_from_pre=_delta(post_reward, pre_reward),
            )
        )
    return effects


def _rejected_reflection_audits(
    rows: list[Mapping[str, Any]],
) -> list[RejectedReflectionAudit]:
    rejected_reflections = []
    for row in rows:
        rejection_id = _as_str_or_none(row.get("rejection_id"))
        batch_id = _as_str_or_none(row.get("batch_id"))
        iteration = _as_int(row.get("iteration"))
        skill_id = _as_str_or_none(row.get("skill_id"))
        agent_role = _as_str_or_none(row.get("agent_role"))
        if (
            rejection_id is None
            or batch_id is None
            or iteration is None
            or skill_id is None
            or agent_role is None
        ):
            continue

        validation = _mapping_or_none(row.get("validation"))
        rejected_reflections.append(
            RejectedReflectionAudit(
                rejection_id=rejection_id,
                batch_id=batch_id,
                iteration=iteration,
                skill_id=skill_id,
                agent_role=agent_role,
                task_ids=_str_list(row.get("task_ids")),
                reason=_as_str(row.get("reason")),
                selected_candidate_id=_as_str_or_none(
                    row.get("selected_candidate_id")
                ),
                selected_update_kind=_as_str_or_none(
                    row.get("selected_update_kind")
                ),
                candidate_batch_id=_as_str_or_none(row.get("candidate_batch_id")),
                candidate_batch_path=_as_str_or_none(row.get("candidate_batch_path")),
                validation_decision=(
                    _as_str_or_none(validation.get("decision"))
                    if validation
                    else None
                ),
                validation_reason=(
                    _as_str_or_none(validation.get("reason"))
                    if validation
                    else None
                ),
                validation_mean_delta=(
                    _as_float(validation.get("mean_delta")) if validation else None
                ),
                candidate_refs=_candidate_ref_audits(row),
                timestamp=_as_str_or_none(row.get("timestamp")),
            )
        )
    return rejected_reflections


def _mediator_report_effect_audits(
    iterations: list[IterationRewardAudit],
    history_rows: list[Mapping[str, Any]],
    judge_reward_rows: list[Mapping[str, Any]],
) -> list[MediatorReportEffectAudit]:
    effects = []
    by_iteration = {summary.iteration: summary for summary in iterations}
    sorted_iterations = [summary.iteration for summary in iterations]
    judge_rewards = _judge_rewards_by_task_iteration(judge_reward_rows)

    for row in history_rows:
        if _as_str(row.get("agent_role")) != "mediator":
            continue
        entry_id = _as_str_or_none(row.get("entry_id"))
        report_iteration = _as_int(row.get("iteration"))
        outcome_reward = _as_float(row.get("reward"))
        metadata = _mapping_or_none(row.get("metadata"))
        if (
            entry_id is None
            or report_iteration is None
            or outcome_reward is None
            or metadata is None
        ):
            continue

        task_id = _as_str_or_none(metadata.get("outcome_task_id"))
        if task_id is None:
            task_id = _as_str_or_none(metadata.get("task_id"))
        outcome_iteration = _as_int(metadata.get("outcome_iteration"))
        if task_id is None or outcome_iteration is None:
            continue

        reward_source = _as_str_or_none(metadata.get("reward_source"))
        previous_iteration = _previous_task_iteration(
            sorted_iterations,
            by_iteration,
            task_id=task_id,
            before_iteration=outcome_iteration,
        )
        previous_reward = None
        previous_reward_source = None
        if previous_iteration is not None:
            if reward_source == "judge":
                previous_reward = judge_rewards.get((task_id, previous_iteration))
                previous_reward_source = "judge" if previous_reward is not None else None
            if previous_reward is None:
                previous_summary = by_iteration[previous_iteration]
                previous_reward = previous_summary.task_mean_rewards.get(task_id)
                previous_reward_source = "metrics" if previous_reward is not None else None

        effects.append(
            MediatorReportEffectAudit(
                entry_id=entry_id,
                report_iteration=report_iteration,
                task_id=task_id,
                outcome_iteration=outcome_iteration,
                outcome_reward=outcome_reward,
                previous_iteration=previous_iteration,
                previous_reward=previous_reward,
                previous_reward_source=previous_reward_source,
                delta_from_previous=_delta(outcome_reward, previous_reward),
                reward_source=reward_source,
                verifier_reward=_as_float(metadata.get("verifier_reward")),
                judge_reward=_as_float(metadata.get("judge_reward")),
                headline=_mediator_headline(row),
            )
        )
    return effects


def _judge_rewards_by_task_iteration(
    rows: list[Mapping[str, Any]],
) -> dict[tuple[str, int], float]:
    rewards = {}
    for row in rows:
        task_id = _as_str_or_none(row.get("task_id"))
        iteration = _as_int(row.get("iteration"))
        judge_reward = _as_float(row.get("judge_reward"))
        if task_id is None or iteration is None or judge_reward is None:
            continue
        rewards[(task_id, iteration)] = judge_reward
    return rewards


def _previous_task_iteration(
    sorted_iterations: list[int],
    by_iteration: Mapping[int, IterationRewardAudit],
    *,
    task_id: str,
    before_iteration: int,
) -> int | None:
    previous_iterations = [
        iteration
        for iteration in sorted_iterations
        if iteration < before_iteration
        and task_id in by_iteration[iteration].task_mean_rewards
    ]
    if not previous_iterations:
        return None
    return previous_iterations[-1]


def _mediator_headline(row: Mapping[str, Any]) -> str:
    payload = _mapping_or_none(row.get("payload"))
    if payload is None:
        return ""
    return _as_str(payload.get("headline"))


def _skill_update_audit(
    raw_update: Mapping[str, Any],
    *,
    row_iteration: int | None,
    row_task_id: str,
) -> SkillUpdateAudit:
    provenance = _mapping_or_none(raw_update.get("provenance"))
    validation = _mapping_or_none(provenance.get("validation")) if provenance else None

    update_iteration = _as_int(raw_update.get("iteration"))
    if update_iteration is None:
        update_iteration = row_iteration if row_iteration is not None else 0

    skill_id = _first_str(
        raw_update.get("skill_id"),
        provenance.get("skill_id") if provenance else None,
        default="unknown",
    )
    task_id = _first_str(raw_update.get("task_id"), row_task_id, default="")

    validation_task_ids = _str_list(validation.get("task_ids")) if validation else []
    validation_task_results = validation.get("task_results") if validation else None
    validation_task_count = (
        len(validation_task_results) if isinstance(validation_task_results, list) else 0
    )

    return SkillUpdateAudit(
        iteration=update_iteration,
        task_id=task_id,
        skill_id=skill_id,
        provenance_kind=_as_str_or_none(provenance.get("kind")) if provenance else None,
        decision=_as_str_or_none(provenance.get("decision")) if provenance else None,
        selected_candidate_id=(
            _as_str_or_none(provenance.get("selected_candidate_id"))
            if provenance
            else None
        ),
        selected_update_kind=(
            _as_str_or_none(provenance.get("selected_update_kind"))
            if provenance
            else None
        ),
        candidate_batch_id=(
            _as_str_or_none(provenance.get("candidate_batch_id"))
            if provenance
            else None
        ),
        has_validation=validation is not None,
        validation_decision=(
            _as_str_or_none(validation.get("decision")) if validation else None
        ),
        validation_reason=_as_str_or_none(validation.get("reason")) if validation else None,
        current_mean_reward=(
            _as_float(validation.get("current_mean_reward")) if validation else None
        ),
        candidate_mean_reward=(
            _as_float(validation.get("candidate_mean_reward")) if validation else None
        ),
        mean_delta=_as_float(validation.get("mean_delta")) if validation else None,
        validation_task_ids=validation_task_ids,
        validation_task_count=validation_task_count,
        candidate_refs=_candidate_ref_audits(provenance),
    )


def _candidate_ref_audits(
    provenance: Mapping[str, Any] | None,
) -> list[CandidateRefAudit]:
    if provenance is None:
        return []
    raw_refs = provenance.get("candidate_refs")
    if not isinstance(raw_refs, list):
        return []

    refs = []
    for raw_ref in raw_refs:
        if not isinstance(raw_ref, Mapping):
            continue
        candidate_id = _as_str_or_none(raw_ref.get("candidate_id"))
        if candidate_id is None:
            continue
        refs.append(
            CandidateRefAudit(
                candidate_id=candidate_id,
                update_kind=_as_str_or_none(raw_ref.get("update_kind")),
                audit_score=_as_float(raw_ref.get("audit_score")),
                selected=bool(raw_ref.get("selected", False)),
                rejection_reason=_as_str_or_none(raw_ref.get("rejection_reason")),
            )
        )
    return refs


def _evolution_warnings(
    iterations: list[IterationRewardAudit],
    skill_updates: list[SkillUpdateAudit],
    *,
    regression_tolerance: float,
) -> list[EvolutionAuditWarning]:
    warnings: list[EvolutionAuditWarning] = []
    by_iteration = {summary.iteration: summary for summary in iterations}
    sorted_iterations = [summary.iteration for summary in iterations]

    for update in skill_updates:
        if not update.has_validation:
            warnings.append(_missing_validation_warning(update))
        elif update.validation_reason == "validation_disabled":
            warnings.append(_disabled_validation_warning(update))
        elif update.validation_decision == "rejected":
            warnings.append(_rejected_validation_warning(update))

        current = by_iteration.get(update.iteration)
        if current is None or current.mean_reward is None:
            continue

        previous_iteration = _neighbor_iteration(sorted_iterations, update.iteration, -1)
        if previous_iteration is not None:
            previous = by_iteration[previous_iteration]
            warning = _regression_warning(
                update,
                kind="update_on_regressed_iteration",
                current=current,
                comparison=previous,
                regression_tolerance=regression_tolerance,
            )
            if warning is not None:
                warnings.append(warning)

        next_iteration = _neighbor_iteration(sorted_iterations, update.iteration, 1)
        if next_iteration is not None:
            next_summary = by_iteration[next_iteration]
            warning = _regression_warning(
                update,
                kind="post_update_regression",
                current=next_summary,
                comparison=current,
                regression_tolerance=regression_tolerance,
            )
            if warning is not None:
                warnings.append(warning)
    return warnings


def _missing_validation_warning(update: SkillUpdateAudit) -> EvolutionAuditWarning:
    return EvolutionAuditWarning(
        kind="missing_validation",
        iteration=update.iteration,
        skill_id=update.skill_id,
        message=(
            f"{update.skill_id} update committed without validation provenance "
            f"at iteration {update.iteration}."
        ),
    )


def _disabled_validation_warning(update: SkillUpdateAudit) -> EvolutionAuditWarning:
    return EvolutionAuditWarning(
        kind="validation_disabled",
        iteration=update.iteration,
        skill_id=update.skill_id,
        message=(
            f"{update.skill_id} update validation was recorded as disabled "
            f"at iteration {update.iteration}."
        ),
    )


def _rejected_validation_warning(update: SkillUpdateAudit) -> EvolutionAuditWarning:
    return EvolutionAuditWarning(
        kind="committed_rejected_validation",
        iteration=update.iteration,
        skill_id=update.skill_id,
        message=(
            f"{update.skill_id} update has rejected validation provenance "
            f"at iteration {update.iteration}."
        ),
    )


def _regression_warning(
    update: SkillUpdateAudit,
    *,
    kind: str,
    current: IterationRewardAudit,
    comparison: IterationRewardAudit,
    regression_tolerance: float,
) -> EvolutionAuditWarning | None:
    if current.mean_reward is None or comparison.mean_reward is None:
        return None
    delta = current.mean_reward - comparison.mean_reward
    if delta >= -regression_tolerance:
        return None

    return EvolutionAuditWarning(
        kind=kind,
        iteration=update.iteration,
        skill_id=update.skill_id,
        comparison_iteration=comparison.iteration,
        current_mean_reward=current.mean_reward,
        comparison_mean_reward=comparison.mean_reward,
        delta=delta,
        message=(
            f"{update.skill_id} update at iteration {update.iteration} is near "
            f"a reward regression: iteration {current.iteration} mean "
            f"{current.mean_reward:.3f} vs iteration {comparison.iteration} mean "
            f"{comparison.mean_reward:.3f} (delta {delta:.3f})."
        ),
    )


def _neighbor_iteration(
    sorted_iterations: list[int],
    iteration: int,
    offset: int,
) -> int | None:
    try:
        index = sorted_iterations.index(iteration)
    except ValueError:
        return None
    neighbor_index = index + offset
    if neighbor_index < 0 or neighbor_index >= len(sorted_iterations):
        return None
    return sorted_iterations[neighbor_index]


def _scored_reward(row: Mapping[str, Any]) -> float | None:
    reward = _as_float(row.get("reward"))
    if reward is None:
        return None
    status = _first_str(row.get("verifier_status"), row.get("status"), default="")
    if status in ENV_FAILURE_STATUSES:
        return None
    return reward


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return _required_mean(values)


def _required_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _delta(current: float | None, comparison: float | None) -> float | None:
    if current is None or comparison is None:
        return None
    return current - comparison


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_as_str(item) for item in value]


def _first_str(*values: Any, default: str) -> str:
    for value in values:
        text = _as_str_or_none(value)
        if text is not None:
            return text
    return default


def _as_str(value: Any) -> str:
    return _as_str_or_none(value) or ""


def _as_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    """Run a metrics evolution audit from the command line."""
    parser = ArgumentParser(
        description="Audit reward trends and skill-update provenance in metrics.jsonl."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Experiment directory or metrics.jsonl path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the audit JSON. Defaults to stdout.",
    )
    args = parser.parse_args(argv)

    metrics_path, experiment_dir = _resolve_audit_target(args.path, parser)
    audit = audit_metrics_file(metrics_path, experiment_dir=experiment_dir)

    if args.output is not None:
        write_evolution_audit(audit, args.output)
        return 0

    write_evolution_audit_json(audit, sys.stdout)
    sys.stdout.write("\n")
    return 0


def _resolve_audit_target(
    path: Path,
    parser: ArgumentParser,
) -> tuple[Path, Path]:
    if path.is_dir():
        metrics_path = path / "metrics.jsonl"
        experiment_dir = path
    else:
        metrics_path = path
        experiment_dir = path.parent

    if not metrics_path.exists():
        parser.error(f"metrics file not found: {metrics_path}")
    return metrics_path, experiment_dir


if __name__ == "__main__":
    raise SystemExit(main())
