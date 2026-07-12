"""Compact metrics serialization helpers."""

from __future__ import annotations

from typing import Any

from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.runtime.token_budget import TokenBudgetEvent


def metric_row(record: IterationRecord) -> dict[str, Any]:
    """Return a compact metrics row without bulky artifacts or prompts."""
    trace = record.execution_trace
    token_usage_by_agent: dict[str, dict[str, int]] = {}
    if trace is not None:
        _add_token_usage(
            token_usage_by_agent,
            "executor",
            prompt_tokens=trace.token_usage.input_tokens,
            completion_tokens=trace.token_usage.output_tokens,
        )
    for event in record.llm_token_events:
        _add_token_usage(
            token_usage_by_agent,
            event.label.split(".", 1)[0] if event.label else "llm",
            prompt_tokens=event.prompt_tokens,
            completion_tokens=event.completion_tokens,
        )

    skill_updates = []
    if record.skill_update:
        skill_updates.append(record.skill_update)
    skill_updates.extend(record.skill_updates)
    skill_update_summaries = [
        {
            "skill_id": update.skill_id,
            "task_id": update.task_id,
            "iteration": update.iteration,
            "old_skill_hash": update.old_skill_hash,
            "new_skill_hash": update.new_skill_hash,
            "skill_version": update.skill_version,
            "reasoning": update.reasoning,
            "provenance": (
                update.provenance.model_dump(mode="json")
                if update.provenance is not None
                else None
            ),
        }
        for update in skill_updates
    ]
    success = record.success
    verifier_status = record.verifier_status
    harbor_job_path = None
    harbor_trial_path = None
    harbor_trial_id = None
    trace_artifact_path = None
    trace_metadata: dict[str, str] = {}
    if trace is not None:
        success = metric_success(trace)
        verifier_status = metric_verifier_status(trace)
        harbor_job_path = trace.harbor_paths.get("job")
        harbor_trial_path = trace.harbor_paths.get("trial")
        harbor_trial_id = trace.harbor_trial_id
        trace_metadata = trace.harbor_metadata
        trace_artifact_path = (
            f"artifacts/traces/{record.task_id}_iter{record.iteration:04d}.json"
        )

    return {
        "iteration": record.iteration,
        "task_id": record.task_id,
        "timestamp": record.timestamp.isoformat(),
        "run_id": record.run_id,
        "condition_name": record.condition_name,
        "seed": record.seed,
        "baseline_preset": record.baseline_preset,
        "diffusion_enabled": record.diffusion_enabled,
        "diffusion_policy": record.diffusion_policy,
        "diffusion_graph": record.diffusion_graph,
        "graph_snapshot_id": record.graph_snapshot_id,
        "diffusion_artifacts_eligible": record.diffusion_artifacts_eligible,
        "diffusion_artifacts_selected": record.diffusion_artifacts_selected,
        "diffusion_artifacts_rendered": record.diffusion_artifacts_rendered,
        "diffusion_artifact_store_path": record.diffusion_artifact_store_path,
        "diffusion_artifact_store_count": record.diffusion_artifact_store_count,
        "diffusion_artifact_store_frozen": record.diffusion_artifact_store_frozen,
        "transfer_context_kind": record.transfer_context_kind,
        "transfer_context_tokens": record.transfer_context_tokens,
        "same_task_prior_tokens": record.same_task_prior_tokens,
        "total_planner_prior_context_tokens": (
            record.total_planner_prior_context_tokens
        ),
        "max_same_task_prior_tokens": record.max_same_task_prior_tokens,
        "max_transfer_context_tokens": record.max_transfer_context_tokens,
        "max_total_prior_context_tokens": record.max_total_prior_context_tokens,
        "context_budget_violation": record.context_budget_violation,
        "compacted_diffusion_artifact_ids": record.compacted_diffusion_artifact_ids,
        "dropped_for_budget_artifact_ids": record.dropped_for_budget_artifact_ids,
        "source_task_ids": record.source_task_ids,
        "reward_after_diffusion_context": record.reward_after_diffusion_context,
        "regression_after_diffusion_context": (
            record.regression_after_diffusion_context
        ),
        "skill_update_policy": record.skill_update_policy,
        "planner_model": record.models.get("planner"),
        "executor_model": record.models.get("executor"),
        "mediator_model": record.models.get("mediator"),
        "executor_agent": record.executor_agent,
        "skill_hashes": record.skill_hashes,
        "skill_version": record.skill_version,
        "mediator_history_entry_id": record.mediator_history_entry_id,
        "planner_history_entry_id": record.planner_history_entry_id,
        "history_entry_ids": record.history_entry_ids,
        "mediator_report_id": record.mediator_report_id,
        "proposal_ids": record.proposal_ids,
        "reward": record.reward,
        "delta_reward": record.delta_reward,
        "success": success,
        "verifier_status": verifier_status,
        "duration_sec": record.duration_sec,
        "total_tokens": record.total_tokens,
        "prompt_tokens_by_agent": {
            agent: usage["prompt_tokens"]
            for agent, usage in token_usage_by_agent.items()
        },
        "completion_tokens_by_agent": {
            agent: usage["completion_tokens"]
            for agent, usage in token_usage_by_agent.items()
        },
        "total_tokens_by_agent": {
            agent: usage["total_tokens"]
            for agent, usage in token_usage_by_agent.items()
        },
        "llm_token_events": [
            event.model_dump(mode="json") for event in record.llm_token_events
        ],
        "harbor_job_path": harbor_job_path,
        "harbor_trial_path": harbor_trial_path,
        "harbor_trial_id": harbor_trial_id,
        "executor_policy_hash": trace_metadata.get("executor_policy_hash"),
        "executor_policy_injected": trace_metadata.get("executor_policy_injected"),
        "executor_policy_injection": trace_metadata.get("executor_policy_injection"),
        "executor_token_source": trace_metadata.get("executor_token_source"),
        "executor_cache_read_tokens": trace_metadata.get(
            "executor_session_cache_read_tokens"
        ),
        "executor_reported_cost_usd": trace_metadata.get("agent_result.cost_usd"),
        "executor_reported_cost_source": trace_metadata.get(
            "executor_reported_cost_source"
        ),
        "task_resource_count": trace_metadata.get("task_resource_count"),
        "task_resource_names": trace_metadata.get("task_resource_names"),
        "verifier_contract_kind": trace_metadata.get("verifier_contract_kind"),
        "trace_artifact_path": trace_artifact_path,
        "advisor_decision": record.advisor_decision,
        "advisor_reason": record.advisor_reason,
        "advisor_rejection_id": record.advisor_rejection_id,
        "task_category": record.task_category,
        "task_difficulty": record.task_difficulty,
        "expected_reward_range": (
            list(record.expected_reward_range)
            if record.expected_reward_range is not None
            else None
        ),
        "verifier_type": record.verifier_type,
        "skill_updates": skill_update_summaries,
    }


def metric_success(trace: ExecutionTrace) -> bool:
    """Return whether a scored task run succeeded."""
    return trace.status == "ok" and trace.reward is not None and trace.reward > 0


def metric_verifier_status(trace: ExecutionTrace) -> str:
    """Distinguish valid zero-reward task failures from environment failures."""
    if trace.status == "ok" and trace.reward == 0:
        return "task_failed"
    return trace.status


def token_totals_by_agent(
    trace: ExecutionTrace | None,
    llm_token_events: list[TokenBudgetEvent],
) -> dict[str, int]:
    """Return compact token totals grouped by component."""
    totals: dict[str, int] = {}
    executor_tokens = 0
    if trace:
        executor_tokens = (
            trace.token_usage.input_tokens + trace.token_usage.output_tokens
        )
    if executor_tokens:
        totals["executor"] = executor_tokens
    for event in llm_token_events:
        agent = event.label.split(".", 1)[0] if event.label else "llm"
        totals[agent] = totals.get(agent, 0) + event.total_tokens
    return totals


def _add_token_usage(
    usage_by_agent: dict[str, dict[str, int]],
    agent: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    usage = usage_by_agent.setdefault(
        agent,
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    usage["prompt_tokens"] += prompt_tokens
    usage["completion_tokens"] += completion_tokens
    usage["total_tokens"] += prompt_tokens + completion_tokens
