"""Full iteration record — snapshot of one plan→execute→feedback cycle."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from mediated_coevo.experiment.conditions import ConditionName
from mediated_coevo.runtime.token_budget import TokenBudgetEvent

from .report import MediatorReport
from .task import TaskSpec
from .trace import ExecutionTrace

TransferContextKind = Literal["diffusion", "cross_task_prior", "none"]


class IterationRecord(BaseModel):
    """Complete record of a single iteration for metrics and analysis."""

    iteration: int
    task_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    task_spec: TaskSpec | None = None
    execution_trace: ExecutionTrace | None = None
    mediator_report: MediatorReport | None = None

    reward: float | None = None
    total_tokens: int = 0
    llm_token_events: list[TokenBudgetEvent] = Field(default_factory=list)
    duration_sec: float = 0.0
    run_id: str | None = None

    mediator_report_id: str | None = None
    condition_name: ConditionName = "learned_mediator"
    seed: int | None = None
    models: dict[str, str] = Field(default_factory=dict)
    executor_agent: str | None = None
    baseline_preset: str | None = None
    diffusion_policy: str = "none"
    diffusion_enabled: bool = False
    diffusion_graph: str | None = None
    graph_snapshot_id: str | None = None
    diffusion_artifacts_eligible: int = 0
    diffusion_artifacts_selected: int = 0
    diffusion_artifacts_rendered: int = 0
    diffusion_artifact_store_path: str | None = None
    diffusion_artifact_store_count: int = 0
    diffusion_artifact_store_frozen: bool = False
    transfer_context_kind: TransferContextKind = "none"
    transfer_context_tokens: int = 0
    same_task_prior_tokens: int = 0
    total_planner_prior_context_tokens: int = 0
    max_same_task_prior_tokens: int = 0
    max_transfer_context_tokens: int = 0
    max_total_prior_context_tokens: int = 0
    context_budget_violation: bool = False
    compacted_diffusion_artifact_ids: list[str] = Field(default_factory=list)
    dropped_for_budget_artifact_ids: list[str] = Field(default_factory=list)
    source_task_ids: list[str] = Field(default_factory=list)
    reward_after_diffusion_context: float | None = None
    regression_after_diffusion_context: bool | None = None
    skill_hashes: dict[str, str] = Field(default_factory=dict)
    success: bool | None = None
    verifier_status: str | None = None
    delta_reward: float | None = None
    token_totals_by_agent: dict[str, int] = Field(default_factory=dict)

    task_category: str | None = None
    task_difficulty: str | None = None
    expected_reward_range: tuple[float, float] | None = None
    verifier_type: str | None = None
