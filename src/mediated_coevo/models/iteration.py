"""Full iteration record — snapshot of one plan→execute→feedback→update cycle."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from mediated_coevo.experiment.conditions import ConditionName
from mediated_coevo.runtime.token_budget import TokenBudgetEvent
from .task import TaskSpec
from .trace import ExecutionTrace
from .report import MediatorReport
from .skill import SkillUpdate


class IterationRecord(BaseModel):
    """Complete record of a single iteration for metrics and analysis."""

    iteration: int
    task_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    task_spec: TaskSpec | None = None
    execution_trace: ExecutionTrace | None = None
    mediator_report: MediatorReport | None = None
    skill_update: SkillUpdate | None = None
    skill_updates: list[SkillUpdate] = Field(default_factory=list)

    reward: float | None = None
    total_tokens: int = 0
    llm_token_events: list[TokenBudgetEvent] = Field(default_factory=list)
    duration_sec: float = 0.0
    run_id: str | None = None

    mediator_history_entry_id: str | None = None
    planner_history_entry_id: str | None = None
    history_entry_ids: dict[str, str] = Field(default_factory=dict)
    mediator_report_id: str | None = None
    proposal_ids: list[str] = Field(default_factory=list)
    condition_name: ConditionName = "learned_mediator"
    seed: int | None = None
    models: dict[str, str] = Field(default_factory=dict)
    executor_agent: str | None = None
    baseline_preset: str | None = None
    cross_task_feedback_enabled: bool = False
    skill_update_policy: dict[str, bool] = Field(default_factory=dict)
    skill_hashes: dict[str, str] = Field(default_factory=dict)
    skill_version: str | None = None
    success: bool | None = None
    verifier_status: str | None = None
    delta_reward: float | None = None
    token_totals_by_agent: dict[str, int] = Field(default_factory=dict)
    advisor_decision: str | None = None
    advisor_reason: str | None = None
    advisor_rejection_id: str | None = None

    task_category: str | None = None
    task_difficulty: str | None = None
    expected_reward_range: tuple[float, float] | None = None
    verifier_type: str | None = None
