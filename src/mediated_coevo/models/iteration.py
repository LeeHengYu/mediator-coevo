"""Full iteration record — snapshot of one plan→execute→feedback→update cycle."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from mediated_coevo.conditions import ConditionName
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

    reward: float | None = None
    total_tokens: int = 0
    duration_sec: float = 0.0

    mediator_history_entry_id: str | None = None
    planner_history_entry_id: str | None = None
    condition_name: ConditionName = "learned_mediator"
