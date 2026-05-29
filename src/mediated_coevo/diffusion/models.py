"""Typed diffusion records for future graph-aware context routing."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _record_id() -> str:
    return uuid4().hex


class DiffusionArtifactType(str, Enum):
    """Planner-safe artifact categories allowed to diffuse across tasks."""

    FAILURE_SIGNATURE = "failure_signature"
    DEBUG_HINT = "debug_hint"
    MEDIATOR_REPORT_SUMMARY = "mediator_report_summary"
    REGRESSION_WARNING = "regression_warning"
    OTHER = "other"


class DiffusionRiskLevel(str, Enum):
    """Operational risk of exposing an artifact to another task."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskGraphEdgeRecord(BaseModel):
    """One directed relation visible to a graph snapshot."""

    source_task_id: str
    target_task_id: str
    relation: str
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiffusionArtifact(BaseModel):
    """A planner-visible artifact emitted from a source task."""

    artifact_id: str = Field(default_factory=_record_id)
    source_task_id: str
    source_iteration: int = Field(ge=0)
    source_run_id: str | None = None
    artifact_type: DiffusionArtifactType
    risk_level: DiffusionRiskLevel
    content: str
    evidence_trace_ids: list[str] = Field(default_factory=list)
    evidence_report_ids: list[str] = Field(default_factory=list)
    verifier_reward: float | None = None
    judge_reward: float | None = None
    token_cost: int = Field(default=0, ge=0)
    ttl_iterations: int | None = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskGraphSnapshot(BaseModel):
    """Time-safe task graph state used during candidate selection."""

    snapshot_id: str = Field(default_factory=_record_id)
    run_id: str
    iteration: int = Field(ge=0)
    task_ids: list[str] = Field(default_factory=list)
    feature_cutoff: datetime | None = None
    edge_records: list[TaskGraphEdgeRecord] = Field(default_factory=list)
    graph_policy: str
    seed: int | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiffusedRecord(BaseModel):
    """Unified audit record for one artifact routed toward one target task."""

    record_id: str = Field(default_factory=_record_id)
    artifact_id: str
    source_task_id: str
    source_iteration: int = Field(ge=0)
    target_task_id: str
    target_iteration: int = Field(ge=0)
    source_run_id: str | None = None
    target_run_id: str | None = None
    snapshot_id: str | None = None
    policy_name: str
    relation: str | None = None
    reason: str = ""
    eligible: bool = True
    selected: bool = False
    rendered: bool = False
    rendered_section: str = ""
    token_count: int = Field(default=0, ge=0)
    cited_by: str | None = None
    citation_text: str = ""
    verifier_reward: float | None = None
    judge_reward: float | None = None
    success: bool | None = None
    regression: bool | None = None
    notes: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
