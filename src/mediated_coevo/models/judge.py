"""Models for LLM judge reward annotation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

JudgeCapFlag = Literal[
    "benchmark_gaming_or_obscured_failure",
    "no_meaningful_progress",
    "brittle_or_one_off_patch",
    "unverifiable_outcome",
]


class JudgeAxisScores(BaseModel):
    """Normalized rubric axis scores returned by the judge."""

    model_config = ConfigDict(extra="forbid")

    task_outcome: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    token_efficiency: float = Field(ge=0.0, le=1.0)


class JudgeCapFlags(BaseModel):
    """Semantic cap flags returned by the judge."""

    model_config = ConfigDict(extra="forbid")

    benchmark_gaming_or_obscured_failure: bool = False
    no_meaningful_progress: bool = False
    brittle_or_one_off_patch: bool = False
    unverifiable_outcome: bool = False

    def enabled_flags(self) -> list[JudgeCapFlag]:
        flags: list[JudgeCapFlag] = []
        if self.benchmark_gaming_or_obscured_failure:
            flags.append("benchmark_gaming_or_obscured_failure")
        if self.no_meaningful_progress:
            flags.append("no_meaningful_progress")
        if self.brittle_or_one_off_patch:
            flags.append("brittle_or_one_off_patch")
        if self.unverifiable_outcome:
            flags.append("unverifiable_outcome")
        return flags


class JudgeLLMResponse(BaseModel):
    """Strict structured response expected from the judge LLM."""

    model_config = ConfigDict(extra="forbid")

    axis_scores: JudgeAxisScores
    flags: JudgeCapFlags = Field(default_factory=JudgeCapFlags)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    flag_evidence: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_evidence_for_true_flags(self) -> "JudgeLLMResponse":
        missing = [
            flag
            for flag in self.flags.enabled_flags()
            if not self.flag_evidence.get(flag, "").strip()
        ]
        if missing:
            raise ValueError(
                "true judge cap flags require evidence: " + ", ".join(missing)
            )
        return self


class JudgeRewardRecord(BaseModel):
    """Persisted judge reward sidecar row."""

    record_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    iteration: int
    run_id: str | None = None
    raw_reward: float
    judge_reward: float
    base_reward: float
    applied_cap: str | None = None
    axis_scores: JudgeAxisScores
    flags: JudgeCapFlags
    confidence: float
    rationale: str
    flag_evidence: dict[str, str] = Field(default_factory=dict)
    judge_model: str
    rubric_version: str
    trace_path: str | None = None
    metrics_path: str | None = None
    verifier_status: str | None = None
    trace_status: str | None = None
    total_tokens: int = 0
    task_category: str | None = None
    task_difficulty: str | None = None
    expected_reward_range: tuple[float, float] | None = None
    verifier_type: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
