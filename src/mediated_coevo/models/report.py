"""Mediator report models."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class AbstractionLevel(str, Enum):
    """Abstraction level of the Mediator's report to the Planner."""

    TRACE = "trace"           # Verbatim excerpts from execution
    REFLECTION = "reflection" # Distilled single-run insight
    PATTERN = "pattern"       # Cross-run trend with evidence


class OutcomeTag(BaseModel):
    """Retroactively added by outcome_tagger after downstream results are known."""

    reward: float       # Reward on the iteration after this report
    skill_changed: bool       # Whether the Planner edited skills after this report


class MediatorReport(BaseModel):
    """The Mediator's curated output sent to the Planner."""

    report_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str = ""
    iteration: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)
    abstraction_level: AbstractionLevel = AbstractionLevel.REFLECTION
    content: str = ""         # The actual report text sent to Claude
    token_count: int = 0
    artifacts_referenced: list[str] = Field(default_factory=list)
    withheld: bool = False    # True if mediator decided to expose nothing
    reasoning: str = ""       # Mediator's internal reasoning (for analysis)
    outcome_tag: OutcomeTag | None = None
