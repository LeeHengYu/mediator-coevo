"""Skill update models."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

SkillUpdateDecision = Literal["approved", "committed", "rejected", "no_change"]
SkillValidationDecision = Literal["accepted", "rejected"]


class SkillEdit(BaseModel):
    """Shared base for any skill edit — draft or committed."""

    old_content: str
    new_content: str
    reasoning: str = ""


class SkillProposal(SkillEdit):
    """Buffered, unreviewed proposal for an executor skill edit."""

    proposal_id: str = Field(default_factory=lambda: str(uuid4()))
    iteration: int
    task_id: str
    reward: float | None = None


class ProposalRef(BaseModel):
    """Compact pointer to a buffered executor skill proposal."""

    proposal_id: str
    task_id: str
    iteration: int
    reward: float | None = None


class SkillValidationTaskResult(BaseModel):
    """One current-vs-candidate executor validation comparison."""

    task_id: str
    current_reward: float | None = None
    candidate_reward: float | None = None
    current_status: str
    candidate_status: str
    current_trace_path: str | None = None
    candidate_trace_path: str | None = None
    usable: bool = False
    regressed: bool = False


class SkillValidationResult(BaseModel):
    """Empirical validation evidence for an executor skill candidate."""

    validation_id: str
    task_ids: list[str] = Field(default_factory=list)
    decision: SkillValidationDecision
    reason: str = ""
    current_mean_reward: float | None = None
    candidate_mean_reward: float | None = None
    mean_delta: float | None = None
    min_mean_delta: float = 0.0
    reward_tolerance: float = 1e-9
    task_results: list[SkillValidationTaskResult] = Field(default_factory=list)


class RejectedSkillProposal(SkillProposal):
    """Durable copy of a proposal that was reviewed but not committed."""

    old_skill_hash: str | None = None
    new_skill_hash: str | None = None


class RejectedProposalBatch(BaseModel):
    """Rejected executor proposal batch kept for later analysis/reflection."""

    rejection_id: str = Field(default_factory=lambda: str(uuid4()))
    batch_id: str
    iteration: int
    skill_id: str = "executor"
    task_ids: list[str] = Field(default_factory=list)
    base_skill_hash: str
    reason: str = ""
    advisor_feedback: str | None = None
    validation: SkillValidationResult | None = None
    proposals: list[RejectedSkillProposal] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class ContrastivePairRef(BaseModel):
    """Compact pointer to one selected contrastive history pair."""

    worse_entry_id: str
    better_entry_id: str
    task_id: str
    worse_reward: float
    better_reward: float
    reward_gap: float


class SkillUpdateProvenance(BaseModel):
    """Shared fields for concise skill update provenance."""

    kind: str
    batch_id: str
    iteration: int
    skill_id: str
    task_ids: list[str] = Field(default_factory=list)
    base_skill_hash: str
    decision: SkillUpdateDecision
    reason: str = ""
    rollback_snapshot: str | None = None


class AdvisorBatchProvenance(SkillUpdateProvenance):
    """Provenance for executor skill commits caused by advisor-reviewed proposals."""

    kind: Literal["advisor_batch"] = "advisor_batch"
    proposal_refs: list[ProposalRef] = Field(default_factory=list)
    validation: SkillValidationResult | None = None


class ContrastiveReflectionProvenance(SkillUpdateProvenance):
    """Provenance for mediator/planner meta-skill reflection commits."""

    kind: Literal["contrastive_reflection"] = "contrastive_reflection"
    contrastive_pair_refs: list[ContrastivePairRef] = Field(default_factory=list)
    max_pairs: int = 0
    selection_seed: int | None = None


SkillProvenance = Annotated[
    AdvisorBatchProvenance | ContrastiveReflectionProvenance,
    Field(discriminator="kind"),
]


class SkillUpdate(SkillEdit):
    """A committed skill edit written to disk after advisor approval."""

    skill_id: str
    task_id: str = ""
    iteration: int = 0
    old_skill_hash: str | None = None
    new_skill_hash: str | None = None
    skill_version: str | None = None
    provenance: SkillProvenance | None = None
