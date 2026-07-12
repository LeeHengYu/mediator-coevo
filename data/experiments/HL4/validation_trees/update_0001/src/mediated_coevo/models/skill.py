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
    reward_source: str | None = None
    verifier_reward: float | None = None
    judge_reward: float | None = None


class SkillUpdateCandidate(SkillEdit):
    """One audit candidate for a possible skill update."""

    candidate_id: str = Field(default_factory=lambda: str(uuid4()))
    skill_id: str
    update_kind: str = "unspecified"
    hypothesis: str = ""
    risk: str = ""
    audit_score: float = 0.0
    selected: bool = False
    rejection_reason: str | None = None


class SkillUpdateCandidateRef(BaseModel):
    """Compact pointer to a persisted skill-update candidate."""

    candidate_id: str
    update_kind: str
    audit_score: float = 0.0
    selected: bool = False
    rejection_reason: str | None = None


class SkillUpdateCandidateBatch(BaseModel):
    """Run-local audit artifact containing candidate skill updates."""

    batch_id: str
    iteration: int
    skill_id: str
    agent_role: str
    task_ids: list[str] = Field(default_factory=list)
    selection_seed: int | None = None
    selection_policy: str = "random_top_quartile"
    selected_candidate_id: str | None = None
    candidates: list[SkillUpdateCandidate] = Field(default_factory=list)


class ProposalRef(BaseModel):
    """Compact pointer to a buffered executor skill proposal."""

    proposal_id: str
    task_id: str
    iteration: int
    reward: float | None = None
    reward_source: str | None = None
    verifier_reward: float | None = None
    judge_reward: float | None = None


class SkillValidationTaskResult(BaseModel):
    """One current-vs-candidate executor validation comparison."""

    task_id: str
    current_reward: float | None = None
    candidate_reward: float | None = None
    current_reward_source: str | None = None
    candidate_reward_source: str | None = None
    current_verifier_reward: float | None = None
    candidate_verifier_reward: float | None = None
    current_judge_reward: float | None = None
    candidate_judge_reward: float | None = None
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
    min_mean_delta: float = 0.01
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
    candidate_batch_id: str | None = None
    candidate_batch_path: str | None = None
    selected_candidate_id: str | None = None
    proposals: list[RejectedSkillProposal] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class RejectedReflectionBatch(BaseModel):
    """Rejected planner/mediator reflection batch kept as negative evidence."""

    rejection_id: str = Field(default_factory=lambda: str(uuid4()))
    batch_id: str
    iteration: int
    skill_id: str
    agent_role: str
    task_ids: list[str] = Field(default_factory=list)
    base_skill_hash: str
    reason: str = ""
    validation: SkillValidationResult | None = None
    candidate_batch_id: str | None = None
    candidate_batch_path: str | None = None
    selected_candidate_id: str | None = None
    selected_update_kind: str | None = None
    candidate_refs: list[SkillUpdateCandidateRef] = Field(default_factory=list)
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
    candidate_batch_id: str | None = None
    candidate_batch_path: str | None = None
    selected_candidate_id: str | None = None
    selected_update_kind: str | None = None
    candidate_refs: list[SkillUpdateCandidateRef] = Field(default_factory=list)


class AdvisorBatchProvenance(SkillUpdateProvenance):
    """Provenance for executor skill commits caused by advisor-reviewed proposals."""

    kind: Literal["advisor_batch"] = "advisor_batch"
    proposal_refs: list[ProposalRef] = Field(default_factory=list)
    validation: SkillValidationResult | None = None


class ContrastiveReflectionProvenance(SkillUpdateProvenance):
    """Provenance for mediator/planner meta-skill reflection commits."""

    kind: Literal["contrastive_reflection"] = "contrastive_reflection"
    contrastive_pair_refs: list[ContrastivePairRef] = Field(default_factory=list)
    validation: SkillValidationResult | None = None
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


class SkillUpdateLedgerEntry(BaseModel):
    """Run-local ledger entry for one committed skill update."""

    update_id: str
    iteration: int
    skill_id: str
    skill_version: str | None = None
    record_task_id: str
    update_task_id: str = ""
    task_ids: list[str] = Field(default_factory=list)
    old_skill_hash: str | None = None
    new_skill_hash: str | None = None
    provenance_kind: str | None = None
    decision: str | None = None
    reason: str = ""
    rollback_snapshot: str | None = None
    candidate_batch_id: str | None = None
    candidate_batch_path: str | None = None
    selected_candidate_id: str | None = None
    selected_update_kind: str | None = None
    candidate_count: int = 0
    validation_decision: str | None = None
    validation_reason: str | None = None
    validation_current_mean_reward: float | None = None
    validation_candidate_mean_reward: float | None = None
    validation_mean_delta: float | None = None
    validation_task_ids: list[str] = Field(default_factory=list)
    validation_task_count: int = 0
    reward: float | None = None
    delta_reward: float | None = None
    success: bool | None = None
    verifier_status: str | None = None
    artifact_path: str | None = None
    diff_path: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
