"""Skill update models."""

from __future__ import annotations

from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field


SkillUpdateDecision = Literal["approved", "committed", "rejected", "no_change"]


class SkillEdit(BaseModel):
    """Shared base for any skill edit — draft or committed."""

    old_content: str
    new_content: str
    reasoning: str = ""


class SkillProposal(SkillEdit):
    """Buffered, unreviewed proposal — never written to HistoryStore."""

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


class ContrastiveReflectionProvenance(SkillUpdateProvenance):
    """Provenance for mediator/planner meta-skill reflection commits."""

    kind: Literal["contrastive_reflection"] = "contrastive_reflection"
    contrastive_pair_refs: list[ContrastivePairRef] = Field(default_factory=list)
    max_pairs: int = 0
    selection_seed: int | None = None


SkillProvenance = Annotated[
    Union[AdvisorBatchProvenance, ContrastiveReflectionProvenance],
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
