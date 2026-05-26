"""Candidate-batch auditing and selection helpers."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from math import ceil
from typing import Any

from mediated_coevo.core.utils import as_optional_float
from mediated_coevo.models.skill import (
    SkillProposal,
    SkillUpdateCandidate,
    SkillUpdateCandidateBatch,
    SkillUpdateCandidateRef,
)

SELECTION_POLICY_RANDOM_TOP_QUARTILE = "random_top_quartile"
_MEDIATOR_REQUIRED_KEYS = ("abstraction_level", "content", "withheld", "reasoning")


def stable_selection_seed(
    *,
    base_seed: int | None,
    iteration: int,
    skill_id: str,
    batch_id: str,
) -> int:
    """Derive a deterministic seed for a candidate batch."""
    raw = f"{base_seed or 0}:{iteration}:{skill_id}:{batch_id}".encode()
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "big")


def proposal_to_candidate(
    proposal: SkillProposal,
    *,
    skill_id: str = "executor",
    update_kind: str = "legacy_single_update",
    audit_score: float = 1.0,
) -> SkillUpdateCandidate:
    """Wrap an existing single proposal as a candidate batch item."""
    return SkillUpdateCandidate(
        skill_id=skill_id,
        update_kind=update_kind,
        hypothesis=proposal.reasoning,
        old_content=proposal.old_content,
        new_content=proposal.new_content,
        reasoning=proposal.reasoning,
        audit_score=audit_score,
    )


def candidate_from_mapping(
    raw_candidate: object,
    *,
    skill_id: str,
    current_skill: str,
) -> SkillUpdateCandidate | None:
    """Parse one JSON-like candidate mapping from an LLM response."""
    if not isinstance(raw_candidate, Mapping):
        return None

    update_kind = str(raw_candidate.get("update_kind", "unspecified")).strip()
    new_content = str(raw_candidate.get("new_content", "")).strip()
    if update_kind == "no_update" and not new_content:
        new_content = current_skill

    candidate_fields: dict[str, Any] = dict(
        skill_id=skill_id,
        update_kind=update_kind or "unspecified",
        hypothesis=str(raw_candidate.get("hypothesis", "")).strip(),
        risk=str(raw_candidate.get("risk", "")).strip(),
        old_content=current_skill,
        new_content=new_content,
        reasoning=str(raw_candidate.get("reasoning", "")).strip(),
        audit_score=as_optional_float(raw_candidate.get("audit_score")) or 0.0,
    )
    if candidate_id := str(raw_candidate.get("candidate_id", "")).strip():
        candidate_fields["candidate_id"] = candidate_id
    return SkillUpdateCandidate(**candidate_fields)


def build_candidate_batch(
    *,
    batch_id: str,
    iteration: int,
    skill_id: str,
    agent_role: str,
    task_ids: list[str],
    current_skill: str,
    candidates: list[SkillUpdateCandidate],
    selection_seed: int | None,
    select_candidate: bool = True,
) -> SkillUpdateCandidateBatch:
    """Audit candidates and select one from the top quartile."""
    audited = [
        audit_candidate(
            candidate,
            current_skill=current_skill,
            skill_id=skill_id,
        )
        for candidate in candidates
    ]
    batch = SkillUpdateCandidateBatch(
        batch_id=batch_id,
        iteration=iteration,
        skill_id=skill_id,
        agent_role=agent_role,
        task_ids=list(task_ids),
        selection_seed=selection_seed,
        selection_policy=SELECTION_POLICY_RANDOM_TOP_QUARTILE,
        candidates=audited,
    )
    if select_candidate:
        selected = select_top_quartile_candidate(batch)
        if selected is not None:
            selected.selected = True
            batch.selected_candidate_id = selected.candidate_id
    return batch


def audit_candidate(
    candidate: SkillUpdateCandidate,
    *,
    current_skill: str,
    skill_id: str,
) -> SkillUpdateCandidate:
    """Return a candidate copy with deterministic audit metadata filled in."""
    reason = candidate.rejection_reason
    new_content = candidate.new_content.strip()
    if candidate.update_kind == "no_update":
        reason = reason or "no_update"
    elif not new_content:
        reason = reason or "empty_content"
    elif _normalized(new_content) == _normalized(current_skill):
        reason = reason or "unchanged"
    elif _drops_mediator_contract(
        skill_id=skill_id,
        current_skill=current_skill,
        new_content=new_content,
    ):
        reason = reason or "missing_mediator_output_contract"

    return candidate.model_copy(
        update={
            "new_content": new_content,
            "audit_score": _audit_score(candidate),
            "rejection_reason": reason,
            "selected": False,
        }
    )


def select_top_quartile_candidate(
    batch: SkillUpdateCandidateBatch,
) -> SkillUpdateCandidate | None:
    """Select one valid candidate from the top score quartile."""
    valid = [
        candidate
        for candidate in batch.candidates
        if not candidate.rejection_reason
    ]
    if not valid:
        return None

    ranked = sorted(
        valid,
        key=lambda candidate: (candidate.audit_score, candidate.candidate_id),
        reverse=True,
    )
    top_count = max(1, ceil(len(ranked) * 0.25))
    top_quartile = ranked[:top_count]
    return random.Random(batch.selection_seed).choice(top_quartile)


def candidate_refs(
    batch: SkillUpdateCandidateBatch,
) -> list[SkillUpdateCandidateRef]:
    """Return compact candidate refs safe to include in metrics provenance."""
    return [
        SkillUpdateCandidateRef(
            candidate_id=candidate.candidate_id,
            update_kind=candidate.update_kind,
            audit_score=candidate.audit_score,
            selected=candidate.selected,
            rejection_reason=candidate.rejection_reason,
        )
        for candidate in batch.candidates
    ]


def selected_candidate(
    batch: SkillUpdateCandidateBatch,
) -> SkillUpdateCandidate | None:
    """Return the selected candidate, if one exists."""
    if batch.selected_candidate_id is None:
        return None
    for candidate in batch.candidates:
        if candidate.candidate_id == batch.selected_candidate_id:
            return candidate
    return None


def candidate_batch_artifact_path(batch_id: str) -> str:
    """Return the run-relative candidate batch artifact path."""
    return f"artifacts/candidate_batches/{batch_id}.json"


def _audit_score(candidate: SkillUpdateCandidate) -> float:
    score = candidate.audit_score
    if score <= 0:
        score = 0.5
        if candidate.hypothesis.strip():
            score += 0.15
        if candidate.risk.strip():
            score += 0.10
        if candidate.reasoning.strip():
            score += 0.10
    return min(1.0, max(0.0, score))


def _drops_mediator_contract(
    *,
    skill_id: str,
    current_skill: str,
    new_content: str,
) -> bool:
    if skill_id != "mediator":
        return False
    current_has_contract = all(key in current_skill for key in _MEDIATOR_REQUIRED_KEYS)
    if not current_has_contract:
        return False
    return any(key not in new_content for key in _MEDIATOR_REQUIRED_KEYS)


def _normalized(text: str) -> str:
    return " ".join(text.split())
