"""Reflector — text-level policy gradient via contrastive reflection.

Uses outcome-tagged history to build contrastive prompts and
calls the appropriate LLM to rewrite the agent's meta-skill.

Reads structured ``HistoryEntry.payload`` slots (``MediatorSignal`` /
``PlannerSignal``) rather than chopping free-text blobs — the Planner's
reasoning is preserved in full, and long mediator reports have already
been compacted upstream by ``evolution/compactor.py``.
"""

from __future__ import annotations

import difflib
import logging
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mediated_coevo.evolution.candidates import (
    build_candidate_batch,
    candidate_batch_artifact_path,
    candidate_from_mapping,
    candidate_refs,
    selected_candidate,
    stable_selection_seed,
)
from mediated_coevo.models.history_signals import MediatorSignal, PlannerSignal
from mediated_coevo.models.skill import (
    ContrastivePairRef,
    ContrastiveReflectionProvenance,
    RejectedReflectionBatch,
    SkillUpdateCandidate,
    SkillUpdateCandidateBatch,
    SkillValidationResult,
)
from mediated_coevo.prompt_text import PromptText
from mediated_coevo.stores.skill_store import SkillStore

if TYPE_CHECKING:
    from mediated_coevo.core.config import BudgetsConfig
    from mediated_coevo.llm.client import LLMClient
    from mediated_coevo.stores.artifact_store import ArtifactStore
    from mediated_coevo.stores.history_store import (
        ContrastivePair,
        HistoryEntry,
        HistoryStore,
    )

logger = logging.getLogger(__name__)


_NO_CHANGE_SENTINEL = "NO_CHANGE"
_SIMILARITY_THRESHOLD = 0.95


@dataclass(frozen=True)
class ReflectionResult:
    """Committed reflection result plus concise provenance."""

    skill_id: str
    old_content: str
    new_content: str
    provenance: ContrastiveReflectionProvenance


@dataclass(frozen=True)
class ReflectionDraft:
    """Uncommitted reflection candidate batch plus evidence context."""

    agent_role: str
    old_content: str
    base_skill_hash: str
    pairs: list[ContrastivePair]
    candidate_batch: SkillUpdateCandidateBatch
    candidate_batch_path: str | None
    iteration: int
    max_pairs: int
    selection_seed: int | None


@dataclass(frozen=True)
class RejectedReflectionSummary:
    """Compact negative evidence from a prior rejected reflection."""

    iteration: int
    skill_id: str
    reason: str
    selected_candidate_id: str | None
    selected_update_kind: str | None
    validation_reason: str | None
    validation_mean_delta: float | None
    task_ids: list[str]


class Reflector:
    """Builds contrastive reflection prompts and updates skills."""

    def __init__(
        self,
        history_store: HistoryStore,
        skill_store: SkillStore,
        similarity_threshold: float = _SIMILARITY_THRESHOLD,
        budgets: BudgetsConfig | None = None,
        condition_name: str | None = None,
        artifact_store: ArtifactStore | None = None,
        base_seed: int | None = None,
    ) -> None:
        self._history_store = history_store
        self._skill_store = skill_store
        self._similarity_threshold = similarity_threshold
        self._budgets = budgets
        self._condition_name = condition_name
        self._artifact_store = artifact_store
        self._base_seed = base_seed

    async def reflect(
        self,
        agent_role: str,
        llm_client: LLMClient,
        iteration: int = 0,
        max_pairs: int = 5,
        selection_seed: int | None = None,
    ) -> ReflectionResult | None:
        """Reflect on past actions for the given role, update the skill file.

        Returns the committed reflection result, or None if no update was made.
        """
        draft = await self.draft_reflection(
            agent_role=agent_role,
            llm_client=llm_client,
            iteration=iteration,
            max_pairs=max_pairs,
            selection_seed=selection_seed,
        )
        if draft is None:
            return None
        return self.commit_draft(draft)

    async def draft_reflection(
        self,
        *,
        agent_role: str,
        llm_client: LLMClient,
        iteration: int = 0,
        max_pairs: int = 5,
        selection_seed: int | None = None,
        candidate_count: int | None = None,
        select_candidate: bool = True,
    ) -> ReflectionDraft | None:
        """Build an audited reflection candidate batch without committing it."""
        pairs = self._history_store.contrastive_pairs(
            agent_role,
            max_pairs=max_pairs,
            selection_seed=selection_seed,
        )
        if not pairs:
            logger.info("No contrastive pairs for %s; skipping reflection.", agent_role)
            return None

        logger.info(
            "Reflector found %d contrastive pairs for %s.", len(pairs), agent_role
        )

        current_skill = self._skill_store.read_skill(agent_role) or ""
        base_skill_hash = SkillStore.content_hash(current_skill)

        messages = self._build_reflection_prompt(
            agent_role,
            current_skill,
            pairs,
            model=llm_client.model,
            candidate_count=candidate_count,
        )

        try:
            raw_content = await self._complete_reflection(
                agent_role=agent_role,
                llm_client=llm_client,
                messages=messages,
            )
            candidates = _parse_reflection_candidates(
                agent_role=agent_role,
                current_skill=current_skill,
                raw_content=raw_content,
            )
            if candidate_count is not None:
                candidates = candidates[:candidate_count]
            candidate_batch = self._build_candidate_batch(
                agent_role=agent_role,
                current_skill=current_skill,
                candidates=candidates,
                iteration=iteration,
                pairs=pairs,
                selection_seed=selection_seed,
                select_candidate=select_candidate,
            )
            candidate_batch_path = None
            if self._artifact_store is not None:
                self._artifact_store.store_candidate_batch(
                    candidate_batch,
                    overwrite=True,
                )
                candidate_batch_path = candidate_batch_artifact_path(
                    candidate_batch.batch_id
                )
            return ReflectionDraft(
                agent_role=agent_role,
                old_content=current_skill,
                base_skill_hash=base_skill_hash,
                pairs=pairs,
                candidate_batch=candidate_batch,
                candidate_batch_path=candidate_batch_path,
                iteration=iteration,
                max_pairs=max_pairs,
                selection_seed=selection_seed,
            )
        except Exception as e:
            logger.error("Failed to reflect for %s: %s", agent_role, e)
            return None

    def commit_draft(
        self,
        draft: ReflectionDraft,
        *,
        selected_candidate_id: str | None = None,
        validation: SkillValidationResult | None = None,
    ) -> ReflectionResult | None:
        """Write a selected reflection candidate and return committed provenance."""
        if selected_candidate_id is not None:
            for candidate in draft.candidate_batch.candidates:
                candidate.selected = candidate.candidate_id == selected_candidate_id
            draft.candidate_batch.selected_candidate_id = selected_candidate_id

        selected = selected_candidate(draft.candidate_batch)
        if selected is None:
            return None
        if draft.old_content and _is_semantically_similar(
            draft.old_content,
            selected.new_content,
            self._similarity_threshold,
        ):
            logger.info(
                "%s reflection: selected candidate too similar to current "
                "(threshold=%.2f); skipping.",
                draft.agent_role,
                self._similarity_threshold,
            )
            return None

        if self._artifact_store is not None:
            self._artifact_store.store_candidate_batch(
                draft.candidate_batch,
                overwrite=True,
            )
        self._skill_store.write_skill(draft.agent_role, selected.new_content)
        logger.info(
            "%s skill updated via reflection (%d chars).",
            draft.agent_role,
            len(selected.new_content),
        )
        provenance = self._build_reflection_provenance(
            agent_role=draft.agent_role,
            base_skill_hash=draft.base_skill_hash,
            pairs=draft.pairs,
            candidate_batch=draft.candidate_batch,
            candidate_batch_path=draft.candidate_batch_path,
            iteration=draft.iteration,
            max_pairs=draft.max_pairs,
            selection_seed=draft.selection_seed,
            validation=validation,
        )
        return ReflectionResult(
            skill_id=draft.agent_role,
            old_content=draft.old_content,
            new_content=selected.new_content,
            provenance=provenance,
        )

    def _build_candidate_batch(
        self,
        *,
        agent_role: str,
        current_skill: str,
        candidates: list[SkillUpdateCandidate],
        iteration: int,
        pairs: list[ContrastivePair],
        selection_seed: int | None,
        select_candidate: bool = True,
    ) -> SkillUpdateCandidateBatch:
        batch_id = f"reflect-{agent_role}-iter-{iteration:04d}"
        batch_seed = stable_selection_seed(
            base_seed=self._base_seed,
            iteration=iteration,
            skill_id=agent_role,
            batch_id=batch_id,
        )
        if selection_seed is not None:
            batch_seed ^= selection_seed
        return build_candidate_batch(
            batch_id=batch_id,
            iteration=iteration,
            skill_id=agent_role,
            agent_role=agent_role,
            task_ids=_task_ids_from_pairs(pairs),
            current_skill=current_skill,
            candidates=candidates,
            selection_seed=batch_seed,
            select_candidate=select_candidate,
        )

    def _build_reflection_prompt(
        self,
        agent_role: str,
        current_skill: str,
        pairs: list[ContrastivePair],
        *,
        model: str = "",
        candidate_count: int | None = None,
    ) -> list[dict[str, Any]]:
        rejected_history = self._compact_rejected_reflection_history(agent_role)
        if agent_role == "mediator":
            return self._build_mediator_prompt(
                current_skill,
                pairs,
                rejected_history=rejected_history,
                model=model,
                budgets=self._budgets,
            )
        return self._build_planner_prompt(
            current_skill,
            pairs,
            rejected_history=rejected_history,
            model=model,
            budgets=self._budgets,
            candidate_count=candidate_count,
        )

    def record_rejected_draft(
        self,
        draft: ReflectionDraft,
        *,
        reason: str,
        validation: SkillValidationResult | None = None,
        selected_candidate_id: str | None = None,
    ) -> str:
        """Persist a rejected reflection draft as future negative evidence."""
        selected = _candidate_by_id(draft.candidate_batch, selected_candidate_id)
        if selected is None:
            selected = selected_candidate(draft.candidate_batch)
        rejected = RejectedReflectionBatch(
            batch_id=draft.candidate_batch.batch_id,
            iteration=draft.iteration,
            skill_id=draft.agent_role,
            agent_role=draft.agent_role,
            task_ids=_task_ids_from_pairs(draft.pairs),
            base_skill_hash=draft.base_skill_hash,
            reason=reason,
            validation=validation,
            candidate_batch_id=draft.candidate_batch.batch_id,
            candidate_batch_path=draft.candidate_batch_path,
            selected_candidate_id=selected.candidate_id if selected else None,
            selected_update_kind=selected.update_kind if selected else None,
            candidate_refs=candidate_refs(draft.candidate_batch),
        )
        return self._history_store.record_rejected_reflection(rejected)

    def _compact_rejected_reflection_history(
        self,
        agent_role: str,
    ) -> list[RejectedReflectionSummary]:
        return [
            RejectedReflectionSummary(
                iteration=batch.iteration,
                skill_id=batch.skill_id,
                reason=batch.reason,
                selected_candidate_id=batch.selected_candidate_id,
                selected_update_kind=batch.selected_update_kind,
                validation_reason=(
                    batch.validation.reason if batch.validation is not None else None
                ),
                validation_mean_delta=(
                    batch.validation.mean_delta if batch.validation is not None else None
                ),
                task_ids=list(batch.task_ids),
            )
            for batch in self._history_store.query_rejected_reflections(
                agent_role=agent_role,
                recent=5,
            )
        ]

    async def _complete_reflection(
        self,
        *,
        agent_role: str,
        llm_client: LLMClient,
        messages: list[dict[str, Any]],
    ) -> str:
        if self._budgets:
            response = await llm_client.complete(
                messages=messages,
                temperature=0.4,
                max_tokens=self._budgets.reflector_completion_tokens,
                prompt_budget=self._budgets.reflector_prompt_tokens,
                budget_label=f"reflector.{agent_role}",
                budget_overflow_strategy="section_pack",
                condition_name=self._condition_name,
            )
        else:
            response = await llm_client.complete(
                messages=messages,
                temperature=0.4,
                max_tokens=4096,
            )
        return response["content"].strip()

    @staticmethod
    def _build_reflection_provenance(
        *,
        agent_role: str,
        base_skill_hash: str,
        pairs: list[ContrastivePair],
        candidate_batch: SkillUpdateCandidateBatch,
        candidate_batch_path: str | None,
        iteration: int,
        max_pairs: int,
        selection_seed: int | None,
        validation: SkillValidationResult | None = None,
    ) -> ContrastiveReflectionProvenance:
        pair_refs = _contrastive_pair_refs(pairs)
        task_ids = sorted({ref.task_id for ref in pair_refs if ref.task_id})
        selected = selected_candidate(candidate_batch)
        return ContrastiveReflectionProvenance(
            batch_id=f"reflect-{agent_role}-iter-{iteration:04d}",
            iteration=iteration,
            skill_id=agent_role,
            task_ids=task_ids,
            base_skill_hash=base_skill_hash,
            decision="committed",
            reason=(
                f"Updated via {len(pair_refs)} contrastive history pair(s); "
                f"selected {selected.update_kind if selected else 'none'} candidate."
            ),
            contrastive_pair_refs=pair_refs,
            validation=validation,
            max_pairs=max_pairs,
            selection_seed=selection_seed,
            candidate_batch_id=candidate_batch.batch_id,
            candidate_batch_path=candidate_batch_path,
            selected_candidate_id=(
                candidate_batch.selected_candidate_id
                if selected is not None
                else None
            ),
            selected_update_kind=selected.update_kind if selected is not None else None,
            candidate_refs=candidate_refs(candidate_batch),
        )

    # ── Prompt Builders ──

    @staticmethod
    def _build_contrastive_prompt(
        *,
        system_text: str,
        current_skill_heading: str,
        current_skill: str,
        evidence_intro: str,
        instructions: str,
        pairs: list[ContrastivePair],
        formatter: Callable[[HistoryEntry], str],
        rejected_history: list[RejectedReflectionSummary] | None = None,
        model: str = "",
        budgets: BudgetsConfig | None = None,
    ) -> list[dict[str, str]]:
        """Shared builder for contrastive reflection prompts."""
        contrastive_parts: list[str] = []
        for i, pair in enumerate(pairs, 1):
            task_id = pair.worse.metadata.get("task_id", "?")
            worse_context = _format_reward_context(
                pair.worse_reward,
                pair.worse_relative_reward,
            )
            better_context = _format_reward_context(
                pair.better_reward,
                pair.better_relative_reward,
            )
            contrastive_parts.append(
                PromptText.reflection_pair(
                    index=i,
                    task_id=task_id,
                    worse_context=worse_context,
                    worse_entry=formatter(pair.worse),
                    better_context=better_context,
                    better_entry=formatter(pair.better),
                )
            )
        rejected_section = _format_rejected_reflection_history(rejected_history or [])

        user_content = PromptText.reflection_user_content(
            current_skill_heading=current_skill_heading,
            current_skill=current_skill,
            evidence_intro=evidence_intro,
            contrastive_parts=contrastive_parts,
            rejected_section=rejected_section,
            instructions=instructions,
        )
        if budgets:
            from mediated_coevo.runtime.token_budget import BudgetSection, pack_sections

            sections = [
                BudgetSection(
                    current_skill_heading,
                    PromptText.reflection_current_skill_section(
                        current_skill_heading=current_skill_heading,
                        current_skill=current_skill,
                    ),
                    required=True,
                    max_tokens=budgets.max_skill_tokens,
                ),
                BudgetSection(
                    "contrastive_evidence",
                    PromptText.reflection_evidence_section(
                        evidence_intro=evidence_intro,
                        contrastive_parts=contrastive_parts,
                    ),
                    required=True,
                    max_tokens=budgets.historical_summary_tokens,
                ),
            ]
            if rejected_section:
                sections.append(
                    BudgetSection(
                        "rejected_reflection_history",
                        rejected_section.strip(),
                        required=True,
                        max_tokens=min(1200, budgets.historical_summary_tokens),
                    )
                )
            sections.append(
                BudgetSection(
                    "instructions",
                    PromptText.reflection_instructions_section(instructions),
                    required=True,
                )
            )
            user_content = pack_sections(
                model,
                sections,
                budgets.reflector_prompt_tokens,
            )

        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _build_mediator_prompt(
        current_skill: str,
        pairs: list[ContrastivePair],
        *,
        rejected_history: list[RejectedReflectionSummary] | None = None,
        model: str = "",
        budgets: BudgetsConfig | None = None,
    ) -> list[dict[str, str]]:
        return Reflector._build_contrastive_prompt(
            system_text=PromptText.MEDIATOR_REFLECTION_SYSTEM,
            current_skill_heading=PromptText.CURRENT_COORDINATION_PROTOCOL_HEADING,
            current_skill=current_skill,
            evidence_intro=PromptText.MEDIATOR_REFLECTION_EVIDENCE_INTRO,
            instructions=PromptText.MEDIATOR_REFLECTION_INSTRUCTIONS,
            pairs=pairs,
            formatter=_format_mediator_entry,
            rejected_history=rejected_history,
            model=model,
            budgets=budgets,
        )

    @staticmethod
    def _build_planner_prompt(
        current_skill: str,
        pairs: list[ContrastivePair],
        *,
        rejected_history: list[RejectedReflectionSummary] | None = None,
        model: str = "",
        budgets: BudgetsConfig | None = None,
        candidate_count: int | None = None,
    ) -> list[dict[str, str]]:
        return Reflector._build_contrastive_prompt(
            system_text=PromptText.reflection_planner_system(
                PromptText.reflection_candidate_instruction(candidate_count)
            ),
            current_skill_heading=PromptText.CURRENT_SKILL_REFINER_GUIDELINES_HEADING,
            current_skill=current_skill,
            evidence_intro=PromptText.PLANNER_REFLECTION_EVIDENCE_INTRO,
            instructions=PromptText.PLANNER_REFLECTION_INSTRUCTIONS,
            pairs=pairs,
            formatter=_format_planner_entry,
            rejected_history=rejected_history,
            model=model,
            budgets=budgets,
        )


def _contrastive_pair_refs(
    pairs: list[ContrastivePair],
) -> list[ContrastivePairRef]:
    """Convert selected history pairs into compact persisted references."""
    refs: list[ContrastivePairRef] = []
    for pair in pairs:
        worse = pair.worse
        better = pair.better
        task_id = str(
            worse.metadata.get("task_id") or better.metadata.get("task_id") or ""
        )
        refs.append(
            ContrastivePairRef(
                worse_entry_id=worse.entry_id,
                better_entry_id=better.entry_id,
                task_id=task_id,
                worse_reward=pair.worse_reward,
                better_reward=pair.better_reward,
                reward_gap=pair.reward_gap,
            )
        )
    return refs


def _parse_reflection_candidates(
    *,
    agent_role: str,
    current_skill: str,
    raw_content: str,
) -> list[SkillUpdateCandidate]:
    """Parse a reflection response into candidate updates."""
    if raw_content == _NO_CHANGE_SENTINEL:
        return [
            SkillUpdateCandidate(
                skill_id=agent_role,
                update_kind="no_update",
                hypothesis="Current skill already captures the evidence.",
                old_content=current_skill,
                new_content=current_skill,
                reasoning="The reflector explicitly signalled no change.",
                audit_score=1.0,
            )
        ]

    from mediated_coevo.core.utils import parse_json_object

    parsed = parse_json_object(raw_content)
    raw_candidates = parsed.get("candidates")
    if isinstance(raw_candidates, list):
        candidates = []
        for raw_candidate in raw_candidates:
            candidate = candidate_from_mapping(
                raw_candidate,
                skill_id=agent_role,
                current_skill=current_skill,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    new_content = _parse_skill_content(raw_content)
    if new_content is None:
        return []
    return [
        SkillUpdateCandidate(
            skill_id=agent_role,
            update_kind="legacy_reflection",
            hypothesis="Single reflected update from legacy Markdown response.",
            old_content=current_skill,
            new_content=new_content,
            reasoning="Parsed from the reflector's legacy Markdown response.",
            audit_score=1.0,
        )
    ]


def _task_ids_from_pairs(pairs: list[ContrastivePair]) -> list[str]:
    return sorted(
        {
            str(
                pair.worse.metadata.get("task_id")
                or pair.better.metadata.get("task_id")
            )
            for pair in pairs
            if pair.worse.metadata.get("task_id") or pair.better.metadata.get("task_id")
        }
    )


def _candidate_by_id(
    batch: SkillUpdateCandidateBatch,
    candidate_id: str | None,
) -> SkillUpdateCandidate | None:
    if candidate_id is None:
        return None
    for candidate in batch.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def _format_rejected_reflection_history(
    rejected_history: list[RejectedReflectionSummary],
) -> str:
    if not rejected_history:
        return ""

    lines = PromptText.rejected_reflection_intro()
    for item in rejected_history:
        task_ids = ",".join(item.task_ids) if item.task_ids else "n/a"
        lines.append(
            PromptText.rejected_reflection_item(
                iteration=item.iteration,
                skill_id=item.skill_id,
                selected_candidate_id=item.selected_candidate_id,
                selected_update_kind=item.selected_update_kind,
                task_ids=task_ids,
                reason=item.reason,
                validation_reason=item.validation_reason,
                validation_mean_delta=_format_optional_delta(
                    item.validation_mean_delta
                ),
            )
        )
    return "\n".join(lines)


def _format_optional_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}"


def _format_reward_context(reward: float, relative_reward: float) -> str:
    if relative_reward > 0:
        position = "above task average"
    elif relative_reward < 0:
        position = "below task average"
    else:
        position = "at task average"
    return (
        f"evolution reward: {reward:.2f}; "
        f"relative task delta: {relative_reward:+.2f} {position}"
    )


def _is_semantically_similar(old: str, new: str, threshold: float) -> bool:
    """Check whether *old* and *new* are similar enough to skip the update.

    Uses ``SequenceMatcher`` on whitespace-normalised text so that
    cosmetic reformatting (extra blank lines, indentation tweaks,
    trailing spaces) does not count as a real change.
    """
    norm_old = " ".join(old.split())
    norm_new = " ".join(new.split())
    ratio = difflib.SequenceMatcher(None, norm_old, norm_new).ratio()
    logger.debug("Skill similarity ratio: %.4f (threshold %.4f)", ratio, threshold)
    return ratio >= threshold


def _format_mediator_entry(entry: HistoryEntry) -> str:
    """Render a mediator HistoryEntry as typed-slot bullets for the prompt."""
    p = entry.payload
    if not isinstance(p, MediatorSignal):
        return "- (no mediator signal)"

    lines: list[str] = []
    if p.abstraction_level:
        lines.append(f"- Abstraction: {p.abstraction_level}")
    if p.withheld:
        lines.append("- Decision: WITHHELD (exposed nothing to the Planner)")
    if p.headline:
        lines.append(f"- Headline: {p.headline}")
    if p.evidence:
        sample_note = (
            ""
            if p.raw_length <= len(p.evidence)
            else f" (sampled from {p.raw_length} chars)"
        )
        lines.append(f"- Evidence{sample_note}:")
        lines.append(textwrap.indent(p.evidence.rstrip(), "    "))
    if p.mediator_reasoning:
        lines.append(f"- Mediator reasoning: {p.mediator_reasoning}")
    return "\n".join(lines) if lines else "- (empty mediator signal)"


def _format_planner_entry(entry: HistoryEntry) -> str:
    """Render a planner HistoryEntry as typed-slot bullets for the prompt."""
    p = entry.payload
    if not isinstance(p, PlannerSignal):
        return "- (no planner signal)"

    lines: list[str] = []
    if p.reasoning:
        lines.append(f"- Reasoning: {p.reasoning}")
    diff_line = f"- Diff: +{p.lines_added}/-{p.lines_removed}"
    lines.append(diff_line)
    if p.diff_excerpt.strip():
        lines.append("- Diff excerpt:")
        lines.append("```diff")
        lines.append(p.diff_excerpt.rstrip())
        lines.append("```")
    return "\n".join(lines)


def _parse_skill_content(response_text: str) -> str | None:
    """Extract Markdown content from ```markdown ... ``` fences.

    Finds the first opening fence and its matching close, so inner
    code fences (e.g., ```json examples) are not confused with the
    outer wrapper.
    """
    text = response_text.strip()

    # Look for ```markdown opening fence
    for marker in ("```markdown", "```md"):
        if marker in text:
            start = text.index(marker) + len(marker)
            # Skip to next line
            nl = text.find("\n", start)
            if nl == -1:
                break
            start = nl + 1
            # Find matching close — scan for ``` at start of a line
            end = _find_closing_fence(text, start)
            if end is not None:
                return text[start:end].strip()

    # No markdown fence — try generic ``` fence
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            start = nl + 1
            end = _find_closing_fence(text, start)
            if end is not None:
                return text[start:end].strip()

    # No fences — accept raw text if it looks like Markdown
    if text.startswith("#"):
        return text

    return None


def _find_closing_fence(text: str, start: int) -> int | None:
    """Find the position of the closing ``` that matches an opening fence.

    Skips over inner fenced blocks (``` that open and close within)
    using a simple depth counter.
    """
    pos = start
    depth = 0
    while pos < len(text):
        line_end = text.find("\n", pos)
        if line_end == -1:
            line_end = len(text)
        line = text[pos:line_end].strip()

        if line.startswith("```"):
            if depth > 0:
                depth -= 1
            elif line == "```":
                return pos
            else:
                depth += 1

        pos = line_end + 1

    return None
