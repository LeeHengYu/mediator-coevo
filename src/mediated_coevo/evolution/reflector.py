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
    SkillUpdateCandidate,
    SkillUpdateCandidateBatch,
)
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
        )

        try:
            raw_content = await self._complete_reflection(
                agent_role=agent_role,
                llm_client=llm_client,
                messages=messages,
            )
            candidate_batch = self._build_candidate_batch(
                agent_role=agent_role,
                current_skill=current_skill,
                raw_content=raw_content,
                iteration=iteration,
                pairs=pairs,
                selection_seed=selection_seed,
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
            selected = selected_candidate(candidate_batch)
            if selected is None:
                return None
            if current_skill and _is_semantically_similar(
                current_skill,
                selected.new_content,
                self._similarity_threshold,
            ):
                logger.info(
                    "%s reflection: selected candidate too similar to current "
                    "(threshold=%.2f); skipping.",
                    agent_role,
                    self._similarity_threshold,
                )
                return None

            self._skill_store.write_skill(agent_role, selected.new_content)
            logger.info(
                "%s skill updated via reflection (%d chars).",
                agent_role,
                len(selected.new_content),
            )
            provenance = self._build_reflection_provenance(
                agent_role=agent_role,
                base_skill_hash=base_skill_hash,
                pairs=pairs,
                candidate_batch=candidate_batch,
                candidate_batch_path=candidate_batch_path,
                iteration=iteration,
                max_pairs=max_pairs,
                selection_seed=selection_seed,
            )
            return ReflectionResult(
                skill_id=agent_role,
                old_content=current_skill,
                new_content=selected.new_content,
                provenance=provenance,
            )
        except Exception as e:
            logger.error("Failed to reflect for %s: %s", agent_role, e)
            return None

    def _build_candidate_batch(
        self,
        *,
        agent_role: str,
        current_skill: str,
        raw_content: str,
        iteration: int,
        pairs: list[ContrastivePair],
        selection_seed: int | None,
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
            candidates=_parse_reflection_candidates(
                agent_role=agent_role,
                current_skill=current_skill,
                raw_content=raw_content,
            ),
            selection_seed=batch_seed,
        )

    def _build_reflection_prompt(
        self,
        agent_role: str,
        current_skill: str,
        pairs: list[ContrastivePair],
        *,
        model: str = "",
    ) -> list[dict[str, Any]]:
        if agent_role == "mediator":
            return self._build_mediator_prompt(
                current_skill,
                pairs,
                model=model,
                budgets=self._budgets,
            )
        return self._build_planner_prompt(
            current_skill,
            pairs,
            model=model,
            budgets=self._budgets,
        )

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
                f"### Pair {i} — task `{task_id}`\n"
                f"**Worse outcome** ({worse_context}):\n"
                f"{formatter(pair.worse)}\n\n"
                f"**Better outcome** ({better_context}):\n"
                f"{formatter(pair.better)}"
            )

        user_content = (
            f"## {current_skill_heading}\n\n"
            f"{current_skill}\n\n"
            "## Contrastive Evidence\n\n"
            f"{evidence_intro}\n\n"
            + "\n\n".join(contrastive_parts)
            + f"\n\n## Instructions\n\n{instructions}"
        )
        if budgets:
            from mediated_coevo.runtime.token_budget import BudgetSection, pack_sections

            user_content = pack_sections(
                model,
                [
                    BudgetSection(
                        current_skill_heading,
                        f"## {current_skill_heading}\n\n{current_skill}",
                        required=True,
                        max_tokens=budgets.max_skill_tokens,
                    ),
                    BudgetSection(
                        "contrastive_evidence",
                        "## Contrastive Evidence\n\n"
                        f"{evidence_intro}\n\n" + "\n\n".join(contrastive_parts),
                        required=True,
                        max_tokens=budgets.historical_summary_tokens,
                    ),
                    BudgetSection(
                        "instructions",
                        f"## Instructions\n\n{instructions}",
                        required=True,
                    ),
                ],
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
        model: str = "",
        budgets: BudgetsConfig | None = None,
    ) -> list[dict[str, str]]:
        return Reflector._build_contrastive_prompt(
            system_text=(
                "You are reflecting on your performance as a Mediator agent. "
                "Your coordination-protocol skill defines HOW you curate "
                "execution feedback for the Planner. You will see contrastive "
                "pairs: cases where your reporting led to better vs. worse "
                "downstream outcomes. Use these to revise your protocol.\n\n"
                "If you believe the current protocol already captures the "
                "lessons from the evidence, include a no_update candidate. "
                "Otherwise, return JSON with 2-3 candidate protocol updates. "
                "Each candidate must include candidate_id, update_kind, "
                "hypothesis, risk, audit_score, new_content, and reasoning. "
                "new_content must be the complete updated Markdown protocol: "
                "integrate changes into existing sections, resolve duplicate "
                "or conflicting rules, and avoid appended addenda unless a new "
                "section is clearly needed."
            ),
            current_skill_heading="Current Coordination Protocol",
            current_skill=current_skill,
            evidence_intro=(
                "Below are pairs of your past reports. In each pair, one report "
                "led to a WORSE downstream evolution reward and the other to a BETTER one "
                "relative to the same task's average outcome. "
                "Each entry shows the mediator's headline, decision, abstraction "
                "level, evolution reward, task-relative delta, and a diagnostic excerpt "
                "of the report."
            ),
            instructions=(
                "Revise your coordination protocol based on the patterns above. "
                "Keep the same JSON output format. Focus on:\n"
                "1. What reporting style led to better outcomes?\n"
                "2. When should you withhold vs. expose?\n"
                "3. What abstraction level works best?\n"
                "Make minimal, targeted changes. Do not rewrite from scratch."
            ),
            pairs=pairs,
            formatter=_format_mediator_entry,
            model=model,
            budgets=budgets,
        )

    @staticmethod
    def _build_planner_prompt(
        current_skill: str,
        pairs: list[ContrastivePair],
        *,
        model: str = "",
        budgets: BudgetsConfig | None = None,
    ) -> list[dict[str, str]]:
        return Reflector._build_contrastive_prompt(
            system_text=(
                "You are reflecting on your performance as a Planner agent. "
                "Your skill-refiner skill defines HOW you decide to edit the "
                "Executor's skills. You will see contrastive pairs: skill edits "
                "that led to better vs. worse outcomes. Use these to revise "
                "your editing strategy.\n\n"
                "If you believe the current guidelines already capture the "
                "lessons from the evidence, include a no_update candidate. "
                "Otherwise, return JSON with 2-3 candidate skill-refiner "
                "updates. Each candidate must include candidate_id, update_kind, "
                "hypothesis, risk, audit_score, new_content, and reasoning. "
                "new_content must be the complete updated Markdown skill: "
                "integrate changes into existing sections, resolve duplicate "
                "or conflicting rules, and avoid appended addenda unless a new "
                "section is clearly needed."
            ),
            current_skill_heading="Current Skill-Refiner Guidelines",
            current_skill=current_skill,
            evidence_intro=(
                "Below are pairs of your past skill edits. In each pair, one "
                "edit led to a WORSE downstream evolution reward and the other to a "
                "BETTER one relative to the same task's average outcome. Each "
                "entry shows your full reasoning, evolution reward, task-relative "
                "delta, the diff size, and a head+tail excerpt of the diff itself."
            ),
            instructions=(
                "Revise your skill-refiner guidelines based on the patterns "
                "above. Focus on:\n"
                "1. What kinds of edits led to better outcomes?\n"
                "2. What edit patterns should you avoid?\n"
                "3. How should you interpret the Mediator's feedback?\n"
                "Make minimal, targeted changes. Do not rewrite from scratch."
            ),
            pairs=pairs,
            formatter=_format_planner_entry,
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
