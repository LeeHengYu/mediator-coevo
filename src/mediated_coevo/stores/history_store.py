"""History store — outcome-tagged history for co-evolution.

Stores the history of mediator reports and planner skill edits, each tagged
with the downstream reward. Used by the reflector to build contrastive pairs
and reflection prompts. Rejected executor skill proposal batches are stored in
a separate sidecar file so they can be inspected later without becoming
committed skill-update history.

Each ``HistoryEntry`` carries a typed ``payload`` (``MediatorSignal``
or ``PlannerSignal``) instead of a free-text blob, so the Reflector
can pull from per-slot fields with realistic budgets instead of
doing lossy character truncation on a single string.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from mediated_coevo.models.history_signals import HistorySignal
from mediated_coevo.models.skill import RejectedProposalBatch, RejectedReflectionBatch

if TYPE_CHECKING:
    from mediated_coevo.models.skill import SkillProposal
    from mediated_coevo.models.trace import ExecutionTrace

logger = logging.getLogger(__name__)


class HistoryEntry(BaseModel):
    """One outcome-tagged entry in the co-evolution history."""

    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    iteration: int
    agent_role: str
    payload: HistorySignal
    reward: float | None = None  # Filled by tag_outcome_by_id
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


@dataclass(frozen=True)
class _RewardedEntry:
    """History entry paired with its non-optional reward."""

    entry: HistoryEntry
    reward: float
    relative_reward: float = 0.0


@dataclass(frozen=True)
class ContrastivePair:
    """Same-task contrastive pair with transient group-relative scores."""

    worse: HistoryEntry
    better: HistoryEntry
    worse_reward: float
    better_reward: float
    worse_relative_reward: float
    better_relative_reward: float

    def __iter__(self) -> Iterator[HistoryEntry]:
        """Allow existing tuple-unpacking call sites to keep working."""
        yield self.worse
        yield self.better

    @property
    def reward_gap(self) -> float:
        return self.better_reward - self.worse_reward

    @property
    def relative_reward_gap(self) -> float:
        return self.better_relative_reward - self.worse_relative_reward


class HistoryStore:
    """File-backed history of outcome-tagged actions for co-evolution."""

    _HISTORY_FILE = "history.jsonl"
    _REJECTED_PROPOSALS_FILE = "rejected_proposals.jsonl"
    _REJECTED_REFLECTIONS_FILE = "rejected_reflections.jsonl"

    def __init__(self, history_dir: Path) -> None:
        self._history_dir = history_dir
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[HistoryEntry] = []
        self._rejected_proposal_batches: list[RejectedProposalBatch] = []
        self._rejected_reflection_batches: list[RejectedReflectionBatch] = []
        self._pending_mediator_entry_id_by_task: dict[str, str] = {}
        self._pending_planner_entry_id_by_task: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load all history entries from disk."""
        path = self._history_dir / self._HISTORY_FILE
        if path.exists():
            for line in path.read_text().strip().split("\n"):
                if line.strip():
                    try:
                        self._entries.append(HistoryEntry.model_validate_json(line))
                    except Exception as e:
                        logger.warning("Failed to parse history entry: %s", e)

        rejected_path = self._history_dir / self._REJECTED_PROPOSALS_FILE
        if rejected_path.exists():
            for line in rejected_path.read_text().strip().split("\n"):
                if line.strip():
                    try:
                        self._rejected_proposal_batches.append(
                            RejectedProposalBatch.model_validate_json(line)
                        )
                    except Exception as e:
                        logger.warning("Failed to parse rejected proposal batch: %s", e)

        rejected_reflections_path = self._history_dir / self._REJECTED_REFLECTIONS_FILE
        if rejected_reflections_path.exists():
            for line in rejected_reflections_path.read_text().strip().split("\n"):
                if line.strip():
                    try:
                        self._rejected_reflection_batches.append(
                            RejectedReflectionBatch.model_validate_json(line)
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to parse rejected reflection batch: %s",
                            e,
                        )

    def _save(self) -> None:
        """Persist all entries to disk."""
        path = self._history_dir / self._HISTORY_FILE
        lines = [entry.model_dump_json() for entry in self._entries]
        path.write_text("\n".join(lines) + "\n")

    def _save_rejected_proposals(self) -> None:
        """Persist rejected proposal batches to disk."""
        path = self._history_dir / self._REJECTED_PROPOSALS_FILE
        lines = [batch.model_dump_json() for batch in self._rejected_proposal_batches]
        path.write_text("\n".join(lines) + "\n")

    def _save_rejected_reflections(self) -> None:
        """Persist rejected reflection batches to disk."""
        path = self._history_dir / self._REJECTED_REFLECTIONS_FILE
        lines = [batch.model_dump_json() for batch in self._rejected_reflection_batches]
        path.write_text("\n".join(lines) + "\n")

    def add(self, entry: HistoryEntry) -> str:
        self._entries.append(entry)
        self._save()
        return entry.entry_id

    def record_rejected_proposals(self, batch: RejectedProposalBatch) -> str:
        """Persist a reviewed proposal batch that was not committed."""
        self._rejected_proposal_batches.append(batch)
        self._save_rejected_proposals()
        return batch.rejection_id

    def record_rejected_reflection(self, batch: RejectedReflectionBatch) -> str:
        """Persist a reflected meta-skill batch that was not committed."""
        self._rejected_reflection_batches.append(batch)
        self._save_rejected_reflections()
        return batch.rejection_id

    def record_signal(
        self,
        *,
        iteration: int,
        agent_role: str,
        task_id: str,
        condition: str,
        payload: HistorySignal,
    ) -> str:
        """Persist one role signal with standard experiment metadata."""
        return self.add(
            HistoryEntry(
                iteration=iteration,
                agent_role=agent_role,
                payload=payload,
                metadata={"task_id": task_id, "condition": condition},
            )
        )

    def remember_pending_outcome(
        self,
        task_id: str,
        *,
        mediator_entry_id: str | None = None,
        planner_entry_id: str | None = None,
    ) -> None:
        """Remember entries that should be tagged by the next clean reward."""
        if mediator_entry_id:
            self._pending_mediator_entry_id_by_task[task_id] = mediator_entry_id
        if planner_entry_id:
            self._pending_planner_entry_id_by_task[task_id] = planner_entry_id

    def tag_pending_outcome(
        self,
        task_id: str,
        trace: ExecutionTrace,
        *,
        proposals: list[SkillProposal] | None = None,
        outcome_reward: float | None = None,
        outcome_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Tag pending role entries with this trace's evolution reward."""
        if trace.iteration <= 0:
            return

        mediator_entry_id = self._pending_mediator_entry_id_by_task.pop(task_id, None)
        planner_entry_id = self._pending_planner_entry_id_by_task.pop(task_id, None)

        reward = trace.reward if outcome_reward is None else outcome_reward
        if not trace.is_usable_feedback_signal or reward is None:
            if mediator_entry_id or planner_entry_id:
                logger.info(
                    "Dropping carry-forward entry IDs untagged for task=%s "
                    "(trace status=%s reward=%s)",
                    task_id,
                    trace.status,
                    trace.reward,
                )
            return

        metadata: dict[str, Any] = {
            "verifier_reward": trace.reward,
            "reward_source": "verifier",
            "outcome_task_id": task_id,
            "outcome_iteration": trace.iteration,
            "trace_status": trace.status,
        }
        if outcome_metadata:
            metadata.update(outcome_metadata)

        if mediator_entry_id:
            self.tag_outcome_by_id(
                mediator_entry_id,
                reward=reward,
                metadata=metadata,
            )
        if planner_entry_id:
            self.tag_outcome_by_id(
                planner_entry_id,
                reward=reward,
                metadata=metadata,
            )
        for proposal in proposals or []:
            if (
                proposal.iteration == trace.iteration - 1
                and proposal.task_id == task_id
            ):
                proposal.reward = reward
                reward_source = metadata.get("reward_source")
                proposal.reward_source = (
                    reward_source if isinstance(reward_source, str) else None
                )
                proposal.verifier_reward = trace.reward
                proposal.judge_reward = _float_metadata(metadata.get("judge_reward"))

    def tag_outcome_by_id(
        self,
        entry_id: str,
        reward: float,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Tag a specific entry by its stable ID."""
        for entry in reversed(self._entries):
            if entry.entry_id == entry_id:
                entry.reward = reward
                if metadata:
                    entry.metadata.update(metadata)
                self._save()
                return
        logger.warning("No history entry found for entry_id=%s", entry_id)

    def annotate_judge_reward(
        self,
        *,
        task_id: str,
        iteration: int,
        judge_reward: float,
        judge_reward_record_id: str,
        rubric_version: str,
        confidence: float,
        applied_cap: str | None,
        reward_source: str = "judge",
    ) -> int:
        """Promote judge reward onto verifier-tagged history entries."""
        updated = 0
        for entry in self._entries:
            if (
                entry.reward is None
                or entry.metadata.get("outcome_task_id") != task_id
                or entry.metadata.get("outcome_iteration") != iteration
            ):
                continue
            entry.metadata.setdefault("verifier_reward", entry.reward)
            entry.reward = judge_reward
            entry.metadata.update(
                {
                    "reward_source": reward_source,
                    "judge_reward": judge_reward,
                    "judge_reward_record_id": judge_reward_record_id,
                    "judge_rubric_version": rubric_version,
                    "judge_confidence": confidence,
                    "judge_applied_cap": applied_cap,
                }
            )
            updated += 1
        if updated:
            self._save()
        return updated

    def query(
        self,
        agent_role: str | None = None,
        recent: int = 20,
        tagged_only: bool = False,
    ) -> list[HistoryEntry]:
        """Query history entries, most recent first."""
        entries = self._entries
        if agent_role:
            entries = [e for e in entries if e.agent_role == agent_role]
        if tagged_only:
            entries = [e for e in entries if e.reward is not None]
        return entries[-recent:]

    def tagged_task_counts(self, agent_role: str) -> dict[str, int]:
        """Return tagged same-role history counts grouped by task ID."""
        counts: dict[str, int] = defaultdict(int)
        for entry in self._entries:
            if entry.agent_role != agent_role or entry.reward is None:
                continue
            task_id = entry.metadata.get("task_id")
            if isinstance(task_id, str) and task_id:
                counts[task_id] += 1
        return dict(counts)

    def query_rejected_proposals(
        self,
        *,
        skill_id: str | None = None,
        recent: int = 20,
    ) -> list[RejectedProposalBatch]:
        """Query rejected proposal batches, oldest-to-newest within the window."""
        batches = self._rejected_proposal_batches
        if skill_id:
            batches = [batch for batch in batches if batch.skill_id == skill_id]
        return batches[-recent:]

    def query_rejected_reflections(
        self,
        *,
        agent_role: str | None = None,
        recent: int = 20,
    ) -> list[RejectedReflectionBatch]:
        """Query rejected reflection batches, oldest-to-newest within the window."""
        batches = self._rejected_reflection_batches
        if agent_role:
            batches = [batch for batch in batches if batch.agent_role == agent_role]
        return batches[-recent:]

    def contrastive_pairs(
        self,
        agent_role: str,
        max_pairs: int = 5,
        task_id: str | None = None,
        top_frac: float = 0.3,
        bot_frac: float = 0.3,
        selection_seed: int | None = None,
    ) -> list[ContrastivePair]:
        """Form same-task contrastive pairs from top/bottom reward buckets.

        For each task with at least two tagged entries, compute same-role,
        same-task relative scores as ``reward - task_mean_reward``. Then sort
        by evolution reward and take the bottom ``bot_frac`` and top ``top_frac`` as
        disjoint buckets. Build all cross-bucket candidates with a strict
        reward gap, pool them across tasks, and return up to ``max_pairs``
        pairs ordered by descending relative-score gap. The persisted verifier
        rewards are kept in metadata when an alternate reward source is used.

        Args:
            agent_role: Role to filter entries by.
            max_pairs: Maximum number of pairs to return.
            task_id: If set, restrict to entries whose metadata task_id matches.
            top_frac: Fraction of each task group to treat as the "better" bucket.
            bot_frac: Fraction of each task group to treat as the "worse" bucket.
            selection_seed: Deterministic seed used to break equal-gap ties.
        """
        tagged: list[_RewardedEntry] = []
        for entry in self._entries:
            if entry.agent_role == agent_role and entry.reward is not None:
                tagged.append(_RewardedEntry(entry=entry, reward=entry.reward))
        if task_id is not None:
            tagged = [
                item for item in tagged if item.entry.metadata.get("task_id") == task_id
            ]

        by_task: dict[str, list[_RewardedEntry]] = defaultdict(list)
        dropped_untagged = 0
        for item in tagged:
            tid = item.entry.metadata.get("task_id", "")
            if not tid:
                dropped_untagged += 1
                continue
            by_task[tid].append(item)
        if dropped_untagged:
            logger.debug(
                "Dropped %d entries with no task_id from contrastive pairing.",
                dropped_untagged,
            )

        pool: list[ContrastivePair] = []
        for entries in by_task.values():
            n = len(entries)
            if n < 2:
                continue

            task_mean_reward = sum(item.reward for item in entries) / n
            relative_entries = [
                _RewardedEntry(
                    entry=item.entry,
                    reward=item.reward,
                    relative_reward=item.reward - task_mean_reward,
                )
                for item in entries
            ]
            k_bot = max(1, ceil(n * bot_frac))
            k_top = max(1, ceil(n * top_frac))

            # Enforce disjointness: buckets must not overlap.
            if k_bot + k_top > n:
                k_bot = max(1, min(k_bot, n - 1))
                k_top = max(1, n - k_bot)

            sorted_entries = sorted(relative_entries, key=lambda item: item.reward)
            bot = sorted_entries[:k_bot]
            top = sorted_entries[-k_top:]

            for worse in bot:
                for better in top:
                    if better.reward <= worse.reward:
                        continue
                    pool.append(
                        ContrastivePair(
                            worse=worse.entry,
                            better=better.entry,
                            worse_reward=worse.reward,
                            better_reward=better.reward,
                            worse_relative_reward=worse.relative_reward,
                            better_relative_reward=better.relative_reward,
                        )
                    )

        if not pool:
            return []

        if len(pool) > max_pairs:
            random.Random(selection_seed).shuffle(pool)

        pool.sort(key=lambda pair: pair.relative_reward_gap, reverse=True)
        return pool[:max_pairs]


def _float_metadata(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
