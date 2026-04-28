"""History store — outcome-tagged history for co-evolution.

Stores the history of mediator reports and planner skill edits,
each tagged with the downstream reward. Used by the reflector
to build contrastive pairs and reflection prompts.

Each ``HistoryEntry`` carries a typed ``payload`` (``MediatorSignal``
or ``PlannerSignal``) instead of a free-text blob, so the Reflector
can pull from per-slot fields with realistic budgets instead of
doing lossy character truncation on a single string.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from mediated_coevo.models.history_signals import HistorySignal

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


class HistoryStore:
    """File-backed history of outcome-tagged actions for co-evolution."""

    _HISTORY_FILE = "history.jsonl"

    def __init__(self, history_dir: Path) -> None:
        self._history_dir = history_dir
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[HistoryEntry] = []
        self._pending_mediator_entry_id_by_task: dict[str, str] = {}
        self._pending_planner_entry_id_by_task: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load all history entries from disk."""
        path = self._history_dir / self._HISTORY_FILE
        if not path.exists():
            return
        for line in path.read_text().strip().split("\n"):
            if line.strip():
                try:
                    self._entries.append(HistoryEntry.model_validate_json(line))
                except Exception as e:
                    logger.warning("Failed to parse history entry: %s", e)

    def _save(self) -> None:
        """Persist all entries to disk."""
        path = self._history_dir / self._HISTORY_FILE
        lines = [entry.model_dump_json() for entry in self._entries]
        path.write_text("\n".join(lines) + "\n")

    def add(self, entry: HistoryEntry) -> str:
        self._entries.append(entry)
        self._save()
        return entry.entry_id

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
        return self.add(HistoryEntry(
            iteration=iteration,
            agent_role=agent_role,
            payload=payload,
            metadata={"task_id": task_id, "condition": condition},
        ))

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
        trace: "ExecutionTrace",
        *,
        proposals: list["SkillProposal"] | None = None,
    ) -> None:
        """Tag pending role entries with this trace's reward, when usable."""
        if trace.iteration <= 0:
            return

        mediator_entry_id = self._pending_mediator_entry_id_by_task.pop(task_id, None)
        planner_entry_id = self._pending_planner_entry_id_by_task.pop(task_id, None)

        reward = trace.reward
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

        if mediator_entry_id:
            self.tag_outcome_by_id(mediator_entry_id, reward=reward)
        if planner_entry_id:
            self.tag_outcome_by_id(planner_entry_id, reward=reward)
        for proposal in proposals or []:
            if proposal.iteration == trace.iteration - 1 and proposal.task_id == task_id:
                proposal.reward = reward

    def tag_outcome_by_id(self, entry_id: str, reward: float) -> None:
        """Tag a specific entry by its stable ID."""
        for entry in reversed(self._entries):
            if entry.entry_id == entry_id:
                entry.reward = reward
                self._save()
                return
        logger.warning("No history entry found for entry_id=%s", entry_id)

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

    def contrastive_pairs(
        self,
        agent_role: str,
        max_pairs: int = 5,
        task_id: str | None = None,
        top_frac: float = 0.3,
        bot_frac: float = 0.3,
        rng: random.Random | None = None,
    ) -> list[tuple[HistoryEntry, HistoryEntry]]:
        """Form same-task contrastive pairs from top/bottom reward buckets.

        For each task with at least two tagged entries, sort by reward and
        take the bottom ``bot_frac`` and top ``top_frac`` as disjoint buckets.
        Build all cross-bucket ``(worse, better)`` candidates with a strict
        reward gap, pool them across tasks, then randomly sample at most
        ``max_pairs`` and return them sorted by descending gap.

        Args:
            agent_role: Role to filter entries by.
            max_pairs: Maximum number of pairs to return.
            task_id: If set, restrict to entries whose metadata task_id matches.
            top_frac: Fraction of each task group to treat as the "better" bucket.
            bot_frac: Fraction of each task group to treat as the "worse" bucket.
            rng: Injectable RNG for deterministic sampling.
        """
        rng = rng or random.Random()

        tagged = [
            _RewardedEntry(entry=e, reward=e.reward)
            for e in self._entries
            if e.agent_role == agent_role and e.reward is not None
        ]
        if task_id is not None:
            tagged = [
                item for item in tagged
                if item.entry.metadata.get("task_id") == task_id
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

        pool: list[tuple[_RewardedEntry, _RewardedEntry]] = []
        for entries in by_task.values():
            n = len(entries)
            if n < 2:
                continue

            sorted_entries = sorted(entries, key=lambda item: item.reward)
            k_bot = max(1, ceil(n * bot_frac))
            k_top = max(1, ceil(n * top_frac))

            # Enforce disjointness: buckets must not overlap.
            if k_bot + k_top > n:
                k_bot = max(1, min(k_bot, n - 1))
                k_top = max(1, n - k_bot)

            bot = sorted_entries[:k_bot]
            top = sorted_entries[-k_top:]

            for worse in bot:
                for better in top:
                    if better.reward > worse.reward:
                        pool.append((worse, better))

        if not pool:
            return []

        if len(pool) > max_pairs:
            pool = rng.sample(pool, max_pairs)

        pool.sort(key=lambda p: p[1].reward - p[0].reward, reverse=True)
        return [(worse.entry, better.entry) for worse, better in pool]
