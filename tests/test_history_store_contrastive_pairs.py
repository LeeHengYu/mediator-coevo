from __future__ import annotations

import pytest

from mediated_coevo.models.history_signals import PlannerSignal
from mediated_coevo.stores.history_store import HistoryEntry, HistoryStore


def _add_entry(
    store: HistoryStore,
    *,
    task_id: str | None,
    reward: float | None,
    iteration: int,
) -> str:
    metadata = {} if task_id is None else {"task_id": task_id}
    return store.add(
        HistoryEntry(
            iteration=iteration,
            agent_role="planner",
            payload=PlannerSignal(reasoning=f"entry {iteration}"),
            reward=reward,
            metadata=metadata,
        )
    )


def _reward_gap(pair: tuple[HistoryEntry, HistoryEntry]) -> float:
    worse, better = pair
    assert worse.reward is not None
    assert better.reward is not None
    return better.reward - worse.reward


def test_contrastive_pairs_are_same_task_and_sorted_by_gap(tmp_path):
    store = HistoryStore(history_dir=tmp_path / "history")
    _add_entry(store, task_id="task-A", reward=0.1, iteration=0)
    _add_entry(store, task_id="task-A", reward=0.9, iteration=1)
    _add_entry(store, task_id="task-B", reward=0.2, iteration=2)
    _add_entry(store, task_id="task-B", reward=0.7, iteration=3)

    pairs = store.contrastive_pairs("planner", max_pairs=10)

    assert len(pairs) == 2
    assert [worse.metadata["task_id"] for worse, _ in pairs] == ["task-A", "task-B"]
    assert [
        better.metadata["task_id"] for _, better in pairs
    ] == ["task-A", "task-B"]
    assert [_reward_gap(pair) for pair in pairs] == pytest.approx([0.8, 0.5])


def test_contrastive_pairs_respect_task_filter(tmp_path):
    store = HistoryStore(history_dir=tmp_path / "history")
    task_a_low = _add_entry(store, task_id="task-A", reward=0.2, iteration=0)
    task_a_high = _add_entry(store, task_id="task-A", reward=0.6, iteration=1)
    _add_entry(store, task_id="task-B", reward=0.0, iteration=2)
    _add_entry(store, task_id="task-B", reward=1.0, iteration=3)

    pairs = store.contrastive_pairs("planner", task_id="task-A", max_pairs=10)

    assert [(worse.entry_id, better.entry_id) for worse, better in pairs] == [
        (task_a_low, task_a_high)
    ]


def test_contrastive_pairs_ignore_unusable_entries(tmp_path):
    store = HistoryStore(history_dir=tmp_path / "history")

    _add_entry(store, task_id=None, reward=0.0, iteration=0)
    _add_entry(store, task_id=None, reward=1.0, iteration=1)
    _add_entry(store, task_id="equal-rewards", reward=0.5, iteration=2)
    _add_entry(store, task_id="equal-rewards", reward=0.5, iteration=3)
    _add_entry(store, task_id="untagged", reward=0.1, iteration=4)
    _add_entry(store, task_id="untagged", reward=None, iteration=5)

    assert store.contrastive_pairs("planner", max_pairs=10) == []


def test_contrastive_pair_sampling_is_seeded_bounded_and_sorted(tmp_path):
    store = HistoryStore(history_dir=tmp_path / "history")
    for index, reward in enumerate([0.0, 0.1, 0.2, 0.8, 0.9, 1.0]):
        _add_entry(store, task_id="task-A", reward=reward, iteration=index)

    first = store.contrastive_pairs(
        "planner",
        max_pairs=3,
        top_frac=0.5,
        bot_frac=0.5,
        selection_seed=123,
    )
    second = store.contrastive_pairs(
        "planner",
        max_pairs=3,
        top_frac=0.5,
        bot_frac=0.5,
        selection_seed=123,
    )

    assert len(first) == 3
    assert [(w.entry_id, b.entry_id) for w, b in first] == [
        (w.entry_id, b.entry_id) for w, b in second
    ]
    assert [_reward_gap(pair) for pair in first] == sorted(
        [_reward_gap(pair) for pair in first],
        reverse=True,
    )
