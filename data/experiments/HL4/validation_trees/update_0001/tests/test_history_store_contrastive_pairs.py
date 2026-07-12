from __future__ import annotations

import pytest

from mediated_coevo.models.history_signals import MediatorSignal, PlannerSignal
from mediated_coevo.models.skill import (
    RejectedReflectionBatch,
    SkillProposal,
    SkillValidationResult,
)
from mediated_coevo.models.trace import ExecutionTrace
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


def _trace(task_id: str, *, iteration: int, reward: float) -> ExecutionTrace:
    return ExecutionTrace(
        task_id=task_id,
        iteration=iteration,
        reward=reward,
        status="ok",
    )


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


def test_contrastive_pairs_use_judge_evolution_reward_over_verifier_metadata(
    tmp_path,
):
    store = HistoryStore(history_dir=tmp_path / "history")
    first = _add_entry(store, task_id="task-A", reward=None, iteration=0)
    store.tag_outcome(
        "task-A",
        _trace("task-A", iteration=0, reward=0.0),
        entry_ids=[first],
        outcome_reward=0.7,
        outcome_metadata={"reward_source": "judge", "judge_reward": 0.7},
    )
    second = _add_entry(store, task_id="task-A", reward=None, iteration=1)
    store.tag_outcome(
        "task-A",
        _trace("task-A", iteration=1, reward=1.0),
        entry_ids=[second],
        outcome_reward=0.4,
        outcome_metadata={"reward_source": "judge", "judge_reward": 0.4},
    )

    pairs = store.contrastive_pairs("planner", max_pairs=10)

    assert [(worse.entry_id, better.entry_id) for worse, better in pairs] == [
        (second, first)
    ]
    assert pairs[0].worse.metadata["verifier_reward"] == pytest.approx(1.0)
    assert pairs[0].better.metadata["verifier_reward"] == pytest.approx(0.0)
    assert pairs[0].worse.iteration == pairs[0].worse.metadata["outcome_iteration"]
    assert pairs[0].better.iteration == pairs[0].better.metadata["outcome_iteration"]
    assert _reward_gap(pairs[0]) == pytest.approx(0.3)


def test_tag_outcome_updates_current_iteration_proposals(tmp_path):
    store = HistoryStore(history_dir=tmp_path / "history")
    current = SkillProposal(
        old_content="old",
        new_content="new",
        iteration=2,
        task_id="task-A",
    )
    previous = SkillProposal(
        old_content="old",
        new_content="new",
        iteration=1,
        task_id="task-A",
    )

    store.tag_outcome(
        "task-A",
        _trace("task-A", iteration=2, reward=1.0),
        proposals=[current, previous],
        outcome_reward=0.8,
        outcome_metadata={"reward_source": "judge", "judge_reward": 0.8},
    )

    assert current.reward == pytest.approx(0.8)
    assert current.reward_source == "judge"
    assert current.verifier_reward == pytest.approx(1.0)
    assert current.judge_reward == pytest.approx(0.8)
    assert previous.reward is None


def test_tag_outcome_persists_multiple_entry_tags_with_one_save(
    tmp_path,
    monkeypatch,
):
    store = HistoryStore(history_dir=tmp_path / "history")
    first = _add_entry(store, task_id="task-A", reward=None, iteration=0)
    second = _add_entry(store, task_id="task-A", reward=None, iteration=0)
    save_calls = 0
    original_save = store._save

    def counted_save():
        nonlocal save_calls
        save_calls += 1
        original_save()

    monkeypatch.setattr(store, "_save", counted_save)

    store.tag_outcome(
        "task-A",
        _trace("task-A", iteration=0, reward=1.0),
        entry_ids=[first, second],
    )

    assert save_calls == 1
    reloaded = HistoryStore(history_dir=tmp_path / "history")
    rewards = [
        entry.reward
        for entry in reloaded.query(recent=10)
        if entry.entry_id in {first, second}
    ]
    assert rewards == [1.0, 1.0]


def test_tagged_task_counts_group_by_role_and_task(tmp_path):
    store = HistoryStore(history_dir=tmp_path / "history")
    _add_entry(store, task_id="task-A", reward=0.1, iteration=0)
    _add_entry(store, task_id="task-A", reward=0.2, iteration=1)
    _add_entry(store, task_id="task-B", reward=0.3, iteration=2)
    _add_entry(store, task_id="task-B", reward=None, iteration=3)
    store.add(
        HistoryEntry(
            iteration=4,
            agent_role="mediator",
            payload=MediatorSignal(headline="wrong role"),
            reward=0.9,
            metadata={"task_id": "task-A"},
        )
    )

    assert store.tagged_task_counts("planner") == {"task-A": 2, "task-B": 1}


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


def test_rejected_reflections_are_persisted_and_queryable(tmp_path):
    store = HistoryStore(history_dir=tmp_path / "history")
    rejection_id = store.record_rejected_reflection(
        RejectedReflectionBatch(
            batch_id="reflect-planner-iter-0002",
            iteration=2,
            skill_id="planner",
            agent_role="planner",
            task_ids=["task-A"],
            base_skill_hash="old-hash",
            reason="validation_rejected_all_candidates",
            selected_candidate_id="candidate-1",
            selected_update_kind="broad_rewrite",
            validation=SkillValidationResult(
                validation_id="validation-1",
                decision="rejected",
                reason="mean_delta_below_threshold",
                mean_delta=-0.2,
            ),
        )
    )

    reloaded = HistoryStore(history_dir=tmp_path / "history")
    rejected = reloaded.query_rejected_reflections(agent_role="planner")

    assert rejected[0].rejection_id == rejection_id
    assert rejected[0].selected_candidate_id == "candidate-1"
    assert rejected[0].validation is not None
    assert rejected[0].validation.reason == "mean_delta_below_threshold"
    assert reloaded.query_rejected_reflections(agent_role="mediator") == []
