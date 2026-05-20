from __future__ import annotations

import json

from mediated_coevo.evolution.candidates import (
    build_candidate_batch,
    selected_candidate,
)
from mediated_coevo.models.skill import SkillUpdateCandidate
from mediated_coevo.stores.artifact_store import ArtifactStore


def _candidate(candidate_id: str, score: float) -> SkillUpdateCandidate:
    return SkillUpdateCandidate(
        candidate_id=candidate_id,
        skill_id="planner",
        update_kind=f"kind-{candidate_id}",
        hypothesis=f"hypothesis {candidate_id}",
        old_content="# Skill\n",
        new_content=f"# Skill\n\n{candidate_id}\n",
        reasoning=f"reasoning {candidate_id}",
        audit_score=score,
    )


def test_random_top_quartile_selection_is_seeded_and_bounded():
    candidates = [
        _candidate("best", 0.95),
        _candidate("second", 0.90),
        _candidate("third", 0.50),
        _candidate("fourth", 0.40),
        _candidate("fifth", 0.10),
    ]

    first = build_candidate_batch(
        batch_id="batch-1",
        iteration=3,
        skill_id="planner",
        agent_role="planner",
        task_ids=["task-A"],
        current_skill="# Skill\n",
        candidates=candidates,
        selection_seed=123,
    )
    second = build_candidate_batch(
        batch_id="batch-1",
        iteration=3,
        skill_id="planner",
        agent_role="planner",
        task_ids=["task-A"],
        current_skill="# Skill\n",
        candidates=candidates,
        selection_seed=123,
    )

    assert first.selected_candidate_id == second.selected_candidate_id
    assert first.selected_candidate_id in {"best", "second"}
    assert selected_candidate(first) is not None
    assert [candidate.selected for candidate in first.candidates].count(True) == 1


def test_candidate_batch_artifact_round_trips(tmp_path):
    batch = build_candidate_batch(
        batch_id="batch-1",
        iteration=3,
        skill_id="planner",
        agent_role="planner",
        task_ids=["task-A"],
        current_skill="# Skill\n",
        candidates=[_candidate("best", 0.95)],
        selection_seed=123,
    )
    store = ArtifactStore(tmp_path / "artifacts")

    path = store.store_candidate_batch(batch)

    loaded = json.loads(path.read_text())
    assert loaded["batch_id"] == "batch-1"
    assert loaded["selected_candidate_id"] == "best"
    assert loaded["candidates"][0]["new_content"] == "# Skill\n\nbest"
