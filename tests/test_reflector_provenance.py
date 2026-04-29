from __future__ import annotations

import pytest

from mediated_coevo.evolution.reflector import Reflector
from mediated_coevo.models.history_signals import MediatorSignal, PlannerSignal
from mediated_coevo.stores.history_store import HistoryEntry, HistoryStore
from mediated_coevo.stores.skill_store import SkillStore


class _MarkdownLLM:
    model = "test-model"

    async def complete(self, *args, **kwargs):
        return {
            "content": "```markdown\n# New Mediator Protocol\n\nUse clearer feedback.\n```"
        }


def _skill_store(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "mediator"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Old Mediator Protocol\n")
    return SkillStore(skills_dir)


@pytest.mark.asyncio
async def test_reflector_returns_concise_contrastive_provenance(tmp_path):
    history = HistoryStore(history_dir=tmp_path / "history")
    worse_id = history.add(HistoryEntry(
        iteration=0,
        agent_role="mediator",
        payload=MediatorSignal(headline="too vague"),
        reward=0.1,
        metadata={"task_id": "task-A"},
    ))
    better_id = history.add(HistoryEntry(
        iteration=1,
        agent_role="mediator",
        payload=MediatorSignal(headline="actionable"),
        reward=0.9,
        metadata={"task_id": "task-A"},
    ))
    store = _skill_store(tmp_path)

    result = await Reflector(history, store).reflect(
        "mediator",
        _MarkdownLLM(),
        iteration=5,
        selection_seed=123,
    )

    assert result is not None
    assert result.skill_id == "mediator"
    assert result.old_content == "# Old Mediator Protocol\n"
    assert result.new_content == "# New Mediator Protocol\n\nUse clearer feedback."
    assert result.provenance.kind == "contrastive_reflection"
    assert result.provenance.batch_id == "reflect-mediator-iter-0005"
    assert result.provenance.task_ids == ["task-A"]
    assert result.provenance.base_skill_hash == SkillStore.content_hash(
        "# Old Mediator Protocol\n"
    )
    assert result.provenance.selection_seed == 123
    assert len(result.provenance.contrastive_pair_refs) == 1
    pair_ref = result.provenance.contrastive_pair_refs[0]
    assert pair_ref.worse_entry_id == worse_id
    assert pair_ref.better_entry_id == better_id
    assert pair_ref.worse_reward == pytest.approx(0.1)
    assert pair_ref.better_reward == pytest.approx(0.9)
    assert pair_ref.reward_gap == pytest.approx(0.8)


def test_contrastive_pair_sampling_is_repeatable_with_seed(tmp_path):
    history = HistoryStore(history_dir=tmp_path / "history")
    for index, reward in enumerate([0.1, 0.2, 0.3, 0.8, 0.9, 1.0]):
        history.add(HistoryEntry(
            iteration=index,
            agent_role="planner",
            payload=PlannerSignal(reasoning=f"entry {index}"),
            reward=reward,
            metadata={"task_id": "task-A"},
        ))

    first = history.contrastive_pairs(
        "planner",
        max_pairs=2,
        selection_seed=99,
    )
    second = history.contrastive_pairs(
        "planner",
        max_pairs=2,
        selection_seed=99,
    )

    assert [(w.entry_id, b.entry_id) for w, b in first] == [
        (w.entry_id, b.entry_id) for w, b in second
    ]
