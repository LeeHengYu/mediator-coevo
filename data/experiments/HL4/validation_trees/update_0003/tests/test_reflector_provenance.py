from __future__ import annotations

from typing import Any

import pytest

from mediated_coevo.evolution.reflector import Reflector
from mediated_coevo.models.history_signals import MediatorSignal, PlannerSignal
from mediated_coevo.models.skill import SkillValidationResult
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryEntry, HistoryStore
from mediated_coevo.stores.skill_store import SkillStore
from tests.prompt_helpers import assert_contains_all


class _MarkdownLLM:
    model = "test-model"

    async def complete(self, *args, **kwargs):
        return {
            "content": "```markdown\n# New Mediator Protocol\n\nUse clearer feedback.\n```"
        }


class _CandidateBatchLLM:
    model = "test-model"

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def complete(self, *args, **kwargs):
        self.messages = kwargs.get("messages") or args[0]
        return {
            "content": (
                '{"candidates": ['
                '{"candidate_id": "broad", "update_kind": "edit_scope_policy", '
                '"hypothesis": "large change", "risk": "too broad", '
                '"audit_score": 0.2, "new_content": "# Broad Planner\\n", '
                '"reasoning": "broad"}, '
                '{"candidate_id": "targeted", "update_kind": "minimal_change_rule", '
                '"hypothesis": "targeted change", "risk": "low", '
                '"audit_score": 0.9, "new_content": "# Targeted Planner\\n", '
                '"reasoning": "targeted"}'
                "]}"
            )
        }


def _skill_store(tmp_path, skill_name: str = "mediator", content: str | None = None):
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / skill_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(content or "# Old Mediator Protocol\n")
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


@pytest.mark.asyncio
async def test_reflector_selects_candidate_batch_and_persists_artifact(tmp_path):
    history = HistoryStore(history_dir=tmp_path / "history")
    history.add(HistoryEntry(
        iteration=0,
        agent_role="planner",
        payload=PlannerSignal(reasoning="over-edited"),
        reward=0.2,
        metadata={"task_id": "task-A"},
    ))
    history.add(HistoryEntry(
        iteration=1,
        agent_role="planner",
        payload=PlannerSignal(reasoning="targeted"),
        reward=0.8,
        metadata={"task_id": "task-A"},
    ))
    artifacts = ArtifactStore(tmp_path / "artifacts")

    result = await Reflector(
        history,
        _skill_store(tmp_path, skill_name="planner", content="# Old Planner\n"),
        artifact_store=artifacts,
        base_seed=42,
    ).reflect(
        "planner",
        _CandidateBatchLLM(),
        iteration=5,
    )

    assert result is not None
    assert result.new_content == "# Targeted Planner"
    assert result.provenance.selected_candidate_id == "targeted"
    assert result.provenance.selected_update_kind == "minimal_change_rule"
    assert result.provenance.candidate_batch_id == "reflect-planner-iter-0005"
    assert [ref.candidate_id for ref in result.provenance.candidate_refs] == [
        "broad",
        "targeted",
    ]

    artifact_path = (
        tmp_path
        / "artifacts"
        / "candidate_batches"
        / "reflect-planner-iter-0005.json"
    )
    assert artifact_path.exists()


@pytest.mark.asyncio
async def test_reflector_records_rejected_draft_and_prompts_with_negative_history(
    tmp_path,
):
    history = HistoryStore(history_dir=tmp_path / "history")
    history.add(HistoryEntry(
        iteration=0,
        agent_role="planner",
        payload=PlannerSignal(reasoning="over-edited"),
        reward=0.2,
        metadata={"task_id": "task-A"},
    ))
    history.add(HistoryEntry(
        iteration=1,
        agent_role="planner",
        payload=PlannerSignal(reasoning="targeted"),
        reward=0.8,
        metadata={"task_id": "task-A"},
    ))
    skill_store = _skill_store(tmp_path, skill_name="planner", content="# Old Planner\n")
    reflector = Reflector(history, skill_store, base_seed=42)
    draft = await reflector.draft_reflection(
        agent_role="planner",
        llm_client=_CandidateBatchLLM(),
        iteration=2,
        select_candidate=False,
    )
    assert draft is not None

    rejection_id = reflector.record_rejected_draft(
        draft,
        reason="validation_rejected_all_candidates",
        validation=SkillValidationResult(
            validation_id="validation-1",
            decision="rejected",
            reason="mean_delta_below_threshold",
            mean_delta=-0.2,
        ),
        selected_candidate_id="broad",
    )

    rejected = history.query_rejected_reflections(agent_role="planner")
    assert rejected[0].rejection_id == rejection_id
    assert rejected[0].selected_candidate_id == "broad"
    assert rejected[0].validation is not None
    assert rejected[0].validation.mean_delta == pytest.approx(-0.2)

    llm = _CandidateBatchLLM()
    await reflector.draft_reflection(
        agent_role="planner",
        llm_client=llm,
        iteration=3,
        select_candidate=False,
    )
    user_prompt = llm.messages[1]["content"]

    assert_contains_all(
        user_prompt,
        [
            "Recently Rejected Reflection Updates",
            "validation_rejected_all_candidates",
            "mean_delta_below_threshold",
            "broad",
        ],
    )
