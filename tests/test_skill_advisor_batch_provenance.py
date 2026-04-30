from __future__ import annotations

import pytest

from mediated_coevo.config import Config
from mediated_coevo.evolution.skill_advisor import SkillAdvisor
from mediated_coevo.models.skill import SkillProposal
from mediated_coevo.orchestrator import Orchestrator
from mediated_coevo.stores.history_store import HistoryStore


class _LLM:
    model = "test-model"

    def __init__(self, *, content: str | None = None, exc: Exception | None = None):
        self.content = content
        self.exc = exc

    async def complete(self, **kwargs):
        if self.exc:
            raise self.exc
        return {"content": self.content or ""}


class _SkillStore:
    def __init__(self) -> None:
        self.content = "# Executor\n"
        self.writes: list[tuple[str, str]] = []

    def read_skill(self, skill_name: str) -> str | None:
        assert skill_name == "executor"
        return self.content

    def write_skill(self, skill_name: str, content: str):
        self.writes.append((skill_name, content))
        self.content = content


class _NoCallPlanner:
    async def suggest_skill_revision(self, *args, **kwargs):
        raise AssertionError("planner should not patch when advisor rejects")


def _proposal(task_id: str, reward: float) -> SkillProposal:
    return SkillProposal(
        iteration=0,
        task_id=task_id,
        old_content="# Executor\n",
        new_content=f"# Executor\n\nHandle {task_id}.\n",
        reasoning=f"reason for {task_id}",
        reward=reward,
    )


def _orchestrator(tmp_path, advisor: SkillAdvisor) -> tuple[Orchestrator, _SkillStore]:
    skill_store = _SkillStore()
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = Config()
    orch.config.experiment.advisor_buffer_max = 2
    orch.skill_store = skill_store
    orch.history_store = HistoryStore(history_dir=tmp_path / "history")
    orch.skill_advisor = advisor
    orch.planner = _NoCallPlanner()
    orch._proposal_buffer = [
        _proposal("task-A", 0.2),
        _proposal("task-B", 0.8),
    ]
    return orch, skill_store


@pytest.mark.asyncio
async def test_advisor_rejection_clears_buffer_without_skill_update(tmp_path):
    advisor = SkillAdvisor(
        _LLM(content='{"approve": false, "feedback": ""}')  # type: ignore[arg-type]
    )
    orch, skill_store = _orchestrator(tmp_path, advisor)
    original_rewards = [proposal.reward for proposal in orch._proposal_buffer]

    update = await orch._review_proposals_and_patch_skill(iteration=3)

    assert update is None
    assert orch._proposal_buffer == []
    assert skill_store.content == "# Executor\n"
    assert skill_store.writes == []
    assert original_rewards == [0.2, 0.8]


@pytest.mark.asyncio
async def test_advisor_llm_failure_clears_buffer_without_skill_update(tmp_path):
    advisor = SkillAdvisor(
        _LLM(exc=RuntimeError("advisor unavailable"))  # type: ignore[arg-type]
    )
    orch, skill_store = _orchestrator(tmp_path, advisor)

    update = await orch._review_proposals_and_patch_skill(iteration=3)

    assert update is None
    assert orch._proposal_buffer == []
    assert skill_store.content == "# Executor\n"
    assert skill_store.writes == []
