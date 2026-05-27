from __future__ import annotations

from typing import Any

import pytest

from mediated_coevo.core.config import Config, ModelsConfig
from mediated_coevo.evolution.executor_skill_gate import ExecutorSkillGate
from mediated_coevo.evolution.skill_advisor import SkillAdvisor
from mediated_coevo.models.skill import (
    RejectedProposalBatch,
    SkillProposal,
    SkillValidationResult,
)
from mediated_coevo.experiment.orchestrator import Orchestrator
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from mediated_coevo.stores.skill_store import SkillStore
from tests.config_helpers import diffusion_config, experiment_config


def _models_config() -> ModelsConfig:
    return ModelsConfig(
        planner="test-planner",
        executor="test-executor",
        mediator="test-mediator",
        judge="test-judge",
    )


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


class _RecordingBatchPlanner:
    def __init__(self) -> None:
        self.rejected_update_history: list[dict[str, Any]] | None = None

    async def suggest_skill_revision_batch(self, *args, **kwargs):
        self.rejected_update_history = kwargs.get("rejected_update_history")
        return []


class _MissingTaskRepo:
    def resolve(self, task_id: str) -> None:
        raise FileNotFoundError(task_id)


_ADVISOR_REJECTION_REASON = "Proposal rewards conflict with the suggested edit."


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
    orch: Any = Orchestrator.__new__(Orchestrator)
    orch.config = Config(
        models=ModelsConfig(
            planner="test-planner",
            executor="test-executor",
            mediator="test-mediator",
            judge="test-judge",
        ),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )
    orch.config.experiment.advisor_buffer_max = 2
    orch.skill_store = skill_store
    orch.history_store = HistoryStore(history_dir=tmp_path / "history")
    orch.artifact_store = ArtifactStore(base_dir=tmp_path / "artifacts")
    orch.executor = object()
    orch.benchmark_repo = object()
    orch.skill_advisor = advisor
    orch.planner = _NoCallPlanner()
    orch.executor_skill_gate = ExecutorSkillGate(
        config=orch.config,
        skill_store=orch.skill_store,
        history_store=orch.history_store,
        planner=orch.planner,
        skill_advisor=orch.skill_advisor,
        executor=orch.executor,
        benchmark_repo=orch.benchmark_repo,
        artifact_store=orch.artifact_store,
    )
    orch._proposal_buffer = [
        _proposal("task-A", 0.2),
        _proposal("task-B", 0.8),
    ]
    return orch, skill_store


@pytest.mark.asyncio
async def test_advisor_rejection_clears_buffer_without_skill_update(tmp_path):
    advisor = SkillAdvisor(
        _LLM(content=f'{{"approve": false, "feedback": "{_ADVISOR_REJECTION_REASON}"}}')  # type: ignore[arg-type]
    )
    orch, skill_store = _orchestrator(tmp_path, advisor)
    proposal_ids = [
        proposal.proposal_id
        for proposal in orch._proposal_buffer
    ]

    gate = orch.executor_skill_gate
    update = await gate.review_and_patch(
        iteration=3,
        proposal_buffer=orch._proposal_buffer,
    )

    assert update is None
    assert orch._proposal_buffer == []
    assert skill_store.content == "# Executor\n"
    assert skill_store.writes == []
    assert gate.last_advisor_decision == "rejected"
    assert gate.last_advisor_reason == _ADVISOR_REJECTION_REASON
    assert gate.last_proposal_ids == proposal_ids

    rejections = orch.history_store.query_rejected_proposals()
    assert len(rejections) == 1
    rejection = rejections[0]
    assert gate.last_rejection_id == rejection.rejection_id
    assert rejection.batch_id.startswith("coevo-iter-0003-batch-")
    assert rejection.iteration == 3
    assert rejection.skill_id == "executor"
    assert rejection.task_ids == ["task-A", "task-B"]
    assert rejection.base_skill_hash == SkillStore.content_hash("# Executor\n")
    assert rejection.reason == _ADVISOR_REJECTION_REASON
    assert rejection.validation is None
    assert len(rejection.proposals) == 2


@pytest.mark.asyncio
async def test_advisor_rejection_without_reason_is_recorded_as_protocol_violation(
    tmp_path,
):
    advisor = SkillAdvisor(
        _LLM(content='{"approve": false, "feedback": ""}')  # type: ignore[arg-type]
    )
    orch, skill_store = _orchestrator(tmp_path, advisor)

    gate = orch.executor_skill_gate
    update = await gate.review_and_patch(
        iteration=3,
        proposal_buffer=orch._proposal_buffer,
    )

    assert update is None
    assert orch._proposal_buffer == []
    assert skill_store.content == "# Executor\n"
    assert skill_store.writes == []
    rejections = orch.history_store.query_rejected_proposals()
    assert len(rejections) == 1
    assert rejections[0].reason == "advisor_rejected_without_reason"
    assert gate.last_advisor_decision == "rejected"
    assert gate.last_advisor_reason == "advisor_rejected_without_reason"
    assert gate.last_rejection_id == rejections[0].rejection_id


@pytest.mark.asyncio
async def test_advisor_llm_failure_clears_buffer_without_skill_update(tmp_path):
    advisor = SkillAdvisor(
        _LLM(exc=RuntimeError("advisor unavailable"))  # type: ignore[arg-type]
    )
    orch, skill_store = _orchestrator(tmp_path, advisor)

    gate = orch.executor_skill_gate
    update = await gate.review_and_patch(
        iteration=3,
        proposal_buffer=orch._proposal_buffer,
    )

    assert update is None
    assert orch._proposal_buffer == []
    assert skill_store.content == "# Executor\n"
    assert skill_store.writes == []
    rejections = orch.history_store.query_rejected_proposals()
    assert len(rejections) == 1
    assert rejections[0].reason == "advisor_error: advisor unavailable"
    assert gate.last_advisor_decision == "rejected"
    assert gate.last_advisor_reason == "advisor_error: advisor unavailable"
    assert gate.last_rejection_id == rejections[0].rejection_id


@pytest.mark.asyncio
async def test_approved_batch_forwards_rejected_validation_history_to_planner(
    tmp_path,
):
    advisor = SkillAdvisor(
        _LLM(content='{"approve": true, "feedback": "try a narrower edit"}')  # type: ignore[arg-type]
    )
    orch, _skill_store = _orchestrator(tmp_path, advisor)
    planner = _RecordingBatchPlanner()
    orch.planner = planner
    orch.executor_skill_gate.planner = planner  # type: ignore[assignment]
    orch.executor_skill_gate.benchmark_repo = _MissingTaskRepo()  # type: ignore[assignment]
    orch.history_store.record_rejected_proposals(
        RejectedProposalBatch(
            batch_id="old-batch",
            iteration=1,
            skill_id="executor",
            task_ids=["task-A"],
            base_skill_hash=SkillStore.content_hash("# Executor\n"),
            reason="validation: task_regression",
            advisor_feedback="avoid broad parsing rewrite",
            validation=SkillValidationResult(
                validation_id="validation-old",
                task_ids=["heldout-A"],
                decision="rejected",
                reason="task_regression",
                current_mean_reward=0.8,
                candidate_mean_reward=0.7,
                mean_delta=-0.1,
            ),
        )
    )

    update = await orch.executor_skill_gate.review_and_patch(
        iteration=3,
        proposal_buffer=orch._proposal_buffer,
    )

    assert update is None
    rejected_history = planner.rejected_update_history
    assert rejected_history is not None
    rejected_summary = rejected_history[0]
    validation_summary = rejected_summary["validation"]
    assert isinstance(validation_summary, dict)
    assert rejected_summary["batch_id"] == "old-batch"
    assert rejected_summary["reason"] == "validation: task_regression"
    assert validation_summary["reason"] == "task_regression"
