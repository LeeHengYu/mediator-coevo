from __future__ import annotations

from typing import Any

import pytest

from mediated_coevo.core.config import Config
from mediated_coevo.evolution.candidates import build_candidate_batch
from mediated_coevo.evolution.reflector import (
    ReflectionDraft,
    ReflectionResult,
    Reflector,
)
from mediated_coevo.experiment.orchestrator import Orchestrator
from mediated_coevo.models.history_signals import MediatorSignal
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.skill import SkillUpdateCandidate
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryEntry, HistoryStore
from mediated_coevo.stores.skill_store import SkillStore
from tests.config_helpers import budgets_config, diffusion_config, experiment_config


class _LLM:
    model = "test-model"

    def drain_token_events(self) -> list[Any]:
        return []


class _Task:
    instruction = "validate mediator behavior"
    task_config = {
        "metadata": {"category": "validation", "difficulty": "easy"},
        "verifier": {"type": "custom_pytest"},
    }


class _TaskRepo:
    def resolve(self, task_id: str) -> _Task:
        return _Task()


class _Planner:
    llm_client = _LLM()

    def __init__(self) -> None:
        self.feedback_seen: list[str | None] = []

    def set_skill_context(
        self,
        executor_skills: str,
        skill_refiner: str | None = None,
    ) -> None:
        pass

    async def plan_task(
        self,
        task_id: str,
        base_instruction: str,
        prior_context: str | None = None,
        current_skills: list[str] | None = None,
        iteration: int = 0,
    ) -> TaskSpec:
        return TaskSpec(
            task_id=task_id,
            instruction=base_instruction,
            iteration=iteration,
        )

    async def suggest_skill_revision_batch(
        self,
        current_skill_content: str,
        feedback: str | None,
        edit_history: list[Any],
        rejected_update_history: list[dict[str, Any]] | None = None,
        *,
        skill_id: str,
        task_ids: list[str],
        iteration: int = 0,
    ) -> list[SkillUpdateCandidate]:
        self.feedback_seen.append(feedback)
        score = 0.9 if feedback and "better" in feedback else 0.2
        return [
            SkillUpdateCandidate(
                candidate_id=f"executor-{len(self.feedback_seen)}",
                skill_id=skill_id,
                update_kind="validation_feedback",
                hypothesis="candidate from mediator feedback",
                old_content=current_skill_content,
                new_content=f"# Executor\n\nscore={score}\n",
                reasoning=f"feedback score {score}",
                audit_score=score,
            )
        ]


class _Executor:
    def __init__(self) -> None:
        self.execution_count = 0

    async def execute_task(
        self,
        task_spec: TaskSpec,
        skill_texts: list[str],
    ) -> ExecutionTrace:
        self.execution_count += 1
        skill_text = skill_texts[0] if skill_texts else ""
        reward = 0.9 if "score=0.9" in skill_text else 0.2
        return ExecutionTrace(
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            reward=reward,
            status="ok",
        )


class _Mediator:
    llm_client = _LLM()

    def __init__(self) -> None:
        self.protocol = "# Old Mediator\n"
        self.loaded_protocols: list[str] = []

    def load_protocol(self, content: str) -> None:
        self.protocol = content
        self.loaded_protocols.append(content)

    async def mediate_trace(
        self,
        condition: str,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport:
        content = (
            "better feedback"
            if "Better Mediator" in self.protocol
            else "weak feedback"
        )
        return MediatorReport(
            task_id=trace.task_id,
            iteration=trace.iteration,
            content=content,
            withheld=False,
        )


class _SkillStore:
    def __init__(self) -> None:
        self.skills = {
            "executor": "# Executor\n",
            "mediator": "# Old Mediator\n",
        }
        self.writes: list[tuple[str, str]] = []

    def read_skill(self, skill_name: str) -> str | None:
        return self.skills.get(skill_name)

    def write_skill(self, skill_name: str, content: str) -> None:
        self.skills[skill_name] = content
        self.writes.append((skill_name, content))

    def skill_hashes(self) -> dict[str, str]:
        return {
            skill_name: SkillStore.content_hash(content)
            for skill_name, content in self.skills.items()
        }


class _Advisor:
    llm_client = _LLM()


class _Reflector:
    def __init__(self, draft: ReflectionDraft, skill_store: _SkillStore) -> None:
        self.draft = draft
        self.skill_store = skill_store

    async def draft_reflection(self, **kwargs: Any) -> ReflectionDraft:
        return self.draft

    def commit_draft(
        self,
        draft: ReflectionDraft,
        *,
        selected_candidate_id: str | None = None,
        validation=None,
    ):
        if selected_candidate_id is not None:
            for candidate in draft.candidate_batch.candidates:
                candidate.selected = candidate.candidate_id == selected_candidate_id
            draft.candidate_batch.selected_candidate_id = selected_candidate_id
        selected = next(
            candidate
            for candidate in draft.candidate_batch.candidates
            if candidate.selected
        )
        provenance = Reflector._build_reflection_provenance(
            agent_role=draft.agent_role,
            base_skill_hash=draft.base_skill_hash,
            pairs=draft.pairs,
            candidate_batch=draft.candidate_batch,
            candidate_batch_path=draft.candidate_batch_path,
            iteration=draft.iteration,
            max_pairs=draft.max_pairs,
            selection_seed=draft.selection_seed,
            validation=validation,
        )
        self.skill_store.write_skill(draft.agent_role, selected.new_content)
        return ReflectionResult(
            skill_id=draft.agent_role,
            old_content=draft.old_content,
            new_content=selected.new_content,
            provenance=provenance,
        )


def _orchestrator(tmp_path):
    config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
            "judge": "test-judge",
        },
        budgets=budgets_config(),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )
    config.experiment.skill_validation.tasks = ["task-A"]
    config.experiment.skill_validation.sample_size = 1
    config.experiment.skill_validation.allow_contributing_fallback = True
    config.experiment.skill_validation.min_mean_delta = 0.01
    planner = _Planner()
    executor = _Executor()
    mediator = _Mediator()
    skill_store = _SkillStore()
    history_store = HistoryStore(tmp_path / "history")
    history_store.add(
        HistoryEntry(
            iteration=0,
            agent_role="mediator",
            payload=MediatorSignal(headline="weak prior feedback"),
            reward=0.2,
            metadata={"task_id": "task-A"},
        )
    )
    history_store.add(
        HistoryEntry(
            iteration=1,
            agent_role="mediator",
            payload=MediatorSignal(headline="strong prior feedback"),
            reward=0.9,
            metadata={"task_id": "task-A"},
        )
    )
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    orchestrator = Orchestrator(
        planner=planner,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        mediator=mediator,  # type: ignore[arg-type]
        skill_store=skill_store,  # type: ignore[arg-type]
        artifact_store=artifact_store,
        history_store=history_store,
        benchmark_repo=_TaskRepo(),  # type: ignore[arg-type]
        config=config,
        experiment_dir=tmp_path,
        skill_advisor=_Advisor(),  # type: ignore[arg-type]
    )
    return orchestrator, planner, executor, mediator, skill_store


def _draft() -> ReflectionDraft:
    current_skill = "# Old Mediator\n"
    candidates = [
        SkillUpdateCandidate(
            candidate_id="weak",
            skill_id="mediator",
            update_kind="weak_reporting",
            old_content=current_skill,
            new_content="# Weak Mediator\n",
            reasoning="weak",
            audit_score=0.2,
        ),
        SkillUpdateCandidate(
            candidate_id="better",
            skill_id="mediator",
            update_kind="better_reporting",
            old_content=current_skill,
            new_content="# Better Mediator\n",
            reasoning="better",
            audit_score=0.9,
        ),
    ]
    batch = build_candidate_batch(
        batch_id="reflect-mediator-iter-0004",
        iteration=4,
        skill_id="mediator",
        agent_role="mediator",
        task_ids=["task-A"],
        current_skill=current_skill,
        candidates=candidates,
        selection_seed=123,
        select_candidate=False,
    )
    return ReflectionDraft(
        agent_role="mediator",
        old_content=current_skill,
        base_skill_hash=SkillStore.content_hash(current_skill),
        pairs=[],
        candidate_batch=batch,
        candidate_batch_path=None,
        iteration=4,
        max_pairs=5,
        selection_seed=123,
    )


@pytest.mark.asyncio
async def test_mediator_reflection_commits_best_validated_candidate(tmp_path):
    orchestrator, planner, executor, mediator, skill_store = _orchestrator(tmp_path)

    update = await orchestrator._reflect_and_validate_mediator(
        _Reflector(_draft(), skill_store),  # type: ignore[arg-type]
        iteration=4,
        selection_seed=123,
    )

    assert update is not None
    assert update.skill_id == "mediator"
    assert update.new_content == "# Better Mediator"
    assert update.provenance is not None
    assert update.provenance.selected_candidate_id == "better"
    assert update.provenance.validation is not None
    assert update.provenance.validation.decision == "accepted"
    assert skill_store.writes == [("mediator", "# Better Mediator")]
    assert mediator.protocol == "# Better Mediator"
    assert executor.execution_count == 7
    assert planner.feedback_seen == ["weak feedback", "weak feedback", "better feedback"]
    source_traces = list(
        (tmp_path / "artifacts" / "validation").glob("*/source/*.json")
    )
    assert len(source_traces) == 1


@pytest.mark.asyncio
async def test_mediator_validation_rejects_regressing_candidate(tmp_path):
    orchestrator, _planner, _executor, mediator, skill_store = _orchestrator(tmp_path)
    draft = _draft()
    draft.candidate_batch.candidates = [
        candidate
        for candidate in draft.candidate_batch.candidates
        if candidate.candidate_id == "weak"
    ]

    update = await orchestrator._reflect_and_validate_mediator(
        _Reflector(draft, skill_store),  # type: ignore[arg-type]
        iteration=4,
        selection_seed=123,
    )

    assert update is None
    assert skill_store.writes == []
    assert mediator.protocol == "# Old Mediator\n"
