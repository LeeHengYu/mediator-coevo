from __future__ import annotations

import json

import pytest

from mediated_coevo.config import Config
from mediated_coevo.models.history_signals import MediatorSignal
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.skill import SkillProposal
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.orchestrator import Orchestrator
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from mediated_coevo.stores.skill_store import SkillStore


class _LLM:
    model = "test-model"

    def drain_token_events(self):
        return []


class _Task:
    def __init__(self, task_id: str) -> None:
        self.instruction = f"base instruction for {task_id}"


class _TaskRepo:
    def resolve(self, task_id: str) -> _Task:
        return _Task(task_id)


class _Planner:
    def __init__(self) -> None:
        self.llm_client = _LLM()
        self.prior_contexts: list[tuple[str, int, str | None]] = []

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
        self.prior_contexts.append((task_id, iteration, prior_context))
        return TaskSpec(
            task_id=task_id,
            instruction=base_instruction,
            iteration=iteration,
        )

    async def suggest_skill_revision(
        self,
        current_skill_content: str,
        feedback: str | None,
        edit_history: list,
        task_id: str = "",
        iteration: int = 0,
    ) -> SkillProposal | None:
        return None


class _Executor:
    def __init__(self) -> None:
        self.rewards = {
            ("task-A", 0): 0.10,
            ("task-B", 0): 0.80,
            ("task-A", 1): 0.25,
            ("task-B", 1): 0.95,
        }

    async def execute_task(
        self,
        task_spec: TaskSpec,
        skill_texts: list[str],
    ) -> ExecutionTrace:
        return ExecutionTrace(
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            reward=self.rewards[(task_spec.task_id, task_spec.iteration)],
            status="ok",
        )


class _Mediator:
    def __init__(self) -> None:
        self.llm_client = _LLM()

    async def mediate_trace(
        self,
        condition: str,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport:
        return MediatorReport(
            task_id=trace.task_id,
            iteration=trace.iteration,
            content=f"report for {trace.task_id} iter {trace.iteration}",
        )

    async def compact_feedback(self, report: MediatorReport) -> MediatorSignal:
        return MediatorSignal(headline=report.content, evidence=report.content)


class _SkillStore:
    def __init__(self) -> None:
        self.snapshots: list[int] = []

    def read_skill(self, skill_name: str) -> str | None:
        if skill_name == "executor":
            return "# Executor\n"
        return None

    def skill_hashes(self) -> dict[str, str]:
        return {"executor": SkillStore.content_hash("# Executor\n")}

    def snapshot(self, iteration: int, snapshot_dir) -> None:
        self.snapshots.append(iteration)


class _Advisor:
    llm_client = _LLM()


@pytest.mark.asyncio
async def test_run_experiment_two_tasks_keeps_feedback_and_metrics_task_scoped(
    tmp_path,
):
    config = Config()
    config.experiment.condition_name = "learned_mediator"
    config.experiment.coevo_interval = 99
    config.experiment.advisor_buffer_max = 99
    planner = _Planner()
    skill_store = _SkillStore()
    artifact_store = ArtifactStore(base_dir=tmp_path / "artifacts")
    history_store = HistoryStore(history_dir=tmp_path / "history")
    orchestrator = Orchestrator(
        planner=planner,  # type: ignore[arg-type]
        executor=_Executor(),  # type: ignore[arg-type]
        mediator=_Mediator(),  # type: ignore[arg-type]
        skill_store=skill_store,  # type: ignore[arg-type]
        artifact_store=artifact_store,
        history_store=history_store,
        benchmark_repo=_TaskRepo(),  # type: ignore[arg-type]
        config=config,
        experiment_dir=tmp_path,
        skill_advisor=_Advisor(),  # type: ignore[arg-type]
    )

    records = await orchestrator.run_experiment(
        ["task-A", "task-B"],
        num_iterations=2,
    )

    assert [(record.task_id, record.iteration) for record in records] == [
        ("task-A", 0),
        ("task-B", 0),
        ("task-A", 1),
        ("task-B", 1),
    ]
    assert planner.prior_contexts == [
        ("task-A", 0, None),
        ("task-B", 0, None),
        ("task-A", 1, "report for task-A iter 0"),
        ("task-B", 1, "report for task-B iter 0"),
    ]

    tagged_iter_zero = {
        entry.metadata["task_id"]: entry.reward
        for entry in history_store._entries
        if entry.iteration == 0
    }
    assert tagged_iter_zero == {
        "task-A": pytest.approx(0.25),
        "task-B": pytest.approx(0.95),
    }
    assert sorted(
        (trace.task_id, trace.iteration)
        for trace in artifact_store.query_traces(recent=10)
    ) == [
        ("task-A", 0),
        ("task-A", 1),
        ("task-B", 0),
        ("task-B", 1),
    ]
    assert skill_store.snapshots == [0, 1]
    assert [record.skill_version for record in records] == [
        "iter_0000",
        "iter_0000",
        "iter_0001",
        "iter_0001",
    ]

    metrics = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert [(row["task_id"], row["iteration"]) for row in metrics] == [
        ("task-A", 0),
        ("task-B", 0),
        ("task-A", 1),
        ("task-B", 1),
    ]
    assert {row["skill_hashes"]["executor"] for row in metrics} == {
        SkillStore.content_hash("# Executor\n")
    }
