from __future__ import annotations

import json

import pytest

from mediated_coevo.core.config import Config
from mediated_coevo.models.history_signals import MediatorSignal
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.skill import (
    AdvisorBatchProvenance,
    SkillProposal,
    SkillUpdate,
    SkillUpdateCandidateRef,
    SkillValidationResult,
    SkillValidationTaskResult,
)
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.experiment.orchestrator import Orchestrator
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from mediated_coevo.stores.skill_store import SkillStore
from tests.config_helpers import budgets_config, diffusion_config, experiment_config


class _LLM:
    model = "test-model"

    def drain_token_events(self):
        return []


class _Task:
    def __init__(self, task_id: str) -> None:
        self.instruction = f"base instruction for {task_id}"
        self.task_config = {
            "metadata": {
                "category": f"category-{task_id}",
                "difficulty": "medium",
            },
            "verifier": {
                "type": "custom_pytest",
            },
        }


class _TaskRepo:
    def resolve(self, task_id: str) -> _Task:
        return _Task(task_id)


class _Planner:
    def __init__(self, metrics_path=None) -> None:
        self.llm_client = _LLM()
        self.prior_contexts: list[tuple[str, int, str | None]] = []
        self.metrics_path = metrics_path
        self.metrics_rows_before_plan: list[
            tuple[str, int, list[tuple[str, int]]]
        ] = []

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
        if self.metrics_path is not None:
            rows = []
            if self.metrics_path.exists():
                rows = [
                    json.loads(line)
                    for line in self.metrics_path.read_text().splitlines()
                ]
            self.metrics_rows_before_plan.append(
                (
                    task_id,
                    iteration,
                    [(row["task_id"], row["iteration"]) for row in rows],
                )
            )
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
async def test_learned_mediator_cross_task_feedback_is_round_causal(tmp_path):
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
    config.experiment.condition_name = "learned_mediator"
    config.experiment.coevo_interval = 99
    config.experiment.advisor_buffer_max = 99
    planner = _Planner()
    artifact_store = ArtifactStore(base_dir=tmp_path / "artifacts")
    orchestrator = Orchestrator(
        planner=planner,  # type: ignore[arg-type]
        executor=_Executor(),  # type: ignore[arg-type]
        mediator=_Mediator(),  # type: ignore[arg-type]
        skill_store=_SkillStore(),  # type: ignore[arg-type]
        artifact_store=artifact_store,
        history_store=HistoryStore(history_dir=tmp_path / "history"),
        benchmark_repo=_TaskRepo(),  # type: ignore[arg-type]
        config=config,
        experiment_dir=tmp_path,
        skill_advisor=_Advisor(),  # type: ignore[arg-type]
    )

    await orchestrator.run_experiment(["task-A", "task-B"], num_iterations=2)

    contexts = {
        (task_id, iteration): prior_context
        for task_id, iteration, prior_context in planner.prior_contexts
    }

    assert contexts[("task-A", 0)] is None
    assert contexts[("task-B", 0)] is None

    task_a_iter_1 = contexts[("task-A", 1)]
    task_b_iter_1 = contexts[("task-B", 1)]

    assert task_a_iter_1 is not None
    assert "report for task-A iter 0" in task_a_iter_1
    assert "source_task=task-B iter=0" in task_a_iter_1
    assert "source_task=task-B iter=1" not in task_a_iter_1

    assert task_b_iter_1 is not None
    assert "report for task-B iter 0" in task_b_iter_1
    assert "source_task=task-A iter=0" in task_b_iter_1
    assert "source_task=task-A iter=1" not in task_b_iter_1


@pytest.mark.asyncio
async def test_run_experiment_two_tasks_keeps_feedback_and_metrics_task_scoped(
    tmp_path,
):
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
    config.experiment.condition_name = "learned_mediator"
    config.experiment.coevo_interval = 99
    config.experiment.advisor_buffer_max = 99
    planner = _Planner(metrics_path=tmp_path / "metrics.jsonl")
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
    contexts = {
        (task_id, iteration): prior_context
        for task_id, iteration, prior_context in planner.prior_contexts
    }
    assert contexts[("task-A", 0)] is None
    assert contexts[("task-B", 0)] is None
    task_a_iter_1 = contexts[("task-A", 1)]
    task_b_iter_1 = contexts[("task-B", 1)]
    assert task_a_iter_1 is not None
    assert "report for task-A iter 0" in task_a_iter_1
    assert "source_task=task-B iter=0" in task_a_iter_1
    assert task_b_iter_1 is not None
    assert "report for task-B iter 0" in task_b_iter_1
    assert "source_task=task-A iter=0" in task_b_iter_1

    tagged_rewards = {
        (entry.metadata["task_id"], entry.iteration): entry.reward
        for entry in history_store._entries
    }
    assert tagged_rewards == {
        ("task-A", 0): pytest.approx(0.10),
        ("task-B", 0): pytest.approx(0.80),
        ("task-A", 1): pytest.approx(0.25),
        ("task-B", 1): pytest.approx(0.95),
    }
    assert all(
        entry.metadata["outcome_iteration"] == entry.iteration
        for entry in history_store._entries
    )
    assert sorted(
        (trace.task_id, trace.iteration)
        for trace in artifact_store.query_traces(recent=10)
    ) == [
        ("task-A", 0),
        ("task-A", 1),
        ("task-B", 0),
        ("task-B", 1),
    ]
    assert skill_store.snapshots == [0, 0, 1, 1]
    assert planner.metrics_rows_before_plan == [
        ("task-A", 0, []),
        ("task-B", 0, [("task-A", 0)]),
        ("task-A", 1, [("task-A", 0), ("task-B", 0)]),
        ("task-B", 1, [("task-A", 0), ("task-B", 0), ("task-A", 1)]),
    ]
    assert [record.skill_version for record in records] == [
        "iter_0000",
        "iter_0000",
        "iter_0001",
        "iter_0001",
    ]
    assert [
        (
            record.task_id,
            record.task_category,
            record.task_difficulty,
            record.expected_reward_range,
            record.verifier_type,
        )
        for record in records
    ] == [
        ("task-A", "category-task-A", "medium", (0.0, 1.0), "custom_pytest"),
        ("task-B", "category-task-B", "medium", (0.0, 1.0), "custom_pytest"),
        ("task-A", "category-task-A", "medium", (0.0, 1.0), "custom_pytest"),
        ("task-B", "category-task-B", "medium", (0.0, 1.0), "custom_pytest"),
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
    assert [
        (
            row["task_id"],
            row["task_category"],
            row["task_difficulty"],
            row["expected_reward_range"],
            row["verifier_type"],
        )
        for row in metrics
    ] == [
        ("task-A", "category-task-A", "medium", [0.0, 1.0], "custom_pytest"),
        ("task-B", "category-task-B", "medium", [0.0, 1.0], "custom_pytest"),
        ("task-A", "category-task-A", "medium", [0.0, 1.0], "custom_pytest"),
        ("task-B", "category-task-B", "medium", [0.0, 1.0], "custom_pytest"),
    ]


def test_snapshot_and_write_metrics_records_skill_update_ledger(tmp_path):
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
    artifact_store = ArtifactStore(base_dir=tmp_path / "artifacts")
    skill_store = _SkillStore()
    orchestrator = Orchestrator(
        planner=_Planner(),  # type: ignore[arg-type]
        executor=_Executor(),  # type: ignore[arg-type]
        mediator=_Mediator(),  # type: ignore[arg-type]
        skill_store=skill_store,  # type: ignore[arg-type]
        artifact_store=artifact_store,
        history_store=HistoryStore(history_dir=tmp_path / "history"),
        benchmark_repo=_TaskRepo(),  # type: ignore[arg-type]
        config=config,
        experiment_dir=tmp_path,
        skill_advisor=_Advisor(),  # type: ignore[arg-type]
    )
    update = SkillUpdate(
        skill_id="executor",
        task_id="task-A",
        iteration=1,
        old_content="# Executor\n\nold\n",
        new_content="# Executor\n\nnew\n",
        old_skill_hash="old-hash",
        new_skill_hash="new-hash",
        provenance=AdvisorBatchProvenance(
            batch_id="batch-1",
            iteration=1,
            skill_id="executor",
            task_ids=["task-A"],
            base_skill_hash="old-hash",
            decision="approved",
            reason="validated improvement",
            selected_candidate_id="candidate-1",
            selected_update_kind="add_guardrail",
            candidate_refs=[
                SkillUpdateCandidateRef(
                    candidate_id="candidate-1",
                    update_kind="add_guardrail",
                    selected=True,
                )
            ],
            validation=SkillValidationResult(
                validation_id="validation-1",
                task_ids=["task-B"],
                decision="accepted",
                reason="mean_delta_met",
                current_mean_reward=0.2,
                candidate_mean_reward=0.5,
                mean_delta=0.3,
                task_results=[
                    SkillValidationTaskResult(
                        task_id="task-B",
                        current_status="ok",
                        candidate_status="ok",
                    )
                ],
            ),
        ),
    )
    record = IterationRecord(
        iteration=1,
        task_id="task-A",
        reward=0.5,
        delta_reward=0.25,
        success=True,
        verifier_status="ok",
        skill_update=update,
    )

    orchestrator._snapshot_and_write_metrics(1, [record])

    history_path = tmp_path / "artifacts" / "skill_updates" / "skill_update_history.jsonl"
    history_rows = [json.loads(line) for line in history_path.read_text().splitlines()]
    ledger_entry = history_rows[0]
    assert ledger_entry["update_id"] == "iter0001-task-a-executor-candidate-1"
    assert ledger_entry["skill_version"] == "iter_0001"
    assert ledger_entry["selected_candidate_id"] == "candidate-1"
    assert ledger_entry["validation_decision"] == "accepted"
    assert ledger_entry["validation_mean_delta"] == pytest.approx(0.3)
    assert ledger_entry["reward"] == pytest.approx(0.5)
    assert ledger_entry["delta_reward"] == pytest.approx(0.25)

    update_path = tmp_path / ledger_entry["artifact_path"]
    diff_path = tmp_path / ledger_entry["diff_path"]
    assert update_path.exists()
    assert diff_path.exists()
    assert "+new" in diff_path.read_text()


@pytest.mark.asyncio
async def test_full_traces_cross_task_feedback_is_round_causal(tmp_path):
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
    config.experiment.condition_name = "full_traces"
    config.experiment.coevo_interval = 99
    config.experiment.advisor_buffer_max = 99
    planner = _Planner()
    artifact_store = ArtifactStore(base_dir=tmp_path / "artifacts")
    orchestrator = Orchestrator(
        planner=planner,  # type: ignore[arg-type]
        executor=_Executor(),  # type: ignore[arg-type]
        mediator=_Mediator(),  # type: ignore[arg-type]
        skill_store=_SkillStore(),  # type: ignore[arg-type]
        artifact_store=artifact_store,
        history_store=HistoryStore(history_dir=tmp_path / "history"),
        benchmark_repo=_TaskRepo(),  # type: ignore[arg-type]
        config=config,
        experiment_dir=tmp_path,
        skill_advisor=_Advisor(),  # type: ignore[arg-type]
    )

    await orchestrator.run_experiment(["task-A", "task-B"], num_iterations=2)

    contexts = {
        (task_id, iteration): prior_context
        for task_id, iteration, prior_context in planner.prior_contexts
    }
    task_a_iter_1 = contexts[("task-A", 1)]
    task_b_iter_1 = contexts[("task-B", 1)]

    assert task_a_iter_1 is not None
    assert "source_task=task-B iter=0" in task_a_iter_1
    assert "source_task=task-B iter=1" not in task_a_iter_1

    assert task_b_iter_1 is not None
    assert "source_task=task-A iter=0" in task_b_iter_1
    assert "source_task=task-A iter=1" not in task_b_iter_1
