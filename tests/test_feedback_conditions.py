from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer
from pydantic import ValidationError

from mediated_coevo.core.config import Config, ModelsConfig
from mediated_coevo.diffusion import (
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
)
from mediated_coevo.experiment.conditions import get_executor_proposal_feedback
from mediated_coevo.main import _validate_condition_name
from mediated_coevo.models.history_signals import MediatorSignal
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.evolution.executor_skill_gate import ExecutorSkillGate
from mediated_coevo.experiment.orchestrator import Orchestrator
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from tests.config_helpers import diffusion_config, experiment_config


class _Task:
    instruction = "base instruction"
    task_config: dict = {}


def _write_graph_artifacts(graph_dir: Path, task_ids: list[str]) -> None:
    graph_dir.mkdir(parents=True, exist_ok=True)
    profiles = {
        task_id: {
            "task_id": task_id,
            "category": "build",
            "difficulty": "easy",
            "tags": ["python", task_id],
            "skills": [],
            "environment_files": [],
            "output_types": ["patch"],
            "domain_terms": ["python", task_id],
            "capability_labels": ["build-debugging"],
        }
        for task_id in task_ids
    }
    pairs = []
    for source_task_id in task_ids:
        for target_task_id in task_ids:
            if source_task_id == target_task_id:
                continue
            pairs.append(
                {
                    "source": source_task_id,
                    "target": target_task_id,
                    "score": 0.6,
                    "components": {"category": 1.0, "tags": 0.5},
                    "shared": {"tags": ["python"]},
                    "kept_after_p20_cut": True,
                    "kept_after_threshold_cut": True,
                }
            )
    (graph_dir / "task_profiles.json").write_text(
        json.dumps({"task_count": len(task_ids), "profiles": profiles})
    )
    (graph_dir / "pairwise_similarity.json").write_text(
        json.dumps(
            {
                "pair_count": len(pairs),
                "p20_threshold": 0.01,
                "edge_score_threshold": 0.05,
                "active_threshold": 0.05,
                "threshold_kind": "absolute_score",
                "pairs": pairs,
            }
        )
    )


def _store_diffusion_artifact(
    orch: Any,
    *,
    artifact_id: str,
    source_task_id: str,
    source_iteration: int = 0,
    content: str | None = None,
) -> None:
    orch._diffusion_store.store_artifact(
        DiffusionArtifact(
            artifact_id=artifact_id,
            source_task_id=source_task_id,
            source_iteration=source_iteration,
            artifact_type=DiffusionArtifactType.DEBUG_HINT,
            risk_level=DiffusionRiskLevel.LOW,
            content=content or artifact_id,
        )
    )


def _models_config() -> ModelsConfig:
    return ModelsConfig(
        planner="test-planner",
        executor="test-executor",
        mediator="test-mediator",
        judge="test-judge",
    )


class _TaskRepo:
    def resolve(self, task_id: str):
        return _Task()


class _SkillStore:
    def read_skill(self, skill_name: str) -> str | None:
        if skill_name == "executor":
            return "# Executor\n"
        return None

    def skill_hashes(self) -> dict[str, str]:
        return {}


class _PlannerLLM:
    model = "test-model"

    def drain_token_events(self):
        return []


class _Planner:
    def __init__(self) -> None:
        self.prior_contexts: dict[str, str | None] = {}
        self.proposal_feedback: list[str] = []
        self.llm_client = _PlannerLLM()

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
        self.prior_contexts[task_id] = prior_context
        return TaskSpec(task_id=task_id, instruction=base_instruction, iteration=iteration)

    async def suggest_skill_revision(
        self,
        *,
        current_skill_content: str,
        feedback: str,
        edit_history: list,
        task_id: str,
        iteration: int,
    ):
        self.proposal_feedback.append(feedback)
        return None


class _Executor:
    async def execute_task(
        self,
        task_spec: TaskSpec,
        skill_texts: list[str],
    ) -> ExecutionTrace:
        return ExecutionTrace(
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            reward=0.5,
            status="ok",
        )


class _Mediator:
    def __init__(self, llm_client=None) -> None:
        self.process_calls = 0
        self.compact_calls = 0
        self.llm_client = llm_client or _PlannerLLM()

    async def process_trace(
        self,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport:
        self.process_calls += 1
        return MediatorReport(
            task_id=trace.task_id,
            iteration=trace.iteration,
            content=f"fresh report for {trace.task_id}",
        )

    async def mediate_trace(
        self,
        condition: str,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport | None:
        if condition not in {"static_mediator", "learned_mediator"}:
            return None
        if not trace.is_usable_feedback_signal:
            return None
        return await self.process_trace(trace, task_context)

    async def compact_feedback(
        self,
        report: MediatorReport,
    ) -> MediatorSignal:
        self.compact_calls += 1
        return MediatorSignal(headline=report.exposed_content or "")


class _TraceHistoryInspectingMediator:
    llm_client = _PlannerLLM()

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self.artifact_store = artifact_store
        self.trace_iterations_seen: list[int] = []

    async def mediate_trace(
        self,
        condition: str,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport | None:
        self.trace_iterations_seen = [
            item.iteration
            for item in self.artifact_store.query_traces(
                task_id=trace.task_id,
                before_iteration=trace.iteration,
            )
        ]
        return None


class _WithholdingMediator:
    llm_client = _PlannerLLM()

    async def mediate_trace(
        self,
        condition: str,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport:
        return MediatorReport(
            task_id=trace.task_id,
            iteration=trace.iteration,
            content="withheld content",
            withheld=True,
            reasoning="not useful for planner",
        )

    async def compact_feedback(
        self,
        report: MediatorReport,
    ) -> MediatorSignal:
        return MediatorSignal(
            withheld=report.withheld,
            mediator_reasoning=report.reasoning,
        )


class _FailingMediator:
    llm_client = _PlannerLLM()

    async def mediate_trace(
        self,
        condition: str,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport | None:
        raise RuntimeError("mediator failed")


class _LLMCompactor:
    model = "test-model"

    def __init__(self, *, content: str, raise_exc: Exception | None = None) -> None:
        self.content = content
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_exc:
            raise self.raise_exc
        return {
            "content": self.content,
            "input_tokens": 1,
            "output_tokens": 1,
            "model": "test",
            "raw": {},
        }

    def drain_token_events(self):
        return []


class _Advisor:
    llm_client = _PlannerLLM()


def _orchestrator(
    tmp_path: Path,
    condition: str,
    *,
    llm_client=None,
) -> tuple[Orchestrator, _Planner, _Mediator]:
    planner = _Planner()
    mediator = _Mediator(llm_client=llm_client)
    orch: Any = Orchestrator.__new__(Orchestrator)
    orch.planner = planner
    orch.executor = _Executor()
    orch.mediator = mediator
    orch.skill_store = _SkillStore()
    orch.artifact_store = ArtifactStore(base_dir=tmp_path / "artifacts")
    orch.history_store = HistoryStore(history_dir=tmp_path / "history")
    orch.benchmark_repo = _TaskRepo()
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
    orch.config.experiment.condition_name = condition
    orch.config.experiment.shared_notes = "shared note"
    orch.experiment_dir = tmp_path
    orch.skill_advisor = _Advisor()
    orch._proposal_buffer = []
    orch._previous_report_by_task = {}
    orch._released_cross_task_reports_by_task = {}
    orch._staged_cross_task_reports_by_task = {}
    orch._previous_reward_by_task = {}
    orch._diffusion_context_by_target = {}
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
    return orch, planner, mediator


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["no_feedback", "shared_notes"])
async def test_non_proposal_conditions_return_no_executor_proposal_feedback(
    tmp_path,
    condition,
):
    store = ArtifactStore(base_dir=tmp_path / "artifacts")
    store.store_trace(ExecutionTrace(task_id="task-A", iteration=1, reward=0.5, status="ok"))
    report = MediatorReport(task_id="task-A", iteration=1, content="mediator report")

    feedback = await get_executor_proposal_feedback(
        condition=condition,
        task_id="task-A",
        artifact_store=store,
        mediator_report=report,
        model="test-model",
    )

    assert feedback is None


@pytest.mark.asyncio
async def test_full_traces_returns_current_usable_trace_summary_for_proposal(
    tmp_path,
):
    store = ArtifactStore(base_dir=tmp_path / "artifacts")
    store.store_trace(ExecutionTrace(task_id="task-A", iteration=2, reward=0.2, status="ok"))
    store.store_trace(ExecutionTrace(task_id="task-A", iteration=3, reward=0.5, status="ok"))

    feedback = await get_executor_proposal_feedback(
        condition="full_traces",
        task_id="task-A",
        artifact_store=store,
        mediator_report=None,
        model="test-model",
    )

    assert feedback is not None
    assert "task-A" in feedback
    assert "iter=3" in feedback
    assert "reward=0.50" in feedback


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "trace",
    [
        ExecutionTrace(task_id="task-A", iteration=3, reward=None, status="env_failure"),
        ExecutionTrace(task_id="task-A", iteration=3, reward=None, status="ok"),
        ExecutionTrace(task_id="task-A", iteration=3, reward=0.4, status="harbor_failed"),
    ],
)
async def test_full_traces_returns_no_proposal_feedback_for_unusable_trace(
    tmp_path,
    trace,
):
    store = ArtifactStore(base_dir=tmp_path / "artifacts")
    store.store_trace(trace)

    feedback = await get_executor_proposal_feedback(
        condition="full_traces",
        task_id="task-A",
        artifact_store=store,
        mediator_report=None,
        model="test-model",
    )

    assert feedback is None


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["static_mediator", "learned_mediator"])
async def test_mediator_conditions_return_exposed_report_content_for_proposal(
    tmp_path,
    condition,
):
    store = ArtifactStore(base_dir=tmp_path / "artifacts")
    report = MediatorReport(task_id="task-A", iteration=1, content="use this insight")

    feedback = await get_executor_proposal_feedback(
        condition=condition,
        task_id="task-A",
        artifact_store=store,
        mediator_report=report,
        model="test-model",
    )

    assert feedback == "use this insight"


@pytest.mark.asyncio
async def test_mediator_conditions_return_no_proposal_feedback_for_withheld_report(
    tmp_path,
):
    store = ArtifactStore(base_dir=tmp_path / "artifacts")
    report = MediatorReport(
        task_id="task-A",
        iteration=1,
        content="hidden insight",
        withheld=True,
    )

    feedback = await get_executor_proposal_feedback(
        condition="learned_mediator",
        task_id="task-A",
        artifact_store=store,
        mediator_report=report,
        model="test-model",
    )

    assert feedback is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("condition", "expected_context", "expected_mediator_calls"),
    [
        ("no_feedback", None, 0),
        ("full_traces", "iter=0 reward=0.25 OK", 0),
        ("shared_notes", "shared note", 0),
        ("static_mediator", "prior same-task report", 1),
        ("learned_mediator", "prior same-task report", 1),
    ],
)
async def test_feedback_conditions_control_planner_context_and_mediator_calls(
    tmp_path,
    condition,
    expected_context,
    expected_mediator_calls,
):
    orch, planner, mediator = _orchestrator(tmp_path, condition)
    if condition == "full_traces":
        orch.artifact_store.store_trace(
            ExecutionTrace(
                task_id="task-A",
                iteration=0,
                reward=0.25,
                status="ok",
            )
        )
    if condition in {"static_mediator", "learned_mediator"}:
        orch._previous_report_by_task["task-A"] = MediatorReport(
            task_id="task-A",
            iteration=0,
            content="prior same-task report",
        )

    record = await orch._run_iteration("task-A", 1)

    assert planner.prior_contexts["task-A"] == expected_context
    assert mediator.process_calls == expected_mediator_calls
    assert record.condition_name == condition
    assert record.cross_task_feedback_enabled is False
    assert record.diffusion_policy == "none"
    assert record.execution_trace is not None
    assert record.execution_trace.iteration == 1


@pytest.mark.asyncio
async def test_run_iteration_full_traces_feeds_executor_proposal_from_current_trace(
    tmp_path,
):
    orch, planner, _ = _orchestrator(tmp_path, "full_traces")
    orch.config.experiment.skill_updates.executor = True

    await orch._run_iteration("task-A", 1)

    assert len(planner.proposal_feedback) == 1
    assert "task-A" in planner.proposal_feedback[0]
    assert "iter=1" in planner.proposal_feedback[0]
    assert "reward=0.50" in planner.proposal_feedback[0]


@pytest.mark.asyncio
async def test_executor_proposal_skip_log_uses_proposal_feedback_wording(
    tmp_path,
    caplog,
):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.config.experiment.skill_updates.executor = True
    caplog.set_level("INFO")

    await orch._ask_planner_for_skill_proposal(
        task_id="task-A",
        iteration=0,
        executor_skill="# Executor\n",
        feedback=None,
    )

    assert "no proposal feedback" in caplog.text
    assert "no mediator feedback" not in caplog.text


@pytest.mark.asyncio
async def test_mediator_history_excludes_current_trace(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.artifact_store.store_trace(
        ExecutionTrace(task_id="task-A", iteration=0, reward=0.25, status="ok")
    )
    mediator = _TraceHistoryInspectingMediator(orch.artifact_store)
    orch.mediator = mediator

    await orch._run_iteration("task-A", 1)

    assert mediator.trace_iterations_seen == [0]
    assert orch.artifact_store.load_trace("task-A", 1) is not None


@pytest.mark.asyncio
async def test_trace_is_stored_when_mediator_fails(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.mediator = _FailingMediator()

    with pytest.raises(RuntimeError, match="mediator failed"):
        await orch._run_iteration("task-A", 1)

    assert orch.artifact_store.load_trace("task-A", 1) is not None


@pytest.mark.asyncio
async def test_withheld_mediator_report_is_recorded_for_reflection(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.mediator = _WithholdingMediator()

    record = await orch._run_iteration("task-A", 1)

    assert record.mediator_report is None
    assert record.mediator_history_entry_id is not None
    assert "task-A" not in orch._previous_report_by_task
    entry = next(
        item
        for item in orch.history_store._entries
        if item.entry_id == record.mediator_history_entry_id
    )
    assert isinstance(entry.payload, MediatorSignal)
    assert entry.payload.withheld is True


@pytest.mark.asyncio
async def test_cross_task_feedback_is_opt_in_and_labeled(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch._released_cross_task_reports_by_task["task-B"] = MediatorReport(
        task_id="task-B",
        iteration=2,
        content="cross-task report",
    )

    assert (
        await orch._build_prior_context(
            "learned_mediator",
            "task-A",
            current_iteration=3,
        )
        is None
    )

    orch.config.experiment.allow_cross_task_feedback = True
    context = await orch._build_prior_context(
        "learned_mediator",
        "task-A",
        current_iteration=3,
    )

    assert context is not None
    assert "Explicit Cross-Task Feedback" in context
    assert "allow_cross_task_feedback=true" in context
    assert "source_task=task-B" in context
    assert "cross-task report" in context


@pytest.mark.asyncio
async def test_cross_task_mediator_reports_are_staged_until_next_iteration(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.config.experiment.allow_cross_task_feedback = True
    orch._staged_cross_task_reports_by_task["task-B"] = MediatorReport(
        task_id="task-B",
        iteration=0,
        content="staged cross-task report",
    )

    same_iteration_context = await orch._build_prior_context(
        "learned_mediator",
        "task-A",
        current_iteration=0,
    )
    orch._release_staged_cross_task_reports()
    next_iteration_context = await orch._build_prior_context(
        "learned_mediator",
        "task-A",
        current_iteration=1,
    )

    assert same_iteration_context is None
    assert next_iteration_context is not None
    assert "source_task=task-B iter=0" in next_iteration_context
    assert "staged cross-task report" in next_iteration_context


@pytest.mark.asyncio
async def test_cross_task_mediator_reports_exclude_current_iteration(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.config.experiment.allow_cross_task_feedback = True
    orch._released_cross_task_reports_by_task["task-B"] = MediatorReport(
        task_id="task-B",
        iteration=1,
        content="same-iteration report",
    )
    orch._released_cross_task_reports_by_task["task-C"] = MediatorReport(
        task_id="task-C",
        iteration=0,
        content="previous-iteration report",
    )

    context = await orch._build_prior_context(
        "learned_mediator",
        "task-A",
        current_iteration=1,
    )

    assert context is not None
    assert "source_task=task-C iter=0" in context
    assert "previous-iteration report" in context
    assert "source_task=task-B iter=1" not in context
    assert "same-iteration report" not in context


@pytest.mark.asyncio
async def test_same_task_prior_context_is_unchanged_without_diffusion_integration(
    tmp_path,
):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch._previous_report_by_task["task-A"] = MediatorReport(
        task_id="task-A",
        iteration=0,
        content="same-task report",
    )

    context = await orch._build_prior_context(
        "learned_mediator",
        "task-A",
        current_iteration=1,
    )

    assert context == "same-task report"
    assert "Diffused Cross-Task Context" not in context


@pytest.mark.asyncio
async def test_capped_broadcast_builds_diffused_cross_task_context(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.config.experiment.allow_cross_task_feedback = True
    orch.config.diffusion.enabled = True
    orch.config.diffusion.policy = "capped_broadcast"
    orch.config.diffusion.max_artifacts = 1
    _write_graph_artifacts(tmp_path / "task-graph", ["task-A", "task-B", "task-C"])
    orch._ensure_diffusion_runtime_state()
    orch._diffusion_store.store_artifact(
        DiffusionArtifact(
            artifact_id="task-b-artifact",
            source_task_id="task-B",
            source_iteration=0,
            artifact_type=DiffusionArtifactType.DEBUG_HINT,
            risk_level=DiffusionRiskLevel.LOW,
            content="hint from task-B",
        )
    )
    orch._diffusion_store.store_artifact(
        DiffusionArtifact(
            artifact_id="task-c-artifact",
            source_task_id="task-C",
            source_iteration=0,
            artifact_type=DiffusionArtifactType.DEBUG_HINT,
            risk_level=DiffusionRiskLevel.LOW,
            content="hint from task-C",
        )
    )

    context = await orch._build_prior_context(
        "learned_mediator",
        "task-A",
        current_iteration=1,
    )

    assert context is not None
    assert "Diffused Cross-Task Context" in context
    assert "policy=capped_broadcast" in context
    assert "hint from task-B" in context or "hint from task-C" in context
    records = orch._diffusion_store.query_diffused_records(target_task_id="task-A")
    assert len([record for record in records if record.selected]) == 1


@pytest.mark.asyncio
async def test_capped_broadcast_excludes_same_task_and_same_iteration_artifacts(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.config.experiment.allow_cross_task_feedback = True
    orch.config.diffusion.enabled = True
    orch.config.diffusion.policy = "capped_broadcast"
    orch.config.diffusion.max_artifacts = 3
    _write_graph_artifacts(tmp_path / "task-graph", ["task-A", "task-B"])
    orch._ensure_diffusion_runtime_state()
    orch._diffusion_store.store_artifact(
        DiffusionArtifact(
            artifact_id="same-task",
            source_task_id="task-A",
            source_iteration=0,
            artifact_type=DiffusionArtifactType.DEBUG_HINT,
            risk_level=DiffusionRiskLevel.LOW,
            content="same task",
        )
    )
    orch._diffusion_store.store_artifact(
        DiffusionArtifact(
            artifact_id="same-iteration",
            source_task_id="task-B",
            source_iteration=1,
            artifact_type=DiffusionArtifactType.DEBUG_HINT,
            risk_level=DiffusionRiskLevel.LOW,
            content="same iteration",
        )
    )
    orch._diffusion_store.store_artifact(
        DiffusionArtifact(
            artifact_id="eligible",
            source_task_id="task-B",
            source_iteration=0,
            artifact_type=DiffusionArtifactType.DEBUG_HINT,
            risk_level=DiffusionRiskLevel.LOW,
            content="eligible artifact",
        )
    )

    context = await orch._build_prior_context(
        "learned_mediator",
        "task-A",
        current_iteration=1,
    )

    assert context is not None
    assert "eligible artifact" in context
    assert "same task" not in context
    assert "same iteration" not in context


@pytest.mark.asyncio
async def test_random_k_builds_seeded_cross_task_context(tmp_path):
    async def selected_artifacts(run_dir: Path) -> tuple[list[str], str]:
        orch, _, _ = _orchestrator(run_dir, "learned_mediator")
        orch.config.experiment.allow_cross_task_feedback = True
        orch.config.diffusion.enabled = True
        orch.config.diffusion.policy = "random_k"
        orch.config.diffusion.max_artifacts = 2
        _write_graph_artifacts(
            run_dir / "task-graph",
            ["task-A", "task-B", "task-C", "task-D"],
        )
        orch._ensure_diffusion_runtime_state()
        _store_diffusion_artifact(
            orch,
            artifact_id="task-b-artifact",
            source_task_id="task-B",
            content="hint from task-B",
        )
        _store_diffusion_artifact(
            orch,
            artifact_id="task-c-artifact",
            source_task_id="task-C",
            content="hint from task-C",
        )
        _store_diffusion_artifact(
            orch,
            artifact_id="task-d-artifact",
            source_task_id="task-D",
            content="hint from task-D",
        )

        context = await orch._build_prior_context(
            "learned_mediator",
            "task-A",
            current_iteration=1,
        )

        assert context is not None
        records = orch._diffusion_store.query_diffused_records(
            target_task_id="task-A",
            recent=None,
        )
        selected = sorted(record.artifact_id for record in records if record.selected)
        return selected, context

    first_selection, first_context = await selected_artifacts(tmp_path / "first")
    second_selection, second_context = await selected_artifacts(tmp_path / "second")

    assert first_selection == second_selection
    assert len(first_selection) == 2
    assert "Diffused Cross-Task Context" in first_context
    assert "policy=random_k" in first_context
    assert "relation=random" in first_context
    assert "policy=random_k" in second_context


@pytest.mark.asyncio
async def test_random_k_excludes_same_task_and_same_iteration_artifacts(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.config.experiment.allow_cross_task_feedback = True
    orch.config.diffusion.enabled = True
    orch.config.diffusion.policy = "random_k"
    orch.config.diffusion.max_artifacts = 3
    _write_graph_artifacts(tmp_path / "task-graph", ["task-A", "task-B"])
    orch._ensure_diffusion_runtime_state()
    _store_diffusion_artifact(
        orch,
        artifact_id="same-task",
        source_task_id="task-A",
        content="same task",
    )
    _store_diffusion_artifact(
        orch,
        artifact_id="same-iteration",
        source_task_id="task-B",
        source_iteration=1,
        content="same iteration",
    )
    _store_diffusion_artifact(
        orch,
        artifact_id="eligible",
        source_task_id="task-B",
        content="eligible artifact",
    )

    context = await orch._build_prior_context(
        "learned_mediator",
        "task-A",
        current_iteration=1,
    )

    assert context is not None
    assert "eligible artifact" in context
    assert "same task" not in context
    assert "same iteration" not in context
    records = orch._diffusion_store.query_diffused_records(target_task_id="task-A")
    assert all(record.policy_name == "random_k" for record in records)
    assert len([record for record in records if record.selected]) == 1


@pytest.mark.asyncio
async def test_cross_task_full_traces_exclude_target_task(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "full_traces")
    orch.config.experiment.allow_cross_task_feedback = True
    orch.artifact_store.store_trace(
        ExecutionTrace(task_id="task-A", iteration=0, reward=0.1, status="ok")
    )
    orch.artifact_store.store_trace(
        ExecutionTrace(task_id="task-B", iteration=0, reward=0.9, status="ok")
    )

    context = await orch._build_prior_context(
        "full_traces",
        "task-A",
        current_iteration=1,
    )

    assert context is not None
    assert "source_task=task-B" in context
    assert "reward=0.90" in context
    assert "source_task=task-A" not in context


@pytest.mark.asyncio
async def test_full_traces_prior_context_excludes_current_and_future_iterations(
    tmp_path,
):
    orch, _, _ = _orchestrator(tmp_path, "full_traces")
    orch.artifact_store.store_trace(
        ExecutionTrace(task_id="task-A", iteration=0, reward=0.1, status="ok")
    )
    orch.artifact_store.store_trace(
        ExecutionTrace(task_id="task-A", iteration=2, reward=0.9, status="ok")
    )

    context = await orch._build_prior_context(
        "full_traces",
        "task-A",
        current_iteration=1,
    )

    assert context is not None
    assert "iter=0" in context
    assert "reward=0.10" in context
    assert "iter=2" not in context
    assert "reward=0.90" not in context


@pytest.mark.asyncio
async def test_cross_task_full_traces_respect_round_causality(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "full_traces")
    orch.config.experiment.allow_cross_task_feedback = True
    orch.artifact_store.store_trace(
        ExecutionTrace(task_id="task-B", iteration=0, reward=0.8, status="ok")
    )
    orch.artifact_store.store_trace(
        ExecutionTrace(task_id="task-B", iteration=1, reward=0.95, status="ok")
    )
    orch.artifact_store.store_trace(
        ExecutionTrace(task_id="task-B", iteration=2, reward=0.99, status="ok")
    )

    context = await orch._build_prior_context(
        "full_traces",
        "task-A",
        current_iteration=1,
    )

    assert context is not None
    assert "source_task=task-B iter=0" in context
    assert "reward=0.80" in context
    assert "source_task=task-B iter=1" not in context
    assert "reward=0.95" not in context
    assert "source_task=task-B iter=2" not in context
    assert "reward=0.99" not in context


@pytest.mark.asyncio
async def test_long_trace_stderr_uses_llm_compactor(tmp_path):
    llm = _LLMCompactor(
        content='{"headline": "Build failed.", "evidence": "Missing dependency xyz."}'
    )
    orch, _, _ = _orchestrator(tmp_path, "full_traces", llm_client=llm)
    orch.artifact_store.store_trace(
        ExecutionTrace(
            task_id="task-A",
            iteration=0,
            reward=0.0,
            status="ok",
            stderr="dependency xyz missing\n" * 100,
        )
    )

    context = await orch._build_prior_context(
        "full_traces",
        "task-A",
        current_iteration=1,
    )

    assert context is not None
    assert "Build failed." in context
    assert "Missing dependency xyz." in context
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_short_trace_stderr_does_not_call_llm_compactor(tmp_path):
    llm = _LLMCompactor(content='{"headline": "unused", "evidence": "unused"}')
    orch, _, _ = _orchestrator(tmp_path, "full_traces", llm_client=llm)
    orch.artifact_store.store_trace(
        ExecutionTrace(
            task_id="task-A",
            iteration=0,
            reward=0.0,
            status="ok",
            stderr="short stderr",
        )
    )

    context = await orch._build_prior_context(
        "full_traces",
        "task-A",
        current_iteration=1,
    )

    assert context is not None
    assert "short stderr" in context
    assert llm.calls == []


@pytest.mark.asyncio
async def test_long_trace_stderr_falls_back_when_llm_compactor_fails(tmp_path):
    llm = _LLMCompactor(content="", raise_exc=RuntimeError("llm unavailable"))
    orch, _, _ = _orchestrator(tmp_path, "full_traces", llm_client=llm)
    orch.artifact_store.store_trace(
        ExecutionTrace(
            task_id="task-A",
            iteration=0,
            reward=0.0,
            status="ok",
            stderr=("START-" + ("x" * 900) + "-END"),
        )
    )

    context = await orch._build_prior_context(
        "full_traces",
        "task-A",
        current_iteration=1,
    )

    assert context is not None
    assert "START-" in context
    assert "-END" in context
    assert "\n...\n" in context or "\n…\n" in context
    assert len(llm.calls) == 1


def test_condition_assignment_and_cli_validation_reject_unknown_names():
    config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
            "judge": "test-judge",
        },
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )
    with pytest.raises(ValidationError):
        config.experiment.condition_name = "bad-condition"

    with pytest.raises(typer.BadParameter):
        _validate_condition_name("bad-condition")

    assert _validate_condition_name("no_feedback") == "no_feedback"
