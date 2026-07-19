from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer
from pydantic import ValidationError

from mediated_coevo.core.config import Config
from mediated_coevo.diffusion import (
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    DiffusionStore,
    DiffusionSubscription,
    LangChainGraphPolicyResult,
    REUSE_SUCCESS_CHANNEL,
    TaskGraphSnapshot,
)
from mediated_coevo.cli.config import _run_config_overrides
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.experiment.orchestrator import Orchestrator
from mediated_coevo.agents.prompt_context import PlannerPriorContextBundle
from mediated_coevo.runtime.token_budget import TokenBudgetExceeded, count_text_tokens
from mediated_coevo.stores.artifact_store import ArtifactStore
from tests.config_helpers import (
    budgets_config,
    diffusion_config,
    experiment_config,
    models_config,
)


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
                "graph_kind": "skillflow_ranked_similarity",
                "pair_count": len(pairs),
                "p20_threshold": 0.01,
                "edge_score_threshold": 0.05,
                "active_threshold": 0.05,
                "threshold_kind": "absolute_score",
                "pairs": pairs,
            }
        )
    )


def _write_weighted_graph_artifacts(
    graph_dir: Path,
    *,
    task_ids: list[str],
    weighted_pairs: list[tuple[str, str, float]],
) -> None:
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
    pairs = [
        {
            "source": source_task_id,
            "target": target_task_id,
            "score": score,
            "components": {"category": 1.0, "tags": 0.5},
            "shared": {"tags": ["python"]},
            "kept_after_p20_cut": True,
            "kept_after_threshold_cut": True,
        }
        for source_task_id, target_task_id, score in weighted_pairs
    ]
    (graph_dir / "task_profiles.json").write_text(
        json.dumps({"task_count": len(task_ids), "profiles": profiles})
    )
    (graph_dir / "pairwise_similarity.json").write_text(
        json.dumps(
            {
                "graph_kind": "skillflow_ranked_similarity",
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
    artifact_type: DiffusionArtifactType = DiffusionArtifactType.DEBUG_HINT,
    content: str | None = None,
    verifier_reward: float = 1.0,
    judge_reward: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    orch._diffusion_store.store_artifact(
        DiffusionArtifact(
            artifact_id=artifact_id,
            source_task_id=source_task_id,
            source_iteration=source_iteration,
            artifact_type=artifact_type,
            risk_level=DiffusionRiskLevel.LOW,
            content=content or artifact_id,
            verifier_reward=verifier_reward,
            judge_reward=judge_reward,
            metadata=metadata or {},
        )
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
        self.llm_client = _PlannerLLM()

    def set_skill_context(
        self,
        executor_skills: str,
        planner_skill: str | None = None,
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
        return TaskSpec(
            task_id=task_id, instruction=base_instruction, iteration=iteration
        )


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


class _FakeLangChainGraphPolicy:
    def __init__(self) -> None:
        self.artifact_ids_seen: list[str] = []

    async def prepare(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        previous_snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> LangChainGraphPolicyResult:
        self.artifact_ids_seen = [artifact.artifact_id for artifact in artifacts]
        selected = next(
            artifact
            for artifact in artifacts
            if artifact.artifact_id == "old-same-task"
        )
        return LangChainGraphPolicyResult(
            snapshot=TaskGraphSnapshot(
                run_id="run-1",
                iteration=current_iteration,
                task_ids=["node-task-A"],
                graph_policy="langchain_graph",
                metadata={"current_node_id": "node-task-A"},
            ),
            subscriptions=[
                DiffusionSubscription(
                    artifact=selected,
                    policy_name="langchain_graph",
                    relation="same_node",
                    reason="fake selected old same-task artifact",
                    context_channel=REUSE_SUCCESS_CHANNEL,
                )
            ],
        )


class _FixedGraphLangChainPolicy:
    def __init__(self) -> None:
        self.prepare_called = False
        self.selected_snapshot_id: str | None = None
        self.artifact_ids_seen: list[str] = []

    async def prepare(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        previous_snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> LangChainGraphPolicyResult:
        self.prepare_called = True
        raise AssertionError("validation must not update the task graph")

    async def select_with_fixed_graph(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        snapshot: TaskGraphSnapshot,
        artifacts: list[DiffusionArtifact],
    ) -> LangChainGraphPolicyResult:
        self.selected_snapshot_id = snapshot.snapshot_id
        self.artifact_ids_seen = [artifact.artifact_id for artifact in artifacts]
        selected = next(
            artifact
            for artifact in artifacts
            if artifact.artifact_id == "old-same-task"
        )
        return LangChainGraphPolicyResult(
            snapshot=snapshot,
            subscriptions=[
                DiffusionSubscription(
                    artifact=selected,
                    policy_name="langchain_graph",
                    relation="fixed_graph_selection",
                    reason="selected using fixed validation graph",
                    context_channel=REUSE_SUCCESS_CHANNEL,
                )
            ],
        )


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
    orch.benchmark_repo = _TaskRepo()
    orch.config = Config(
        models=models_config(),
        budgets=budgets_config(),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )
    orch.config.experiment.condition_name = condition
    orch.config.experiment.shared_notes = "shared note"
    orch.experiment_dir = tmp_path
    orch._previous_report_by_task = {}
    orch._released_cross_task_reports_by_task = {}
    orch._staged_cross_task_reports_by_task = {}
    orch._previous_reward_by_task = {}
    orch._prior_context_by_target = {}
    orch._diffusion_context_by_target = {}
    return orch, planner, mediator


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
    assert record.diffusion_policy == "none"
    assert record.execution_trace is not None
    assert record.execution_trace.iteration == 1


@pytest.mark.asyncio
async def test_prior_context_bundle_sections_follow_condition_matrix(tmp_path):
    bundles: dict[str, PlannerPriorContextBundle] = {}

    orch, _, _ = _orchestrator(tmp_path / "no-feedback", "no_feedback")
    bundles["no_feedback"] = await orch._build_prior_context_bundle(
        "no_feedback",
        "task-A",
        current_iteration=1,
    )

    orch, _, _ = _orchestrator(tmp_path / "shared-notes", "shared_notes")
    bundles["shared_notes"] = await orch._build_prior_context_bundle(
        "shared_notes",
        "task-A",
        current_iteration=1,
    )

    orch, _, _ = _orchestrator(tmp_path / "full-traces", "full_traces")
    orch.artifact_store.store_trace(
        ExecutionTrace(task_id="task-A", iteration=0, reward=0.25, status="ok")
    )
    orch.artifact_store.store_trace(
        ExecutionTrace(task_id="task-B", iteration=0, reward=0.75, status="ok")
    )
    bundles["full_traces"] = await orch._build_prior_context_bundle(
        "full_traces",
        "task-A",
        current_iteration=1,
    )

    for condition in ("static_mediator", "learned_mediator"):
        orch, _, _ = _orchestrator(tmp_path / condition, condition)
        orch._previous_report_by_task["task-A"] = MediatorReport(
            task_id="task-A",
            iteration=0,
            content=f"{condition} same-task report",
        )
        orch._released_cross_task_reports_by_task["task-B"] = MediatorReport(
            task_id="task-B",
            iteration=0,
            content=f"{condition} cross-task report",
        )
        bundles[condition] = await orch._build_prior_context_bundle(
            condition,
            "task-A",
            current_iteration=1,
        )

    assert [section.kind for section in bundles["no_feedback"].sections()] == []
    assert [section.kind for section in bundles["shared_notes"].sections()] == [
        "same_task_prior"
    ]
    assert [section.kind for section in bundles["full_traces"].sections()] == [
        "same_task_prior",
        "cross_task_prior",
    ]
    assert [section.kind for section in bundles["static_mediator"].sections()] == [
        "same_task_prior",
        "cross_task_prior",
    ]
    assert [section.kind for section in bundles["learned_mediator"].sections()] == [
        "same_task_prior",
        "cross_task_prior",
    ]


@pytest.mark.asyncio
async def test_diffusion_context_is_condition_independent_and_priority_routed(
    tmp_path,
):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.config.diffusion.enabled = True
    orch.config.diffusion.policy = "capped_broadcast"
    orch.config.diffusion.max_artifacts = 1
    orch._released_cross_task_reports_by_task["task-B"] = MediatorReport(
        task_id="task-B",
        iteration=0,
        content="explicit cross-task report",
    )
    orch._ensure_diffusion_runtime_state()
    _store_diffusion_artifact(
        orch,
        artifact_id="task-c-artifact",
        source_task_id="task-C",
        content="diffused hint",
    )

    bundle = await orch._build_prior_context_bundle(
        "learned_mediator",
        "task-A",
        current_iteration=1,
    )

    assert [section.kind for section in bundle.sections()] == ["diffusion_context"]
    assert bundle.diffusion_context is not None
    assert "diffused hint" in bundle.diffusion_context
    assert bundle.cross_task_prior is None
    assert "explicit cross-task report" not in bundle.flatten()


@pytest.mark.asyncio
async def test_langchain_graph_policy_gets_same_task_causal_artifacts_only(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    fake_policy = _FakeLangChainGraphPolicy()
    orch.config.diffusion.enabled = True
    orch.config.diffusion.policy = "langchain_graph"
    orch._langchain_graph_policy = fake_policy
    orch._ensure_diffusion_runtime_state()
    _store_diffusion_artifact(
        orch,
        artifact_id="old-same-task",
        source_task_id="task-A",
        source_iteration=0,
        content="same-node prior hint",
    )
    _store_diffusion_artifact(
        orch,
        artifact_id="future-artifact",
        source_task_id="task-B",
        source_iteration=2,
        content="future hint",
    )

    bundle = await orch._build_prior_context_bundle(
        "learned_mediator",
        "task-A",
        current_iteration=1,
    )

    assert fake_policy.artifact_ids_seen == ["old-same-task"]
    assert bundle.diffusion_context is not None
    assert "same-node prior hint" in bundle.diffusion_context


def test_langchain_graph_uses_carried_seed_snapshot_at_first_iteration(tmp_path):
    orch = Orchestrator.__new__(Orchestrator)
    orch._diffusion_store = DiffusionStore(tmp_path / "diffusion")
    seed_snapshot = TaskGraphSnapshot(
        snapshot_id="seed-snapshot",
        run_id="old-run",
        iteration=7,
        task_ids=["old-node"],
        graph_policy="langchain_graph",
    )
    current_snapshot = TaskGraphSnapshot(
        snapshot_id="current-snapshot",
        run_id="new-run",
        iteration=0,
        task_ids=["new-node"],
        graph_policy="langchain_graph",
    )
    orch._diffusion_store.store_graph_snapshot(seed_snapshot)

    assert (
        orch._latest_langchain_graph_snapshot(current_iteration=0).snapshot_id
        == "seed-snapshot"
    )

    orch._diffusion_store.store_graph_snapshot(current_snapshot)

    assert (
        orch._latest_langchain_graph_snapshot(current_iteration=1).snapshot_id
        == "current-snapshot"
    )


@pytest.mark.asyncio
async def test_validation_langchain_graph_uses_fixed_snapshot_for_context(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    fake_policy = _FixedGraphLangChainPolicy()
    orch.config.experiment.benchmark_selection.split = "validation"
    orch.config.diffusion.enabled = True
    orch.config.diffusion.policy = "langchain_graph"
    orch._langchain_graph_policy = fake_policy
    orch._ensure_diffusion_runtime_state()
    seed_snapshot = TaskGraphSnapshot(
        snapshot_id="seed-snapshot",
        run_id="train-run",
        iteration=7,
        task_ids=["node-task-A"],
        graph_policy="langchain_graph",
        metadata={"current_node_id": "node-task-A"},
    )
    orch._diffusion_store.store_graph_snapshot(seed_snapshot)
    _store_diffusion_artifact(
        orch,
        artifact_id="old-same-task",
        source_task_id="task-A",
        source_iteration=0,
        content="validation should use fixed graph",
    )

    bundle = await orch._build_prior_context_bundle(
        "learned_mediator",
        "task-A",
        current_iteration=1,
    )

    snapshots = orch._diffusion_store.query_graph_snapshots(recent=None)
    assert fake_policy.prepare_called is False
    assert fake_policy.selected_snapshot_id == "seed-snapshot"
    assert fake_policy.artifact_ids_seen == ["old-same-task"]
    assert [snapshot.snapshot_id for snapshot in snapshots] == ["seed-snapshot"]
    assert bundle.diffusion_context is not None
    assert "validation should use fixed graph" in bundle.diffusion_context


@pytest.mark.asyncio
async def test_prior_context_fit_uses_same_and_transfer_slots(tmp_path):
    llm = _LLMCompactor(content='{"headline":"compact","evidence":"context"}')
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator", llm_client=llm)
    orch.config.budgets.max_same_task_prior_tokens = 20
    orch.config.budgets.max_transfer_context_tokens = 30
    bundle = PlannerPriorContextBundle(
        same_task_prior="same task prior " * 200,
        diffusion_context="diffused transfer " * 200,
        cross_task_prior="cross task transfer " * 200,
    )

    fitted = await orch._fit_prior_context_bundle(bundle)
    flattened = fitted.flatten()
    assert flattened is not None

    model = orch.planner.llm_client.model
    assert count_text_tokens(model, fitted.same_task_prior or "") <= 20
    assert count_text_tokens(model, fitted.diffusion_context or "") <= 30
    assert fitted.cross_task_prior is None

    orch._record_prior_context_metrics(
        task_id="task-A",
        current_iteration=1,
        bundle=fitted,
        flattened=flattened,
    )
    metrics = orch._prior_context_by_target[("task-A", 1)]
    assert metrics["transfer_context_kind"] == "diffusion"
    assert metrics["max_total_prior_context_tokens"] == 50
    assert metrics["context_budget_violation"] is True
    assert len(llm.calls) == 2


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
async def test_cross_task_feedback_is_available_and_labeled(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch._released_cross_task_reports_by_task["task-B"] = MediatorReport(
        task_id="task-B",
        iteration=2,
        content="cross-task report",
    )

    context = await orch._build_prior_context(
        "learned_mediator",
        "task-A",
        current_iteration=3,
    )

    assert context is not None
    assert "Explicit Cross-Task Feedback" in context
    assert "allow_cross_task_feedback" not in context
    assert "source_task=task-B" in context
    assert "cross-task report" in context


@pytest.mark.asyncio
async def test_cross_task_mediator_reports_are_staged_until_next_iteration(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
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
    orch.config.diffusion.enabled = True
    orch.config.diffusion.policy = "capped_broadcast"
    orch.config.diffusion.max_artifacts = 1
    _write_graph_artifacts(tmp_path / "task-graph", ["task-A", "task-B", "task-C"])
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
    assert len(records) == 2
    assert sum(1 for record in records if record.eligible) == 2
    assert sum(1 for record in records if record.selected) == 1
    assert sum(1 for record in records if record.rendered) == 1


@pytest.mark.asyncio
async def test_capped_broadcast_excludes_same_task_and_same_iteration_artifacts(
    tmp_path,
):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.config.diffusion.enabled = True
    orch.config.diffusion.policy = "capped_broadcast"
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


@pytest.mark.asyncio
async def test_random_k_builds_seeded_cross_task_context(tmp_path):
    async def selected_artifacts(run_dir: Path) -> tuple[list[str], str]:
        orch, _, _ = _orchestrator(run_dir, "learned_mediator")
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
    assert len(records) == 1
    assert records[0].selected is True


@pytest.mark.asyncio
async def test_top_k_similarity_records_eligible_selected_and_transfer_metrics(
    tmp_path,
):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.config.diffusion.enabled = True
    orch.config.diffusion.policy = "top_k_similarity"
    orch.config.diffusion.graph = "task_similarity"
    orch.config.diffusion.max_artifacts = 3
    orch.config.diffusion.top_k_neighbors = 1
    _write_weighted_graph_artifacts(
        tmp_path / "task-graph",
        task_ids=["task-A", "task-B", "task-C", "task-D"],
        weighted_pairs=[
            ("task-B", "task-A", 0.9),
            ("task-C", "task-A", 0.7),
            ("task-D", "task-A", 0.2),
        ],
    )
    orch._ensure_diffusion_runtime_state()
    _store_diffusion_artifact(
        orch,
        artifact_id="task-b-duplicate-debug",
        source_task_id="task-B",
        artifact_type=DiffusionArtifactType.DEBUG_HINT,
        content="duplicate debug from task-B",
    )
    _store_diffusion_artifact(
        orch,
        artifact_id="task-b-debug",
        source_task_id="task-B",
        artifact_type=DiffusionArtifactType.DEBUG_HINT,
        content="hint from task-B",
    )
    _store_diffusion_artifact(
        orch,
        artifact_id="task-b-outcome",
        source_task_id="task-B",
        artifact_type=DiffusionArtifactType.RUN_OUTCOME,
        content="outcome from task-B",
    )
    _store_diffusion_artifact(
        orch,
        artifact_id="task-b-regressed-outcome",
        source_task_id="task-B",
        artifact_type=DiffusionArtifactType.RUN_OUTCOME,
        content="warning from task-B",
        verifier_reward=0.0,
        metadata={"regression": True},
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
    assert "outcome from task-B" in context
    assert "warning from task-B" in context
    assert "Avoid/Recheck Artifacts" in context
    assert "hint from task-B" not in context
    assert "duplicate debug from task-B" not in context
    assert "hint from task-C" not in context
    assert "hint from task-D" not in context
    records = orch._diffusion_store.query_diffused_records(
        target_task_id="task-A",
        recent=None,
    )
    selected_records = [record for record in records if record.selected]
    selected_artifact_ids = {record.artifact_id for record in selected_records}
    assert sum(1 for record in records if record.eligible) == 6
    assert len(selected_records) == 2
    assert sum(1 for record in records if record.rendered) == 2
    assert selected_artifact_ids == {
        "task-b-outcome",
        "task-b-regressed-outcome",
    }

    record = IterationRecord(
        iteration=1,
        task_id="task-A",
        reward=0.0,
        delta_reward=-1.0,
    )
    orch._attach_diffusion_context_metrics(record)

    assert record.diffusion_artifacts_eligible == 6
    assert record.diffusion_artifacts_selected == 2
    assert record.diffusion_artifacts_rendered == 2
    assert record.reward_after_diffusion_context == 0.0
    assert record.regression_after_diffusion_context is True
    assert record.source_task_ids == ["task-B"]


@pytest.mark.asyncio
async def test_top_k_similarity_prepares_per_target_subscription_board(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.config.diffusion.enabled = True
    orch.config.diffusion.policy = "top_k_similarity"
    orch.config.diffusion.graph = "task_similarity"
    orch.config.diffusion.max_artifacts = 3
    orch.config.diffusion.top_k_neighbors = 1
    orch._diffusion_target_task_ids = ["task-A", "task-B", "task-C"]
    _write_weighted_graph_artifacts(
        tmp_path / "task-graph",
        task_ids=["task-A", "task-B", "task-C"],
        weighted_pairs=[
            ("task-B", "task-A", 0.9),
            ("task-C", "task-A", 0.4),
            ("task-C", "task-B", 0.8),
            ("task-A", "task-B", 0.6),
            ("task-A", "task-C", 0.7),
            ("task-B", "task-C", 0.5),
        ],
    )
    orch._ensure_diffusion_runtime_state()
    _store_diffusion_artifact(
        orch,
        artifact_id="task-a-artifact",
        source_task_id="task-A",
        content="hint from task-A",
    )
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

    context_a = await orch._build_prior_context(
        "learned_mediator",
        "task-A",
        current_iteration=1,
    )
    context_b = await orch._build_prior_context(
        "learned_mediator",
        "task-B",
        current_iteration=1,
    )
    context_c = await orch._build_prior_context(
        "learned_mediator",
        "task-C",
        current_iteration=1,
    )

    assert context_a is not None
    assert "hint from task-B" in context_a
    assert "hint from task-C" not in context_a
    assert context_b is not None
    assert "hint from task-C" in context_b
    assert "hint from task-A" not in context_b
    assert context_c is not None
    assert "hint from task-A" in context_c
    assert "hint from task-B" not in context_c

    records = orch._diffusion_store.query_diffused_records(recent=None)
    selected_by_target = {
        record.target_task_id: record.source_task_id
        for record in records
        if record.selected and record.rendered
    }
    assert selected_by_target == {
        "task-A": "task-B",
        "task-B": "task-C",
        "task-C": "task-A",
    }
    assert all(
        record.metadata.get("edge_rank") == 1 for record in records if record.selected
    )
    assert sum(1 for record in records if record.eligible) == 6
    assert sum(1 for record in records if record.selected) == 3
    assert sum(1 for record in records if record.rendered) == 3


@pytest.mark.asyncio
async def test_diffusion_subscription_board_consumes_target_entry(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.config.diffusion.enabled = True
    orch.config.diffusion.policy = "capped_broadcast"
    orch.config.diffusion.max_artifacts = 1
    orch._diffusion_target_task_ids = ["task-A", "task-C"]
    orch._ensure_diffusion_runtime_state()
    _store_diffusion_artifact(
        orch,
        artifact_id="task-b-artifact",
        source_task_id="task-B",
        content="hint from task-B",
    )

    context = await orch._build_prior_context(
        "learned_mediator",
        "task-A",
        current_iteration=1,
    )

    assert context is not None
    assert "hint from task-B" in context
    assert (1, "task-A") not in orch._diffusion_sub_board
    assert (1, "task-C") in orch._diffusion_sub_board


@pytest.mark.asyncio
async def test_diffusion_subscription_board_queries_artifacts_once_per_iteration(
    tmp_path,
):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.config.diffusion.enabled = True
    orch.config.diffusion.policy = "capped_broadcast"
    orch.config.diffusion.max_artifacts = 1
    orch._diffusion_target_task_ids = ["task-A", "task-C"]
    orch._ensure_diffusion_runtime_state()
    _store_diffusion_artifact(
        orch,
        artifact_id="task-b-artifact",
        source_task_id="task-B",
        content="hint from task-B",
    )

    original_query_artifacts = orch._diffusion_store.query_artifacts
    query_count = 0

    def counting_query_artifacts(**kwargs):
        nonlocal query_count
        query_count += 1
        return original_query_artifacts(**kwargs)

    orch._diffusion_store.query_artifacts = counting_query_artifacts

    await orch._build_prior_context(
        "learned_mediator",
        "task-A",
        current_iteration=1,
    )
    await orch._build_prior_context(
        "learned_mediator",
        "task-C",
        current_iteration=1,
    )

    assert query_count == 1


@pytest.mark.asyncio
async def test_cross_task_full_traces_exclude_target_task(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "full_traces")
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
async def test_long_trace_stderr_raises_when_llm_compactor_fails(tmp_path):
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

    with pytest.raises(TokenBudgetExceeded):
        await orch._build_prior_context(
            "full_traces",
            "task-A",
            current_iteration=1,
        )

    assert len(llm.calls) == 3


def test_condition_assignment_and_cli_validation_reject_unknown_names():
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
    with pytest.raises(ValidationError):
        config.experiment.condition_name = "bad-condition"

    with pytest.raises(typer.BadParameter):
        _run_config_overrides(
            iterations=None,
            seed=None,
            condition="bad-condition",
            diffusion_enabled=None,
            diffusion_policy=None,
            diffusion_graph=None,
            diffusion_max_artifacts=None,
            diffusion_top_k_neighbors=None,
            harbor_agent_setup_timeout_multiplier=None,
        )

    assert _run_config_overrides(
        iterations=None,
        seed=None,
        condition="no_feedback",
        diffusion_enabled=None,
        diffusion_policy=None,
        diffusion_graph=None,
        diffusion_max_artifacts=None,
        diffusion_top_k_neighbors=None,
        harbor_agent_setup_timeout_multiplier=None,
    ) == {"experiment": {"condition_name": "no_feedback"}}
