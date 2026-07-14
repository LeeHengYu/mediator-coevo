from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import mediated_coevo.experiment.sample_runtime as sample_runtime_module
from mediated_coevo.artifacts.adapters import DiffusionArtifactBankUpdater
from mediated_coevo.diffusion.models import (
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
)
from mediated_coevo.diffusion.policy import (
    REUSE_SUCCESS_CHANNEL,
    DiffusionSubscription,
)
from mediated_coevo.diffusion.renderer import render_diffusion_subscriptions
from mediated_coevo.diffusion.store import DiffusionStore
from mediated_coevo.execution.adapters import ExplicitContextOrchestratorExecutionAgent
from mediated_coevo.execution.models import (
    ContextPack,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskProfile,
)
from mediated_coevo.experiment.sample_archive import (
    load_sample_result,
    load_warmup_bundle,
)
from mediated_coevo.experiment.sample_models import (
    FailureRecord,
    FailureStage,
    PositionJournal,
    RunProgress,
    SampleRunError,
    SampleSpec,
    SequenceSpec,
)
from mediated_coevo.experiment.sample_runner import SampleRunner
from mediated_coevo.experiment.sample_runtime import SampleRuntime, build_sample_runtime
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.orchestration.adapters import (
    DiffusionContextPacker,
    RandomPolicyAgent,
)
from mediated_coevo.orchestration.arms import OrchestrationArm
from mediated_coevo.orchestration.contracts import (
    GraphAgentRequest,
    GraphAgentResponse,
    PolicyAgentRequest,
    PolicyAgentResponse,
)
from tests.test_feedback_conditions import _orchestrator


@pytest.mark.asyncio
async def test_renderer_audits_policy_context_without_a_graph_snapshot(tmp_path):
    artifact = DiffusionArtifact(
        artifact_id="artifact-0",
        source_task_id="warmup-task",
        source_iteration=0,
        artifact_type=DiffusionArtifactType.RUN_OUTCOME,
        risk_level=DiffusionRiskLevel.LOW,
        content="reusable warm-up outcome",
        verifier_reward=1.0,
    )
    subscription = DiffusionSubscription(
        artifact=artifact,
        policy_name="langchain_diffusion_policy",
        relation="agent_selected",
        reason="selected without graph state",
        context_channel=REUSE_SUCCESS_CHANNEL,
    )
    store = DiffusionStore(tmp_path / "diffusion")

    bundle = await render_diffusion_subscriptions(
        store=store,
        snapshot=None,
        graph_policy="no_graph",
        model="test-model",
        target_task_id="target-task",
        target_iteration=1,
        target_run_id="sample-1",
        subscriptions=[subscription],
        eligible_count=1,
    )

    assert bundle.snapshot_id is None
    assert bundle.graph_policy == "no_graph"
    assert bundle.selected_count == 1
    assert bundle.rendered_count == 1
    assert bundle.rendered_artifact_ids == ["artifact-0"]
    records = store.query_diffused_records(recent=None)
    assert len(records) == 1
    assert records[0].snapshot_id is None
    assert records[0].target_run_id == "sample-1"


@dataclass
class _UnusedGraphAgent:
    async def update(self, request: GraphAgentRequest) -> GraphAgentResponse:
        raise AssertionError(f"graph agent must not be called at {request.position}")


@dataclass
class _UnusedDiffusionPolicyAgent:
    async def select(self, request: PolicyAgentRequest) -> PolicyAgentResponse:
        raise AssertionError(
            f"diffusion policy must not be called at {request.position}"
        )


@dataclass
class _RuntimeProjector:
    async def project(
        self,
        *,
        task: TaskProfile,
        execution: TaskExecutionResult,
    ) -> tuple[DiffusionArtifact, ...]:
        return (
            DiffusionArtifact(
                artifact_id=f"artifact-{execution.run_id}-{execution.position}",
                source_task_id=task.task_id,
                source_iteration=execution.position,
                source_run_id=execution.run_id,
                artifact_type=DiffusionArtifactType.RUN_OUTCOME,
                risk_level=DiffusionRiskLevel.LOW,
                content=f"portable outcome for {task.task_id}",
                verifier_reward=execution.reward,
            ),
        )


@dataclass
class _RuntimeOrchestrator:
    experiment_dir: Path
    _diffusion_store: DiffusionStore = field(init=False)
    execution_calls: list[tuple[int, ContextPack]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self._diffusion_store = DiffusionStore(self.experiment_dir / "diffusion")
        self.history_store = SimpleNamespace(
            _entries=[],
            _rejected_proposal_batches=[],
            _rejected_reflection_batches=[],
        )
        self.config = SimpleNamespace(
            models=SimpleNamespace(
                planner="fake-planner",
                mediator="fake-mediator",
                judge="fake-judge",
                executor="fake-executor",
            ),
            executor_runtime=SimpleNamespace(
                jobs_dir="jobs",
                agent_name="fake-agent",
            ),
        )

    @property
    def diffusion_store(self) -> DiffusionStore:
        return self._diffusion_store

    async def execute_task_with_context(
        self,
        *,
        task_id: str,
        position: int,
        context: ContextPack,
        task: TaskProfile,
    ) -> IterationRecord:
        assert task.task_id == task_id
        self.execution_calls.append((position, context))
        job_dir = self.experiment_dir / "jobs" / f"position-{position:04d}"
        job_dir.mkdir(parents=True)
        (job_dir / "evidence.json").write_text(
            json.dumps({"task_id": task_id, "position": position}),
            encoding="utf-8",
        )
        trace = ExecutionTrace(
            task_id=task_id,
            iteration=position,
            reward=float(position),
            status="ok",
            harbor_paths={"job": str(job_dir.resolve())},
        )
        trace_dir = self.experiment_dir / "artifacts" / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        portable_trace = trace.model_copy(
            update={
                "harbor_paths": {
                    "job": job_dir.relative_to(self.experiment_dir).as_posix()
                }
            }
        )
        (trace_dir / f"{task_id}-{position:04d}.json").write_text(
            portable_trace.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return IterationRecord(
            iteration=position,
            task_id=task_id,
            reward=float(position),
            execution_trace=trace,
            graph_snapshot_id=context.snapshot_id,
            diffusion_policy=context.policy_name,
            diffusion_enabled=context.policy_name != "none",
            diffusion_artifacts_eligible=len(context.eligible_artifact_ids),
            diffusion_artifacts_selected=len(context.selected_artifact_ids),
            diffusion_artifacts_rendered=len(context.rendered_artifact_ids),
            transfer_context_kind="diffusion" if context.text else "none",
            transfer_context_tokens=context.token_count,
            max_transfer_context_tokens=context.max_context_tokens or 0,
            context_budget_violation=context.budget_violation,
            compacted_diffusion_artifact_ids=list(context.compacted_artifact_ids),
            dropped_for_budget_artifact_ids=list(
                context.dropped_for_budget_artifact_ids
            ),
            source_task_ids=list(context.source_task_ids),
        )


def _runtime_runner(orchestrator: _RuntimeOrchestrator) -> SampleRunner:
    store = orchestrator.diffusion_store
    return SampleRunner(
        graph_agent=_UnusedGraphAgent(),
        diffusion_policy_agent=_UnusedDiffusionPolicyAgent(),
        random_policy_agent=RandomPolicyAgent(max_artifacts=1),
        context_packer=DiffusionContextPacker(store=store, model="fake-model"),
        execution_agent=ExplicitContextOrchestratorExecutionAgent(orchestrator),
        artifact_projector=_RuntimeProjector(),
        artifact_bank_updater=DiffusionArtifactBankUpdater(store),
    )


def _runtime(
    *,
    workspace: Path,
    sequence_dir: Path,
    run_id: str,
) -> tuple[SampleRuntime, _RuntimeOrchestrator]:
    orchestrator = _RuntimeOrchestrator(workspace)
    return (
        SampleRuntime(
            orchestrator=cast(Any, orchestrator),
            run_id=run_id,
            sequence_dir=sequence_dir,
            implementation_revision="abc123",
            implementation_dirty=False,
            runner=_runtime_runner(orchestrator),
            diffusion_store=orchestrator.diffusion_store,
        ),
        orchestrator,
    )


def _runtime_sequence(*, warmup_count: int = 1) -> SequenceSpec:
    return SequenceSpec(
        sequence_id="sequence-runtime",
        tasks=tuple(
            TaskProfile(task_id=f"task-{position}", instruction=f"task {position}")
            for position in range(3)
        ),
        warmup_count=warmup_count,
        policy_seed=19,
    )


@pytest.mark.asyncio
async def test_runtime_builds_warmup_from_portable_task_stores_without_execution(
    tmp_path,
):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=2)
    store_root = tmp_path / "base-artifacts"
    for position, task in enumerate(sequence.tasks[:2]):
        source = DiffusionStore(tmp_path / f"source-{position}")
        source.store_artifact(
            DiffusionArtifact(
                artifact_id=f"artifact-{position}",
                source_task_id=task.task_id,
                source_iteration=0,
                source_run_id=f"base-run-{position}",
                artifact_type=DiffusionArtifactType.RUN_OUTCOME,
                risk_level=DiffusionRiskLevel.LOW,
                content=f"stored outcome {position}",
                verifier_reward=1.0,
            )
        )
        source.save_artifact_store(store_root / task.task_id, store_id=task.task_id)

    runtime, orchestrator = _runtime(
        workspace=sequence_dir / "warmup" / "warmup-run",
        sequence_dir=sequence_dir,
        run_id="warmup-run",
    )
    bundle = await runtime.prepare_warmup_from_stores(
        sequence,
        artifact_store_root=store_root,
    )

    assert orchestrator.execution_calls == []
    assert [record.execution for record in bundle.task_records] == [None, None]
    assert [record.artifact_store_id for record in bundle.task_records] == [
        "task-0",
        "task-1",
    ]
    assert [artifact.source_iteration for artifact in bundle.final_artifact_bank] == [
        0,
        1,
    ]
    assert {artifact.source_run_id for artifact in bundle.final_artifact_bank} == {
        "warmup-run"
    }
    assert (
        load_warmup_bundle(
            sequence_dir / "warmup" / "warmup-run" / "warmup_bundle.json"
        )
        == bundle
    )


@pytest.mark.asyncio
async def test_runtime_e2e_reuses_one_portable_warmup_without_copying_harbor(tmp_path):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence()
    warmup_runtime, warmup_orchestrator = _runtime(
        workspace=sequence_dir / "warmup" / "warmup-run",
        sequence_dir=sequence_dir,
        run_id="warmup-run",
    )

    bundle = await warmup_runtime.prepare_warmup(sequence)

    warmup_terminal = sequence_dir / "warmup" / "warmup-run" / "warmup_bundle.json"
    loaded_bundle = load_warmup_bundle(warmup_terminal)
    assert loaded_bundle == bundle
    warmup_paths = {entry.relative_path for entry in bundle.archive_manifest.entries}
    assert "sequence_spec.json" in warmup_paths
    assert "warmup/warmup-run/journal/position-0000.json" in warmup_paths
    assert "warmup/warmup-run/jobs/position-0000/evidence.json" in warmup_paths
    assert warmup_orchestrator.execution_calls[0][1].policy_name == "none"
    assert str((sequence_dir / "warmup" / "warmup-run").resolve()) not in (
        warmup_terminal.read_text(encoding="utf-8")
    )

    spec = SampleSpec(
        sample_id="sample-random",
        sequence=sequence,
        arm=OrchestrationArm.RANDOM_POLICY,
        warmup_bundle_id=bundle.bundle_id,
    )
    sample_runtime, sample_orchestrator = _runtime(
        workspace=sequence_dir / "samples" / spec.sample_id,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    completed_positions: list[int] = []

    def on_position_complete(journal: PositionJournal) -> None:
        completed_positions.append(journal.position)

    result = await sample_runtime.run(
        spec,
        warmup=bundle,
        on_position_complete=on_position_complete,
    )

    sample_terminal = sequence_dir / "samples" / spec.sample_id / "sample_result.json"
    assert load_sample_result(sample_terminal) == result
    sample_paths = {entry.relative_path for entry in result.archive_manifest.entries}
    assert "sequence_spec.json" in sample_paths
    assert "samples/sample-random/warmup_ref.json" in sample_paths
    assert "samples/sample-random/jobs/position-0001/evidence.json" in sample_paths
    assert not (
        sequence_dir / "samples" / spec.sample_id / "jobs" / "position-0000"
    ).exists()
    assert len(sample_orchestrator.execution_calls) == 2
    assert completed_positions == [1, 2]
    assert all(
        context.policy_name == "random_uniform"
        for _, context in sample_orchestrator.execution_calls
    )
    assert str((sequence_dir / "samples" / spec.sample_id).resolve()) not in (
        sample_terminal.read_text(encoding="utf-8")
    )
    assert result.rewards.valid_for_reporting is True

    tampered = json.loads(sample_terminal.read_text(encoding="utf-8"))
    tampered["final_artifact_bank"][0]["content"] = "changed shared prefix"
    sample_terminal.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="bank|prefix|transition"):
        load_sample_result(sample_terminal)

    with pytest.raises(RuntimeError, match="single-use"):
        await sample_runtime.run(spec, warmup=bundle)


@dataclass
class _FailingRunner:
    stage: FailureStage
    calls: int = 0

    async def run(
        self,
        spec: SampleSpec,
        *,
        warmup: Any = None,
        on_position_complete: Any = None,
    ) -> Any:
        del warmup, on_position_complete
        self.calls += 1
        position = spec.sequence.warmup_count
        raise SampleRunError(
            stage=self.stage,
            position=position,
            task_id=spec.sequence.tasks[position].task_id,
            progress=RunProgress(
                run_id=spec.sample_id,
                sequence_id=spec.sequence.sequence_id,
                sample_id=spec.sample_id,
            ),
            cause=RuntimeError(f"{self.stage.value} failed with token=super-secret"),
        )


@dataclass
class _MalformedExternalRefExecutionAgent:
    async def execute(self, request: TaskExecutionRequest) -> TaskExecutionResult:
        valid = TaskExecutionResult(
            run_id=request.run_id,
            position=request.position,
            task_id=request.task.task_id,
            record=IterationRecord(
                iteration=request.position,
                task_id=request.task.task_id,
                reward=1.0,
            ),
            metadata={"phase": request.phase},
        )
        return valid.model_copy(
            update={
                "metadata": {
                    "phase": request.phase,
                    "arm": request.arm,
                    "external_archive_refs": (
                        {"kind": "remote", "uri": "relative/not-portable"},
                    ),
                }
            }
        )


@pytest.mark.asyncio
async def test_invalid_external_provenance_fails_before_journal_and_seals_failure(
    tmp_path,
):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-invalid-external-ref",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, _ = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    runtime.runner._execution_agent = _MalformedExternalRefExecutionAgent()

    with pytest.raises(SampleRunError) as raised:
        await runtime.run(spec)

    assert raised.value.stage is FailureStage.EXECUTE
    assert not list((workspace / "journal").glob("*.json"))
    assert (workspace / "sample_failure.json").is_file()
    assert (workspace / "archive_manifest.json").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    [
        FailureStage.RESOLVE,
        FailureStage.GRAPH,
        FailureStage.POLICY,
        FailureStage.PACK,
        FailureStage.EXECUTE,
        FailureStage.PROJECT,
        FailureStage.PERSIST,
        FailureStage.FINALIZE,
    ],
)
async def test_runtime_persists_each_staged_failure_without_success_terminal(
    tmp_path,
    stage: FailureStage,
):
    sequence_dir = tmp_path / stage.value
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id=f"sample-{stage.value}",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    failing = _FailingRunner(stage)
    runtime.runner = cast(Any, failing)

    with pytest.raises(SampleRunError) as raised:
        await runtime.run(spec)

    assert raised.value.stage is stage
    failure_path = workspace / "sample_failure.json"
    failure = FailureRecord.model_validate_json(
        failure_path.read_text(encoding="utf-8")
    )
    assert failure.stage is stage
    assert failure.message == f"{stage.value} failed with token=[redacted]"
    assert not (workspace / "sample_result.json").exists()
    assert (workspace / "archive_manifest.json").is_file()
    assert failing.calls == 1
    assert orchestrator.execution_calls == []


@pytest.mark.asyncio
async def test_secondary_manifest_error_cannot_replace_staged_sample_failure(
    tmp_path,
    monkeypatch,
):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-failure-manifest-fallback",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, _ = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    runtime.runner = cast(Any, _FailingRunner(FailureStage.EXECUTE))

    def fail_manifest(self, workspace, *, records):
        del self, workspace, records
        raise ValueError("malformed completed-journal provenance")

    monkeypatch.setattr(SampleRuntime, "_manifest", fail_manifest)

    with pytest.raises(SampleRunError) as raised:
        await runtime.run(spec)

    assert raised.value.stage is FailureStage.EXECUTE
    assert (workspace / "sample_failure.json").is_file()
    assert (workspace / "archive_manifest.json").is_file()


@dataclass
class _FailingWarmupRunner:
    calls: int = 0

    async def prepare_warmup(
        self,
        sequence: SequenceSpec,
        *,
        warmup_run_id: str,
        on_position_complete: Any = None,
    ) -> Any:
        del on_position_complete
        self.calls += 1
        raise SampleRunError(
            stage=FailureStage.EXECUTE,
            position=0,
            task_id=sequence.tasks[0].task_id,
            progress=RunProgress(
                run_id=warmup_run_id,
                sequence_id=sequence.sequence_id,
            ),
            cause=RuntimeError("warm-up failed with api_key=super-secret"),
        )


@pytest.mark.asyncio
async def test_runtime_persists_warmup_failure_without_success_terminal(tmp_path):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence()
    workspace = sequence_dir / "warmup" / "warmup-failure"
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id="warmup-failure",
    )
    failing = _FailingWarmupRunner()
    runtime.runner = cast(Any, failing)

    with pytest.raises(SampleRunError) as raised:
        await runtime.prepare_warmup(sequence)

    assert raised.value.stage is FailureStage.EXECUTE
    failure = FailureRecord.model_validate_json(
        (workspace / "warmup_failure.json").read_text(encoding="utf-8")
    )
    assert failure.stage is FailureStage.EXECUTE
    assert failure.message == "warm-up failed with api_key=[redacted]"
    assert not (workspace / "warmup_bundle.json").exists()
    assert (workspace / "archive_manifest.json").is_file()
    assert failing.calls == 1
    assert orchestrator.execution_calls == []


@pytest.mark.asyncio
async def test_transfer_materialization_rolls_back_files_when_store_raises_late(
    tmp_path,
    monkeypatch,
):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=2)
    warmup_runtime, _ = _runtime(
        workspace=sequence_dir / "warmup" / "warmup-run",
        sequence_dir=sequence_dir,
        run_id="warmup-run",
    )
    bundle = await warmup_runtime.prepare_warmup(sequence)
    spec = SampleSpec(
        sample_id="sample-rollback",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
        warmup_bundle_id=bundle.bundle_id,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, _ = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    original = runtime.diffusion_store.store_artifact
    calls = 0

    def write_then_raise(artifact):
        nonlocal calls
        calls += 1
        path = original(artifact)
        if calls == 2:
            raise RuntimeError("late persistence failure")
        return path

    monkeypatch.setattr(runtime.diffusion_store, "store_artifact", write_then_raise)

    with pytest.raises(SampleRunError) as raised:
        await runtime.run(spec, warmup=bundle)

    assert raised.value.stage is FailureStage.PERSIST
    artifact_dir = workspace / "diffusion" / "artifacts"
    assert not list(artifact_dir.glob("*.json"))
    assert (workspace / "sample_failure.json").is_file()


@pytest.mark.asyncio
async def test_finalize_late_write_cleans_success_before_failure_terminal(
    tmp_path,
    monkeypatch,
):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-finalize",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, _ = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    original = sample_runtime_module.write_model_atomic

    def write_then_raise(path, model, *, exists_error_prefix="Archive model"):
        result = original(
            path,
            model,
            exists_error_prefix=exists_error_prefix,
        )
        if path.name == "sample_result.json":
            raise RuntimeError("terminal writer raised after publication")
        return result

    monkeypatch.setattr(
        sample_runtime_module,
        "write_model_atomic",
        write_then_raise,
    )

    with pytest.raises(SampleRunError) as raised:
        await runtime.run(spec)

    assert raised.value.stage is FailureStage.FINALIZE
    assert not (workspace / "sample_result.json").exists()
    assert (workspace / "sample_failure.json").is_file()
    assert (workspace / "archive_manifest.json").is_file()


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "..", "bad\\run"])
def test_runtime_rejects_nonportable_run_ids(tmp_path, run_id: str):
    with pytest.raises(ValueError, match="path component"):
        _runtime(
            workspace=tmp_path / "workspace",
            sequence_dir=tmp_path / "sequence",
            run_id=run_id,
        )


def test_provenance_config_hash_redacts_credential_values(tmp_path):
    first, first_orchestrator = _runtime(
        workspace=tmp_path / "one" / "samples" / "sample-1",
        sequence_dir=tmp_path / "one",
        run_id="sample-1",
    )
    first_orchestrator.config.executor_runtime.agent_env = {
        "OPENAI_API_KEY": "first-secret",
        "PUBLIC_SETTING": "stable",
    }
    second, second_orchestrator = _runtime(
        workspace=tmp_path / "two" / "samples" / "sample-1",
        sequence_dir=tmp_path / "two",
        run_id="sample-1",
    )
    second_orchestrator.config.executor_runtime.agent_env = {
        "OPENAI_API_KEY": "second-secret",
        "PUBLIC_SETTING": "stable",
    }
    third, third_orchestrator = _runtime(
        workspace=tmp_path / "three" / "samples" / "sample-1",
        sequence_dir=tmp_path / "three",
        run_id="sample-1",
    )
    third_orchestrator.config.executor_runtime.agent_env = {
        "OPENAI_API_KEY": "third-secret",
        "PUBLIC_SETTING": "changed",
    }
    started_at = datetime.now(UTC)

    first_hash = first._provenance(started_at=started_at).config_hash
    second_hash = second._provenance(started_at=started_at).config_hash
    third_hash = third._provenance(started_at=started_at).config_hash

    assert first_hash == second_hash
    assert third_hash != first_hash


@pytest.mark.asyncio
async def test_runtime_rejects_existing_state_before_any_agent_call(tmp_path):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-stale",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    (workspace / "metrics.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fresh Orchestrator/workspace"):
        await runtime.run(spec)

    assert orchestrator.execution_calls == []
    assert not (workspace / "sample_failure.json").exists()


@pytest.mark.asyncio
async def test_runtime_rejects_executor_workspace_outside_claimed_run(tmp_path):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-layout",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    orchestrator.executor = SimpleNamespace(_workspace_root=tmp_path / "outside")

    with pytest.raises(ValueError, match="outside its claimed workspace"):
        await runtime.run(spec)

    assert orchestrator.execution_calls == []


@pytest.mark.asyncio
async def test_sample_spec_persist_failure_writes_terminal_failure(
    tmp_path,
    monkeypatch,
):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-bootstrap-persist",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    original = sample_runtime_module.write_model_atomic

    def fail_sample_spec(path, model, *, exists_error_prefix="Archive model"):
        if path.name == "sample_spec.json":
            raise OSError("sample spec persistence failed")
        return original(path, model, exists_error_prefix=exists_error_prefix)

    monkeypatch.setattr(sample_runtime_module, "write_model_atomic", fail_sample_spec)

    with pytest.raises(SampleRunError) as raised:
        await runtime.run(spec)

    assert raised.value.stage is FailureStage.PERSIST
    assert (workspace / "sample_failure.json").is_file()
    assert orchestrator.execution_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relative",
    [
        "artifacts/validation/result.json",
        "artifacts/candidate_batches/batch.json",
        "artifacts/skill_updates/update.json",
        "skills_snapshots/position-0000.json",
        "benchmarks/task/output.txt",
        "history/history.jsonl",
        "diffusion/artifacts/stale.json",
    ],
)
async def test_runtime_rejects_every_durable_output_surface(tmp_path, relative: str):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-stale",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    stale = workspace / relative
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("stale\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fresh Orchestrator/workspace"):
        await runtime.run(spec)

    assert orchestrator.execution_calls == []


@pytest.mark.asyncio
async def test_runtime_rejects_preloaded_or_explicit_in_memory_state(tmp_path):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-stale",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    orchestrator.preloaded_diffusion_artifact_store_path = "/tmp/old-store"
    orchestrator.preloaded_diffusion_artifact_store_count = 1
    orchestrator.freeze_diffusion_artifact_store = True
    orchestrator._explicit_execution_provenance_by_key = {
        ("task-0", 0): {"judge_reward": 1.0}
    }

    with pytest.raises(ValueError, match="fresh Orchestrator/workspace"):
        await runtime.run(spec)

    assert orchestrator.execution_calls == []


@pytest.mark.asyncio
async def test_runtime_rejects_a_diffusion_store_outside_the_claimed_workspace(
    tmp_path,
):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-layout",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    orchestrator = _RuntimeOrchestrator(workspace)
    runtime = SampleRuntime(
        orchestrator=cast(Any, orchestrator),
        run_id=spec.sample_id,
        sequence_dir=sequence_dir,
        implementation_revision="abc123",
        implementation_dirty=False,
        runner=_runtime_runner(orchestrator),
        diffusion_store=DiffusionStore(tmp_path / "outside-diffusion"),
    )

    with pytest.raises(ValueError, match="outside its claimed workspace"):
        await runtime.run(spec)

    assert orchestrator.execution_calls == []


@pytest.mark.asyncio
async def test_runtime_rejects_harbor_jobs_outside_claimed_workspace(tmp_path):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-jobs-layout",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    orchestrator.executor = SimpleNamespace(
        _workspace_root=workspace / "benchmarks",
        _harbor_runner=SimpleNamespace(jobs_dir=tmp_path / "outside-jobs"),
    )

    with pytest.raises(ValueError, match="outside its claimed workspace"):
        await runtime.run(spec)

    assert orchestrator.execution_calls == []


@pytest.mark.asyncio
async def test_runtime_rejects_skill_store_outside_claimed_workspace(tmp_path):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-skills-layout",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    orchestrator.skill_store = SimpleNamespace(_skills_dir=tmp_path / "outside-skills")

    with pytest.raises(ValueError, match="outside its claimed workspace"):
        await runtime.run(spec)

    assert orchestrator.execution_calls == []


@pytest.mark.asyncio
async def test_runtime_rejects_sensitive_paths_before_agent_calls(tmp_path):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-sensitive-path",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    (workspace / ".env.local").write_text(
        "OPENAI_API_KEY=must-not-run\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fresh Orchestrator/workspace"):
        await runtime.run(spec)

    assert orchestrator.execution_calls == []
    assert not (workspace / "sample_failure.json").exists()


@pytest.mark.asyncio
async def test_runtime_redacts_config_and_host_secrets_from_success_archive(
    tmp_path,
    monkeypatch,
):
    secret = "sk-host-success-secret"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-redacted-config",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    orchestrator.config.executor_runtime.agent_env = {
        "OPENAI_API_KEY": secret,
        "PUBLIC_SETTING": "stable",
    }
    (workspace / "config.toml").write_text(
        f'OPENAI_API_KEY = "{secret}"\n',
        encoding="utf-8",
    )

    await runtime.run(spec)

    for path in workspace.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), path


@pytest.mark.asyncio
async def test_runtime_reloads_sanitized_journals_before_success_terminal(
    tmp_path,
):
    secret = "config-only-neutral-secret"
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-sanitized-terminal",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    orchestrator.config.executor_runtime.agent_env = {"OPENAI_API_KEY": secret}
    original = orchestrator.execute_task_with_context

    async def execute_with_echo(**kwargs):
        record = await original(**kwargs)
        trace = record.execution_trace
        assert trace is not None
        return record.model_copy(
            update={
                "execution_trace": trace.model_copy(
                    update={"stdout": f"diagnostic {secret}"}
                )
            }
        )

    orchestrator.execute_task_with_context = execute_with_echo

    result = await runtime.run(spec)

    assert load_sample_result(workspace / "sample_result.json") == result
    for path in workspace.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), path


@pytest.mark.asyncio
async def test_redacted_executor_output_remains_stable_across_archive_round_trips(
    tmp_path,
):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-idempotent-redaction",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    original = orchestrator.execute_task_with_context

    async def execute_with_credentials(**kwargs):
        record = await original(**kwargs)
        trace = record.execution_trace
        assert trace is not None
        return record.model_copy(
            update={
                "execution_trace": trace.model_copy(
                    update={
                        "stdout": ("Bearer one-time-secret token=assignment-secret")
                    }
                )
            }
        )

    orchestrator.execute_task_with_context = execute_with_credentials

    result = await runtime.run(spec)
    loaded = load_sample_result(workspace / "sample_result.json")

    assert loaded == result
    encoded = loaded.model_dump_json()
    assert "one-time-secret" not in encoded
    assert "assignment-secret" not in encoded
    assert "[redacted]]" not in encoded


@pytest.mark.asyncio
async def test_archive_sanitizer_cannot_remove_declared_path_and_publish_success(
    tmp_path,
):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-sensitive-output",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    original = orchestrator.execute_task_with_context

    async def execute_with_sensitive_file(**kwargs):
        record = await original(**kwargs)
        position = kwargs["position"]
        sensitive = workspace / "jobs" / f"position-{position:04d}" / ".env"
        sensitive.write_text("OPENAI_API_KEY=must-not-archive\n", encoding="utf-8")
        return record

    orchestrator.execute_task_with_context = execute_with_sensitive_file

    with pytest.raises(SampleRunError) as raised:
        await runtime.run(spec)

    assert raised.value.stage is FailureStage.FINALIZE
    assert not (workspace / "sample_result.json").exists()
    assert (workspace / "sample_failure.json").is_file()


@pytest.mark.asyncio
async def test_executor_dot_tmp_evidence_is_redacted_and_manifested(
    tmp_path,
):
    secret = "config-dot-tmp-secret"
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-dot-tmp-evidence",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    orchestrator.config.executor_runtime.agent_env = {"OPENAI_API_KEY": secret}
    original = orchestrator.execute_task_with_context

    async def execute_with_dot_tmp(**kwargs):
        record = await original(**kwargs)
        position = kwargs["position"]
        evidence = workspace / "jobs" / f"position-{position:04d}" / ".evidence.tmp"
        evidence.write_text(f"diagnostic {secret}\n", encoding="utf-8")
        return record

    orchestrator.execute_task_with_context = execute_with_dot_tmp

    result = await runtime.run(spec)

    manifest_paths = {entry.relative_path for entry in result.archive_manifest.entries}
    expected = "samples/sample-dot-tmp-evidence/jobs/position-0000/.evidence.tmp"
    assert expected in manifest_paths
    assert secret not in (
        workspace / "jobs" / "position-0000" / ".evidence.tmp"
    ).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_nonportable_executor_filename_fails_before_journal_and_is_quarantined(
    tmp_path,
):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-nonportable-output",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    original = orchestrator.execute_task_with_context

    async def execute_with_nonportable_file(**kwargs):
        record = await original(**kwargs)
        position = kwargs["position"]
        invalid = workspace / "jobs" / f"position-{position:04d}" / "bad\\name.txt"
        invalid.write_text("non-portable filename\n", encoding="utf-8")
        return record

    orchestrator.execute_task_with_context = execute_with_nonportable_file

    with pytest.raises(SampleRunError) as raised:
        await runtime.run(spec)

    assert raised.value.stage is FailureStage.PERSIST
    assert not list((workspace / "journal").glob("*.json"))
    assert not (workspace / "jobs" / "position-0000" / "bad\\name.txt").exists()
    assert (workspace / "sample_failure.json").is_file()
    assert (workspace / "archive_manifest.json").is_file()


@pytest.mark.asyncio
async def test_host_credential_in_executor_filename_fails_without_archive_leak(
    tmp_path,
    monkeypatch,
):
    secret = "host-path-secret"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-secret-filename",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    original = orchestrator.execute_task_with_context

    async def execute_with_secret_filename(**kwargs):
        record = await original(**kwargs)
        position = kwargs["position"]
        output = workspace / "jobs" / f"position-{position:04d}" / f"{secret}.log"
        output.write_text(f"diagnostic {secret}\n", encoding="utf-8")
        return record

    orchestrator.execute_task_with_context = execute_with_secret_filename

    with pytest.raises(SampleRunError) as raised:
        await runtime.run(spec)

    assert raised.value.stage is FailureStage.PERSIST
    assert secret not in str(raised.value)
    assert not list((workspace / "journal").glob("*.json"))
    assert (workspace / "sample_failure.json").is_file()
    assert (workspace / "archive_manifest.json").is_file()
    for path in workspace.rglob("*"):
        assert secret not in path.relative_to(workspace).as_posix()
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), path


@pytest.mark.asyncio
async def test_failure_archive_has_provenance_and_no_host_secret(
    tmp_path,
    monkeypatch,
):
    secret = "super-secret"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-redacted-failure",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, orchestrator = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    orchestrator.config.executor_runtime.agent_env = {"OPENAI_API_KEY": secret}
    (workspace / "config.toml").write_text(
        f'OPENAI_API_KEY = "{secret}"\n',
        encoding="utf-8",
    )
    runtime.runner = cast(Any, _FailingRunner(FailureStage.EXECUTE))

    with pytest.raises(SampleRunError):
        await runtime.run(spec)

    failure = FailureRecord.model_validate_json(
        (workspace / "sample_failure.json").read_text(encoding="utf-8")
    )
    assert failure.provenance.implementation_revision == "abc123"
    assert failure.provenance.executor_backend
    for path in workspace.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), path


@pytest.mark.asyncio
async def test_sequence_accepts_only_one_successful_shared_warmup(tmp_path):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence()
    first, _ = _runtime(
        workspace=sequence_dir / "warmup" / "warmup-a",
        sequence_dir=sequence_dir,
        run_id="warmup-a",
    )
    await first.prepare_warmup(sequence)
    second, second_orchestrator = _runtime(
        workspace=sequence_dir / "warmup" / "warmup-b",
        sequence_dir=sequence_dir,
        run_id="warmup-b",
    )

    with pytest.raises(ValueError, match="already has a successful shared warm-up"):
        await second.prepare_warmup(sequence)

    assert second_orchestrator.execution_calls == []


@pytest.mark.asyncio
async def test_success_terminal_digest_rejects_provenance_tampering(tmp_path):
    sequence_dir = tmp_path / "sequence"
    sequence = _runtime_sequence(warmup_count=0)
    spec = SampleSpec(
        sample_id="sample-terminal-digest",
        sequence=sequence,
        arm=OrchestrationArm.EXECUTION_ONLY,
    )
    workspace = sequence_dir / "samples" / spec.sample_id
    runtime, _ = _runtime(
        workspace=workspace,
        sequence_dir=sequence_dir,
        run_id=spec.sample_id,
    )
    await runtime.run(spec)
    terminal = workspace / "sample_result.json"
    payload = json.loads(terminal.read_text(encoding="utf-8"))
    payload["provenance"]["implementation_revision"] = "forged"
    terminal.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        load_sample_result(terminal)


def test_build_sample_runtime_wires_direct_agents_without_invocation(tmp_path):
    orchestrator, _, _ = _orchestrator(tmp_path, "no_feedback")
    orchestrator.config.experiment.skill_updates.executor = False
    orchestrator.config.experiment.skill_updates.planner = False
    orchestrator.config.experiment.skill_updates.mediator = False

    runtime = build_sample_runtime(
        orchestrator=orchestrator,
        run_id="sample-builder",
        sequence_dir=tmp_path / "sequence",
        implementation_revision="abc123",
    )

    assert runtime.run_id == "sample-builder"
    assert type(runtime.runner._graph_agent).__name__ == "LangChainTaskGraphAdapter"
    assert type(runtime.runner._diffusion_policy_agent).__name__ == (
        "LangChainDiffusionPolicyAdapter"
    )
    assert type(runtime.runner._random_policy_agent).__name__ == "RandomPolicyAgent"


def test_build_sample_runtime_rejects_legacy_baseline_overlay(tmp_path):
    orchestrator, _, _ = _orchestrator(tmp_path, "no_feedback")
    orchestrator.config.experiment.skill_updates.executor = False
    orchestrator.config.experiment.skill_updates.planner = False
    orchestrator.config.experiment.skill_updates.mediator = False
    orchestrator.config.experiment.baseline_preset = "skill_all_diffusion_none"

    with pytest.raises(ValueError, match="legacy baseline preset"):
        build_sample_runtime(
            orchestrator=orchestrator,
            run_id="sample-builder",
            sequence_dir=tmp_path / "sequence",
            implementation_revision="abc123",
        )
