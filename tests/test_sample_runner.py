from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from mediated_coevo.artifacts.models import ArtifactBankUpdate
from mediated_coevo.diffusion.models import (
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    TaskGraphSnapshot,
)
from mediated_coevo.diffusion.policy import DiffusionSubscription
from mediated_coevo.execution.models import (
    ContextPack,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskProfile,
)
from mediated_coevo.experiment.sample_models import (
    ArchiveManifest,
    FailureStage,
    OrchestrationArm,
    PositionJournal,
    RuntimeProvenance,
    SampleRunError,
    SampleSpec,
    SequenceSpec,
    WarmupBundle,
)
from mediated_coevo.experiment.sample_runner import SampleRunner
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.orchestration.contracts import (
    GraphAgentRequest,
    GraphAgentResponse,
    PolicyAgentRequest,
    PolicyAgentResponse,
)


@dataclass
class _GraphAgent:
    events: list[str]
    fail: bool = False
    wrong_run: bool = False
    calls: list[GraphAgentRequest] = field(default_factory=list)

    async def update(self, request: GraphAgentRequest) -> GraphAgentResponse:
        self.events.append(f"graph:{request.position}")
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("graph exploded")
        task_ids = (
            list(request.previous_graph.task_ids) if request.previous_graph else []
        )
        for artifact in request.artifacts:
            if artifact.source_task_id not in task_ids:
                task_ids.append(artifact.source_task_id)
        if request.task.task_id not in task_ids:
            task_ids.append(request.task.task_id)
        return GraphAgentResponse(
            snapshot=TaskGraphSnapshot(
                run_id="wrong-run" if self.wrong_run else request.run_id,
                iteration=request.position,
                task_ids=task_ids,
                graph_policy="fake_graph",
                metadata={"current_node_id": request.task.task_id},
            ),
            raw_decision={"node_id": request.task.task_id},
        )


@dataclass
class _PolicyAgent:
    events: list[str]
    policy_name: str
    fail: bool = False
    ghost: DiffusionArtifact | None = None
    duplicate: bool = False
    calls: list[PolicyAgentRequest] = field(default_factory=list)

    async def select(self, request: PolicyAgentRequest) -> PolicyAgentResponse:
        self.events.append(f"{self.policy_name}:{request.position}")
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("policy exploded")
        selected = (self.ghost,) if self.ghost is not None else request.artifacts[-1:]
        subscriptions = tuple(
            DiffusionSubscription(
                artifact=artifact,
                policy_name=self.policy_name,
                relation="test_selection",
                reason="selected by fake policy",
            )
            for artifact in selected
        )
        if self.duplicate and subscriptions:
            subscriptions = (*subscriptions, subscriptions[0])
        return PolicyAgentResponse(
            policy_name=self.policy_name,
            subscriptions=subscriptions,
            raw_decision={
                "selected_artifact_ids": tuple(
                    artifact.artifact_id for artifact in selected
                )
            },
        )


@dataclass
class _ContextPacker:
    events: list[str]
    fail: bool = False
    wrong_candidates: bool = False
    mutate_inputs: bool = False
    calls: list[tuple[TaskProfile, PolicyAgentResponse]] = field(default_factory=list)

    async def pack(
        self,
        *,
        run_id: str,
        position: int,
        task: TaskProfile,
        graph: TaskGraphSnapshot | None,
        policy: PolicyAgentResponse,
        eligible_artifacts: tuple[DiffusionArtifact, ...],
    ) -> ContextPack:
        del run_id
        self.events.append(f"pack:{position}")
        self.calls.append((task, policy))
        if self.fail:
            raise RuntimeError("packer exploded")
        if self.mutate_inputs:
            if graph is not None:
                graph.task_ids.append("packer-corruption")
            if eligible_artifacts:
                eligible_artifacts[0].content = "packer-corruption"
            if policy.subscriptions:
                policy.subscriptions[0].artifact.content = "packer-corruption"
        selected_ids = policy.selected_artifact_ids
        rendered = selected_ids
        selected_by_id = {
            subscription.artifact.artifact_id: subscription.artifact
            for subscription in policy.subscriptions
        }
        source_ids = tuple(
            dict.fromkeys(
                selected_by_id[artifact_id].source_task_id for artifact_id in rendered
            )
        )
        eligible_ids = tuple(artifact.artifact_id for artifact in eligible_artifacts)
        if self.wrong_candidates:
            eligible_ids = eligible_ids[:-1]
        return ContextPack(
            text=f"context for {task.task_id}" if rendered else None,
            eligible_artifact_ids=eligible_ids,
            selected_artifact_ids=selected_ids,
            rendered_artifact_ids=rendered,
            source_task_ids=source_ids,
            snapshot_id=graph.snapshot_id if graph else None,
            policy_name=policy.policy_name,
            token_count=1 if rendered else 0,
        )


@dataclass
class _ExecutionAgent:
    events: list[str]
    fail: bool = False
    infrastructure_failure: bool = False
    missing_reward_positions: set[int] = field(default_factory=set)
    calls: list[TaskExecutionRequest] = field(default_factory=list)

    async def execute(self, request: TaskExecutionRequest) -> TaskExecutionResult:
        self.events.append(f"execute:{request.position}")
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("executor exploded")
        trace = None
        if self.infrastructure_failure:
            trace = ExecutionTrace(
                task_id=request.task.task_id,
                iteration=request.position,
                status="env_failure",
                reward=None,
            )
        return TaskExecutionResult(
            run_id=request.run_id,
            position=request.position,
            task_id=request.task.task_id,
            record=IterationRecord(
                iteration=request.position,
                task_id=request.task.task_id,
                reward=(
                    None
                    if request.position in self.missing_reward_positions
                    else 0.0
                    if request.position == 0
                    else float(request.position)
                ),
                execution_trace=trace,
            ),
            archive_paths=(f"jobs/position-{request.position:04d}.json",),
        )


@dataclass
class _Projector:
    events: list[str]
    fail: bool = False
    calls: list[tuple[TaskProfile, TaskExecutionResult]] = field(default_factory=list)

    async def project(
        self,
        *,
        task: TaskProfile,
        execution: TaskExecutionResult,
    ) -> tuple[DiffusionArtifact, ...]:
        self.events.append(f"project:{execution.position}")
        self.calls.append((task, execution))
        if self.fail:
            raise RuntimeError("projector exploded")
        return (
            DiffusionArtifact(
                artifact_id=f"artifact-{execution.run_id}-{execution.position}",
                source_task_id=task.task_id,
                source_iteration=execution.position,
                source_run_id=execution.run_id,
                artifact_type=DiffusionArtifactType.RUN_OUTCOME,
                risk_level=DiffusionRiskLevel.LOW,
                content=f"outcome for {task.task_id}",
                verifier_reward=execution.reward,
            ),
        )


@dataclass
class _BankUpdater:
    events: list[str]
    fail_prepare: bool = False
    fail_persist: bool = False
    fail_rollback: bool = False
    mutate_current_bank: bool = False
    mutate_persist_input: bool = False
    persisted: list[ArtifactBankUpdate] = field(default_factory=list)
    rolled_back: list[tuple[Path, ...]] = field(default_factory=list)

    def prepare(
        self,
        *,
        run_id: str,
        position: int,
        task: TaskProfile,
        execution: TaskExecutionResult,
        current_bank: tuple[DiffusionArtifact, ...],
        projected_artifacts: tuple[DiffusionArtifact, ...],
    ) -> ArtifactBankUpdate:
        del execution
        self.events.append(f"prepare:{position}")
        if self.fail_prepare:
            raise RuntimeError("transition invalid")
        before = tuple(artifact.artifact_id for artifact in current_bank)
        if self.mutate_current_bank and current_bank:
            current_bank[0].content = "updater-corruption"
        return ArtifactBankUpdate(
            run_id=run_id,
            position=position,
            task_id=task.task_id,
            before_artifact_ids=before,
            added_artifacts=projected_artifacts,
            after_artifact_ids=(
                *before,
                *(artifact.artifact_id for artifact in projected_artifacts),
            ),
        )

    def persist(self, update: ArtifactBankUpdate) -> tuple[Path, ...]:
        self.events.append(f"persist:{update.position}")
        if self.fail_persist:
            raise RuntimeError("persistence exploded")
        if self.mutate_persist_input and update.added_artifacts:
            update.added_artifacts[0].content = "persist-corruption"
        self.persisted.append(update)
        return tuple(
            Path("artifacts") / f"{artifact.artifact_id}.json"
            for artifact in update.added_artifacts
        )

    def rollback(self, paths: tuple[Path, ...]) -> None:
        self.events.append("rollback")
        self.rolled_back.append(paths)
        if self.fail_rollback:
            raise OSError("rollback exploded")


@dataclass
class _Components:
    runner: SampleRunner
    graph: _GraphAgent
    diffusion: _PolicyAgent
    random: _PolicyAgent
    packer: _ContextPacker
    executor: _ExecutionAgent
    projector: _Projector
    updater: _BankUpdater
    events: list[str]


def _components() -> _Components:
    events: list[str] = []
    graph = _GraphAgent(events)
    diffusion = _PolicyAgent(events, "diffusion_policy")
    random_policy = _PolicyAgent(events, "random_uniform")
    packer = _ContextPacker(events)
    executor = _ExecutionAgent(events)
    projector = _Projector(events)
    updater = _BankUpdater(events)
    return _Components(
        runner=SampleRunner(
            graph_agent=graph,
            diffusion_policy_agent=diffusion,
            random_policy_agent=random_policy,
            context_packer=packer,
            execution_agent=executor,
            artifact_projector=projector,
            artifact_bank_updater=updater,
        ),
        graph=graph,
        diffusion=diffusion,
        random=random_policy,
        packer=packer,
        executor=executor,
        projector=projector,
        updater=updater,
        events=events,
    )


def _sequence(*, warmup_count: int = 2) -> SequenceSpec:
    return SequenceSpec(
        sequence_id="sequence-1",
        tasks=tuple(
            TaskProfile(
                task_id=f"task-{position}",
                instruction=f"execute task {position}",
                task_config={"position": position},
            )
            for position in range(5)
        ),
        warmup_count=warmup_count,
        policy_seed=71,
    )


def _provenance() -> RuntimeProvenance:
    now = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
    return RuntimeProvenance(
        implementation_revision="abc123",
        implementation_dirty=False,
        config_hash="1" * 64,
        model_mapping={},
        executor_backend="fake",
        executor_agent="fake",
        python_version="3.13",
        package_version="0.1.0",
        started_at=now,
        finished_at=now,
    )


async def _bundle(
    components: _Components,
    sequence: SequenceSpec,
) -> WarmupBundle:
    warmup = await components.runner.prepare_warmup(
        sequence,
        warmup_run_id="warmup-run-1",
    )
    return WarmupBundle.create(
        sequence_id=sequence.sequence_id,
        warmup_run_id=warmup.warmup_run_id,
        warmup_count=sequence.warmup_count,
        task_records=warmup.task_records,
        final_artifact_bank=warmup.final_artifact_bank,
        archive_manifest=ArchiveManifest(),
        provenance=_provenance(),
    )


def _spec(
    sequence: SequenceSpec,
    bundle: WarmupBundle | None,
    *,
    arm: OrchestrationArm = OrchestrationArm.FULL_ORCHESTRATION,
    sample_id: str = "sample-1",
) -> SampleSpec:
    return SampleSpec(
        sample_id=sample_id,
        sequence=sequence,
        arm=arm,
        warmup_bundle_id=bundle.bundle_id if bundle else None,
    )


@pytest.mark.asyncio
async def test_warmup_is_arm_neutral_runs_once_and_suffix_sees_only_causal_bank():
    components = _components()
    sequence = _sequence()
    bundle = await _bundle(components, sequence)

    execution = await components.runner.run(_spec(sequence, bundle), warmup=bundle)

    assert [call.position for call in components.executor.calls] == list(range(5))
    assert all(call.phase == "warmup" for call in components.executor.calls[:2])
    assert all(call.arm is None for call in components.executor.calls[:2])
    assert all(
        call.context.policy_name == "none" for call in components.executor.calls[:2]
    )
    assert all("arm" not in record.model_dump() for record in bundle.task_records)
    assert [record.position for record in execution.task_records] == [2, 3, 4]
    assert components.graph.calls[0].position == 2
    assert [
        artifact.source_iteration for artifact in components.graph.calls[0].artifacts
    ] == [0, 1]
    assert [
        artifact.source_iteration for artifact in components.graph.calls[1].artifacts
    ] == [0, 1, 2]
    assert all(
        artifact.source_iteration < call.position
        for call in components.graph.calls
        for artifact in call.artifacts
    )
    assert execution.final_artifact_bank[:2] == bundle.final_artifact_bank


@pytest.mark.parametrize(
    ("arm", "graph_calls", "diffusion_calls", "random_calls", "packer_calls"),
    [
        (OrchestrationArm.EXECUTION_ONLY, 0, 0, 0, 0),
        (OrchestrationArm.RANDOM_POLICY, 0, 0, 3, 3),
        (OrchestrationArm.NO_GRAPH, 0, 3, 0, 3),
        (OrchestrationArm.FULL_ORCHESTRATION, 3, 3, 0, 3),
    ],
)
@pytest.mark.asyncio
async def test_each_arm_invokes_exactly_its_declared_components(
    arm: OrchestrationArm,
    graph_calls: int,
    diffusion_calls: int,
    random_calls: int,
    packer_calls: int,
):
    components = _components()
    sequence = _sequence()
    bundle = await _bundle(components, sequence)
    components.graph.calls.clear()
    components.diffusion.calls.clear()
    components.random.calls.clear()
    components.packer.calls.clear()

    result = await components.runner.run(
        _spec(sequence, bundle, arm=arm),
        warmup=bundle,
    )

    assert len(components.graph.calls) == graph_calls
    assert len(components.diffusion.calls) == diffusion_calls
    assert len(components.random.calls) == random_calls
    assert len(components.packer.calls) == packer_calls
    assert all(call.graph is None for call in components.random.calls)
    if arm is OrchestrationArm.NO_GRAPH:
        assert all(call.graph is None for call in components.diffusion.calls)
    assert all(
        call.policy_seed == sequence.policy_seed for call in components.random.calls
    )
    assert all(
        call.policy_seed == sequence.policy_seed for call in components.diffusion.calls
    )
    assert len(result.task_records) == sequence.suffix_count


@pytest.mark.asyncio
async def test_position_commit_order_and_journal_happen_after_artifact_persistence():
    components = _components()
    sequence = _sequence(warmup_count=0)
    journal_positions: list[int] = []

    async def journal(position: PositionJournal) -> str:
        components.events.append(f"journal:{position.position}")
        journal_positions.append(position.position)
        return f"samples/sample-1/journal/position-{position.position:04d}.json"

    execution = await components.runner.run(
        _spec(sequence, None),
        on_position_complete=journal,
    )

    assert components.events == [
        event
        for position in range(5)
        for event in (
            f"graph:{position}",
            f"diffusion_policy:{position}",
            f"pack:{position}",
            f"execute:{position}",
            f"project:{position}",
            f"prepare:{position}",
            f"persist:{position}",
            f"journal:{position}",
        )
    ]
    assert journal_positions == list(range(5))
    assert execution.completed_journal_paths[-1].endswith("position-0004.json")


@pytest.mark.parametrize(
    ("stage", "configure"),
    [
        (FailureStage.GRAPH, lambda c: setattr(c.graph, "fail", True)),
        (FailureStage.POLICY, lambda c: setattr(c.diffusion, "fail", True)),
        (FailureStage.PACK, lambda c: setattr(c.packer, "fail", True)),
        (FailureStage.EXECUTE, lambda c: setattr(c.executor, "fail", True)),
        (FailureStage.PROJECT, lambda c: setattr(c.projector, "fail", True)),
        (FailureStage.PERSIST, lambda c: setattr(c.updater, "fail_prepare", True)),
        (FailureStage.PERSIST, lambda c: setattr(c.updater, "fail_persist", True)),
    ],
)
@pytest.mark.asyncio
async def test_stage_failures_stop_the_sequence_with_last_committed_state(
    stage: FailureStage,
    configure,
):
    components = _components()
    sequence = _sequence()
    bundle = await _bundle(components, sequence)
    configure(components)

    with pytest.raises(SampleRunError) as raised:
        await components.runner.run(_spec(sequence, bundle), warmup=bundle)

    error = raised.value
    assert error.stage is stage
    assert error.position == sequence.warmup_count
    assert error.task_id == sequence.tasks[sequence.warmup_count].task_id
    assert error.progress.bank_artifact_ids == tuple(
        artifact.artifact_id for artifact in bundle.final_artifact_bank
    )
    assert error.progress.completed_positions == ()
    assert len(components.executor.calls) <= sequence.warmup_count + 1


@pytest.mark.asyncio
async def test_wrong_graph_identity_is_rejected_before_policy_or_next_state():
    components = _components()
    sequence = _sequence()
    bundle = await _bundle(components, sequence)
    components.graph.wrong_run = True

    with pytest.raises(SampleRunError) as raised:
        await components.runner.run(_spec(sequence, bundle), warmup=bundle)

    assert raised.value.stage is FailureStage.GRAPH
    assert components.diffusion.calls == []
    assert len(components.executor.calls) == sequence.warmup_count


@pytest.mark.parametrize("invalid_selection", ["future", "duplicate"])
@pytest.mark.asyncio
async def test_policy_cannot_select_future_or_duplicate_artifacts(
    invalid_selection: str,
):
    components = _components()
    sequence = _sequence()
    bundle = await _bundle(components, sequence)
    if invalid_selection == "future":
        components.diffusion.ghost = DiffusionArtifact(
            artifact_id="future-artifact",
            source_task_id="future-task",
            source_iteration=99,
            source_run_id="other-run",
            artifact_type=DiffusionArtifactType.RUN_OUTCOME,
            risk_level=DiffusionRiskLevel.LOW,
            content="future information",
        )
    else:
        components.diffusion.duplicate = True

    with pytest.raises(SampleRunError) as raised:
        await components.runner.run(_spec(sequence, bundle), warmup=bundle)

    assert raised.value.stage is FailureStage.POLICY
    assert len(components.executor.calls) == sequence.warmup_count


@pytest.mark.asyncio
async def test_context_pack_must_record_exact_candidate_and_selection_sets():
    components = _components()
    sequence = _sequence()
    bundle = await _bundle(components, sequence)
    components.packer.wrong_candidates = True

    with pytest.raises(SampleRunError) as raised:
        await components.runner.run(_spec(sequence, bundle), warmup=bundle)

    assert raised.value.stage is FailureStage.PACK
    assert len(components.executor.calls) == sequence.warmup_count


@pytest.mark.asyncio
async def test_mutating_context_packer_cannot_corrupt_causal_runner_state():
    components = _components()
    sequence = _sequence()
    bundle = await _bundle(components, sequence)
    components.packer.mutate_inputs = True

    result = await components.runner.run(_spec(sequence, bundle), warmup=bundle)

    assert all(
        artifact.content != "packer-corruption"
        for artifact in result.final_artifact_bank
    )
    assert result.final_graph is not None
    assert "packer-corruption" not in result.final_graph.task_ids


@pytest.mark.asyncio
async def test_mutating_bank_updater_cannot_corrupt_committed_runner_state():
    components = _components()
    sequence = _sequence()
    bundle = await _bundle(components, sequence)
    components.updater.mutate_current_bank = True
    components.updater.mutate_persist_input = True

    result = await components.runner.run(_spec(sequence, bundle), warmup=bundle)

    assert all(
        artifact.content not in {"updater-corruption", "persist-corruption"}
        for artifact in result.final_artifact_bank
    )


@pytest.mark.asyncio
async def test_journal_failure_rolls_back_current_artifacts_and_does_not_advance():
    components = _components()
    sequence = _sequence()
    bundle = await _bundle(components, sequence)

    async def fail_journal(journal: PositionJournal) -> str:
        raise OSError(f"cannot write {journal.position}")

    with pytest.raises(SampleRunError) as raised:
        await components.runner.run(
            _spec(sequence, bundle),
            warmup=bundle,
            on_position_complete=fail_journal,
        )

    error = raised.value
    assert error.stage is FailureStage.PERSIST
    assert error.progress.completed_positions == ()
    assert error.progress.bank_artifact_ids == tuple(
        artifact.artifact_id for artifact in bundle.final_artifact_bank
    )
    assert len(components.updater.rolled_back) == 1
    assert components.events[-1] == "rollback"


@pytest.mark.asyncio
async def test_rollback_failure_is_surfaced_with_the_original_persist_failure():
    components = _components()
    sequence = _sequence()
    bundle = await _bundle(components, sequence)
    components.updater.fail_rollback = True

    async def fail_journal(journal: PositionJournal) -> str:
        raise OSError(f"journal exploded at {journal.position}")

    with pytest.raises(SampleRunError) as raised:
        await components.runner.run(
            _spec(sequence, bundle),
            warmup=bundle,
            on_position_complete=fail_journal,
        )

    error = raised.value
    assert error.stage is FailureStage.PERSIST
    assert "journal exploded" in str(error.cause)
    assert "rollback failed" in str(error.cause)
    assert "rollback exploded" in str(error.cause)
    assert error.progress.completed_positions == ()


@pytest.mark.asyncio
async def test_infrastructure_failure_is_execute_failure_but_zero_reward_is_valid():
    valid = _components()
    zero_sequence = _sequence(warmup_count=0)
    result = await valid.runner.run(
        _spec(
            zero_sequence,
            None,
            arm=OrchestrationArm.EXECUTION_ONLY,
        )
    )
    assert result.task_records[0].execution.reward == 0.0

    failed = _components()
    failed.executor.infrastructure_failure = True
    with pytest.raises(SampleRunError) as raised:
        await failed.runner.run(
            _spec(
                zero_sequence,
                None,
                arm=OrchestrationArm.EXECUTION_ONLY,
            )
        )
    assert raised.value.stage is FailureStage.EXECUTE


@pytest.mark.asyncio
async def test_sample_execution_round_trips_suffix_only_with_missing_reward_semantics():
    components = _components()
    sequence = _sequence()
    bundle = await _bundle(components, sequence)

    result = await components.runner.run(_spec(sequence, bundle), warmup=bundle)
    restored = type(result).model_validate_json(result.model_dump_json())

    assert len(restored.task_records) == sequence.suffix_count
    assert all(
        record.position >= sequence.warmup_count for record in restored.task_records
    )
    assert restored.rewards.valid_for_reporting is True
    assert restored.warmup_reference is not None
    assert restored.warmup_reference.bundle_id == bundle.bundle_id

    tampered_rewards = result.model_dump(mode="json")
    tampered_rewards["rewards"]["weighted_sum"] = 999.0
    with pytest.raises(ValidationError, match="aggregates"):
        type(result).model_validate(tampered_rewards)

    tampered_graph = result.model_dump(mode="json")
    tampered_graph["final_graph"]["iteration"] = sequence.warmup_count
    with pytest.raises(ValidationError, match="last suffix position"):
        type(result).model_validate(tampered_graph)

    disconnected_chain = result.model_dump(mode="json")
    middle = disconnected_chain["task_records"][1]
    added_ids = middle["bank_update"]["after_artifact_ids"][
        len(middle["bank_update"]["before_artifact_ids"]) :
    ]
    middle["artifact_ids_before"] = ["disconnected-artifact"]
    middle["bank_update"]["before_artifact_ids"] = ["disconnected-artifact"]
    middle["bank_update"]["after_artifact_ids"] = [
        "disconnected-artifact",
        *added_ids,
    ]
    with pytest.raises(ValidationError, match="causal bank chain"):
        type(result).model_validate(disconnected_chain)

    mismatched_object = result.model_dump(mode="json")
    mismatched_object["final_artifact_bank"][-1]["content"] = "tampered content"
    with pytest.raises(ValidationError, match="artifact objects"):
        type(result).model_validate(mismatched_object)

    first_record = result.task_records[0]
    with pytest.raises(ValidationError, match="graph snapshot"):
        PositionJournal(
            run_id=result.spec.sample_id,
            sequence_id=result.spec.sequence.sequence_id,
            sample_id=result.spec.sample_id,
            position=first_record.position,
            task_record=first_record,
            bank_artifact_ids=first_record.bank_update.after_artifact_ids,
            graph_snapshot_id="wrong-snapshot",
        )


@pytest.mark.asyncio
async def test_complete_suffix_with_missing_reward_is_not_primary_reportable():
    components = _components()
    sequence = _sequence()
    bundle = await _bundle(components, sequence)
    components.executor.missing_reward_positions.add(3)

    result = await components.runner.run(_spec(sequence, bundle), warmup=bundle)

    assert tuple(record.position for record in result.task_records) == (2, 3, 4)
    assert result.rewards.task_rewards == (2.0, None, 4.0)
    assert result.rewards.all_tasks_completed is True
    assert result.rewards.rewards_complete is False
    assert result.rewards.valid_for_reporting is False
    assert result.rewards.unweighted_sum is None
    assert result.rewards.unweighted_mean is None
    assert result.rewards.weighted_sum is None
    assert result.rewards.weighted_mean is None
