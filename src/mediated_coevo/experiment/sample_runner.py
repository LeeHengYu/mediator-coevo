"""Causal warm-up and arm-specific suffix state machine."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TypeAlias, cast

from mediated_coevo.artifacts.models import ArtifactBankUpdate
from mediated_coevo.artifacts.protocols import (
    ArtifactBankUpdater,
    ArtifactProjector,
)
from mediated_coevo.diffusion.models import DiffusionArtifact, TaskGraphSnapshot
from mediated_coevo.diffusion.policy_agent import graph_prior_candidates
from mediated_coevo.execution.models import (
    ContextPack,
    SamplePhaseName,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskProfile,
    empty_context_pack,
    redact_sensitive_data,
)
from mediated_coevo.execution.protocols import TaskExecutionAgent
from mediated_coevo.experiment.sample_models import (
    AgentCallRecord,
    FailureStage,
    PositionJournal,
    RunProgress,
    SampleExecution,
    SampleRunError,
    SampleSpec,
    SampleTaskRecord,
    SequenceSpec,
    TaskRecord,
    WarmupBundle,
    WarmupExecution,
    WarmupTaskRecord,
    calculate_sequence_rewards,
)
from mediated_coevo.orchestration.arms import (
    OrchestrationArm,
    plan_for_arm,
)
from mediated_coevo.orchestration.contracts import (
    ContextPacker,
    DiffusionPolicyAgent,
    GraphAgentRequest,
    GraphAgentResponse,
    PolicyAgentRequest,
    PolicyAgentResponse,
    TaskGraphAgent,
)

JournalResult: TypeAlias = str | Path | None
PositionCompleteCallback: TypeAlias = Callable[
    [PositionJournal],
    JournalResult | Awaitable[JournalResult],
]


@dataclass(frozen=True, slots=True)
class _RunState:
    """Immutable in-memory state; only journal success commits an advance."""

    run_id: str
    sequence: SequenceSpec
    sample_id: str | None
    next_position: int
    artifacts: tuple[DiffusionArtifact, ...] = ()
    graph: TaskGraphSnapshot | None = None
    records: tuple[TaskRecord, ...] = ()
    journal_paths: tuple[str, ...] = ()

    @property
    def artifact_ids(self) -> tuple[str, ...]:
        return tuple(artifact.artifact_id for artifact in self.artifacts)

    @property
    def completed_positions(self) -> tuple[int, ...]:
        return tuple(record.position for record in self.records)

    def preview_advance(
        self,
        *,
        record: TaskRecord,
        graph: TaskGraphSnapshot | None,
    ) -> _RunState:
        """Validate a transition without committing external or local state."""
        if self.next_position >= len(self.sequence.tasks):
            raise ValueError("sample state is already complete")
        if record.run_id != self.run_id:
            raise ValueError("task record belongs to another run")
        if record.sequence_id != self.sequence.sequence_id:
            raise ValueError("task record belongs to another sequence")
        if record.position != self.next_position:
            raise ValueError("task record position does not match the next occurrence")
        if record.task != self.sequence.tasks[self.next_position]:
            raise ValueError("task record differs from the frozen task occurrence")
        if record.artifact_ids_before != self.artifact_ids:
            raise ValueError("task record did not read the current artifact bank")
        if record.bank_update.after_artifact_ids != (
            *self.artifact_ids,
            *record.bank_update.added_artifact_ids,
        ):
            raise ValueError("task record is not an append-only bank transition")
        if graph is not None:
            if graph.run_id != self.run_id:
                raise ValueError("graph belongs to another run")
            if graph.iteration != self.next_position:
                raise ValueError("graph does not represent the current position")
        return replace(
            self,
            next_position=self.next_position + 1,
            artifacts=(*self.artifacts, *record.bank_update.added_artifacts),
            graph=graph,
            records=(*self.records, record),
        )


class SampleRunner:
    """Execute a frozen sequence with no hidden cross-task input.

    The runner is intentionally stateless between calls. Runtime-level freshness,
    single-use behavior, archive paths, and terminal files are enforced by
    :class:`mediated_coevo.experiment.sample_runtime.SampleRuntime`.
    """

    def __init__(
        self,
        *,
        graph_agent: TaskGraphAgent,
        diffusion_policy_agent: DiffusionPolicyAgent,
        random_policy_agent: DiffusionPolicyAgent,
        context_packer: ContextPacker,
        execution_agent: TaskExecutionAgent,
        artifact_projector: ArtifactProjector,
        artifact_bank_updater: ArtifactBankUpdater,
    ) -> None:
        self._graph_agent = graph_agent
        self._diffusion_policy_agent = diffusion_policy_agent
        self._random_policy_agent = random_policy_agent
        self._context_packer = context_packer
        self._execution_agent = execution_agent
        self._artifact_projector = artifact_projector
        self._artifact_bank_updater = artifact_bank_updater

    async def prepare_warmup(
        self,
        sequence: SequenceSpec,
        *,
        warmup_run_id: str,
        on_position_complete: PositionCompleteCallback | None = None,
    ) -> WarmupExecution:
        """Execute ``[0, W)`` once with empty context and no treatment arm."""
        state = _RunState(
            run_id=warmup_run_id,
            sequence=sequence,
            sample_id=None,
            next_position=0,
        )
        try:
            for _ in range(sequence.warmup_count):
                state = await self._run_warmup_position(
                    state=state,
                    on_position_complete=on_position_complete,
                )
            return WarmupExecution(
                sequence_id=sequence.sequence_id,
                warmup_run_id=warmup_run_id,
                task_records=cast(tuple[WarmupTaskRecord, ...], state.records),
                final_artifact_bank=state.artifacts,
                completed_journal_paths=state.journal_paths,
            )
        except SampleRunError:
            raise
        except Exception as exc:
            raise self._error(
                stage=FailureStage.FINALIZE,
                state=state,
                position=min(state.next_position, len(sequence.tasks) - 1),
                task_id=sequence.tasks[
                    min(state.next_position, len(sequence.tasks) - 1)
                ].task_id,
                cause=exc,
            ) from exc

    async def run(
        self,
        spec: SampleSpec,
        *,
        warmup: WarmupBundle | None = None,
        on_position_complete: PositionCompleteCallback | None = None,
    ) -> SampleExecution:
        """Execute only ``[W, N)`` for one treatment arm."""
        sequence = spec.sequence
        try:
            self.validate_warmup(spec, warmup)
        except Exception as exc:
            empty_state = _RunState(
                run_id=spec.sample_id,
                sequence=sequence,
                sample_id=spec.sample_id,
                next_position=sequence.warmup_count,
            )
            raise self._error(
                stage=FailureStage.FINALIZE,
                state=empty_state,
                position=sequence.warmup_count,
                task_id=sequence.tasks[sequence.warmup_count].task_id,
                cause=exc,
            ) from exc

        state = _RunState(
            run_id=spec.sample_id,
            sequence=sequence,
            sample_id=spec.sample_id,
            next_position=sequence.warmup_count,
            artifacts=(
                tuple(
                    DiffusionArtifact.model_validate(artifact.model_dump(mode="python"))
                    for artifact in warmup.final_artifact_bank
                )
                if warmup is not None
                else ()
            ),
        )
        try:
            while state.next_position < len(sequence.tasks):
                state = await self._run_suffix_position(
                    spec=spec,
                    state=state,
                    on_position_complete=on_position_complete,
                )
            records = cast(tuple[SampleTaskRecord, ...], state.records)
            rewards = calculate_sequence_rewards(
                sequence=sequence,
                task_rewards=tuple(record.execution.reward for record in records),
                completed_positions=tuple(record.position for record in records),
            )
            warmup_reference = (
                warmup.reference(
                    relative_path=(f"warmup/{warmup.warmup_run_id}/warmup_bundle.json"),
                    manifest_path=(
                        f"warmup/{warmup.warmup_run_id}/archive_manifest.json"
                    ),
                )
                if warmup is not None and sequence.warmup_count > 0
                else None
            )
            return SampleExecution(
                spec=spec,
                warmup_reference=warmup_reference,
                task_records=records,
                rewards=rewards,
                final_artifact_bank=state.artifacts,
                final_graph=state.graph,
                completed_journal_paths=state.journal_paths,
            )
        except SampleRunError:
            raise
        except Exception as exc:
            position = min(state.next_position, len(sequence.tasks) - 1)
            raise self._error(
                stage=FailureStage.FINALIZE,
                state=state,
                position=position,
                task_id=sequence.tasks[position].task_id,
                cause=exc,
            ) from exc

    @staticmethod
    def validate_warmup(
        spec: SampleSpec,
        warmup: WarmupBundle | None,
    ) -> None:
        """Validate a shared prefix without relabeling any warm-up record."""
        sequence = spec.sequence
        if sequence.warmup_count == 0:
            if warmup is not None:
                raise ValueError("zero-warm-up sample cannot receive a warm-up bundle")
            return
        if warmup is None:
            raise ValueError("sample requires its declared warm-up bundle")
        if spec.warmup_bundle_id != warmup.bundle_id:
            raise ValueError("warm-up bundle does not match SampleSpec")
        if warmup.sequence_id != sequence.sequence_id:
            raise ValueError("warm-up bundle belongs to another sequence")
        if warmup.warmup_count != sequence.warmup_count:
            raise ValueError("warm-up bundle has the wrong prefix length")
        for position, record in enumerate(warmup.task_records):
            if record.position != position or record.task != sequence.tasks[position]:
                raise ValueError("warm-up bundle differs from the frozen task prefix")

    async def _run_warmup_position(
        self,
        *,
        state: _RunState,
        on_position_complete: PositionCompleteCallback | None,
    ) -> _RunState:
        position = state.next_position
        task = self._resolve_task(state=state, position=position)
        context = empty_context_pack()
        execution = await self._execute(
            state=state,
            position=position,
            task=task,
            phase="warmup",
            arm=None,
            context=context,
            graph=None,
        )
        projected = await self._project(
            state=state,
            position=position,
            task=task,
            execution=execution,
            graph=None,
        )
        update = self._prepare_update(
            state=state,
            position=position,
            task=task,
            execution=execution,
            projected=projected,
            graph=None,
        )
        try:
            record = WarmupTaskRecord(
                run_id=state.run_id,
                sequence_id=state.sequence.sequence_id,
                position=position,
                task=task,
                artifact_ids_before=state.artifact_ids,
                context=context,
                execution=execution,
                bank_update=update,
            )
            next_state = state.preview_advance(record=record, graph=None)
        except Exception as exc:
            raise self._error(
                stage=FailureStage.PROJECT,
                state=state,
                position=position,
                task_id=task.task_id,
                cause=exc,
            ) from exc
        return await self._persist_and_commit(
            state=state,
            next_state=next_state,
            record=record,
            graph=None,
            update=update,
            on_position_complete=on_position_complete,
        )

    async def _run_suffix_position(
        self,
        *,
        spec: SampleSpec,
        state: _RunState,
        on_position_complete: PositionCompleteCallback | None,
    ) -> _RunState:
        position = state.next_position
        task = self._resolve_task(state=state, position=position)
        eligible = self._causal_artifacts(state=state, task=task)
        plan = plan_for_arm(spec.arm)
        graph_response: GraphAgentResponse | None = None
        policy_response: PolicyAgentResponse | None = None
        next_graph: TaskGraphSnapshot | None = None
        context = empty_context_pack()

        if plan.graph_agent_enabled:
            graph_request = GraphAgentRequest(
                run_id=spec.sample_id,
                position=position,
                task=task,
                previous_graph=state.graph,
                artifacts=eligible,
            )
            graph_response = await self._graph(
                state=state,
                request=graph_request,
                task=task,
            )
            next_graph = graph_response.snapshot
        else:
            graph_request = None

        if plan.policy_component != "none":
            policy_artifacts = eligible
            if plan.policy_component == "random_uniform":
                assert next_graph is not None
                policy_artifacts = tuple(
                    candidate[0]
                    for candidate in graph_prior_candidates(
                        current_task_id=task.task_id,
                        snapshot=next_graph,
                        artifacts=eligible,
                    )
                )
            policy_request = PolicyAgentRequest(
                run_id=spec.sample_id,
                position=position,
                policy_seed=spec.sequence.policy_seed,
                task=task,
                graph=next_graph,
                artifacts=policy_artifacts,
            )
            policy_agent = (
                self._random_policy_agent
                if plan.policy_component == "random_uniform"
                else self._diffusion_policy_agent
            )
            policy_response = await self._policy(
                state=state,
                request=policy_request,
                task=task,
                graph=next_graph,
                agent=policy_agent,
            )
            if plan.pack_context:
                context = await self._pack(
                    state=state,
                    task=task,
                    graph=next_graph,
                    policy=policy_response,
                    eligible=policy_artifacts,
                )
        else:
            policy_request = None

        execution = await self._execute(
            state=state,
            position=position,
            task=task,
            phase="orchestrated",
            arm=spec.arm,
            context=context,
            graph=next_graph,
        )
        projected = await self._project(
            state=state,
            position=position,
            task=task,
            execution=execution,
            graph=next_graph,
        )
        update = self._prepare_update(
            state=state,
            position=position,
            task=task,
            execution=execution,
            projected=projected,
            graph=next_graph,
        )
        try:
            record = SampleTaskRecord(
                sample_id=spec.sample_id,
                sequence_id=spec.sequence.sequence_id,
                position=position,
                task=task,
                arm=spec.arm,
                artifact_ids_before=state.artifact_ids,
                graph_call=(
                    self._graph_call_record(graph_request, graph_response)
                    if graph_request is not None and graph_response is not None
                    else None
                ),
                policy_call=(
                    self._policy_call_record(policy_request, policy_response)
                    if policy_request is not None and policy_response is not None
                    else None
                ),
                context=context,
                execution=execution,
                bank_update=update,
                graph_snapshot_id_after=(
                    next_graph.snapshot_id if next_graph is not None else None
                ),
            )
            next_state = state.preview_advance(record=record, graph=next_graph)
        except Exception as exc:
            raise self._error(
                stage=FailureStage.PROJECT,
                state=state,
                position=position,
                task_id=task.task_id,
                cause=exc,
                graph=next_graph,
            ) from exc
        return await self._persist_and_commit(
            state=state,
            next_state=next_state,
            record=record,
            graph=next_graph,
            update=update,
            on_position_complete=on_position_complete,
        )

    def _resolve_task(self, *, state: _RunState, position: int) -> TaskProfile:
        try:
            task = state.sequence.tasks[position]
            if task != TaskProfile.model_validate(task.model_dump(mode="python")):
                raise ValueError("frozen task profile is not normalized")
            return task
        except Exception as exc:
            task_id = (
                state.sequence.tasks[position].task_id
                if 0 <= position < len(state.sequence.tasks)
                else "<out-of-range>"
            )
            raise self._error(
                stage=FailureStage.RESOLVE,
                state=state,
                position=max(position, 0),
                task_id=task_id,
                cause=exc,
            ) from exc

    def _causal_artifacts(
        self,
        *,
        state: _RunState,
        task: TaskProfile,
    ) -> tuple[DiffusionArtifact, ...]:
        try:
            if len(state.artifact_ids) != len(set(state.artifact_ids)):
                raise ValueError("artifact bank contains duplicate IDs")
            if any(
                artifact.source_iteration >= state.next_position
                for artifact in state.artifacts
            ):
                raise ValueError("artifact bank contains current or future information")
            return state.artifacts
        except Exception as exc:
            raise self._error(
                stage=FailureStage.RESOLVE,
                state=state,
                position=state.next_position,
                task_id=task.task_id,
                cause=exc,
            ) from exc

    async def _graph(
        self,
        *,
        state: _RunState,
        request: GraphAgentRequest,
        task: TaskProfile,
    ) -> GraphAgentResponse:
        try:
            returned = await self._graph_agent.update(request)
            response = GraphAgentResponse.model_validate(
                returned.model_dump(mode="python")
            )
            snapshot = response.snapshot
            if snapshot.run_id != request.run_id:
                raise ValueError("graph snapshot belongs to another run")
            if snapshot.iteration != request.position:
                raise ValueError("graph snapshot position does not match the request")
            return response
        except Exception as exc:
            raise self._error(
                stage=FailureStage.GRAPH,
                state=state,
                position=request.position,
                task_id=task.task_id,
                cause=exc,
            ) from exc

    async def _policy(
        self,
        *,
        state: _RunState,
        request: PolicyAgentRequest,
        task: TaskProfile,
        graph: TaskGraphSnapshot | None,
        agent: DiffusionPolicyAgent,
    ) -> PolicyAgentResponse:
        try:
            returned = await agent.select(request)
            response = PolicyAgentResponse.model_validate(
                returned.model_dump(mode="python")
            )
            eligible = {
                artifact.artifact_id: artifact for artifact in request.artifacts
            }
            seen: set[str] = set()
            for subscription in response.subscriptions:
                selected = subscription.artifact
                candidate = eligible.get(selected.artifact_id)
                if candidate is None or candidate != selected:
                    raise ValueError(
                        "policy selected an artifact outside the causal candidate pool"
                    )
                if selected.source_iteration >= request.position:
                    raise ValueError("policy selected a current or future artifact")
                if selected.artifact_id in seen:
                    raise ValueError("policy selected the same artifact more than once")
                seen.add(selected.artifact_id)
            return response
        except Exception as exc:
            raise self._error(
                stage=FailureStage.POLICY,
                state=state,
                position=request.position,
                task_id=task.task_id,
                cause=exc,
                graph=graph,
            ) from exc

    async def _pack(
        self,
        *,
        state: _RunState,
        task: TaskProfile,
        graph: TaskGraphSnapshot | None,
        policy: PolicyAgentResponse,
        eligible: tuple[DiffusionArtifact, ...],
    ) -> ContextPack:
        try:
            packer_graph = (
                TaskGraphSnapshot.model_validate(graph.model_dump(mode="python"))
                if graph is not None
                else None
            )
            packer_policy = PolicyAgentResponse.model_validate(
                policy.model_dump(mode="python")
            )
            packer_eligible = tuple(
                DiffusionArtifact.model_validate(artifact.model_dump(mode="python"))
                for artifact in eligible
            )
            packed_context = await self._context_packer.pack(
                run_id=state.run_id,
                position=state.next_position,
                task=task,
                graph=packer_graph,
                policy=packer_policy,
                eligible_artifacts=packer_eligible,
            )
            context = ContextPack.model_validate(
                redact_sensitive_data(packed_context.model_dump(mode="python"))
            )
            self._validate_context_pack(
                context=context,
                eligible=eligible,
                policy=policy,
                graph=graph,
            )
            return context
        except Exception as exc:
            raise self._error(
                stage=FailureStage.PACK,
                state=state,
                position=state.next_position,
                task_id=task.task_id,
                cause=exc,
                graph=graph,
            ) from exc

    async def _execute(
        self,
        *,
        state: _RunState,
        position: int,
        task: TaskProfile,
        phase: SamplePhaseName,
        arm: OrchestrationArm | None,
        context: ContextPack,
        graph: TaskGraphSnapshot | None,
    ) -> TaskExecutionResult:
        try:
            request = TaskExecutionRequest(
                run_id=state.run_id,
                position=position,
                phase=phase,
                arm=arm.value if arm is not None else None,
                task=task,
                context=context,
            )
            result = await self._execution_agent.execute(request)
            existing_metadata = dict(result.metadata)
            expected_metadata: dict[str, str] = {"phase": phase}
            if arm is not None:
                expected_metadata["arm"] = arm.value
            for key, expected in expected_metadata.items():
                actual = existing_metadata.get(key)
                if actual is not None and actual != expected:
                    raise ValueError(
                        f"execution metadata {key} conflicts with its request"
                    )
            if phase == "warmup" and any(
                key.casefold() in {"arm", "treatment_arm", "baseline_preset"}
                for key in existing_metadata
            ):
                raise ValueError("warm-up execution metadata must be arm-neutral")
            result = TaskExecutionResult.model_validate(
                {
                    **result.model_dump(mode="python"),
                    "metadata": {**existing_metadata, **expected_metadata},
                }
            )
            if result.run_id != request.run_id:
                raise ValueError("execution result belongs to another run")
            if result.position != request.position:
                raise ValueError("execution result belongs to another position")
            if result.task_id != request.task.task_id:
                raise ValueError("execution result belongs to another task")
            if result.is_infrastructure_failure:
                trace = result.record.execution_trace
                status = trace.status if trace is not None else "unknown"
                raise RuntimeError(f"task infrastructure status is {status}")
            return result
        except Exception as exc:
            raise self._error(
                stage=FailureStage.EXECUTE,
                state=state,
                position=position,
                task_id=task.task_id,
                cause=exc,
                graph=graph,
            ) from exc

    async def _project(
        self,
        *,
        state: _RunState,
        position: int,
        task: TaskProfile,
        execution: TaskExecutionResult,
        graph: TaskGraphSnapshot | None,
    ) -> tuple[DiffusionArtifact, ...]:
        try:
            projected = await self._artifact_projector.project(
                task=task,
                execution=execution,
            )
            if not isinstance(projected, tuple):
                raise TypeError("artifact projector must return an immutable tuple")
            return tuple(
                DiffusionArtifact.model_validate(
                    redact_sensitive_data(artifact.model_dump(mode="python"))
                )
                for artifact in projected
            )
        except Exception as exc:
            raise self._error(
                stage=FailureStage.PROJECT,
                state=state,
                position=position,
                task_id=task.task_id,
                cause=exc,
                graph=graph,
            ) from exc

    def _prepare_update(
        self,
        *,
        state: _RunState,
        position: int,
        task: TaskProfile,
        execution: TaskExecutionResult,
        projected: tuple[DiffusionArtifact, ...],
        graph: TaskGraphSnapshot | None,
    ) -> ArtifactBankUpdate:
        try:
            current_bank = tuple(
                DiffusionArtifact.model_validate(artifact.model_dump(mode="python"))
                for artifact in state.artifacts
            )
            projected_copy = tuple(
                DiffusionArtifact.model_validate(artifact.model_dump(mode="python"))
                for artifact in projected
            )
            returned = self._artifact_bank_updater.prepare(
                run_id=state.run_id,
                position=position,
                task=task,
                execution=execution,
                current_bank=current_bank,
                projected_artifacts=projected_copy,
            )
            return ArtifactBankUpdate.model_validate(returned.model_dump(mode="python"))
        except Exception as exc:
            raise self._error(
                stage=FailureStage.PERSIST,
                state=state,
                position=position,
                task_id=task.task_id,
                cause=exc,
                graph=graph,
            ) from exc

    async def _persist_and_commit(
        self,
        *,
        state: _RunState,
        next_state: _RunState,
        record: TaskRecord,
        graph: TaskGraphSnapshot | None,
        update: ArtifactBankUpdate,
        on_position_complete: PositionCompleteCallback | None,
    ) -> _RunState:
        position = record.position
        persisted_paths: tuple[Path, ...] = ()
        try:
            persisted_paths = self._artifact_bank_updater.persist(
                ArtifactBankUpdate.model_validate(update.model_dump(mode="python"))
            )
            journal = PositionJournal(
                run_id=state.run_id,
                sequence_id=state.sequence.sequence_id,
                sample_id=state.sample_id,
                position=position,
                task_record=record,
                bank_artifact_ids=next_state.artifact_ids,
                graph_snapshot_id=(graph.snapshot_id if graph is not None else None),
            )
            journal_path = await self._write_journal(
                callback=on_position_complete,
                journal=journal,
            )
            if journal_path is not None:
                next_state = replace(
                    next_state,
                    journal_paths=(*state.journal_paths, journal_path),
                )
            return next_state
        except Exception as exc:
            failure: BaseException = exc
            if persisted_paths:
                try:
                    self._artifact_bank_updater.rollback(persisted_paths)
                except Exception as rollback_exc:
                    failure = RuntimeError(f"{exc}; rollback failed: {rollback_exc}")
                    failure.__cause__ = rollback_exc
            raise self._error(
                stage=FailureStage.PERSIST,
                state=state,
                position=position,
                task_id=record.task_id,
                cause=failure,
                graph=graph,
            ) from failure

    @staticmethod
    async def _write_journal(
        *,
        callback: PositionCompleteCallback | None,
        journal: PositionJournal,
    ) -> str | None:
        if callback is None:
            return None
        result = callback(journal)
        if inspect.isawaitable(result):
            result = await result
        if result is None:
            return None
        value = result.as_posix() if isinstance(result, Path) else result
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts:
            raise ValueError("journal callback must return a relative path")
        return value

    @staticmethod
    def _validate_context_pack(
        *,
        context: ContextPack,
        eligible: tuple[DiffusionArtifact, ...],
        policy: PolicyAgentResponse,
        graph: TaskGraphSnapshot | None,
    ) -> None:
        eligible_ids = tuple(artifact.artifact_id for artifact in eligible)
        if context.eligible_artifact_ids != eligible_ids:
            raise ValueError("context must record the exact causal candidate pool")
        if context.selected_artifact_ids != policy.selected_artifact_ids:
            raise ValueError("context selection must match the policy response")
        expected_snapshot_id = graph.snapshot_id if graph is not None else None
        if context.snapshot_id != expected_snapshot_id:
            raise ValueError("context snapshot must match the policy graph")
        if context.policy_name != policy.policy_name:
            raise ValueError("context policy_name must match the policy response")
        eligible_by_id = {artifact.artifact_id: artifact for artifact in eligible}
        expected_sources = tuple(
            dict.fromkeys(
                eligible_by_id[artifact_id].source_task_id
                for artifact_id in context.rendered_artifact_ids
            )
        )
        if context.source_task_ids != expected_sources:
            raise ValueError("context source_task_ids must match rendered artifacts")

    @staticmethod
    def _graph_call_record(
        request: GraphAgentRequest,
        response: GraphAgentResponse,
    ) -> AgentCallRecord:
        return AgentCallRecord(
            input_payload=request.model_dump(mode="json"),
            output_payload=response.model_dump(mode="json"),
        )

    @staticmethod
    def _policy_call_record(
        request: PolicyAgentRequest,
        response: PolicyAgentResponse,
    ) -> AgentCallRecord:
        return AgentCallRecord(
            input_payload=request.model_dump(mode="json"),
            output_payload=response.model_dump(mode="json"),
        )

    @staticmethod
    def _progress(
        state: _RunState,
        *,
        graph: TaskGraphSnapshot | None = None,
    ) -> RunProgress:
        current_graph = graph if graph is not None else state.graph
        return RunProgress(
            run_id=state.run_id,
            sequence_id=state.sequence.sequence_id,
            sample_id=state.sample_id,
            completed_positions=state.completed_positions,
            completed_journal_paths=state.journal_paths,
            bank_artifact_ids=state.artifact_ids,
            graph_snapshot_id=(
                current_graph.snapshot_id if current_graph is not None else None
            ),
        )

    @classmethod
    def _error(
        cls,
        *,
        stage: FailureStage,
        state: _RunState,
        position: int,
        task_id: str,
        cause: BaseException,
        graph: TaskGraphSnapshot | None = None,
    ) -> SampleRunError:
        if isinstance(cause, SampleRunError):
            return cause
        return SampleRunError(
            stage=stage,
            position=position,
            task_id=task_id,
            progress=cls._progress(state, graph=graph),
            cause=cause,
        )
