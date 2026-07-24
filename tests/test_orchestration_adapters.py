from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from mediated_coevo.artifacts import (
    ArtifactBankUpdate,
    DiffusionArtifactBankUpdater,
    DiffusionEmitterProjector,
)
from mediated_coevo.diffusion import (
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    DiffusionStore,
    TaskGraphSnapshot,
)
from mediated_coevo.diffusion.emitter import DiffusionEmitter
from mediated_coevo.diffusion.policy import DiffusionSubscription
from mediated_coevo.diffusion.policy_agent import (
    LangChainDiffusionPolicyAgent,
    graph_prior_candidates,
)
from mediated_coevo.diffusion.task_graph_agent import LangChainTaskGraphAgent
from mediated_coevo.execution import (
    BenchmarkTaskProfileProvider,
    ContextPack,
    ExplicitContextOrchestratorExecutionAgent,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskProfile,
    empty_context_pack,
)
from mediated_coevo.execution.models import (
    redact_sensitive_data,
    redact_sensitive_text,
)
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.trace import ExecutionTrace, TraceStatus
from mediated_coevo.orchestration import (
    GraphAgentRequest,
    OrchestrationArm,
    PolicyAgentRequest,
    arm_for_flags,
    plan_for_arm,
)
from mediated_coevo.orchestration.adapters import (
    DiffusionContextPacker,
    LangChainDiffusionPolicyAdapter,
    LangChainTaskGraphAdapter,
    RandomPolicyAgent,
)
from mediated_coevo.orchestration.contracts import (
    GraphAgentResponse,
    PolicyAgentResponse,
)


def _artifact(
    artifact_id: str,
    *,
    task_id: str,
    position: int,
    run_id: str = "warmup-run",
    reward: float | None = 1.0,
) -> DiffusionArtifact:
    return DiffusionArtifact(
        artifact_id=artifact_id,
        source_task_id=task_id,
        source_iteration=position,
        source_run_id=run_id,
        artifact_type=DiffusionArtifactType.RUN_OUTCOME,
        risk_level=DiffusionRiskLevel.LOW,
        content=f"outcome for {task_id}",
        verifier_reward=reward,
    )


def _execution(
    *,
    run_id: str,
    position: int,
    task_id: str,
    reward: float | None = 0.0,
    status: TraceStatus = "ok",
    metadata: dict[str, Any] | None = None,
) -> TaskExecutionResult:
    trace = ExecutionTrace(
        task_id=task_id,
        iteration=position,
        reward=reward,
        status=status,
        run_id=f"executor-{run_id}-{position}",
    )
    return TaskExecutionResult(
        run_id=run_id,
        position=position,
        task_id=task_id,
        record=IterationRecord(
            iteration=position,
            task_id=task_id,
            reward=reward,
            execution_trace=trace,
        ),
        metadata=metadata or {},
    )


def test_task_profiles_and_persisted_text_do_not_expose_credentials():
    with pytest.raises(ValidationError, match="must not contain credentials"):
        TaskProfile(
            task_id="task-secret",
            instruction="execute",
            task_config={"runtime": {"api_key": "super-secret"}},
        )

    redacted = redact_sensitive_data(
        {
            "api_key": "super-secret",
            "stdout": (
                "Authorization: Bearer bearer-secret "
                'password="password-secret" '
                '"client_secret": "json-secret"'
            ),
        }
    )
    encoded = json.dumps(redacted)
    assert "super-secret" not in encoded
    assert "bearer-secret" not in encoded
    assert "password-secret" not in encoded
    assert "json-secret" not in encoded
    assert encoded.count("[redacted]") == 4


@pytest.mark.parametrize(
    "value",
    (
        "Bearer original-secret",
        "token=original-secret",
        'password="original-secret"',
        '"client_secret": "original-secret"',
    ),
)
def test_credential_redaction_is_idempotent(value):
    redacted = redact_sensitive_text(value)
    assert redact_sensitive_text(redacted) == redacted


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_normalized_task_profile_rejects_non_finite_numbers(value):
    with pytest.raises(ValidationError, match="JSON-compatible"):
        TaskProfile(
            task_id="task-non-finite",
            instruction="execute",
            task_config={"value": value},
        )


def test_execution_result_rejects_malformed_external_archive_provenance():
    with pytest.raises(ValidationError, match="absolute or have a scheme"):
        _execution(
            run_id="run-1",
            position=0,
            task_id="task-a",
            metadata={
                "external_archive_refs": (
                    {"kind": "remote", "uri": "relative/not-portable"},
                )
            },
        )


def test_orchestration_contracts_detach_and_sanitize_legacy_models():
    artifact = _artifact(
        "artifact-0",
        task_id="task-a",
        position=0,
    )
    artifact.metadata["nested"] = {"value": 1}
    snapshot = TaskGraphSnapshot(
        run_id="sample-1",
        iteration=1,
        task_ids=["task-a"],
        graph_policy="test",
        metadata={"detail": "Authorization: Bearer graph-secret"},
    )
    graph_request = GraphAgentRequest(
        run_id="sample-1",
        position=2,
        task=TaskProfile(task_id="task-b", instruction="execute"),
        previous_graph=snapshot,
        artifacts=(artifact,),
    )

    graph_request.previous_graph.task_ids.append("mutated")
    graph_request.artifacts[0].metadata["nested"]["value"] = 2
    assert snapshot.task_ids == ["task-a"]
    assert artifact.metadata["nested"] == {"value": 1}

    graph_response = GraphAgentResponse(
        snapshot=snapshot,
        raw_decision={"authorization": "Bearer decision-secret"},
    )
    subscription = DiffusionSubscription(
        artifact=artifact,
        policy_name="test",
        relation="selected",
        reason="Bearer policy-secret",
        metadata={"authorization": "Bearer metadata-secret"},
    )
    policy_response = PolicyAgentResponse(
        policy_name="test",
        subscriptions=(subscription,),
        raw_decision={"selected": [artifact.artifact_id]},
    )

    encoded = graph_response.model_dump_json() + policy_response.model_dump_json()
    assert "graph-secret" not in encoded
    assert "decision-secret" not in encoded
    assert "policy-secret" not in encoded
    assert "metadata-secret" not in encoded
    policy_response.subscriptions[0].artifact.content = "mutated"
    assert artifact.content == "outcome for task-a"


class _GraphAgent(LangChainTaskGraphAgent):
    async def decide(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        previous_snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        del current_iteration, previous_snapshot, artifacts
        return {
            "node_id": task_profile["task_id"],
            "node_action": "created",
            "edges": [],
            "reason": "test",
        }


class _WrongGraphAgent(_GraphAgent):
    def materialize_snapshot(self, **kwargs: Any) -> TaskGraphSnapshot:
        del kwargs
        return TaskGraphSnapshot(
            run_id="wrong-run",
            iteration=2,
            graph_policy="test",
        )


class _UnsafeSnapshotIdGraphAgent(_GraphAgent):
    def materialize_snapshot(self, **kwargs: Any) -> TaskGraphSnapshot:
        task_profile = kwargs["task_profile"]
        current_iteration = kwargs["current_iteration"]
        return TaskGraphSnapshot(
            snapshot_id="../../escape",
            run_id=self.run_id,
            iteration=current_iteration,
            task_ids=[task_profile["task_id"]],
            graph_policy="test",
            metadata={
                "task_assignments": {
                    f"{current_iteration}:{task_profile['task_id']}": "node"
                }
            },
        )


class _InvalidRoutingGraphAgent(_GraphAgent):
    def materialize_snapshot(self, **kwargs: Any) -> TaskGraphSnapshot:
        snapshot = super().materialize_snapshot(**kwargs)
        snapshot.metadata["current_node_id"] = "unassigned-node"
        return snapshot


class _MissingSourceGraphAgent(_GraphAgent):
    def materialize_snapshot(self, **kwargs: Any) -> TaskGraphSnapshot:
        snapshot = super().materialize_snapshot(**kwargs)
        source_task_id = kwargs["artifacts"][0].source_task_id
        del snapshot.metadata["task_nodes"][source_task_id]
        return snapshot


class _PolicyAgent(LangChainDiffusionPolicyAgent):
    async def decide(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        del task_profile, current_iteration, snapshot
        return {
            "selected_artifacts": [
                {
                    "artifact_id": artifacts[0].artifact_id,
                    "relation": "test",
                    "reason": "test selection",
                }
            ]
        }


class _EmptyPolicyAgent(LangChainDiffusionPolicyAgent):
    async def decide(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"selected_artifacts": []}


@pytest.mark.asyncio
async def test_graph_adapter_materializes_warmup_nodes_and_validates_before_store(
    tmp_path: Path,
):
    artifact = _artifact("artifact-0", task_id="warmup-0", position=0)
    task = TaskProfile(task_id="task-2", instruction="next")
    store = DiffusionStore(tmp_path / "diffusion")
    request = GraphAgentRequest(
        run_id="sample-1",
        position=2,
        task=task,
        artifacts=(artifact,),
    )
    adapter = LangChainTaskGraphAdapter(
        _GraphAgent(model="openrouter/test/model", run_id="sample-1"),
        store=store,
    )

    response = await adapter.update(request)

    assert response.raw_decision["reason"] == "test"
    assert response.snapshot.run_id == "sample-1"
    assert response.snapshot.iteration == 2
    assert response.snapshot.task_ids == ["warmup-0", "task-2"]
    assert store.load_graph_snapshot(response.snapshot.snapshot_id) == response.snapshot

    bad_store = DiffusionStore(tmp_path / "bad-diffusion")
    bad = LangChainTaskGraphAdapter(
        _WrongGraphAgent(model="openrouter/test/model", run_id="sample-1"),
        store=bad_store,
    )
    with pytest.raises(ValueError, match="run_id"):
        await bad.update(request)
    assert bad_store.query_graph_snapshots(recent=None) == []

    unsafe_store = DiffusionStore(tmp_path / "unsafe-diffusion")
    unsafe = LangChainTaskGraphAdapter(
        _UnsafeSnapshotIdGraphAgent(
            model="openrouter/test/model",
            run_id="sample-1",
        ),
        store=unsafe_store,
    )
    with pytest.raises(ValueError, match="safe path component"):
        await unsafe.update(request)
    assert unsafe_store.query_graph_snapshots(recent=None) == []
    assert not (tmp_path / "escape.json").exists()

    invalid_routing_store = DiffusionStore(tmp_path / "invalid-routing")
    invalid_routing = LangChainTaskGraphAdapter(
        _InvalidRoutingGraphAgent(
            model="openrouter/test/model",
            run_id="sample-1",
        ),
        store=invalid_routing_store,
    )
    with pytest.raises(ValueError, match="differs from its task assignment"):
        await invalid_routing.update(request)
    assert invalid_routing_store.query_graph_snapshots(recent=None) == []

    missing_source_store = DiffusionStore(tmp_path / "missing-source")
    missing_source = LangChainTaskGraphAdapter(
        _MissingSourceGraphAgent(
            model="openrouter/test/model",
            run_id="sample-1",
        ),
        store=missing_source_store,
    )
    with pytest.raises(ValueError, match="omits causal artifact source tasks"):
        await missing_source.update(request)
    assert missing_source_store.query_graph_snapshots(recent=None) == []


@pytest.mark.asyncio
async def test_policy_adapter_supports_graph_none_and_preserves_empty_selection():
    artifact = _artifact("artifact-0", task_id="warmup-0", position=0)
    request = PolicyAgentRequest(
        run_id="sample-1",
        position=1,
        policy_seed=7,
        task=TaskProfile(task_id="task-1", instruction="next"),
        graph=None,
        artifacts=(artifact,),
    )
    selected = await LangChainDiffusionPolicyAdapter(
        _PolicyAgent(
            model="openrouter/test/model",
            max_artifacts=1,
            fallback_strategy="none",
        )
    ).select(request)
    empty = await LangChainDiffusionPolicyAdapter(
        _EmptyPolicyAgent(
            model="openrouter/test/model",
            max_artifacts=1,
            fallback_strategy="none",
        )
    ).select(request)

    assert selected.selected_artifact_ids == ("artifact-0",)
    assert selected.metadata["graph_used"] is False
    assert empty.subscriptions == ()
    assert empty.metadata["fallback_strategy"] == "none"


@pytest.mark.asyncio
async def test_random_policy_requires_graph():
    request = PolicyAgentRequest(
        run_id="sample-1",
        position=1,
        policy_seed=31,
        task=TaskProfile(task_id="task-1", instruction="next"),
        graph=None,
        artifacts=(_artifact("artifact-0", task_id="task-0", position=0),),
    )

    with pytest.raises(ValueError, match="requires a graph snapshot"):
        await RandomPolicyAgent(max_artifacts=1).select(request)


@pytest.mark.asyncio
async def test_random_policy_follows_graph_priors_without_full_pool_fallback():
    artifacts = (
        _artifact("artifact-a1", task_id="task-a", position=0),
        _artifact("artifact-a2", task_id="task-a", position=1),
        _artifact("artifact-same", task_id="task-same", position=2),
        _artifact("artifact-outside", task_id="task-outside", position=3),
    )
    graph = TaskGraphSnapshot(
        run_id="sample-1",
        iteration=4,
        graph_policy="langchain_graph",
        edge_records=[
            {
                "source_task_id": "node-a",
                "target_task_id": "node-current",
                "relation": "transfer_prior",
                "weight": 0.7,
            }
        ],
        metadata={
            "current_node_id": "node-current",
            "task_nodes": {
                "node-a": {"task_ids": ["task-a"]},
                "node-current": {"task_ids": ["task-same", "task-current"]},
                "node-outside": {"task_ids": ["task-outside"]},
            },
        },
    )
    candidates = tuple(
        candidate[0]
        for candidate in graph_prior_candidates(
            current_task_id="task-current",
            snapshot=graph,
            artifacts=artifacts,
        )
    )
    request = PolicyAgentRequest(
        run_id="sample-1",
        position=4,
        policy_seed=31,
        task=TaskProfile(task_id="task-current", instruction="next"),
        graph=graph,
        artifacts=candidates,
    )

    policy = RandomPolicyAgent(max_artifacts=2)
    selected = await policy.select(request)
    repeated = await policy.select(request)

    assert selected == repeated
    assert selected.raw_decision["candidate_artifact_ids"] == (
        "artifact-a1",
        "artifact-a2",
        "artifact-same",
    )
    assert len(selected.subscriptions) == 2
    assert selected.raw_decision["graph_used"] is True
    assert selected.raw_decision["candidate_scope"] == "graph_priors"
    assert all(
        subscription.relation == "graph_random_uniform"
        for subscription in selected.subscriptions
    )

    moved = await policy.select(request.model_copy(update={"position": 5}))
    assert (
        moved.raw_decision["selection_seed"] != selected.raw_decision["selection_seed"]
    )

    empty_graph = graph.model_copy(
        update={
            "edge_records": [],
            "metadata": {
                "current_node_id": "node-current",
                "task_nodes": {"node-current": {"task_ids": ["task-current"]}},
            },
        }
    )
    empty_candidates = tuple(
        candidate[0]
        for candidate in graph_prior_candidates(
            current_task_id="task-current",
            snapshot=empty_graph,
            artifacts=artifacts,
        )
    )
    empty = await policy.select(
        request.model_copy(update={"graph": empty_graph, "artifacts": empty_candidates})
    )
    assert empty.subscriptions == ()
    assert empty.raw_decision["candidate_artifact_ids"] == ()


@pytest.mark.asyncio
async def test_context_packer_supports_no_graph_and_records_exact_budget_state(
    tmp_path: Path,
):
    artifact = _artifact("artifact-0", task_id="task-0", position=0)
    task = TaskProfile(task_id="task-1", instruction="next")
    policy = PolicyAgentResponse(
        policy_name="langchain_graph",
        subscriptions=(
            DiffusionSubscription(
                artifact=artifact,
                policy_name="langchain_graph",
                relation="test_selection",
                reason="selected for packer test",
            ),
        ),
        raw_decision={"selected_artifact_ids": (artifact.artifact_id,)},
    )
    packer = DiffusionContextPacker(
        store=DiffusionStore(tmp_path / "diffusion"),
        model="openrouter/test/model",
        max_context_tokens=1_000,
    )

    context = await packer.pack(
        run_id="sample-1",
        position=1,
        task=task,
        graph=None,
        policy=policy,
        eligible_artifacts=(artifact,),
    )

    assert context.snapshot_id is None
    assert context.policy_name == "langchain_graph"
    assert context.selected_artifact_ids == ("artifact-0",)
    assert context.rendered_artifact_ids == ("artifact-0",)
    assert context.source_task_ids == ("task-0",)
    assert context.max_context_tokens == 1_000
    assert context.text is not None
    assert context.token_count > 0


def test_requests_and_context_reject_future_duplicate_and_inconsistent_ids():
    artifact = _artifact("artifact-0", task_id="future", position=2)
    task = TaskProfile(task_id="task-1", instruction="next")
    with pytest.raises(ValidationError, match="current or future"):
        PolicyAgentRequest(
            run_id="sample-1",
            position=1,
            policy_seed=7,
            task=task,
            artifacts=(artifact,),
        )

    with pytest.raises(ValidationError, match="unique"):
        GraphAgentRequest(
            run_id="sample-1",
            position=3,
            task=task,
            artifacts=(artifact, artifact),
        )

    with pytest.raises(ValidationError, match="selected artifacts must be eligible"):
        ContextPack(
            text="context",
            selected_artifact_ids=("artifact-0",),
            rendered_artifact_ids=("artifact-0",),
            source_task_ids=("task-0",),
            policy_name="test",
            token_count=1,
        )
    with pytest.raises(ValidationError, match="rendered or explicitly"):
        ContextPack(
            eligible_artifact_ids=("artifact-0",),
            selected_artifact_ids=("artifact-0",),
            policy_name="test",
        )


def test_task_profiles_are_detached_deeply_frozen_and_round_trip():
    source = {"z": [3, {"b": 2, "a": 1}], "a": "first"}
    repository = SimpleNamespace(
        resolve=lambda task_id: SimpleNamespace(
            task_id=task_id,
            instruction="execute",
            task_config=source,
        )
    )
    profile = BenchmarkTaskProfileProvider(repository).resolve("task-1")
    source["z"].append(4)

    assert profile.task_config == {
        "a": "first",
        "z": (3, {"a": 1, "b": 2}),
    }
    with pytest.raises(TypeError):
        profile.task_config["new"] = "blocked"
    with pytest.raises(TypeError):
        profile.task_config["z"][1]["new"] = "blocked"
    assert TaskProfile.model_validate_json(profile.model_dump_json()) == profile


@pytest.mark.asyncio
async def test_projector_preserves_normalized_verifier_and_judge_provenance():
    task = TaskProfile(
        task_id="task-1",
        instruction="execute",
        task_config={
            "metadata": {
                "category": "coding",
                "difficulty": "hard",
                "expected_reward_range": [0.0, 1.0],
            },
            "verifier": {"type": "tests"},
        },
    )
    execution = _execution(
        run_id="warmup-run",
        position=0,
        task_id="task-1",
        reward=0.0,
        metadata={
            "judge_reward": 0.25,
            "judge_reward_record_id": "judge-1",
            "reward_source": "verifier",
        },
    )
    projector = DiffusionEmitterProjector(DiffusionEmitter(model="test-model"))

    projected = await projector.project(task=task, execution=execution)

    assert projected
    artifact = projected[0]
    assert artifact.source_run_id == "warmup-run"
    assert artifact.source_task_id == "task-1"
    assert artifact.source_iteration == 0
    assert artifact.verifier_reward == 0.0
    assert artifact.judge_reward == 0.25
    assert artifact.metadata["task_category"] == "coding"
    assert artifact.metadata["task_difficulty"] == "hard"
    assert artifact.metadata["verifier_type"] == "tests"
    assert artifact.metadata["judge_reward_record_id"] == "judge-1"
    assert artifact.metadata["source_execution_run_id"] == "executor-warmup-run-0"


def test_artifact_update_preflights_collisions_and_rolls_back_partial_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    task = TaskProfile(task_id="task-1", instruction="execute")
    execution = _execution(run_id="sample-1", position=1, task_id="task-1")
    projected = (
        _artifact(
            "artifact-a",
            task_id="task-1",
            position=1,
            run_id="sample-1",
        ),
        _artifact(
            "artifact-b",
            task_id="task-1",
            position=1,
            run_id="sample-1",
        ),
    )
    store = DiffusionStore(tmp_path / "diffusion")
    updater = DiffusionArtifactBankUpdater(store)
    update = updater.prepare(
        run_id="sample-1",
        position=1,
        task=task,
        execution=execution,
        current_bank=(
            _artifact(
                "warmup-a",
                task_id="warmup",
                position=0,
                run_id="warmup-run",
            ),
        ),
        projected_artifacts=projected,
    )
    assert update.after_artifact_ids == ("warmup-a", "artifact-a", "artifact-b")

    original_store = store.store_artifact
    calls = 0

    def fail_second(artifact: DiffusionArtifact, *, overwrite: bool = False) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated partial write")
        return original_store(artifact, overwrite=overwrite)

    monkeypatch.setattr(store, "store_artifact", fail_second)
    with pytest.raises(OSError, match="partial write"):
        updater.persist(update)
    assert store.query_artifacts(recent=None) == []

    monkeypatch.setattr(store, "store_artifact", original_store)
    paths = updater.persist(update)
    with pytest.raises(FileExistsError, match="already exists"):
        updater.persist(update)
    updater.rollback(paths)
    assert store.query_artifacts(recent=None) == []

    calls = 0

    def write_then_fail(
        artifact: DiffusionArtifact,
        *,
        overwrite: bool = False,
    ) -> Path:
        nonlocal calls
        calls += 1
        path = original_store(artifact, overwrite=overwrite)
        if calls == 1:
            raise OSError("failure after write")
        return path

    monkeypatch.setattr(store, "store_artifact", write_then_fail)
    with pytest.raises(OSError, match="after write"):
        updater.persist(update)
    assert store.query_artifacts(recent=None) == []

    monkeypatch.setattr(store, "store_artifact", original_store)
    original_store(projected[1])
    with pytest.raises(FileExistsError, match="artifact-b"):
        updater.persist(update)
    assert not (tmp_path / "diffusion" / "artifacts" / "artifact-a.json").exists()


def test_artifact_bank_rejects_path_like_artifact_ids(tmp_path):
    store = DiffusionStore(tmp_path / "diffusion")
    updater = DiffusionArtifactBankUpdater(store)
    artifact = _artifact(
        "../outside",
        task_id="task-b",
        position=1,
        run_id="sample-a",
    )
    update = ArtifactBankUpdate(
        run_id="sample-a",
        position=1,
        task_id="task-b",
        before_artifact_ids=(),
        added_artifacts=(artifact,),
        after_artifact_ids=(artifact.artifact_id,),
    )

    with pytest.raises(ValueError, match="safe path component"):
        updater.persist(update)


class _ExplicitBackend:
    context: ContextPack | None = None
    task: TaskProfile | None = None

    async def execute_task_with_context(
        self,
        *,
        task_id: str,
        position: int,
        context: ContextPack,
        task: TaskProfile,
    ) -> IterationRecord:
        self.context = context
        self.task = task
        return IterationRecord(
            iteration=position,
            task_id=task_id,
            graph_snapshot_id=context.snapshot_id,
            diffusion_policy=context.policy_name,
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


@pytest.mark.asyncio
async def test_execution_adapter_forwards_complete_context_and_warmup_is_arm_neutral():
    backend = _ExplicitBackend()
    agent = ExplicitContextOrchestratorExecutionAgent(backend)
    context = ContextPack(
        text="selected context",
        eligible_artifact_ids=("artifact-0",),
        selected_artifact_ids=("artifact-0",),
        rendered_artifact_ids=("artifact-0",),
        source_task_ids=("task-0",),
        snapshot_id=None,
        policy_name="langchain_graph",
        token_count=2,
        max_context_tokens=10,
    )
    request = TaskExecutionRequest(
        run_id="sample-1",
        position=1,
        phase="orchestrated",
        arm="diffusion_only",
        task=TaskProfile(task_id="task-1", instruction="execute"),
        context=context,
    )

    result = await agent.execute(request)

    assert backend.context == context
    assert backend.task is request.task
    assert result.run_id == "sample-1"
    assert result.position == 1
    assert result.reward is None

    warmup = TaskExecutionRequest(
        run_id="warmup-run",
        position=0,
        phase="warmup",
        arm=None,
        task=TaskProfile(task_id="task-0", instruction="execute"),
        context=empty_context_pack(),
    )
    warmup_result = await agent.execute(warmup)
    assert warmup_result.run_id == "warmup-run"
    assert "arm" not in warmup_result.metadata

    with pytest.raises(ValidationError, match="must not carry a treatment arm"):
        warmup.model_copy(update={"arm": "execution_only"}).model_validate(
            warmup.model_copy(update={"arm": "execution_only"})
        )


@pytest.mark.asyncio
async def test_execution_adapter_redacts_and_cannot_override_canonical_metadata():
    backend = _ExplicitBackend()
    backend.take_explicit_execution_provenance = lambda **kwargs: {
        "OPENAI_API_KEY": "provenance-secret",
        "phase": "forged",
        "arm": "full_orchestration",
        "context_policy": "forged",
        "nested": {"authorization": "Bearer nested-secret"},
    }
    request = TaskExecutionRequest(
        run_id="sample-1",
        position=1,
        phase="orchestrated",
        arm="diffusion_only",
        task=TaskProfile(task_id="task-1", instruction="execute"),
        context=empty_context_pack().model_copy(
            update={"policy_name": "langchain_graph"}
        ),
    )

    result = await ExplicitContextOrchestratorExecutionAgent(backend).execute(request)

    assert result.metadata["phase"] == "orchestrated"
    assert result.metadata["arm"] == "diffusion_only"
    assert result.metadata["context_policy"] == "langchain_graph"
    assert "provenance-secret" not in result.model_dump_json()
    assert "nested-secret" not in result.model_dump_json()


def test_artifact_bank_redacts_projected_content_and_metadata(tmp_path):
    store = DiffusionStore(tmp_path / "diffusion")
    updater = DiffusionArtifactBankUpdater(store)
    task = TaskProfile(task_id="task-b", instruction="execute")
    execution = _execution(run_id="sample-a", position=1, task_id="task-b")
    artifact = _artifact(
        "artifact-secret",
        task_id="task-b",
        position=1,
        run_id="sample-a",
    ).model_copy(
        update={
            "content": "Authorization: Bearer artifact-secret-value",
            "metadata": {"OPENAI_API_KEY": "metadata-secret-value"},
        }
    )

    update = updater.prepare(
        run_id="sample-a",
        position=1,
        task=task,
        execution=execution,
        current_bank=(),
        projected_artifacts=(artifact,),
    )
    updater.persist(update)

    serialized = update.added_artifacts[0].model_dump_json()
    assert "artifact-secret-value" not in serialized
    assert "metadata-secret-value" not in serialized


def test_arm_plans_are_the_fixed_four_treatments():
    assert plan_for_arm(OrchestrationArm.EXECUTION_ONLY).model_dump(
        exclude={"schema_version", "arm"}
    ) == {
        "graph_agent_enabled": False,
        "diffusion_agent_enabled": False,
        "policy_component": "none",
        "pack_context": False,
    }
    assert (
        plan_for_arm(OrchestrationArm.GRAPH_ONLY).policy_component == "random_uniform"
    )
    assert plan_for_arm(OrchestrationArm.DIFFUSION_ONLY).graph_agent_enabled is False
    assert plan_for_arm(OrchestrationArm.FULL_ORCHESTRATION).graph_agent_enabled is True
    assert (
        arm_for_flags(graph_agent_enabled=False, diffusion_agent_enabled=False)
        is OrchestrationArm.EXECUTION_ONLY
    )
    assert (
        arm_for_flags(graph_agent_enabled=True, diffusion_agent_enabled=False)
        is OrchestrationArm.GRAPH_ONLY
    )
    assert (
        arm_for_flags(graph_agent_enabled=False, diffusion_agent_enabled=True)
        is OrchestrationArm.DIFFUSION_ONLY
    )
    assert (
        arm_for_flags(graph_agent_enabled=True, diffusion_agent_enabled=True)
        is OrchestrationArm.FULL_ORCHESTRATION
    )
