"""Adapters from split diffusion components to explicit sample contracts."""

from __future__ import annotations

import random
from dataclasses import dataclass

from mediated_coevo.core.selection import deterministic_seed
from mediated_coevo.diffusion.models import DiffusionArtifact, TaskGraphSnapshot
from mediated_coevo.diffusion.policy import (
    AVOID_RECHECK_CHANNEL,
    REUSE_SUCCESS_CHANNEL,
    DiffusionSubscription,
    is_avoid_recheck_artifact,
)
from mediated_coevo.diffusion.policy_agent import LangChainDiffusionPolicyAgent
from mediated_coevo.diffusion.renderer import (
    DiffusionArtifactCompactor,
    render_diffusion_subscriptions,
)
from mediated_coevo.diffusion.store import DiffusionStore
from mediated_coevo.diffusion.task_graph_agent import LangChainTaskGraphAgent
from mediated_coevo.execution.models import ContextPack, TaskProfile
from mediated_coevo.orchestration.contracts import (
    GraphAgentRequest,
    GraphAgentResponse,
    PolicyAgentRequest,
    PolicyAgentResponse,
)


def _agent_task_profile(task: TaskProfile) -> dict[str, object]:
    return task.model_dump(mode="json", exclude={"schema_version"})


def _validate_path_component(value: str, *, label: str) -> str:
    if (
        not value
        or value != value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be a safe path component")
    return value


@dataclass(frozen=True, slots=True)
class LangChainTaskGraphAdapter:
    """Expose the standalone graph agent through the sample contract."""

    agent: LangChainTaskGraphAgent
    store: DiffusionStore | None = None

    async def update(self, request: GraphAgentRequest) -> GraphAgentResponse:
        """Materialize, validate, then optionally persist one graph decision."""
        if self.agent.run_id != request.run_id:
            raise ValueError("graph agent run_id does not match graph request")
        task_profile = _agent_task_profile(request.task)
        artifacts = list(request.artifacts)
        decision = await self.agent.decide(
            task_profile=task_profile,
            current_iteration=request.position,
            previous_snapshot=request.previous_graph,
            artifacts=artifacts,
        )
        snapshot = self.agent.materialize_snapshot(
            task_profile=task_profile,
            current_iteration=request.position,
            previous_snapshot=request.previous_graph,
            artifacts=artifacts,
            graph_decision=decision,
            materialize_artifact_nodes=True,
        )
        response = GraphAgentResponse(snapshot=snapshot, raw_decision=decision)
        _validate_graph_response(request=request, response=response)
        if self.store is not None:
            self.store.store_graph_snapshot(response.snapshot)
        return response


def _validate_graph_response(
    *, request: GraphAgentRequest, response: GraphAgentResponse
) -> None:
    snapshot = response.snapshot
    _validate_path_component(snapshot.snapshot_id, label="graph snapshot_id")
    if snapshot.run_id != request.run_id:
        raise ValueError("graph snapshot run_id does not match graph request")
    if snapshot.iteration != request.position:
        raise ValueError("graph snapshot iteration does not match graph request")
    assignments = snapshot.metadata.get("task_assignments")
    expected_key = f"{request.position}:{request.task.task_id}"
    if not isinstance(assignments, dict) or expected_key not in assignments:
        raise ValueError("graph snapshot does not assign the current task occurrence")


@dataclass(frozen=True, slots=True)
class LangChainDiffusionPolicyAdapter:
    """Expose the standalone graph-optional policy through sample contracts."""

    agent: LangChainDiffusionPolicyAgent

    async def select(self, request: PolicyAgentRequest) -> PolicyAgentResponse:
        """Run the policy with no implicit fallback and retain its raw output."""
        if self.agent.fallback_strategy != "none":
            raise ValueError("sample policy agent must use fallback_strategy='none'")
        task_profile = _agent_task_profile(request.task)
        artifacts = list(request.artifacts)
        decision = await self.agent.decide(
            task_profile=task_profile,
            current_iteration=request.position,
            snapshot=request.graph,
            artifacts=artifacts,
        )
        subscriptions = self.agent.materialize_subscriptions(
            diffusion_decision=decision,
            task_profile=task_profile,
            snapshot=request.graph,
            artifacts=artifacts,
        )
        response = PolicyAgentResponse(
            policy_name="langchain_graph",
            subscriptions=tuple(subscriptions),
            raw_decision=decision,
            metadata={
                "fallback_strategy": self.agent.fallback_strategy,
                "graph_used": request.graph is not None,
                "artifact_cap": self.agent.max_artifacts,
            },
        )
        _validate_policy_response(request=request, response=response)
        if len(response.subscriptions) > self.agent.max_artifacts:
            raise ValueError("policy selected more artifacts than its total cap")
        return response


def _validate_policy_response(
    *, request: PolicyAgentRequest, response: PolicyAgentResponse
) -> None:
    eligible = {artifact.artifact_id: artifact for artifact in request.artifacts}
    for subscription in response.subscriptions:
        artifact = eligible.get(subscription.artifact.artifact_id)
        if artifact is None or artifact != subscription.artifact:
            raise ValueError("policy selected an artifact outside the causal candidate pool")


@dataclass(frozen=True, slots=True)
class RandomPolicyAgent:
    """Deterministic uniform selection over the exact causal candidate pool."""

    max_artifacts: int

    def __post_init__(self) -> None:
        if self.max_artifacts < 0:
            raise ValueError("max_artifacts must be non-negative")

    async def select(self, request: PolicyAgentRequest) -> PolicyAgentResponse:
        """Select up to the learned policy's total cap without reward quotas."""
        candidates = sorted(
            request.artifacts,
            key=lambda artifact: artifact.artifact_id,
        )
        candidate_ids = tuple(artifact.artifact_id for artifact in candidates)
        route_seed = deterministic_seed(
            request.policy_seed,
            "sample_random_uniform",
            request.position,
            ",".join(candidate_ids),
            self.max_artifacts,
        )
        count = min(self.max_artifacts, len(candidates))
        selected = random.Random(route_seed).sample(candidates, count) if count else []
        subscriptions = tuple(
            DiffusionSubscription(
                artifact=artifact,
                policy_name="random_uniform",
                relation="random_uniform",
                reason="selected uniformly from the complete causal candidate pool",
                context_channel=(
                    AVOID_RECHECK_CHANNEL
                    if is_avoid_recheck_artifact(artifact)
                    else REUSE_SUCCESS_CHANNEL
                ),
                metadata={"selection_seed": route_seed},
            )
            for artifact in selected
        )
        return PolicyAgentResponse(
            policy_name="random_uniform",
            subscriptions=subscriptions,
            raw_decision={
                "candidate_artifact_ids": candidate_ids,
                "selected_artifact_ids": tuple(
                    artifact.artifact_id for artifact in selected
                ),
                "selection_seed": route_seed,
                "artifact_cap": self.max_artifacts,
                "graph_used": False,
            },
            metadata={
                "sampling": "deterministic_uniform_without_replacement",
                "selection_seed": route_seed,
                "artifact_cap": self.max_artifacts,
            },
        )


@dataclass(frozen=True, slots=True)
class DiffusionContextPacker:
    """Render selected routes for graph and graph-free arms."""

    store: DiffusionStore
    model: str
    max_context_tokens: int | None = None
    compact_artifact_content: DiffusionArtifactCompactor | None = None

    def __post_init__(self) -> None:
        if self.max_context_tokens is not None and self.max_context_tokens < 0:
            raise ValueError("max_context_tokens must be non-negative")

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
        """Render a strictly validated selection and retain all budget decisions."""
        request = PolicyAgentRequest(
            run_id=run_id,
            position=position,
            policy_seed=0,
            task=task,
            graph=graph,
            artifacts=eligible_artifacts,
        )
        _validate_policy_response(request=request, response=policy)
        bundle = await render_diffusion_subscriptions(
            store=self.store,
            snapshot=graph,
            graph_policy=policy.policy_name,
            model=self.model,
            target_task_id=task.task_id,
            target_iteration=position,
            target_run_id=run_id,
            subscriptions=list(policy.subscriptions),
            eligible_count=len(eligible_artifacts),
            max_context_tokens=self.max_context_tokens,
            compact_artifact_content=self.compact_artifact_content,
        )
        return ContextPack(
            text=bundle.text,
            eligible_artifact_ids=tuple(
                artifact.artifact_id for artifact in eligible_artifacts
            ),
            selected_artifact_ids=policy.selected_artifact_ids,
            rendered_artifact_ids=tuple(bundle.rendered_artifact_ids or ()),
            source_task_ids=tuple(bundle.source_task_ids),
            snapshot_id=bundle.snapshot_id,
            policy_name=policy.policy_name,
            token_count=bundle.context_tokens,
            max_context_tokens=bundle.max_context_tokens,
            compacted_artifact_ids=tuple(bundle.compacted_artifact_ids or ()),
            dropped_for_budget_artifact_ids=tuple(
                bundle.dropped_for_budget_artifact_ids or ()
            ),
            budget_violation=bundle.budget_violation,
            metadata={
                "graph_policy": bundle.graph_policy,
                "eligible_count": bundle.eligible_count,
                "selected_count": bundle.selected_count,
                "rendered_count": bundle.rendered_count,
            },
        )
