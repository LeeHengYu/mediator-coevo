"""Deterministic diffusion selection policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import exp
import random
from typing import Any

from mediated_coevo.core.selection import deterministic_seed
from mediated_coevo.diffusion.models import (
    DiffusionArtifact,
    DiffusionArtifactType,
    TaskGraphEdgeRecord,
    TaskGraphSnapshot,
)

REUSE_SUCCESS_CHANNEL = "reuse_success"
AVOID_RECHECK_CHANNEL = "avoid_recheck"


@dataclass(frozen=True)
class DiffusionSubscription:
    """One runtime route from a durable artifact to a target context."""

    artifact: DiffusionArtifact
    policy_name: str
    relation: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    context_channel: str = REUSE_SUCCESS_CHANNEL


@dataclass(frozen=True)
class LLMRouterSoftmaxDecision:
    """Audit payload for one source-artifact softmax activation."""

    source_artifact_id: str
    source_task_id: str
    target_task_id: str
    target_iteration: int
    policy_name: str
    random_marker: float
    selection_seed: int
    selected_probability: float
    selected_similarity_index: float
    candidate_distribution: list[dict[str, Any]]


@dataclass(frozen=True)
class LLMRouterSoftmaxRoute:
    """A selected source artifact routed to one activated target task."""

    target_task_id: str
    subscription: DiffusionSubscription
    decision: LLMRouterSoftmaxDecision


def select_capped_broadcast_subscriptions(
    *,
    eligible_artifacts: list[DiffusionArtifact],
    max_artifacts: int,
    avoid_recheck_max_artifacts: int = 1,
) -> list[DiffusionSubscription]:
    """Subscribe the target to recent successful artifacts plus capped failures."""
    reuse_artifacts = _select_source_diverse_artifacts(
        [artifact for artifact in eligible_artifacts if is_reusable_artifact(artifact)],
        max_artifacts=max_artifacts,
        artifact_type_priority=_REUSE_ARTIFACT_TYPE_PRIORITY,
    )
    avoid_artifacts = _select_source_diverse_artifacts(
        [artifact for artifact in eligible_artifacts if is_avoid_recheck_artifact(artifact)],
        max_artifacts=avoid_recheck_max_artifacts,
        artifact_type_priority=_AVOID_RECHECK_ARTIFACT_TYPE_PRIORITY,
    )
    return [
        *_subscriptions(
            artifacts=reuse_artifacts,
            policy_name="capped_broadcast",
            relation="broadcast",
            reason=f"selected_success_in_top_{max_artifacts}_by_recency",
            context_channel=REUSE_SUCCESS_CHANNEL,
        ),
        *_subscriptions(
            artifacts=avoid_artifacts,
            policy_name="capped_broadcast",
            relation="avoid_recheck",
            reason=f"selected_failure_in_top_{avoid_recheck_max_artifacts}_by_recency",
            context_channel=AVOID_RECHECK_CHANNEL,
        ),
    ]


def select_random_k_subscriptions(
    *,
    eligible_artifacts: list[DiffusionArtifact],
    target_task_id: str,
    target_iteration: int,
    max_artifacts: int,
    seed: int | None,
    avoid_recheck_max_artifacts: int = 1,
) -> list[DiffusionSubscription]:
    """Subscribe the target to a reproducible random sample of eligible artifacts."""
    reuse_pool = _best_artifact_per_source(
        [artifact for artifact in eligible_artifacts if is_reusable_artifact(artifact)],
        artifact_type_priority=_REUSE_ARTIFACT_TYPE_PRIORITY,
    )
    avoid_pool = _best_artifact_per_source(
        [artifact for artifact in eligible_artifacts if is_avoid_recheck_artifact(artifact)],
        artifact_type_priority=_AVOID_RECHECK_ARTIFACT_TYPE_PRIORITY,
    )
    reuse_seed = _selection_seed(
        artifact_pool=reuse_pool,
        seed=seed,
        namespace="random_k_reuse_success",
        target_task_id=target_task_id,
        target_iteration=target_iteration,
        max_artifacts=max_artifacts,
    )
    avoid_seed = _selection_seed(
        artifact_pool=avoid_pool,
        seed=seed,
        namespace="random_k_avoid_recheck",
        target_task_id=target_task_id,
        target_iteration=target_iteration,
        max_artifacts=avoid_recheck_max_artifacts,
    )
    reuse_artifacts = _seeded_sample(
        artifact_pool=reuse_pool,
        route_seed=reuse_seed,
        max_artifacts=max_artifacts,
    )
    avoid_artifacts = _seeded_sample(
        artifact_pool=avoid_pool,
        route_seed=avoid_seed,
        max_artifacts=avoid_recheck_max_artifacts,
    )
    return [
        *_subscriptions(
            artifacts=reuse_artifacts,
            policy_name="random_k",
            relation="random",
            reason=f"selected_success_by_seeded_random_k_{max_artifacts}",
            context_channel=REUSE_SUCCESS_CHANNEL,
            metadata={"selection_seed": reuse_seed},
        ),
        *_subscriptions(
            artifacts=avoid_artifacts,
            policy_name="random_k",
            relation="avoid_recheck",
            reason=f"selected_failure_by_seeded_random_k_{avoid_recheck_max_artifacts}",
            context_channel=AVOID_RECHECK_CHANNEL,
            metadata={"selection_seed": avoid_seed},
        ),
    ]


def select_llm_router_softmax_routes(
    *,
    eligible_artifacts: list[DiffusionArtifact],
    snapshot: TaskGraphSnapshot,
    target_task_ids: list[str],
    target_iteration: int,
    top_k_candidates: int,
    temperature: float,
    seed: int | None,
    router_scores: Mapping[tuple[str, str], Mapping[str, Any]],
    router_weight: float = 0.0,
    router_model: str | None = None,
) -> list[LLMRouterSoftmaxRoute]:
    """Route each source artifact to a single LLM-scored target.

    This implements the logical-batch checkpoint used by the tmp WRA runs:
    each source contributes at most one selected transfer target. Candidates
    without parsed LLM router scores are not eligible.
    """
    if top_k_candidates <= 0 or temperature <= 0:
        return []

    routes: list[LLMRouterSoftmaxRoute] = []
    source_artifacts = _best_artifact_per_source(
        [
            artifact
            for artifact in eligible_artifacts
            if diffusion_channel_for_artifact(artifact) is not None
        ],
        artifact_type_priority=_SOFTMAX_ARTIFACT_TYPE_PRIORITY,
    )
    for artifact in source_artifacts:
        candidates = _softmax_candidates_for_source(
            artifact=artifact,
            snapshot=snapshot,
            target_task_ids=target_task_ids,
            router_scores=router_scores,
            router_weight=router_weight,
            router_model=router_model,
        )
        if not candidates:
            continue
        candidates = sorted(
            candidates,
            key=lambda item: (-float(item["similarity_index"]), item["target_task_id"]),
        )[:top_k_candidates]
        distribution = _softmax_distribution(candidates, temperature=temperature)
        selection_seed = deterministic_seed(
            seed or 0,
            "llm_router_softmax",
            target_iteration,
            artifact.artifact_id,
            ",".join(item["target_task_id"] for item in distribution),
        )
        marker = random.Random(selection_seed).random()
        selected = distribution[-1]
        cumulative = 0.0
        for item in distribution:
            cumulative += float(item["probability"])
            if marker <= cumulative:
                selected = item
                break
        channel = diffusion_channel_for_artifact(artifact)
        if channel is None:
            continue
        subscription = DiffusionSubscription(
            artifact=artifact,
            policy_name="llm_router_softmax",
            relation=str(selected["relation"]),
            reason="selected_by_llm_router_softmax",
            metadata={
                "selection_seed": selection_seed,
                "softmax_temperature": temperature,
                "softmax_top_k_candidates": top_k_candidates,
                "selected_target_task_id": selected["target_task_id"],
                "selected_probability": selected["probability"],
                "selected_similarity_index": selected["similarity_index"],
                "candidate_distribution": distribution,
                **(
                    {
                        "router_model": router_model,
                        "router_weight_cap": router_weight,
                    }
                    if router_model
                    else {}
                ),
            },
            context_channel=channel,
        )
        routes.append(
            LLMRouterSoftmaxRoute(
                target_task_id=str(selected["target_task_id"]),
                subscription=subscription,
                decision=LLMRouterSoftmaxDecision(
                    source_artifact_id=artifact.artifact_id,
                    source_task_id=artifact.source_task_id,
                    target_task_id=str(selected["target_task_id"]),
                    target_iteration=target_iteration,
                    policy_name="llm_router_softmax",
                    random_marker=marker,
                    selection_seed=selection_seed,
                    selected_probability=float(selected["probability"]),
                    selected_similarity_index=float(selected["similarity_index"]),
                    candidate_distribution=distribution,
                ),
            )
        )
    return routes


def _seeded_sample(
    *,
    artifact_pool: list[DiffusionArtifact],
    route_seed: int,
    max_artifacts: int,
) -> list[DiffusionArtifact]:
    artifact_pool = sorted(artifact_pool, key=lambda artifact: artifact.artifact_id)
    if max_artifacts <= 0 or not artifact_pool:
        return []
    sample_size = min(max_artifacts, len(artifact_pool))
    return random.Random(route_seed).sample(artifact_pool, sample_size)


def _selection_seed(
    *,
    artifact_pool: list[DiffusionArtifact],
    seed: int | None,
    namespace: str,
    target_task_id: str,
    target_iteration: int,
    max_artifacts: int,
) -> int:
    artifact_pool = sorted(artifact_pool, key=lambda artifact: artifact.artifact_id)
    route_seed = deterministic_seed(
        seed or 0,
        namespace,
        target_task_id,
        target_iteration,
        max_artifacts,
        ",".join(artifact.artifact_id for artifact in artifact_pool),
    )
    return route_seed


def _softmax_candidates_for_source(
    *,
    artifact: DiffusionArtifact,
    snapshot: TaskGraphSnapshot,
    target_task_ids: list[str],
    router_scores: Mapping[tuple[str, str], Mapping[str, Any]],
    router_weight: float,
    router_model: str | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for target_task_id in target_task_ids:
        if target_task_id == artifact.source_task_id:
            continue
        router_payload = router_scores.get((artifact.artifact_id, target_task_id))
        if router_payload is None:
            continue
        deterministic, relation, components = _softmax_deterministic_score(
            artifact=artifact,
            snapshot=snapshot,
            target_task_id=target_task_id,
        )
        confidence = _clamped_float(router_payload.get("confidence"), 0.0, 1.0)
        weight = max(0.0, min(1.0, router_weight * confidence))
        llm_signed = (2.0 * _clamped_float(router_payload.get("score"), 0.0, 1.0)) - 1.0
        similarity_index = _clamped_signed(
            ((1.0 - weight) * deterministic) + (weight * llm_signed)
        )
        components = {
            **components,
            "deterministic_similarity_index": deterministic,
            "llm_router_score": router_payload.get("score"),
            "llm_router_confidence": confidence,
            "llm_router_weight": weight,
            "llm_router_rationale": router_payload.get("rationale") or "",
            **({"router_model": router_model} if router_model else {}),
        }
        candidates.append(
            {
                "target_task_id": target_task_id,
                "similarity_index": similarity_index,
                "relation": relation,
                "score_components": components,
            }
        )
    return candidates


def _softmax_deterministic_score(
    *,
    artifact: DiffusionArtifact,
    snapshot: TaskGraphSnapshot,
    target_task_id: str,
) -> tuple[float, str, dict[str, Any]]:
    edge = _edge_record(
        snapshot=snapshot,
        source_task_id=artifact.source_task_id,
        target_task_id=target_task_id,
    )
    if edge is None:
        graph_score = 0.0
        relation = "llm_router_no_graph_edge"
        edge_weight = None
    else:
        edge_weight = edge.weight
        graph_score = _clamped_signed((2.0 * edge.weight) - 1.0)
        relation = edge.relation
    reward = _reward_score(artifact)
    reward_adjustment = 0.0 if reward < 0 else 0.15 * ((2.0 * reward) - 1.0)
    artifact_type_adjustment = {
        DiffusionArtifactType.RUN_OUTCOME: 0.05,
        DiffusionArtifactType.MEDIATOR_REPORT_SUMMARY: 0.03,
        DiffusionArtifactType.DEBUG_HINT: 0.0,
        DiffusionArtifactType.OTHER: -0.03,
    }.get(artifact.artifact_type, 0.0)
    score = _clamped_signed(graph_score + reward_adjustment + artifact_type_adjustment)
    return (
        score,
        relation,
        {
            "graph_score": graph_score,
            "edge_weight": edge_weight,
            "reward_score": reward,
            "reward_adjustment": reward_adjustment,
            "artifact_type_adjustment": artifact_type_adjustment,
        },
    )


def _edge_record(
    *,
    snapshot: TaskGraphSnapshot,
    source_task_id: str,
    target_task_id: str,
) -> TaskGraphEdgeRecord | None:
    for edge in snapshot.edge_records:
        if (
            edge.source_task_id == source_task_id
            and edge.target_task_id == target_task_id
        ):
            return edge
    return None


def _softmax_distribution(
    candidates: list[dict[str, Any]],
    *,
    temperature: float,
) -> list[dict[str, Any]]:
    max_score = max(float(item["similarity_index"]) for item in candidates)
    weights = [
        exp((float(item["similarity_index"]) - max_score) / temperature)
        for item in candidates
    ]
    total = sum(weights)
    return [
        {**item, "probability": weight / total}
        for item, weight in zip(candidates, weights, strict=True)
    ]


def _clamped_signed(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _clamped_float(value: Any, minimum: float, maximum: float) -> float:
    if not isinstance(value, (int, float)):
        return minimum
    return max(minimum, min(maximum, float(value)))


def select_top_k_similarity_subscriptions(
    *,
    eligible_artifacts: list[DiffusionArtifact],
    snapshot: TaskGraphSnapshot,
    target_task_id: str,
    max_artifacts: int,
    top_k_neighbors: int,
    avoid_recheck_max_artifacts: int = 1,
) -> list[DiffusionSubscription]:
    """Subscribe the target to artifacts from its strongest incoming graph edges."""
    edges_by_source_task = _top_similarity_edges_by_source_task(
        snapshot=snapshot,
        target_task_id=target_task_id,
        top_k_neighbors=top_k_neighbors,
    )
    if not edges_by_source_task:
        return []

    reuse_artifacts = _select_ranked_top_k_artifacts(
        eligible_artifacts=[
            artifact for artifact in eligible_artifacts if is_reusable_artifact(artifact)
        ],
        edges_by_source_task=edges_by_source_task,
        max_artifacts=max_artifacts,
        artifact_type_priority=_REUSE_ARTIFACT_TYPE_PRIORITY,
    )
    avoid_artifacts = _select_ranked_top_k_artifacts(
        eligible_artifacts=[
            artifact
            for artifact in eligible_artifacts
            if is_avoid_recheck_artifact(artifact)
        ],
        edges_by_source_task=edges_by_source_task,
        max_artifacts=avoid_recheck_max_artifacts,
        artifact_type_priority=_AVOID_RECHECK_ARTIFACT_TYPE_PRIORITY,
    )
    return [
        *_top_k_subscriptions(
            artifacts=reuse_artifacts,
            edges_by_source_task=edges_by_source_task,
            top_k_neighbors=top_k_neighbors,
            context_channel=REUSE_SUCCESS_CHANNEL,
            relation_fallback=None,
            reason=f"selected_success_from_top_{top_k_neighbors}_similarity_neighbor",
        ),
        *_top_k_subscriptions(
            artifacts=avoid_artifacts,
            edges_by_source_task=edges_by_source_task,
            top_k_neighbors=top_k_neighbors,
            context_channel=AVOID_RECHECK_CHANNEL,
            relation_fallback="avoid_recheck",
            reason=f"selected_failure_from_top_{top_k_neighbors}_similarity_neighbor",
        ),
    ]


def is_reusable_artifact(artifact: DiffusionArtifact) -> bool:
    """Return whether an artifact is eligible for normal reusable context."""
    return artifact.verifier_reward == 1.0


def is_avoid_recheck_artifact(artifact: DiffusionArtifact) -> bool:
    """Return whether an artifact may be shown only as an avoid/recheck warning."""
    return artifact.verifier_reward is not None and artifact.verifier_reward < 1.0


def diffusion_channel_for_artifact(artifact: DiffusionArtifact) -> str | None:
    """Classify an artifact's diffusion channel for audit metadata."""
    if is_reusable_artifact(artifact):
        return REUSE_SUCCESS_CHANNEL
    if is_avoid_recheck_artifact(artifact):
        return AVOID_RECHECK_CHANNEL
    return None


def _select_ranked_top_k_artifacts(
    *,
    eligible_artifacts: list[DiffusionArtifact],
    edges_by_source_task: dict[str, TaskGraphEdgeRecord],
    max_artifacts: int,
    artifact_type_priority: dict[DiffusionArtifactType, int],
) -> list[DiffusionArtifact]:
    if max_artifacts <= 0:
        return []
    recency_rank = {
        artifact.artifact_id: index for index, artifact in enumerate(eligible_artifacts)
    }
    ranked_artifacts = sorted(
        (
            artifact
            for artifact in eligible_artifacts
            if artifact.source_task_id in edges_by_source_task
        ),
        key=lambda artifact: (
            edges_by_source_task[artifact.source_task_id]
            .metadata["top_k_similarity_rank"],
            -_reward_score(artifact),
            artifact_type_priority.get(artifact.artifact_type, 99),
            recency_rank[artifact.artifact_id],
            artifact.artifact_id,
        ),
    )
    return _select_source_diverse_from_ranked(
        artifacts=ranked_artifacts,
        max_artifacts=max_artifacts,
    )


def _top_k_subscriptions(
    *,
    artifacts: list[DiffusionArtifact],
    edges_by_source_task: dict[str, TaskGraphEdgeRecord],
    top_k_neighbors: int,
    context_channel: str,
    relation_fallback: str | None,
    reason: str,
) -> list[DiffusionSubscription]:
    subscriptions: list[DiffusionSubscription] = []
    for artifact in artifacts:
        edge = edges_by_source_task[artifact.source_task_id]
        subscriptions.append(
            DiffusionSubscription(
                artifact=artifact,
                policy_name="top_k_similarity",
                relation=relation_fallback or edge.relation,
                reason=reason,
                metadata={
                    "edge_weight": edge.weight,
                    "edge_rank": edge.metadata["top_k_similarity_rank"],
                    "edge_relation": edge.relation,
                    "top_k_neighbors": top_k_neighbors,
                },
                context_channel=context_channel,
            )
        )
    return subscriptions


def _subscriptions(
    *,
    artifacts: list[DiffusionArtifact],
    policy_name: str,
    relation: str,
    reason: str,
    context_channel: str,
    metadata: dict[str, Any] | None = None,
) -> list[DiffusionSubscription]:
    return [
        DiffusionSubscription(
            artifact=artifact,
            policy_name=policy_name,
            relation=relation,
            reason=reason,
            metadata=metadata or {},
            context_channel=context_channel,
        )
        for artifact in artifacts
    ]


def _select_source_diverse_artifacts(
    artifacts: list[DiffusionArtifact],
    *,
    max_artifacts: int,
    artifact_type_priority: dict[DiffusionArtifactType, int],
) -> list[DiffusionArtifact]:
    if max_artifacts <= 0:
        return []
    selected: list[DiffusionArtifact] = []
    seen_sources: set[str] = set()
    for artifact in _best_first_artifacts(
        artifacts,
        artifact_type_priority=artifact_type_priority,
    ):
        if artifact.source_task_id in seen_sources:
            continue
        selected.append(artifact)
        seen_sources.add(artifact.source_task_id)
        if len(selected) >= max_artifacts:
            break
    return selected


def _select_source_diverse_from_ranked(
    *,
    artifacts: list[DiffusionArtifact],
    max_artifacts: int,
) -> list[DiffusionArtifact]:
    selected: list[DiffusionArtifact] = []
    seen_sources: set[str] = set()
    for artifact in artifacts:
        if artifact.source_task_id in seen_sources:
            continue
        selected.append(artifact)
        seen_sources.add(artifact.source_task_id)
        if len(selected) >= max_artifacts:
            break
    return selected


def _best_artifact_per_source(
    artifacts: list[DiffusionArtifact],
    *,
    artifact_type_priority: dict[DiffusionArtifactType, int],
) -> list[DiffusionArtifact]:
    return _select_source_diverse_artifacts(
        artifacts,
        max_artifacts=len({artifact.source_task_id for artifact in artifacts}),
        artifact_type_priority=artifact_type_priority,
    )


def _best_first_artifacts(
    artifacts: list[DiffusionArtifact],
    *,
    artifact_type_priority: dict[DiffusionArtifactType, int],
) -> list[DiffusionArtifact]:
    recency_rank = {artifact.artifact_id: index for index, artifact in enumerate(artifacts)}
    return sorted(
        artifacts,
        key=lambda artifact: (
            -_reward_score(artifact),
            artifact_type_priority.get(artifact.artifact_type, 99),
            recency_rank[artifact.artifact_id],
            artifact.artifact_id,
        ),
    )


def _reward_score(artifact: DiffusionArtifact) -> float:
    if artifact.judge_reward is not None:
        return artifact.judge_reward
    if artifact.verifier_reward is not None:
        return artifact.verifier_reward
    return -1.0


def _top_similarity_edges_by_source_task(
    *,
    snapshot: TaskGraphSnapshot,
    target_task_id: str,
    top_k_neighbors: int,
) -> dict[str, TaskGraphEdgeRecord]:
    ranked_edges = sorted(
        (
            edge
            for edge in snapshot.edge_records
            if edge.target_task_id == target_task_id
        ),
        key=lambda edge: (
            -edge.weight,
            edge.source_task_id,
            edge.target_task_id,
            edge.relation,
        ),
    )
    selected_edges: dict[str, TaskGraphEdgeRecord] = {}
    for edge in ranked_edges:
        if edge.source_task_id in selected_edges:
            continue
        selected_edges[edge.source_task_id] = edge.model_copy(
            update={
                "metadata": {
                    **edge.metadata,
                    "top_k_similarity_rank": len(selected_edges) + 1,
                },
            }
        )
        if len(selected_edges) >= top_k_neighbors:
            break
    return selected_edges


_REUSE_ARTIFACT_TYPE_PRIORITY = {
    DiffusionArtifactType.RUN_OUTCOME: 0,
    DiffusionArtifactType.MEDIATOR_REPORT_SUMMARY: 1,
    DiffusionArtifactType.DEBUG_HINT: 2,
    DiffusionArtifactType.OTHER: 3,
}

_AVOID_RECHECK_ARTIFACT_TYPE_PRIORITY = {
    DiffusionArtifactType.RUN_OUTCOME: 0,
    DiffusionArtifactType.DEBUG_HINT: 2,
    DiffusionArtifactType.MEDIATOR_REPORT_SUMMARY: 3,
    DiffusionArtifactType.OTHER: 4,
}

_SOFTMAX_ARTIFACT_TYPE_PRIORITY = {
    DiffusionArtifactType.RUN_OUTCOME: 0,
    DiffusionArtifactType.MEDIATOR_REPORT_SUMMARY: 1,
    DiffusionArtifactType.DEBUG_HINT: 2,
    DiffusionArtifactType.OTHER: 3,
}
