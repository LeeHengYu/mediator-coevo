"""Deterministic diffusion selection policies."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
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
