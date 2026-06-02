"""Deterministic diffusion selection policies."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from mediated_coevo.core.selection import deterministic_seed
from mediated_coevo.diffusion.models import (
    DiffusionArtifact,
    TaskGraphEdgeRecord,
    TaskGraphSnapshot,
)
from mediated_coevo.diffusion.renderer import (
    DiffusionContextBundle,
    render_diffusion_subscriptions,
)
from mediated_coevo.diffusion.store import DiffusionStore


@dataclass(frozen=True)
class DiffusionSubscription:
    """One runtime route from a durable artifact to a target context."""

    artifact: DiffusionArtifact
    policy_name: str
    relation: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


def build_capped_broadcast_context(
    *,
    store: DiffusionStore,
    snapshot: TaskGraphSnapshot,
    model: str,
    target_task_id: str,
    target_iteration: int,
    target_run_id: str | None,
    max_artifacts: int,
) -> DiffusionContextBundle:
    """Select a flat capped set of prior cross-task artifacts and render them."""
    eligible_artifacts = _eligible_artifacts(
        store=store,
        target_task_id=target_task_id,
        target_iteration=target_iteration,
    )
    subscriptions = select_capped_broadcast_subscriptions(
        eligible_artifacts=eligible_artifacts,
        max_artifacts=max_artifacts,
    )
    store.store_graph_snapshot(snapshot, overwrite=True)
    return render_diffusion_subscriptions(
        store=store,
        snapshot=snapshot,
        model=model,
        target_task_id=target_task_id,
        target_iteration=target_iteration,
        target_run_id=target_run_id,
        subscriptions=subscriptions,
        eligible_count=len(eligible_artifacts),
    )


def build_random_k_context(
    *,
    store: DiffusionStore,
    snapshot: TaskGraphSnapshot,
    model: str,
    target_task_id: str,
    target_iteration: int,
    target_run_id: str | None,
    max_artifacts: int,
    seed: int | None,
) -> DiffusionContextBundle:
    """Select up to k prior cross-task artifacts with a reproducible RNG."""
    eligible_artifacts = _eligible_artifacts(
        store=store,
        target_task_id=target_task_id,
        target_iteration=target_iteration,
    )
    subscriptions = select_random_k_subscriptions(
        eligible_artifacts=eligible_artifacts,
        target_task_id=target_task_id,
        target_iteration=target_iteration,
        max_artifacts=max_artifacts,
        seed=seed,
    )
    store.store_graph_snapshot(snapshot, overwrite=True)
    return render_diffusion_subscriptions(
        store=store,
        snapshot=snapshot,
        model=model,
        target_task_id=target_task_id,
        target_iteration=target_iteration,
        target_run_id=target_run_id,
        subscriptions=subscriptions,
        eligible_count=len(eligible_artifacts),
    )


def select_capped_broadcast_subscriptions(
    *,
    eligible_artifacts: list[DiffusionArtifact],
    max_artifacts: int,
) -> list[DiffusionSubscription]:
    """Subscribe the target to the most recent eligible broadcast artifacts."""
    return [
        DiffusionSubscription(
            artifact=artifact,
            policy_name="capped_broadcast",
            relation="broadcast",
            reason=f"selected_in_top_{max_artifacts}_by_recency",
        )
        for artifact in eligible_artifacts[:max_artifacts]
    ]


def select_random_k_subscriptions(
    *,
    eligible_artifacts: list[DiffusionArtifact],
    target_task_id: str,
    target_iteration: int,
    max_artifacts: int,
    seed: int | None,
) -> list[DiffusionSubscription]:
    """Subscribe the target to a reproducible random sample of eligible artifacts."""
    artifact_pool = sorted(eligible_artifacts, key=lambda artifact: artifact.artifact_id)
    route_seed = deterministic_seed(
        seed or 0,
        "random_k",
        target_task_id,
        target_iteration,
        max_artifacts,
        ",".join(artifact.artifact_id for artifact in artifact_pool),
    )
    sample_size = min(max_artifacts, len(artifact_pool))
    selected_artifacts = random.Random(route_seed).sample(artifact_pool, sample_size)
    metadata = {"selection_seed": route_seed}
    return [
        DiffusionSubscription(
            artifact=artifact,
            policy_name="random_k",
            relation="random",
            reason=f"selected_by_seeded_random_k_{max_artifacts}",
            metadata=metadata,
        )
        for artifact in selected_artifacts
    ]


def select_top_k_similarity_subscriptions(
    *,
    eligible_artifacts: list[DiffusionArtifact],
    snapshot: TaskGraphSnapshot,
    target_task_id: str,
    max_artifacts: int,
    top_k_neighbors: int,
) -> list[DiffusionSubscription]:
    """Subscribe the target to artifacts from its strongest incoming graph edges."""
    edges_by_source_task = _top_similarity_edges_by_source_task(
        snapshot=snapshot,
        target_task_id=target_task_id,
        top_k_neighbors=top_k_neighbors,
    )
    if not edges_by_source_task:
        return []

    subscriptions: list[DiffusionSubscription] = []
    for artifact in eligible_artifacts:
        edge = edges_by_source_task.get(artifact.source_task_id)
        if edge is None:
            continue
        subscriptions.append(
            DiffusionSubscription(
                artifact=artifact,
                policy_name="top_k_similarity",
                relation=edge.relation,
                reason=f"selected_from_top_{top_k_neighbors}_similarity_neighbor",
                metadata={
                    "edge_weight": edge.weight,
                    "edge_rank": edge.metadata["top_k_similarity_rank"],
                    "edge_relation": edge.relation,
                },
            )
        )
        if len(subscriptions) >= max_artifacts:
            break
    return subscriptions


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


def _eligible_artifacts(
    *,
    store: DiffusionStore,
    target_task_id: str,
    target_iteration: int,
) -> list[DiffusionArtifact]:
    visible_artifacts = store.query_artifacts(
        recent=None,
        before_source_iteration=target_iteration,
    )
    return [
        artifact
        for artifact in visible_artifacts
        if artifact.source_task_id != target_task_id
    ]

