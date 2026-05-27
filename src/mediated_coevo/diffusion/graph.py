"""Pure constructors for diffusion graph snapshots from precomputed artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from mediated_coevo.analysis.task_similarity import PairSimilarity, TaskSimilarityProfile
from mediated_coevo.diffusion.models import TaskGraphEdgeRecord, TaskGraphSnapshot


class TaskProfilesArtifact(BaseModel):
    """Validated contents of a task profile precompute artifact."""

    task_count: int
    profiles: dict[str, TaskSimilarityProfile]


class PairwiseSimilarityArtifact(BaseModel):
    """Validated contents of a pairwise similarity precompute artifact."""

    pair_count: int
    p20_threshold: float | None = None
    edge_score_threshold: float | None = None
    active_threshold: float | None = None
    threshold_kind: str | None = None
    pairs: list[PairSimilarity]


def load_precomputed_similarity_artifacts(
    graph_dir: Path,
) -> tuple[TaskProfilesArtifact, PairwiseSimilarityArtifact]:
    """Load validated graph artifacts from a precompute directory."""
    profiles = TaskProfilesArtifact.model_validate(
        json.loads((graph_dir / "task_profiles.json").read_text(encoding="utf-8"))
    )
    pairwise = PairwiseSimilarityArtifact.model_validate(
        json.loads(
            (graph_dir / "pairwise_similarity.json").read_text(encoding="utf-8")
        )
    )
    return profiles, pairwise


def construct_feature_index(
    *,
    task_profiles: Mapping[str, TaskSimilarityProfile | Mapping[str, Any]],
    task_ids: Sequence[str] | None = None,
) -> dict[str, TaskSimilarityProfile]:
    """Return a validated task profile index for the selected task IDs."""
    selected_task_ids = _selected_task_ids(task_profiles.keys(), task_ids)
    return {
        task_id: TaskSimilarityProfile.model_validate(task_profiles[task_id])
        for task_id in selected_task_ids
    }


def construct_edge_records(
    *,
    pairwise_similarity: Sequence[PairSimilarity | Mapping[str, Any]],
    task_ids: Sequence[str],
    use_threshold_cut: bool = True,
) -> list[TaskGraphEdgeRecord]:
    """Return directed edge records from precomputed pairwise similarity."""
    selected_task_ids = _normalize_task_ids(task_ids)
    selected_task_id_set = set(selected_task_ids)
    edges: list[TaskGraphEdgeRecord] = []

    for raw_pair in pairwise_similarity:
        pair = PairSimilarity.model_validate(raw_pair)
        if (
            pair.source not in selected_task_id_set
            or pair.target not in selected_task_id_set
        ):
            continue
        if use_threshold_cut and not pair.kept_after_threshold_cut:
            continue
        edges.extend(_pair_edges(pair, threshold_filter_applied=use_threshold_cut))

    return sorted(
        edges,
        key=lambda edge: (edge.source_task_id, edge.target_task_id, edge.relation),
    )


def construct_snapshot(
    *,
    run_id: str,
    iteration: int,
    task_ids: Sequence[str],
    graph_policy: str,
    edge_records: Sequence[TaskGraphEdgeRecord],
    feature_cutoff: datetime | None = None,
    seed: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TaskGraphSnapshot:
    """Freeze a graph structure into a serializable snapshot."""
    selected_task_ids = _normalize_task_ids(task_ids)
    serialized_edges = list(edge_records)
    connected_task_ids = {
        edge.source_task_id for edge in serialized_edges
    } | {edge.target_task_id for edge in serialized_edges}
    snapshot_metadata = dict(metadata or {})
    snapshot_metadata.setdefault(
        "isolated_task_ids",
        [task_id for task_id in selected_task_ids if task_id not in connected_task_ids],
    )
    snapshot_metadata.setdefault("edge_count", len(serialized_edges))
    snapshot_metadata.setdefault("task_count", len(selected_task_ids))

    return TaskGraphSnapshot(
        run_id=run_id,
        iteration=iteration,
        task_ids=selected_task_ids,
        feature_cutoff=feature_cutoff,
        edge_records=serialized_edges,
        graph_policy=graph_policy,
        seed=seed,
        metadata=snapshot_metadata,
    )


def construct_snapshot_from_artifacts(
    *,
    graph_dir: Path,
    task_ids: Sequence[str],
    run_id: str,
    iteration: int,
    graph_policy: str = "precomputed_similarity",
    feature_cutoff: datetime | None = None,
    seed: int | None = None,
    use_threshold_cut: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> TaskGraphSnapshot:
    """Construct one task graph snapshot directly from precomputed JSON artifacts."""
    profiles_artifact, pairwise_artifact = load_precomputed_similarity_artifacts(
        graph_dir
    )
    feature_index = construct_feature_index(
        task_profiles=profiles_artifact.profiles,
        task_ids=task_ids,
    )
    edge_records = construct_edge_records(
        pairwise_similarity=pairwise_artifact.pairs,
        task_ids=list(feature_index),
        use_threshold_cut=use_threshold_cut,
    )
    snapshot_metadata = dict(metadata or {})
    snapshot_metadata.setdefault("active_threshold", pairwise_artifact.active_threshold)
    snapshot_metadata.setdefault(
        "edge_score_threshold",
        pairwise_artifact.edge_score_threshold,
    )
    snapshot_metadata.setdefault("p20_threshold", pairwise_artifact.p20_threshold)
    snapshot_metadata.setdefault("threshold_kind", pairwise_artifact.threshold_kind)

    return construct_snapshot(
        run_id=run_id,
        iteration=iteration,
        task_ids=list(feature_index),
        graph_policy=graph_policy,
        edge_records=edge_records,
        feature_cutoff=feature_cutoff,
        seed=seed,
        metadata=snapshot_metadata,
    )


def _selected_task_ids(
    available_task_ids: Iterable[str],
    requested_task_ids: Sequence[str] | None,
) -> list[str]:
    available = set(available_task_ids)
    if requested_task_ids is None:
        return sorted(available)

    selected: list[str] = []
    seen: set[str] = set()
    missing: list[str] = []
    for task_id in requested_task_ids:
        if task_id in seen:
            continue
        seen.add(task_id)
        if task_id not in available:
            missing.append(task_id)
            continue
        selected.append(task_id)
    if missing:
        raise ValueError(f"unknown task IDs in graph artifact: {sorted(missing)}")
    return selected


def _normalize_task_ids(task_ids: Sequence[str]) -> list[str]:
    return _selected_task_ids(task_ids, task_ids)


def _pair_edges(
    pair: PairSimilarity,
    *,
    threshold_filter_applied: bool,
) -> list[TaskGraphEdgeRecord]:
    metadata = {
        "components": pair.components,
        "shared": pair.shared,
        "kept_after_p20_cut": pair.kept_after_p20_cut,
        "kept_after_threshold_cut": pair.kept_after_threshold_cut,
        "symmetric": True,
        "threshold_filter_applied": threshold_filter_applied,
    }
    return [
        TaskGraphEdgeRecord(
            source_task_id=pair.source,
            target_task_id=pair.target,
            relation="precomputed_similarity",
            weight=pair.score,
            metadata=metadata,
        ),
        TaskGraphEdgeRecord(
            source_task_id=pair.target,
            target_task_id=pair.source,
            relation="precomputed_similarity",
            weight=pair.score,
            metadata=metadata,
        ),
    ]
