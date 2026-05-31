"""Object-oriented PR2 task graph materialization from precomputed artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from mediated_coevo.analysis.task_similarity import PairSimilarity, TaskSimilarityProfile
from mediated_coevo.diffusion.models import TaskGraphEdgeRecord, TaskGraphSnapshot

if TYPE_CHECKING:
    from mediated_coevo.diffusion.store import DiffusionStore


class TaskProfilesArtifact(BaseModel):
    """Validated contents of a task profile precompute artifact."""

    task_count: int
    profiles: dict[str, TaskSimilarityProfile]


class PairwiseSimilarityArtifact(BaseModel):
    """Validated contents of a pairwise similarity precompute artifact."""

    graph_kind: str | None = None
    pair_count: int
    p20_threshold: float | None = None
    edge_score_threshold: float | None = None
    active_threshold: float | None = None
    threshold_kind: str | None = None
    pairs: list[PairSimilarity]


@dataclass(slots=True)
class GraphBuildSpec:
    """Inputs required to build one graph snapshot from precomputed artifacts."""

    graph_dir: Path
    task_ids: list[str]
    run_id: str
    iteration: int
    graph_policy: str = "precomputed_similarity"
    feature_cutoff: datetime | None = None
    seed: int | None = None
    use_threshold_cut: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskGraphNode:
    """One selected task node in the PR2 diffusion graph."""

    task_id: str
    profile: TaskSimilarityProfile


class DiffusionNetwork:
    """In-memory PR2 graph facade over precomputed similarity artifacts."""

    def __init__(self, spec: GraphBuildSpec) -> None:
        self.spec = spec
        self._profiles_artifact: TaskProfilesArtifact | None = None
        self._pairwise_artifact: PairwiseSimilarityArtifact | None = None
        self._task_ids: list[str] = []
        self._nodes_by_task_id: dict[str, TaskGraphNode] = {}
        self._adj_list: defaultdict[str, list[TaskGraphEdgeRecord]] = defaultdict(list)
        self._snapshot: TaskGraphSnapshot | None = None

    @classmethod
    def from_graph_dir(cls, spec: GraphBuildSpec) -> DiffusionNetwork:
        """Build one network directly from a graph artifact directory."""
        return cls(spec).build()

    @classmethod
    def _freeze_snapshot(
        cls,
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
        """Freeze one graph structure into a serializable snapshot."""
        selected_task_ids = cls._normalize_task_ids(task_ids)
        serialized_edges = list(edge_records)
        connected_task_ids = {
            edge.source_task_id for edge in serialized_edges
        } | {edge.target_task_id for edge in serialized_edges}
        snapshot_metadata = dict(metadata or {})
        snapshot_metadata.setdefault(
            "isolated_task_ids",
            [
                task_id
                for task_id in selected_task_ids
                if task_id not in connected_task_ids
            ],
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

    def build(self) -> DiffusionNetwork:
        """Load artifacts and materialize the selected graph view."""
        self._load_artifacts()
        self._task_ids = self._select_task_ids()
        self._nodes_by_task_id = self._build_nodes()
        edge_records = self._build_edge_records()
        self._adj_list = self._build_adj_list(
            task_ids=self._task_ids,
            edge_records=edge_records,
        )
        self._snapshot = None
        return self

    def get_node(self, task_id: str) -> TaskGraphNode:
        """Return one selected node by task ID."""
        self._require_built()
        try:
            return self._nodes_by_task_id[task_id]
        except KeyError as exc:
            raise ValueError(f"unknown task ID in diffusion network: {task_id}") from exc

    def get_nodes(self) -> list[TaskGraphNode]:
        """Return selected nodes in configured task order."""
        self._require_built()
        return [self._nodes_by_task_id[task_id] for task_id in self._task_ids]

    def get_edge_records(self) -> list[TaskGraphEdgeRecord]:
        """Return all directed edge records in stable order."""
        self._require_built()
        return _flatten_adj_list(self._adj_list)

    def get_edges_for(self, task_id: str) -> list[TaskGraphEdgeRecord]:
        """Return outgoing edges for one task."""
        self._require_built()
        self.get_node(task_id)
        return list(self._adj_list[task_id])

    def get_adj_list(self) -> dict[str, list[TaskGraphEdgeRecord]]:
        """Return outgoing graph adjacency keyed by source task ID."""
        self._require_built()
        return _copy_adj_list(self._adj_list, self._task_ids)

    def get_neighbors(self, task_id: str) -> list[TaskGraphNode]:
        """Return neighboring nodes reachable by one outgoing edge."""
        self._require_built()
        neighbor_ids = [edge.target_task_id for edge in self.get_edges_for(task_id)]
        return [self._nodes_by_task_id[neighbor_id] for neighbor_id in neighbor_ids]

    def get_isolated_task_ids(self) -> list[str]:
        """Return selected task IDs with no incident directed edges."""
        self._require_built()
        edge_records = self.get_edge_records()
        connected_task_ids = {edge.source_task_id for edge in edge_records} | {
            edge.target_task_id for edge in edge_records
        }
        return [
            task_id for task_id in self._task_ids if task_id not in connected_task_ids
        ]

    def to_snapshot(self) -> TaskGraphSnapshot:
        """Return the frozen graph snapshot for the current build."""
        self._require_built()
        if self._snapshot is None:
            edge_records = self.get_edge_records()
            self._snapshot = self._freeze_snapshot(
                run_id=self.spec.run_id,
                iteration=self.spec.iteration,
                task_ids=self._task_ids,
                graph_policy=self.spec.graph_policy,
                edge_records=edge_records,
                feature_cutoff=self.spec.feature_cutoff,
                seed=self.spec.seed,
                metadata=self._snapshot_metadata(),
            )
        return self._snapshot.model_copy(deep=True)

    def persist_snapshot(
        self,
        store: DiffusionStore,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist the current snapshot to the diffusion store."""
        return store.store_graph_snapshot(self.to_snapshot(), overwrite=overwrite)

    def _load_artifacts(self) -> None:
        (
            self._profiles_artifact,
            self._pairwise_artifact,
        ) = self._load_precomputed_similarity_artifacts(self.spec.graph_dir)

    def _select_task_ids(self) -> list[str]:
        profiles_artifact = self._require_profiles_artifact()
        return self._selected_task_ids(
            profiles_artifact.profiles.keys(),
            self.spec.task_ids,
        )

    def _build_nodes(self) -> dict[str, TaskGraphNode]:
        profiles_artifact = self._require_profiles_artifact()
        feature_index = self._validate_task_profiles(
            task_profiles=profiles_artifact.profiles,
            task_ids=self._task_ids,
        )
        return {
            task_id: TaskGraphNode(task_id=task_id, profile=feature_index[task_id])
            for task_id in self._task_ids
        }

    def _build_edge_records(self) -> list[TaskGraphEdgeRecord]:
        pairwise_artifact = self._require_pairwise_artifact()
        return self._translate_pairwise_edges(
            pairwise_similarity=pairwise_artifact.pairs,
            task_ids=self._task_ids,
            use_threshold_cut=self.spec.use_threshold_cut,
        )

    @staticmethod
    def _build_adj_list(
        *,
        task_ids: Sequence[str],
        edge_records: Sequence[TaskGraphEdgeRecord],
    ) -> defaultdict[str, list[TaskGraphEdgeRecord]]:
        adjacency: defaultdict[str, list[TaskGraphEdgeRecord]] = defaultdict(list)
        for task_id in task_ids:
            adjacency[task_id]
        for edge in edge_records:
            adjacency[edge.source_task_id].append(edge)
        return adjacency

    def _snapshot_metadata(self) -> dict[str, Any]:
        pairwise_artifact = self._require_pairwise_artifact()
        snapshot_metadata = dict(self.spec.metadata)
        snapshot_metadata.setdefault("graph_kind", pairwise_artifact.graph_kind)
        snapshot_metadata.setdefault("active_threshold", pairwise_artifact.active_threshold)
        snapshot_metadata.setdefault(
            "edge_score_threshold",
            pairwise_artifact.edge_score_threshold,
        )
        snapshot_metadata.setdefault("p20_threshold", pairwise_artifact.p20_threshold)
        snapshot_metadata.setdefault("threshold_kind", pairwise_artifact.threshold_kind)
        return snapshot_metadata

    def _require_built(self) -> None:
        if not self._nodes_by_task_id:
            raise RuntimeError("diffusion network has not been built yet")

    def _require_profiles_artifact(self) -> TaskProfilesArtifact:
        if self._profiles_artifact is None:
            raise RuntimeError("task profiles artifact has not been loaded yet")
        return self._profiles_artifact

    def _require_pairwise_artifact(self) -> PairwiseSimilarityArtifact:
        if self._pairwise_artifact is None:
            raise RuntimeError("pairwise similarity artifact has not been loaded yet")
        return self._pairwise_artifact

    @classmethod
    def _validate_task_profiles(
        cls,
        *,
        task_profiles: Mapping[str, TaskSimilarityProfile | Mapping[str, Any]],
        task_ids: Sequence[str] | None = None,
    ) -> dict[str, TaskSimilarityProfile]:
        selected_task_ids = cls._selected_task_ids(task_profiles.keys(), task_ids)
        return {
            task_id: TaskSimilarityProfile.model_validate(task_profiles[task_id])
            for task_id in selected_task_ids
        }

    @classmethod
    def _translate_pairwise_edges(
        cls,
        *,
        pairwise_similarity: Sequence[PairSimilarity | Mapping[str, Any]],
        task_ids: Sequence[str],
        use_threshold_cut: bool = True,
    ) -> list[TaskGraphEdgeRecord]:
        selected_task_ids = cls._normalize_task_ids(task_ids)
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
            edges.append(
                cls._edge_record(
                    pair,
                    threshold_filter_applied=use_threshold_cut,
                )
            )

        return sorted(
            edges,
            key=lambda edge: (edge.source_task_id, edge.target_task_id, edge.relation),
        )

    @staticmethod
    def _load_precomputed_similarity_artifacts(
        graph_dir: Path,
    ) -> tuple[TaskProfilesArtifact, PairwiseSimilarityArtifact]:
        profiles = TaskProfilesArtifact.model_validate(
            json.loads((graph_dir / "task_profiles.json").read_text(encoding="utf-8"))
        )
        pairwise = PairwiseSimilarityArtifact.model_validate(
            json.loads(
                (graph_dir / "pairwise_similarity.json").read_text(encoding="utf-8")
            )
        )
        return profiles, pairwise

    @staticmethod
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

    @classmethod
    def _normalize_task_ids(cls, task_ids: Sequence[str]) -> list[str]:
        return cls._selected_task_ids(task_ids, task_ids)

    @staticmethod
    def _edge_record(
        pair: PairSimilarity,
        *,
        threshold_filter_applied: bool,
    ) -> TaskGraphEdgeRecord:
        metadata = {
            **pair.metadata,
            "components": pair.components,
            "shared": pair.shared,
            "kept_after_p20_cut": pair.kept_after_p20_cut,
            "kept_after_threshold_cut": pair.kept_after_threshold_cut,
            "threshold_filter_applied": threshold_filter_applied,
        }
        return TaskGraphEdgeRecord(
            source_task_id=pair.source,
            target_task_id=pair.target,
            relation="precomputed_similarity",
            weight=pair.score,
            metadata=metadata,
        )


def adjacency_from_snapshot(
    snapshot: TaskGraphSnapshot,
) -> dict[str, list[TaskGraphEdgeRecord]]:
    """Return outgoing adjacency from a persisted graph snapshot."""
    adjacency = DiffusionNetwork._build_adj_list(
        task_ids=snapshot.task_ids,
        edge_records=snapshot.edge_records,
    )
    return _copy_adj_list(adjacency, snapshot.task_ids)


def _copy_adj_list(
    adjacency: Mapping[str, Sequence[TaskGraphEdgeRecord]],
    task_ids: Sequence[str],
) -> dict[str, list[TaskGraphEdgeRecord]]:
    return {task_id: list(adjacency[task_id]) for task_id in task_ids}


def _flatten_adj_list(
    adjacency: Mapping[str, Sequence[TaskGraphEdgeRecord]],
) -> list[TaskGraphEdgeRecord]:
    edges = [edge for edge_records in adjacency.values() for edge in edge_records]
    return sorted(
        edges,
        key=lambda edge: (edge.source_task_id, edge.target_task_id, edge.relation),
    )
