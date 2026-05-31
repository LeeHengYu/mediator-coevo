from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from mediated_coevo.diffusion import (
    DiffusionStore,
    DiffusionNetwork,
    GraphBuildSpec,
    adjacency_from_snapshot,
)


def test_diffusion_network_filters_selected_task_ids(tmp_path: Path) -> None:
    network = _build_network(
        tmp_path,
        task_ids=["task-b", "task-a"],
        run_id="run-filter",
        iteration=1,
    )

    assert [node.task_id for node in network.get_nodes()] == ["task-b", "task-a"]
    assert network.get_node("task-a").profile.category == "build"
    assert network.get_node("task-b").profile.tags == ["python", "build"]


def test_diffusion_network_rejects_unknown_task_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown task IDs"):
        _build_network(
            tmp_path,
            task_ids=["missing-task"],
            run_id="run-missing",
            iteration=1,
        )


def test_diffusion_network_respects_threshold_cut_for_directed_pairs(
    tmp_path: Path,
) -> None:
    network = _build_network(
        tmp_path,
        task_ids=["task-a", "task-b", "task-c"],
        run_id="run-threshold",
        iteration=2,
        use_threshold_cut=True,
    )
    edges = network.get_edge_records()

    assert [(edge.source_task_id, edge.target_task_id) for edge in edges] == [
        ("task-a", "task-b"),
    ]
    assert all(edge.relation == "precomputed_similarity" for edge in edges)
    assert all(edge.weight == pytest.approx(0.6) for edge in edges)
    assert all(edge.metadata["threshold_filter_applied"] is True for edge in edges)


def test_diffusion_network_can_include_unkept_pairs(tmp_path: Path) -> None:
    network = _build_network(
        tmp_path,
        task_ids=["task-a", "task-b", "task-c"],
        run_id="run-unkept",
        iteration=2,
        use_threshold_cut=False,
    )
    edges = network.get_edge_records()

    assert [(edge.source_task_id, edge.target_task_id) for edge in edges] == [
        ("task-a", "task-b"),
        ("task-b", "task-c"),
    ]
    assert all(edge.metadata["threshold_filter_applied"] is False for edge in edges)


def test_diffusion_network_materializes_directed_skillflow_edges(tmp_path: Path) -> None:
    network = _build_network(
        tmp_path,
        task_ids=["task-a", "task-b", "task-c"],
        run_id="run-directed",
        iteration=2,
        use_threshold_cut=False,
    )
    edges = network.get_edge_records()

    assert [(edge.source_task_id, edge.target_task_id) for edge in edges] == [
        ("task-a", "task-b"),
        ("task-b", "task-c"),
    ]
    assert edges[0].metadata["edge_kind"] == "same_family_forward"
    assert edges[0].metadata["rank_gap"] == 1


def test_diffusion_network_snapshot_records_task_ids_and_isolated_nodes(
    tmp_path: Path,
) -> None:
    network = _build_network(
        tmp_path,
        task_ids=["task-a", "task-b", "task-c"],
        run_id="run-1",
        iteration=3,
    )
    snapshot = network.to_snapshot()

    assert snapshot.task_ids == ["task-a", "task-b", "task-c"]
    assert snapshot.metadata["task_count"] == 3
    assert snapshot.metadata["edge_count"] == 1
    assert snapshot.metadata["isolated_task_ids"] == ["task-c"]


def test_diffusion_network_builds_nodes_neighbors_and_snapshot(tmp_path: Path) -> None:
    network = _build_network(
        tmp_path,
        task_ids=["task-b", "task-a", "task-c"],
        run_id="run-oo",
        iteration=5,
    )

    assert [node.task_id for node in network.get_nodes()] == [
        "task-b",
        "task-a",
        "task-c",
    ]
    assert [node.task_id for node in network.get_neighbors("task-a")] == ["task-b"]
    adj_list = network.get_adj_list()
    assert list(adj_list) == ["task-b", "task-a", "task-c"]
    assert [(edge.source_task_id, edge.target_task_id) for edge in adj_list["task-a"]] == [
        ("task-a", "task-b"),
    ]
    assert adj_list["task-c"] == []
    assert network.get_isolated_task_ids() == ["task-c"]

    snapshot = network.to_snapshot()
    assert snapshot.run_id == "run-oo"
    assert snapshot.iteration == 5
    assert snapshot.task_ids == ["task-b", "task-a", "task-c"]
    assert snapshot.metadata["edge_count"] == 1
    snapshot_adj_list = adjacency_from_snapshot(snapshot)
    assert snapshot_adj_list == adj_list


def test_diffusion_network_snapshot_round_trips_through_store(
    tmp_path: Path,
) -> None:
    network = _build_network(
        tmp_path,
        task_ids=["task-a", "task-b", "task-c"],
        run_id="run-2",
        iteration=4,
        feature_cutoff=datetime(2026, 5, 27, 12, 0, 0),
        seed=42,
    )
    snapshot = network.to_snapshot()

    store = DiffusionStore(tmp_path / "diffusion")
    store.store_graph_snapshot(snapshot)
    loaded = store.load_graph_snapshot(snapshot.snapshot_id)

    assert loaded is not None
    assert loaded.run_id == "run-2"
    assert loaded.iteration == 4
    assert loaded.task_ids == ["task-a", "task-b", "task-c"]
    assert len(loaded.edge_records) == 1
    assert loaded.metadata["active_threshold"] == pytest.approx(0.05)
    assert loaded.metadata["threshold_kind"] == "absolute_score"


def _build_network(
    tmp_path: Path,
    *,
    task_ids: list[str],
    run_id: str,
    iteration: int,
    graph_policy: str = "precomputed_similarity",
    feature_cutoff: datetime | None = None,
    seed: int | None = None,
    use_threshold_cut: bool = True,
    pairwise_similarity: dict[str, Any] | None = None,
) -> DiffusionNetwork:
    graph_dir = tmp_path / f"graph-{run_id}-{iteration}"
    graph_dir.mkdir()
    (graph_dir / "task_profiles.json").write_text(json.dumps(_task_profiles()))
    (graph_dir / "pairwise_similarity.json").write_text(
        json.dumps(pairwise_similarity or _pairwise_similarity())
    )
    return DiffusionNetwork.from_graph_dir(
        GraphBuildSpec(
            graph_dir=graph_dir,
            task_ids=task_ids,
            run_id=run_id,
            iteration=iteration,
            graph_policy=graph_policy,
            feature_cutoff=feature_cutoff,
            seed=seed,
            use_threshold_cut=use_threshold_cut,
        )
    )


def _task_profiles() -> dict[str, Any]:
    return {
        "task_count": 3,
        "profiles": {
            "task-a": {
                "task_id": "task-a",
                "category": "build",
                "difficulty": "easy",
                "tags": ["python", "build"],
                "skills": [],
                "environment_files": [],
                "output_types": ["patch"],
                "domain_terms": ["python", "build"],
                "capability_labels": ["build-debugging"],
            },
            "task-b": {
                "task_id": "task-b",
                "category": "build",
                "difficulty": "easy",
                "tags": ["python", "build"],
                "skills": [],
                "environment_files": [],
                "output_types": ["patch"],
                "domain_terms": ["python", "ci"],
                "capability_labels": ["build-debugging"],
            },
            "task-c": {
                "task_id": "task-c",
                "category": "research",
                "difficulty": "medium",
                "tags": ["citation"],
                "skills": [],
                "environment_files": [],
                "output_types": ["json"],
                "domain_terms": ["citation"],
                "capability_labels": ["structured-json-output"],
            },
        },
    }


def _pairwise_similarity() -> dict[str, Any]:
    return {
        "graph_kind": "skillflow_ranked_similarity",
        "pair_count": 2,
        "p20_threshold": 0.01,
        "edge_score_threshold": 0.05,
        "active_threshold": 0.05,
        "threshold_kind": "absolute_score",
        "pairs": [
            {
                "source": "task-a",
                "target": "task-b",
                "score": 0.6,
                "components": {
                    "category": 1.0,
                    "tags": 1.0,
                    "io_shape": 1.0,
                    "instruction_text": 0.2,
                },
                "shared": {"tags": ["python", "build"], "io_shape": ["patch"]},
                "metadata": {
                    "directed": True,
                    "edge_kind": "same_family_forward",
                    "same_family": True,
                    "source_family": "family-a",
                    "target_family": "family-a",
                    "source_rank": 0,
                    "target_rank": 1,
                    "source_family_size": 2,
                    "target_family_size": 2,
                    "rank_gap": 1,
                    "rank_affinity": 1.0,
                },
                "kept_after_p20_cut": True,
                "kept_after_threshold_cut": True,
            },
            {
                "source": "task-b",
                "target": "task-c",
                "score": 0.02,
                "components": {
                    "category": 0.0,
                    "tags": 0.0,
                    "io_shape": 0.0,
                    "instruction_text": 0.1,
                },
                "shared": {"tags": [], "io_shape": []},
                "kept_after_threshold_cut": False,
                "metadata": {
                    "directed": True,
                    "edge_kind": "cross_family",
                    "same_family": False,
                    "source_family": "family-a",
                    "target_family": "family-b",
                    "source_rank": 1,
                    "target_rank": 0,
                    "source_family_size": 2,
                    "target_family_size": 1,
                    "rank_gap": None,
                    "rank_affinity": 0.0,
                },
                "kept_after_p20_cut": True,
            },
        ],
    }
