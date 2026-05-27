from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from mediated_coevo.diffusion import (
    DiffusionStore,
    construct_edge_records,
    construct_feature_index,
    construct_snapshot,
    construct_snapshot_from_artifacts,
    load_precomputed_similarity_artifacts,
)


def test_construct_feature_index_filters_selected_task_ids() -> None:
    index = construct_feature_index(
        task_profiles=_task_profiles()["profiles"],
        task_ids=["task-b", "task-a"],
    )

    assert list(index) == ["task-b", "task-a"]
    assert index["task-a"].category == "build"
    assert index["task-b"].tags == ["python", "build"]


def test_construct_feature_index_rejects_unknown_task_ids() -> None:
    with pytest.raises(ValueError, match="unknown task IDs"):
        construct_feature_index(
            task_profiles=_task_profiles()["profiles"],
            task_ids=["missing-task"],
        )


def test_construct_edge_records_respects_threshold_cut_and_is_bidirectional() -> None:
    edges = construct_edge_records(
        pairwise_similarity=_pairwise_similarity()["pairs"],
        task_ids=["task-a", "task-b", "task-c"],
        use_threshold_cut=True,
    )

    assert [(edge.source_task_id, edge.target_task_id) for edge in edges] == [
        ("task-a", "task-b"),
        ("task-b", "task-a"),
    ]
    assert all(edge.relation == "precomputed_similarity" for edge in edges)
    assert all(edge.weight == pytest.approx(0.6) for edge in edges)
    assert all(edge.metadata["threshold_filter_applied"] is True for edge in edges)


def test_construct_edge_records_can_include_unkept_pairs() -> None:
    edges = construct_edge_records(
        pairwise_similarity=_pairwise_similarity()["pairs"],
        task_ids=["task-a", "task-b", "task-c"],
        use_threshold_cut=False,
    )

    assert [(edge.source_task_id, edge.target_task_id) for edge in edges] == [
        ("task-a", "task-b"),
        ("task-b", "task-a"),
        ("task-b", "task-c"),
        ("task-c", "task-b"),
    ]
    assert all(edge.metadata["threshold_filter_applied"] is False for edge in edges)


def test_construct_snapshot_records_task_ids_and_isolated_nodes() -> None:
    edges = construct_edge_records(
        pairwise_similarity=_pairwise_similarity()["pairs"],
        task_ids=["task-a", "task-b", "task-c"],
        use_threshold_cut=True,
    )

    snapshot = construct_snapshot(
        run_id="run-1",
        iteration=3,
        task_ids=["task-a", "task-b", "task-c"],
        graph_policy="precomputed_similarity",
        edge_records=edges,
    )

    assert snapshot.task_ids == ["task-a", "task-b", "task-c"]
    assert snapshot.metadata["task_count"] == 3
    assert snapshot.metadata["edge_count"] == 2
    assert snapshot.metadata["isolated_task_ids"] == ["task-c"]


def test_construct_snapshot_from_artifacts_round_trips_through_store(
    tmp_path: Path,
) -> None:
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    (graph_dir / "task_profiles.json").write_text(json.dumps(_task_profiles()))
    (graph_dir / "pairwise_similarity.json").write_text(
        json.dumps(_pairwise_similarity())
    )

    profiles_artifact, pairwise_artifact = load_precomputed_similarity_artifacts(
        graph_dir
    )
    assert profiles_artifact.task_count == 3
    assert pairwise_artifact.pair_count == 2

    snapshot = construct_snapshot_from_artifacts(
        graph_dir=graph_dir,
        task_ids=["task-a", "task-b", "task-c"],
        run_id="run-2",
        iteration=4,
        feature_cutoff=datetime(2026, 5, 27, 12, 0, 0),
        seed=42,
    )

    store = DiffusionStore(tmp_path / "diffusion")
    store.store_graph_snapshot(snapshot)
    loaded = store.load_graph_snapshot(snapshot.snapshot_id)

    assert loaded is not None
    assert loaded.run_id == "run-2"
    assert loaded.iteration == 4
    assert loaded.task_ids == ["task-a", "task-b", "task-c"]
    assert len(loaded.edge_records) == 2
    assert loaded.metadata["active_threshold"] == pytest.approx(0.05)
    assert loaded.metadata["threshold_kind"] == "absolute_score"


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
                "components": {"category": 1.0, "tags": 1.0},
                "shared": {"tags": ["python", "build"]},
                "kept_after_p20_cut": True,
                "kept_after_threshold_cut": True,
            },
            {
                "source": "task-b",
                "target": "task-c",
                "score": 0.02,
                "components": {"category": 0.0, "tags": 0.0},
                "shared": {"tags": []},
                "kept_after_p20_cut": True,
                "kept_after_threshold_cut": False,
            },
        ],
    }
