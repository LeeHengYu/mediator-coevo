from __future__ import annotations

import json

import pytest

from mediated_coevo.diffusion import (
    DiffusedRecord,
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    DiffusionStore,
    TaskGraphEdgeRecord,
    TaskGraphSnapshot,
)


def _artifact(
    artifact_id: str,
    *,
    source_task_id: str = "task-a",
    source_iteration: int = 1,
) -> DiffusionArtifact:
    return DiffusionArtifact(
        artifact_id=artifact_id,
        source_task_id=source_task_id,
        source_iteration=source_iteration,
        source_run_id="run-1",
        artifact_type=DiffusionArtifactType.DEBUG_HINT,
        risk_level=DiffusionRiskLevel.LOW,
        content=f"hint from {source_task_id}",
        evidence_trace_ids=["trace-1"],
        evidence_report_ids=["report-1"],
        verifier_reward=0.7,
        token_cost=32,
    )


def _snapshot(
    snapshot_id: str,
    *,
    iteration: int = 2,
    run_id: str = "run-1",
) -> TaskGraphSnapshot:
    return TaskGraphSnapshot(
        snapshot_id=snapshot_id,
        run_id=run_id,
        iteration=iteration,
        task_ids=["task-a", "task-b"],
        graph_policy="same-benchmark-family",
        seed=42,
        edge_records=[
            TaskGraphEdgeRecord(
                source_task_id="task-a",
                target_task_id="task-b",
                relation="same-benchmark-family",
                weight=0.9,
            )
        ],
    )


def _diffused_record(
    artifact_id: str,
    *,
    target_task_id: str = "task-b",
    target_iteration: int = 3,
) -> DiffusedRecord:
    return DiffusedRecord(
        artifact_id=artifact_id,
        source_task_id="task-a",
        source_iteration=1,
        source_run_id="run-1",
        target_task_id=target_task_id,
        target_iteration=target_iteration,
        target_run_id="run-1",
        snapshot_id="snapshot-1",
        policy_name="capped_broadcast",
        relation="broadcast",
        reason="selected_in_top_3_by_recency",
        eligible=True,
        selected=True,
        rendered=True,
        rendered_section="Diffused Cross-Task Context",
        token_count=24,
        cited_by="planner",
        citation_text="artifact_id=artifact-1",
        verifier_reward=1.0,
        judge_reward=0.8,
        success=True,
        regression=False,
        notes="rendered artifact preceded a clean solve",
    )


def test_store_artifact_round_trips_and_queries_by_source_task(tmp_path):
    store = DiffusionStore(tmp_path / "diffusion")
    store.store_artifact(
        _artifact("artifact-1", source_task_id="task-a", source_iteration=1)
    )
    store.store_artifact(
        _artifact("artifact-2", source_task_id="task-b", source_iteration=2)
    )

    loaded = store.load_artifact("artifact-1")
    artifacts = store.query_artifacts(
        source_task_id="task-a",
        recent=5,
        before_source_iteration=2,
    )

    assert loaded is not None
    assert loaded.artifact_type == DiffusionArtifactType.DEBUG_HINT
    assert [artifact.artifact_id for artifact in artifacts] == ["artifact-1"]


def test_store_artifact_rejects_double_write_without_overwrite(tmp_path):
    store = DiffusionStore(tmp_path / "diffusion")
    artifact = _artifact("artifact-1")

    store.store_artifact(artifact)

    with pytest.raises(FileExistsError):
        store.store_artifact(artifact)


def test_store_graph_snapshot_overwrite_replaces_existing_snapshot(tmp_path):
    store = DiffusionStore(tmp_path / "diffusion")
    first = _snapshot("snapshot-1", iteration=1)
    second = _snapshot("snapshot-1", iteration=4)

    store.store_graph_snapshot(first)
    store.store_graph_snapshot(second, overwrite=True)

    loaded = store.load_graph_snapshot("snapshot-1")
    assert loaded is not None
    assert loaded.iteration == 4


def test_append_diffused_records_round_trip(tmp_path):
    store = DiffusionStore(tmp_path / "diffusion")
    artifact = _artifact("artifact-1")
    store.store_artifact(artifact)

    record = _diffused_record("artifact-1")
    store.append_diffused_record(record)

    loaded = store.query_diffused_records(target_task_id="task-b")

    assert loaded[0].eligible is True
    assert loaded[0].selected is True
    assert loaded[0].rendered is True
    assert loaded[0].token_count == 24
    assert loaded[0].cited_by == "planner"
    assert loaded[0].success is True


def test_query_diffused_records_skips_malformed_jsonl_lines(tmp_path, caplog):
    store = DiffusionStore(tmp_path / "diffusion")
    store.append_diffused_record(_diffused_record("artifact-1", target_iteration=2))
    path = tmp_path / "diffusion" / "diffused_records.jsonl"
    path.write_text(
        path.read_text() + '{"artifact_id": "artifact-2", "target_iteration": "bad"}\n'
    )

    records = store.query_diffused_records(target_task_id="task-b", recent=10)

    assert [record.artifact_id for record in records] == ["artifact-1"]
    assert "Failed to load DiffusedRecord" in caplog.text


def test_saved_artifact_store_round_trips_as_preloaded_iteration(tmp_path):
    source = DiffusionStore(tmp_path / "source")
    source.store_artifact(_artifact("artifact-1", source_iteration=2))
    saved = tmp_path / "saved-store"

    saved_count = source.save_artifact_store(saved, store_id="experiment-1")
    manifest = json.loads((saved / "manifest.json").read_text())
    target = DiffusionStore(tmp_path / "target")
    imported_count = target.import_artifact_store(saved)
    loaded = target.load_artifact("artifact-1")
    visible = target.query_artifacts(before_source_iteration=0)

    assert saved_count == 1
    assert manifest["id"] == "experiment-1"
    assert manifest["artifact_count"] == 1
    assert imported_count == 1
    assert loaded is not None
    assert loaded.source_iteration == -1
    assert loaded.metadata["original_source_iteration"] == 2
    assert loaded.metadata["preloaded_artifact_store_frozen"] is False
    assert [artifact.artifact_id for artifact in visible] == ["artifact-1"]


def test_import_artifact_store_records_frozen_preload_metadata(tmp_path):
    source = DiffusionStore(tmp_path / "source")
    source.store_artifact(_artifact("artifact-1", source_iteration=2))
    saved = tmp_path / "saved-store"
    source.save_artifact_store(saved, store_id="experiment-1")
    target = DiffusionStore(tmp_path / "target")

    target.import_artifact_store(saved, frozen=True)
    loaded = target.load_artifact("artifact-1")

    assert loaded is not None
    assert loaded.metadata["preloaded_from_artifact_store"] == str(saved)
    assert loaded.metadata["original_source_iteration"] == 2
    assert loaded.metadata["preloaded_artifact_store_frozen"] is True


def test_save_artifact_store_rejects_empty_store(tmp_path):
    store = DiffusionStore(tmp_path / "empty")

    with pytest.raises(ValueError, match="contains no diffusion artifacts"):
        store.save_artifact_store(tmp_path / "saved", store_id="empty-run")
