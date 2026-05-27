from __future__ import annotations

import pytest

from mediated_coevo.diffusion import (
    CandidateRecord,
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    DiffusionStore,
    OutcomeAssociation,
    RenderRecord,
    SelectionRecord,
    TaskGraphEdgeRecord,
    TaskGraphSnapshot,
    UseCitationRecord,
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


def _candidate_record(
    artifact_id: str,
    *,
    target_task_id: str = "task-b",
    target_iteration: int = 3,
) -> CandidateRecord:
    return CandidateRecord(
        artifact_id=artifact_id,
        source_task_id="task-a",
        source_iteration=1,
        source_run_id="run-1",
        target_task_id=target_task_id,
        target_iteration=target_iteration,
        target_run_id="run-1",
        snapshot_id="snapshot-1",
        policy_name="top_k_similarity",
        relation="same-benchmark-family",
        eligible=True,
        reason="score above threshold",
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


def test_append_record_ledgers_round_trip(tmp_path):
    store = DiffusionStore(tmp_path / "diffusion")
    artifact = _artifact("artifact-1")
    store.store_artifact(artifact)

    candidate = _candidate_record("artifact-1")
    selection = SelectionRecord(
        artifact_id="artifact-1",
        source_task_id="task-a",
        source_iteration=1,
        source_run_id="run-1",
        target_task_id="task-b",
        target_iteration=3,
        target_run_id="run-1",
        snapshot_id="snapshot-1",
        policy_name="top_k_similarity",
        relation="same-benchmark-family",
        selected=True,
        reason="highest score",
    )
    render = RenderRecord(
        artifact_id="artifact-1",
        source_task_id="task-a",
        source_iteration=1,
        source_run_id="run-1",
        target_task_id="task-b",
        target_iteration=3,
        target_run_id="run-1",
        snapshot_id="snapshot-1",
        policy_name="top_k_similarity",
        relation="same-benchmark-family",
        rendered_section="Diffused Cross-Task Context",
        token_count=24,
    )
    citation = UseCitationRecord(
        artifact_id="artifact-1",
        source_task_id="task-a",
        source_iteration=1,
        source_run_id="run-1",
        target_task_id="task-b",
        target_iteration=3,
        target_run_id="run-1",
        snapshot_id="snapshot-1",
        cited_by="planner",
        citation_text="artifact_id=artifact-1",
    )
    outcome = OutcomeAssociation(
        artifact_id="artifact-1",
        source_task_id="task-a",
        source_iteration=1,
        source_run_id="run-1",
        target_task_id="task-b",
        target_iteration=3,
        target_run_id="run-1",
        snapshot_id="snapshot-1",
        verifier_reward=1.0,
        judge_reward=0.8,
        success=True,
        regression=False,
        notes="rendered artifact preceded a clean solve",
    )

    store.append_candidate_record(candidate)
    store.append_selection_record(selection)
    store.append_render_record(render)
    store.append_use_citation_record(citation)
    store.append_outcome_association(outcome)

    assert store.query_candidate_records(target_task_id="task-b")[0].eligible is True
    assert store.query_selection_records(artifact_id="artifact-1")[0].selected is True
    assert store.query_render_records(target_task_id="task-b")[0].token_count == 24
    assert (
        store.query_use_citation_records(artifact_id="artifact-1")[0].cited_by
        == "planner"
    )
    assert store.query_outcome_associations(target_task_id="task-b")[0].success is True


def test_query_candidate_records_skips_malformed_jsonl_lines(tmp_path, caplog):
    store = DiffusionStore(tmp_path / "diffusion")
    store.append_candidate_record(_candidate_record("artifact-1", target_iteration=2))
    path = tmp_path / "diffusion" / "candidate_records.jsonl"
    path.write_text(
        path.read_text()
        + '{"artifact_id": "artifact-2", "target_iteration": "bad"}\n'
    )

    records = store.query_candidate_records(target_task_id="task-b", recent=10)

    assert [record.artifact_id for record in records] == ["artifact-1"]
    assert "Failed to load CandidateRecord" in caplog.text
