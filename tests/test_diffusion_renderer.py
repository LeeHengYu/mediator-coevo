from __future__ import annotations

from mediated_coevo.diffusion import (
    DIFFUSED_SECTION_NAME,
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    DiffusionStore,
    DiffusionSubscription,
    TaskGraphSnapshot,
    render_diffusion_subscriptions,
)


def test_render_diffusion_subscriptions_writes_context_and_audit_record(tmp_path):
    store = DiffusionStore(tmp_path / "diffusion")
    snapshot = TaskGraphSnapshot(
        snapshot_id="snapshot-1",
        run_id="run-1",
        iteration=2,
        task_ids=["task-a", "task-b"],
        graph_policy="broadcast",
    )
    artifact = DiffusionArtifact(
        artifact_id="artifact-1",
        source_task_id="task-b",
        source_iteration=1,
        source_run_id="source-run",
        artifact_type=DiffusionArtifactType.DEBUG_HINT,
        risk_level=DiffusionRiskLevel.LOW,
        content="reuse the parser guard",
    )
    subscription = DiffusionSubscription(
        artifact=artifact,
        policy_name="capped_broadcast",
        relation="broadcast",
        reason="selected_by_test",
        metadata={"rank": 1},
    )

    bundle = render_diffusion_subscriptions(
        store=store,
        snapshot=snapshot,
        model="openrouter/openai/gpt-5.5",
        target_task_id="task-a",
        target_iteration=2,
        target_run_id="target-run",
        subscriptions=[subscription],
        eligible_count=3,
    )

    assert bundle.text is not None
    assert "Diffused Cross-Task Context" in bundle.text
    assert "artifact_id=artifact-1" in bundle.text
    assert "content=reuse the parser guard" in bundle.text
    assert bundle.eligible_count == 3
    assert bundle.selected_count == 1
    assert bundle.rendered_count == 1
    assert bundle.source_task_ids == ["task-b"]
    assert bundle.context_tokens > 0

    records = store.query_diffused_records(target_task_id="task-a")
    assert len(records) == 1
    record = records[0]
    assert record.artifact_id == "artifact-1"
    assert record.rendered_section == DIFFUSED_SECTION_NAME
    assert record.selected is True
    assert record.rendered is True
    assert record.token_count > 0
    assert record.metadata == {
        "artifact_type": "debug_hint",
        "risk_level": "low",
        "rank": 1,
    }


def test_render_diffusion_subscriptions_returns_empty_bundle_without_subscriptions(
    tmp_path,
):
    store = DiffusionStore(tmp_path / "diffusion")
    snapshot = TaskGraphSnapshot(
        snapshot_id="snapshot-1",
        run_id="run-1",
        iteration=2,
        task_ids=["task-a"],
        graph_policy="broadcast",
    )

    bundle = render_diffusion_subscriptions(
        store=store,
        snapshot=snapshot,
        model="openrouter/openai/gpt-5.5",
        target_task_id="task-a",
        target_iteration=2,
        target_run_id=None,
        subscriptions=[],
    )

    assert bundle.text is None
    assert bundle.eligible_count == 0
    assert bundle.selected_count == 0
    assert bundle.rendered_count == 0
    assert bundle.context_tokens == 0
    assert store.query_diffused_records(target_task_id="task-a") == []
