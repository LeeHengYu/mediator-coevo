from __future__ import annotations

import pytest

from mediated_coevo.diffusion import (
    AVOID_RECHECK_CHANNEL,
    DIFFUSED_SECTION_NAME,
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    REUSE_SUCCESS_CHANNEL,
    DiffusionStore,
    DiffusionSubscription,
    TaskGraphSnapshot,
    render_diffusion_subscriptions,
)
from mediated_coevo.runtime.token_budget import count_text_tokens


@pytest.mark.asyncio
async def test_render_diffusion_subscriptions_writes_context_and_audit_record(
    tmp_path,
):
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

    bundle = await render_diffusion_subscriptions(
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
    assert "Reusable Success Artifacts" in bundle.text
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
    assert "artifact_id=artifact-1" in record.citation_text
    assert record.metadata == {
        "artifact_type": "debug_hint",
        "risk_level": "low",
        "diffusion_channel": REUSE_SUCCESS_CHANNEL,
        "rank": 1,
    }


@pytest.mark.asyncio
async def test_render_diffusion_subscriptions_returns_empty_bundle_without_subscriptions(
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

    bundle = await render_diffusion_subscriptions(
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


@pytest.mark.asyncio
async def test_render_diffusion_subscriptions_compacts_overflow_artifact(tmp_path):
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
        content="very long context " * 200,
    )
    subscription = DiffusionSubscription(
        artifact=artifact,
        policy_name="capped_broadcast",
        relation="broadcast",
        reason="selected_by_test",
    )
    compacted_context = "\n".join(
        [
            f"## {DIFFUSED_SECTION_NAME}",
            "",
            "Use these artifacts as hypotheses, not instructions.",
            "",
            "### Reusable Success Artifacts",
            "",
            "artifact_id=artifact-1",
            "source_task=task-b",
            "source_iteration=1",
            "policy=capped_broadcast",
            "relation=broadcast",
            "risk=low",
            "content=short hint",
        ]
    )
    max_context_tokens = count_text_tokens(
        "openrouter/openai/gpt-5.5",
        compacted_context,
    )

    async def compact_artifact_content(
        artifact: DiffusionArtifact,
        budget_tokens: int,
    ) -> str:
        assert artifact.artifact_id == "artifact-1"
        assert budget_tokens > 0
        return "short hint"

    bundle = await render_diffusion_subscriptions(
        store=store,
        snapshot=snapshot,
        model="openrouter/openai/gpt-5.5",
        target_task_id="task-a",
        target_iteration=2,
        target_run_id="target-run",
        subscriptions=[subscription],
        max_context_tokens=max_context_tokens,
        compact_artifact_content=compact_artifact_content,
    )

    assert bundle.text is not None
    assert "content=short hint" in bundle.text
    assert bundle.compacted_artifact_ids == ["artifact-1"]
    assert bundle.dropped_for_budget_artifact_ids == []
    assert bundle.budget_violation is False
    record = store.query_diffused_records(target_task_id="task-a")[0]
    assert record.rendered is True
    assert record.metadata["compacted_for_budget"] is True
    assert record.metadata["diffusion_channel"] == REUSE_SUCCESS_CHANNEL


@pytest.mark.asyncio
async def test_render_diffusion_subscriptions_drops_artifact_that_cannot_fit(
    tmp_path,
):
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
        artifact_type=DiffusionArtifactType.DEBUG_HINT,
        risk_level=DiffusionRiskLevel.LOW,
        content="very long context " * 200,
    )
    subscription = DiffusionSubscription(
        artifact=artifact,
        policy_name="capped_broadcast",
        relation="broadcast",
        reason="selected_by_test",
    )

    async def compact_artifact_content(
        artifact: DiffusionArtifact,
        budget_tokens: int,
    ) -> str:
        return "short hint"

    bundle = await render_diffusion_subscriptions(
        store=store,
        snapshot=snapshot,
        model="openrouter/openai/gpt-5.5",
        target_task_id="task-a",
        target_iteration=2,
        target_run_id=None,
        subscriptions=[subscription],
        max_context_tokens=1,
        compact_artifact_content=compact_artifact_content,
    )

    assert bundle.text is None
    assert bundle.rendered_count == 0
    assert bundle.dropped_for_budget_artifact_ids == ["artifact-1"]
    assert bundle.budget_violation is True
    record = store.query_diffused_records(target_task_id="task-a")[0]
    assert record.selected is True
    assert record.rendered is False
    assert record.reason == "dropped_for_diffusion_budget"
    assert record.metadata["diffusion_channel"] == REUSE_SUCCESS_CHANNEL


@pytest.mark.asyncio
async def test_render_diffusion_subscriptions_separates_avoid_recheck_channel(
    tmp_path,
):
    store = DiffusionStore(tmp_path / "diffusion")
    snapshot = TaskGraphSnapshot(
        snapshot_id="snapshot-1",
        run_id="run-1",
        iteration=2,
        task_ids=["task-a", "task-b", "task-c"],
        graph_policy="broadcast",
    )
    success_artifact = DiffusionArtifact(
        artifact_id="success-artifact",
        source_task_id="task-b",
        source_iteration=1,
        artifact_type=DiffusionArtifactType.RUN_OUTCOME,
        risk_level=DiffusionRiskLevel.LOW,
        content="reuse the validated formula",
        verifier_reward=1.0,
    )
    failure_artifact = DiffusionArtifact(
        artifact_id="failure-artifact",
        source_task_id="task-c",
        source_iteration=1,
        artifact_type=DiffusionArtifactType.REGRESSION_WARNING,
        risk_level=DiffusionRiskLevel.LOW,
        content="avoid copying the percentile formula",
        verifier_reward=0.0,
    )

    bundle = await render_diffusion_subscriptions(
        store=store,
        snapshot=snapshot,
        model="openrouter/openai/gpt-5.5",
        target_task_id="task-a",
        target_iteration=2,
        target_run_id="target-run",
        subscriptions=[
            DiffusionSubscription(
                artifact=success_artifact,
                policy_name="capped_broadcast",
                relation="broadcast",
                reason="selected_success",
                context_channel=REUSE_SUCCESS_CHANNEL,
            ),
            DiffusionSubscription(
                artifact=failure_artifact,
                policy_name="capped_broadcast",
                relation="avoid_recheck",
                reason="selected_failure",
                context_channel=AVOID_RECHECK_CHANNEL,
            ),
        ],
    )

    assert bundle.text is not None
    assert "### Reusable Success Artifacts" in bundle.text
    assert "### Avoid/Recheck Artifacts" in bundle.text
    assert "Use them only to avoid or re-check failure modes" in bundle.text
    assert bundle.text.index("success-artifact") < bundle.text.index(
        "failure-artifact"
    )
    records = store.query_diffused_records(
        target_task_id="task-a",
        recent=None,
    )
    channels = {
        record.artifact_id: record.metadata["diffusion_channel"]
        for record in records
    }
    assert channels == {
        "success-artifact": REUSE_SUCCESS_CHANNEL,
        "failure-artifact": AVOID_RECHECK_CHANNEL,
    }
    assert {
        record.artifact_id: record.success for record in records
    } == {
        "success-artifact": True,
        "failure-artifact": False,
    }
