from __future__ import annotations

from mediated_coevo.diffusion import (
    AVOID_RECHECK_CHANNEL,
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    REUSE_SUCCESS_CHANNEL,
    TaskGraphEdgeRecord,
    TaskGraphSnapshot,
    select_capped_broadcast_subscriptions,
    select_top_k_similarity_subscriptions,
)


def _artifact(
    artifact_id: str,
    *,
    source_task_id: str,
    source_iteration: int = 0,
    artifact_type: DiffusionArtifactType = DiffusionArtifactType.DEBUG_HINT,
    verifier_reward: float = 1.0,
    judge_reward: float | None = None,
) -> DiffusionArtifact:
    return DiffusionArtifact(
        artifact_id=artifact_id,
        source_task_id=source_task_id,
        source_iteration=source_iteration,
        artifact_type=artifact_type,
        risk_level=DiffusionRiskLevel.LOW,
        content=artifact_id,
        verifier_reward=verifier_reward,
        judge_reward=judge_reward,
    )


def test_capped_broadcast_uses_success_reuse_and_capped_avoid_channels() -> None:
    subscriptions = select_capped_broadcast_subscriptions(
        eligible_artifacts=[
            _artifact("b-debug", source_task_id="task-B"),
            _artifact(
                "b-outcome",
                source_task_id="task-B",
                artifact_type=DiffusionArtifactType.RUN_OUTCOME,
            ),
            _artifact("c-debug", source_task_id="task-C"),
            _artifact(
                "d-warning",
                source_task_id="task-D",
                artifact_type=DiffusionArtifactType.REGRESSION_WARNING,
                verifier_reward=0.0,
            ),
            _artifact("e-debug-failure", source_task_id="task-E", verifier_reward=0.0),
        ],
        max_artifacts=2,
        avoid_recheck_max_artifacts=1,
    )

    assert [
        (subscription.artifact.artifact_id, subscription.context_channel)
        for subscription in subscriptions
    ] == [
        ("b-outcome", REUSE_SUCCESS_CHANNEL),
        ("c-debug", REUSE_SUCCESS_CHANNEL),
        ("d-warning", AVOID_RECHECK_CHANNEL),
    ]
    assert len({subscription.artifact.source_task_id for subscription in subscriptions}) == 3


def test_top_k_similarity_orders_by_edge_reward_type_and_source_diversity() -> None:
    snapshot = TaskGraphSnapshot(
        snapshot_id="snapshot-1",
        run_id="run-1",
        iteration=1,
        task_ids=["task-A", "task-B", "task-C", "task-D"],
        graph_policy="precomputed_similarity",
        edge_records=[
            TaskGraphEdgeRecord(
                source_task_id="task-B",
                target_task_id="task-A",
                relation="similar",
                weight=0.9,
            ),
            TaskGraphEdgeRecord(
                source_task_id="task-C",
                target_task_id="task-A",
                relation="similar",
                weight=0.8,
            ),
            TaskGraphEdgeRecord(
                source_task_id="task-D",
                target_task_id="task-A",
                relation="similar",
                weight=0.7,
            ),
        ],
    )

    subscriptions = select_top_k_similarity_subscriptions(
        eligible_artifacts=[
            _artifact("c-debug", source_task_id="task-C"),
            _artifact("b-debug", source_task_id="task-B"),
            _artifact(
                "b-outcome",
                source_task_id="task-B",
                artifact_type=DiffusionArtifactType.RUN_OUTCOME,
            ),
            _artifact(
                "b-warning",
                source_task_id="task-B",
                artifact_type=DiffusionArtifactType.REGRESSION_WARNING,
                verifier_reward=0.0,
            ),
            _artifact("d-debug", source_task_id="task-D"),
        ],
        snapshot=snapshot,
        target_task_id="task-A",
        max_artifacts=2,
        top_k_neighbors=3,
        avoid_recheck_max_artifacts=1,
    )

    assert [
        (subscription.artifact.artifact_id, subscription.context_channel)
        for subscription in subscriptions
    ] == [
        ("b-outcome", REUSE_SUCCESS_CHANNEL),
        ("c-debug", REUSE_SUCCESS_CHANNEL),
        ("b-warning", AVOID_RECHECK_CHANNEL),
    ]
    assert [subscription.metadata["edge_rank"] for subscription in subscriptions] == [
        1,
        2,
        1,
    ]
