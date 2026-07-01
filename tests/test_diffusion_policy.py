from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from mediated_coevo.core.config import Config, DiffusionConfig
from mediated_coevo.diffusion import (
    AVOID_RECHECK_CHANNEL,
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    DiffusionStore,
    REUSE_SUCCESS_CHANNEL,
    TaskGraphEdgeRecord,
    TaskGraphSnapshot,
    select_capped_broadcast_subscriptions,
    select_llm_router_softmax_routes,
    select_top_k_similarity_subscriptions,
)
from mediated_coevo.experiment.orchestrator import Orchestrator
from tests.config_helpers import budgets_config, experiment_config, models_config


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
                "d-regressed-outcome",
                source_task_id="task-D",
                artifact_type=DiffusionArtifactType.RUN_OUTCOME,
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
        ("d-regressed-outcome", AVOID_RECHECK_CHANNEL),
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
                "b-regressed-outcome",
                source_task_id="task-B",
                artifact_type=DiffusionArtifactType.RUN_OUTCOME,
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
        ("b-regressed-outcome", AVOID_RECHECK_CHANNEL),
    ]
    assert [subscription.metadata["edge_rank"] for subscription in subscriptions] == [
        1,
        2,
        1,
    ]


def test_llm_router_softmax_requires_router_scores() -> None:
    snapshot = TaskGraphSnapshot(
        snapshot_id="snapshot-1",
        run_id="run-1",
        iteration=1,
        task_ids=["task-A", "task-B", "task-C"],
        graph_policy="precomputed_similarity",
    )

    routes = select_llm_router_softmax_routes(
        eligible_artifacts=[
            _artifact(
                "b-outcome",
                source_task_id="task-B",
                artifact_type=DiffusionArtifactType.RUN_OUTCOME,
            )
        ],
        snapshot=snapshot,
        target_task_ids=["task-A", "task-B", "task-C"],
        target_iteration=1,
        top_k_candidates=2,
        temperature=0.35,
        seed=42,
        router_scores={},
    )

    assert routes == []


def test_llm_router_softmax_routes_scored_source_to_one_target() -> None:
    snapshot = TaskGraphSnapshot(
        snapshot_id="snapshot-1",
        run_id="run-1",
        iteration=1,
        task_ids=["task-A", "task-B", "task-C"],
        graph_policy="precomputed_similarity",
        edge_records=[
            TaskGraphEdgeRecord(
                source_task_id="task-B",
                target_task_id="task-A",
                relation="similar",
                weight=0.9,
            ),
            TaskGraphEdgeRecord(
                source_task_id="task-B",
                target_task_id="task-C",
                relation="weak",
                weight=0.1,
            ),
        ],
    )

    routes = select_llm_router_softmax_routes(
        eligible_artifacts=[
            _artifact(
                "b-outcome",
                source_task_id="task-B",
                artifact_type=DiffusionArtifactType.RUN_OUTCOME,
            )
        ],
        snapshot=snapshot,
        target_task_ids=["task-A", "task-B", "task-C"],
        target_iteration=1,
        top_k_candidates=2,
        temperature=0.35,
        seed=42,
        router_scores={
            ("b-outcome", "task-A"): {
                "score": 0.9,
                "confidence": 1.0,
                "rationale": "strong fit",
            },
            ("b-outcome", "task-C"): {
                "score": 0.1,
                "confidence": 1.0,
                "rationale": "weak fit",
            },
        },
        router_weight=0.3,
        router_model="openrouter/openai/gpt-5.2",
    )

    assert len(routes) == 1
    route = routes[0]
    assert route.target_task_id in {"task-A", "task-C"}
    assert route.subscription.artifact.artifact_id == "b-outcome"
    assert route.subscription.policy_name == "llm_router_softmax"
    assert route.subscription.metadata["softmax_temperature"] == 0.35
    assert len(route.decision.candidate_distribution) == 2
    assert all(
        "llm_router_score" in row["score_components"]
        for row in route.decision.candidate_distribution
    )
    assert round(sum(
        row["probability"] for row in route.decision.candidate_distribution
    ), 12) == 1.0


class _RouterLLM:
    async def complete(self, *args, **kwargs):
        return {
            "content": json.dumps(
                {
                    "scores": [
                        {
                            "target": "task-A",
                            "score": 0.0,
                            "confidence": 1.0,
                            "rationale": "already seen",
                        },
                        {
                            "target": "task-B",
                            "score": 0.0,
                            "confidence": 1.0,
                            "rationale": "already seen",
                        },
                        {
                            "target": "task-C",
                            "score": 1.0,
                            "confidence": 1.0,
                            "rationale": "best transfer target",
                        },
                    ]
                }
            )
        }


@pytest.mark.asyncio
async def test_logical_scheduler_activates_only_llm_routed_tasks(tmp_path) -> None:
    config = Config(
        models=models_config(),
        budgets=budgets_config(),
        experiment=experiment_config(),
        diffusion=DiffusionConfig(
            enabled=True,
            policy="llm_router_softmax",
            graph="none",
            max_artifacts=3,
            top_k_neighbors=3,
            logical_seed_count=2,
            softmax_top_k_candidates=1,
        ),
    )
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = config
    orch.mediator = cast(Any, SimpleNamespace(llm_client=_RouterLLM()))
    orch.experiment_dir = tmp_path
    orch._diffusion_store = DiffusionStore(tmp_path / "diffusion")
    orch._diffusion_sub_board = {}
    orch._diffusion_prepared_iterations = set()
    orch._diffusion_snapshot_by_iteration = {}
    orch._diffusion_context_by_target = {}
    orch._diffusion_target_task_ids = ["task-A", "task-B", "task-C"]
    orch._logical_task_run_iterations = {"task-A": [0], "task-B": [0]}

    for artifact in [
        _artifact("a-outcome", source_task_id="task-A", source_iteration=0),
        _artifact("b-outcome", source_task_id="task-B", source_iteration=0),
    ]:
        orch._diffusion_store.store_artifact(artifact)

    active = await orch._logical_next_active_task_ids(
        all_task_ids=["task-A", "task-B", "task-C"],
        next_iteration=1,
    )

    assert active == ["task-C"]
    assert (1, "task-C") in orch._diffusion_sub_board
