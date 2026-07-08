from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from mediated_coevo.diffusion import (
    AVOID_RECHECK_CHANNEL,
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    REUSE_SUCCESS_CHANNEL,
    TaskGraphEdgeRecord,
    TaskGraphSnapshot,
)


def _load_overlay_module() -> Any:
    overlay_root = Path(__file__).resolve().parents[1]
    module_path = (
        overlay_root / "src" / "mediated_coevo" / "diffusion" / "langchain_graph.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_overlay_langchain_graph",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


langchain_graph = _load_overlay_module()


def _artifact(
    artifact_id: str,
    *,
    source_task_id: str = "task-B",
    artifact_type: DiffusionArtifactType = DiffusionArtifactType.DEBUG_HINT,
    verifier_reward: float = 1.0,
) -> DiffusionArtifact:
    return DiffusionArtifact(
        artifact_id=artifact_id,
        source_task_id=source_task_id,
        source_iteration=0,
        artifact_type=artifact_type,
        risk_level=DiffusionRiskLevel.LOW,
        content=f"artifact content for {artifact_id}",
        verifier_reward=verifier_reward,
    )


def _snapshot() -> TaskGraphSnapshot:
    return TaskGraphSnapshot(
        run_id="run-1",
        iteration=1,
        task_ids=["node-A", "node-B"],
        graph_policy="langchain_graph",
        edge_records=[
            TaskGraphEdgeRecord(
                source_task_id="node-B",
                target_task_id="node-A",
                relation="transfer_latent_method",
                weight=0.9,
            )
        ],
        metadata={
            "current_node_id": "node-A",
            "task_nodes": {
                "node-A": {"task_ids": ["task-A"]},
                "node-B": {"task_ids": ["task-B"]},
            },
        },
    )


def test_langchain_graph_channels_follow_artifact_outcomes() -> None:
    subscriptions = langchain_graph._subscriptions_from_diffusion_decision(
        diffusion_decision={
            "selected_artifacts": [
                {
                    "artifact_id": "success-artifact",
                    "context_channel": AVOID_RECHECK_CHANNEL,
                },
                {
                    "artifact_id": "failure-artifact",
                    "context_channel": REUSE_SUCCESS_CHANNEL,
                },
            ]
        },
        artifacts=[
            _artifact("success-artifact", verifier_reward=1.0),
            _artifact("failure-artifact", verifier_reward=0.0),
        ],
        snapshot=_snapshot(),
        target_task_id="task-A",
        max_artifacts=2,
    )

    assert [
        (subscription.artifact.artifact_id, subscription.context_channel)
        for subscription in subscriptions
    ] == [
        ("success-artifact", REUSE_SUCCESS_CHANNEL),
        ("failure-artifact", AVOID_RECHECK_CHANNEL),
    ]
    assert all(
        subscription.metadata["context_channel_overridden_by_verifier_reward"]
        for subscription in subscriptions
    )


def test_langchain_graph_fills_unused_budget_with_actionable_same_source_artifact() -> None:
    subscriptions = langchain_graph._subscriptions_from_diffusion_decision(
        diffusion_decision={
            "selected_artifacts": [
                {
                    "artifact_id": "source-run-outcome",
                    "context_channel": REUSE_SUCCESS_CHANNEL,
                }
            ]
        },
        artifacts=[
            _artifact(
                "source-run-outcome",
                artifact_type=DiffusionArtifactType.RUN_OUTCOME,
            ),
            _artifact(
                "source-report-summary",
                artifact_type=DiffusionArtifactType.MEDIATOR_REPORT_SUMMARY,
            ),
            _artifact(
                "source-debug-hint",
                artifact_type=DiffusionArtifactType.DEBUG_HINT,
            ),
        ],
        snapshot=_snapshot(),
        target_task_id="task-A",
        max_artifacts=2,
    )

    assert [subscription.artifact.artifact_id for subscription in subscriptions] == [
        "source-run-outcome",
        "source-report-summary",
    ]
    assert subscriptions[1].metadata == {"fallback_selection": True}
    assert subscriptions[1].relation == "fallback_actionable"
