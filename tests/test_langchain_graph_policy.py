from __future__ import annotations

from typing import Any

import pytest

from mediated_coevo.diffusion import (
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    LangChainGraphPolicy,
    REUSE_SUCCESS_CHANNEL,
    TaskGraphSnapshot,
)


def _artifact(
    artifact_id: str,
    *,
    source_task_id: str,
    source_iteration: int = 0,
) -> DiffusionArtifact:
    return DiffusionArtifact(
        artifact_id=artifact_id,
        source_task_id=source_task_id,
        source_iteration=source_iteration,
        artifact_type=DiffusionArtifactType.DEBUG_HINT,
        risk_level=DiffusionRiskLevel.LOW,
        content=f"hint from {source_task_id}",
        verifier_reward=1.0,
    )


class _FakeLangChainGraphPolicy(LangChainGraphPolicy):
    async def _run_graph_agent(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        previous_snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        return {
            "node_id": "node-task-A",
            "node_action": "reused",
            "edges": [
                {
                    "source_node_id": "node-task-C",
                    "target_node_id": "node-task-A",
                    "relation": "agent_transfer_prior",
                    "weight": 0.4,
                    "reason": "weak but possible transfer",
                }
            ],
            "reason": "duplicate task belongs to existing node",
        }

    async def _run_diffusion_agent(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        snapshot: TaskGraphSnapshot,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        return {
            "selected_artifacts": [
                {
                    "artifact_id": "artifact-C",
                    "relation": "full_store_override",
                    "reason": "useful despite not being same node",
                    "context_channel": REUSE_SUCCESS_CHANNEL,
                }
            ]
        }


@pytest.mark.asyncio
async def test_langchain_graph_reuses_duplicate_node_and_selects_from_full_store():
    previous = TaskGraphSnapshot(
        run_id="run-1",
        iteration=0,
        task_ids=["node-task-A"],
        graph_policy="langchain_graph",
        metadata={
            "task_nodes": {
                "node-task-A": {
                    "task_ids": ["task-A"],
                    "last_iteration": 0,
                }
            }
        },
    )
    policy = _FakeLangChainGraphPolicy(
        model="openrouter/test/model",
        run_id="run-1",
        max_artifacts=2,
    )

    result = await policy.prepare(
        task_profile={"task_id": "task-A", "instruction": "same task"},
        current_iteration=1,
        previous_snapshot=previous,
        artifacts=[
            _artifact("artifact-A", source_task_id="task-A"),
            _artifact("artifact-C", source_task_id="task-C"),
        ],
    )

    node = result.snapshot.metadata["task_nodes"]["node-task-A"]
    assert node["task_ids"] == ["task-A", "task-A"]
    assert result.snapshot.metadata["current_node_id"] == "node-task-A"
    assert result.snapshot.edge_records[0].weight == 0.4
    assert [sub.artifact.artifact_id for sub in result.subscriptions] == [
        "artifact-C"
    ]
    assert result.subscriptions[0].relation == "full_store_override"
