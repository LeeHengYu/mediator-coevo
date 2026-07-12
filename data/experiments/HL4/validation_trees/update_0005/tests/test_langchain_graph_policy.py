from __future__ import annotations

from typing import Any

import pytest

from mediated_coevo.diffusion import (
    AVOID_RECHECK_CHANNEL,
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    LangChainGraphPolicy,
    REUSE_SUCCESS_CHANNEL,
    TaskGraphEdgeRecord,
    TaskGraphSnapshot,
)
from mediated_coevo.diffusion import langchain_graph as langchain_graph_module


def _artifact(
    artifact_id: str,
    *,
    source_task_id: str,
    source_iteration: int = 0,
    verifier_reward: float = 1.0,
    artifact_type: DiffusionArtifactType = DiffusionArtifactType.DEBUG_HINT,
) -> DiffusionArtifact:
    return DiffusionArtifact(
        artifact_id=artifact_id,
        source_task_id=source_task_id,
        source_iteration=source_iteration,
        artifact_type=artifact_type,
        risk_level=DiffusionRiskLevel.LOW,
        content=f"hint from {source_task_id}",
        verifier_reward=verifier_reward,
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


class _SuffixNodeLangChainGraphPolicy(LangChainGraphPolicy):
    async def _run_graph_agent(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        previous_snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        return {
            "node_id": "weighted-campus-energy-balance-calc",
            "node_action": "created",
            "edges": [
                {
                    "source_node_id": "Excel-Formula-Statistics-Base",
                    "target_node_id": "weighted-campus-energy-balance-calc",
                    "relation": "transfer_prior",
                    "weight": 0.85,
                    "reason": "hallucinated base node",
                }
            ],
            "reason": "agent emitted a task slug instead of the existing node id",
        }

    async def _run_diffusion_agent(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        snapshot: TaskGraphSnapshot,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        return {"selected_artifacts": []}


class _FullStoreArtifactLangChainGraphPolicy(LangChainGraphPolicy):
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
            "edges": [],
            "reason": "same task belongs to existing node",
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
                    "relation": "unrelated_full_store_pick",
                    "reason": "useful despite weak graph prior",
                    "context_channel": REUSE_SUCCESS_CHANNEL,
                },
                {
                    "artifact_id": "artifact-A",
                    "relation": "same_node_history",
                    "reason": "same node artifact should remain eligible",
                    "context_channel": REUSE_SUCCESS_CHANNEL,
                },
            ]
        }


class _EmptySelectionSameTaskLangChainGraphPolicy(LangChainGraphPolicy):
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
            "edges": [],
            "reason": "same task belongs to existing node",
        }

    async def _run_diffusion_agent(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        snapshot: TaskGraphSnapshot,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        return {"selected_artifacts": []}


class _CrossFamilyFailureArtifactLangChainGraphPolicy(LangChainGraphPolicy):
    async def _run_graph_agent(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        previous_snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        return {
            "node_id": "Weighted-Risk-Assessment/task-A",
            "node_action": "reused",
            "edges": [],
            "reason": "same family task belongs to existing node",
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
                    "artifact_id": "artifact-success-cross",
                    "relation": "cross_family_success",
                    "reason": "keep successful template",
                    "context_channel": REUSE_SUCCESS_CHANNEL,
                },
                {
                    "artifact_id": "artifact-failure-cross",
                    "relation": "cross_family_failure",
                    "reason": "should be filtered",
                    "context_channel": REUSE_SUCCESS_CHANNEL,
                },
                {
                    "artifact_id": "artifact-failure-same",
                    "relation": "same_family_failure",
                    "reason": "keep local failure prior",
                    "context_channel": REUSE_SUCCESS_CHANNEL,
                },
            ]
        }


class _CrossFamilyOffPriorSuccessLangChainGraphPolicy(LangChainGraphPolicy):
    async def _run_graph_agent(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        previous_snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        return {
            "node_id": "Weighted-Risk-Assessment/task-A",
            "node_action": "reused",
            "edges": [
                {
                    "source_node_id": "Weighted-Risk-Assessment/task-C",
                    "target_node_id": "Weighted-Risk-Assessment/task-A",
                    "relation": "agent_transfer_prior",
                    "weight": 0.9,
                    "reason": "same-family weighted prior",
                }
            ],
            "reason": "same weighted family node already exists",
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
                    "artifact_id": "artifact-success-cross",
                    "relation": "cross_family_procedural_similarity",
                    "reason": "generic spreadsheet success",
                    "context_channel": REUSE_SUCCESS_CHANNEL,
                }
            ]
        }


class _MissingWeightLangChainGraphPolicy(LangChainGraphPolicy):
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
                    "reason": "missing score should not become a max prior",
                }
            ],
            "reason": "agent forgot the transfer-prior weight",
        }

    async def _run_diffusion_agent(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        snapshot: TaskGraphSnapshot,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        return {"selected_artifacts": []}


@pytest.mark.asyncio
async def test_langchain_graph_reuses_duplicate_node_and_selects_graph_neighbor():
    previous = TaskGraphSnapshot(
        run_id="run-1",
        iteration=0,
        task_ids=["node-task-A", "node-task-C"],
        graph_policy="langchain_graph",
        metadata={
            "task_nodes": {
                "node-task-A": {
                    "task_ids": ["task-A"],
                    "last_iteration": 0,
                },
                "node-task-C": {
                    "task_ids": ["task-C"],
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


@pytest.mark.asyncio
async def test_langchain_graph_allows_agent_selected_full_store_artifacts():
    previous = TaskGraphSnapshot(
        run_id="run-1",
        iteration=0,
        task_ids=["node-task-A", "node-task-C"],
        graph_policy="langchain_graph",
        metadata={
            "task_nodes": {
                "node-task-A": {
                    "task_ids": ["task-A"],
                    "last_iteration": 0,
                },
                "node-task-C": {
                    "task_ids": ["task-C"],
                    "last_iteration": 0,
                },
            }
        },
    )
    policy = _FullStoreArtifactLangChainGraphPolicy(
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

    assert [sub.artifact.artifact_id for sub in result.subscriptions] == [
        "artifact-C",
        "artifact-A"
    ]
    assert result.subscriptions[0].relation == "unrelated_full_store_pick"
    assert result.subscriptions[1].relation == "same_node_history"


@pytest.mark.asyncio
async def test_langchain_graph_falls_back_to_same_task_when_agent_selects_none():
    previous = TaskGraphSnapshot(
        run_id="run-1",
        iteration=0,
        task_ids=["node-task-A", "node-task-C"],
        graph_policy="langchain_graph",
        metadata={
            "task_nodes": {
                "node-task-A": {"task_ids": ["task-A"], "last_iteration": 0},
                "node-task-C": {"task_ids": ["task-C"], "last_iteration": 0},
            }
        },
    )
    policy = _EmptySelectionSameTaskLangChainGraphPolicy(
        model="openrouter/test/model",
        run_id="run-1",
        max_artifacts=1,
    )

    result = await policy.prepare(
        task_profile={"task_id": "task-A", "instruction": "same task"},
        current_iteration=1,
        previous_snapshot=previous,
        artifacts=[
            _artifact("artifact-C", source_task_id="task-C"),
            _artifact("artifact-A", source_task_id="task-A"),
        ],
    )

    assert [sub.artifact.artifact_id for sub in result.subscriptions] == [
        "artifact-A"
    ]
    assert result.subscriptions[0].relation == "same_task_prior"
    assert result.subscriptions[0].metadata == {"fallback": "empty_agent_selection"}


@pytest.mark.asyncio
async def test_langchain_graph_filters_cross_family_failure_artifacts_from_agent_selection():
    previous = TaskGraphSnapshot(
        run_id="run-1",
        iteration=0,
        task_ids=["Weighted-Risk-Assessment/task-A"],
        graph_policy="langchain_graph",
        metadata={
            "task_nodes": {
                "Weighted-Risk-Assessment/task-A": {
                    "task_ids": ["Weighted-Risk-Assessment/task-A"],
                    "last_iteration": 0,
                }
            }
        },
    )
    policy = _CrossFamilyFailureArtifactLangChainGraphPolicy(
        model="openrouter/test/model",
        run_id="run-1",
        max_artifacts=3,
    )

    result = await policy.prepare(
        task_profile={
            "task_id": "Weighted-Risk-Assessment/task-A",
            "instruction": "same weighted workbook family",
        },
        current_iteration=1,
        previous_snapshot=previous,
        artifacts=[
            _artifact(
                "artifact-success-cross",
                source_task_id="Inventory-&-Finance-Integration/task-C",
            ),
            _artifact(
                "artifact-failure-cross",
                source_task_id="Production-Capacity-Planning/task-B",
                verifier_reward=0.0,
            ),
            _artifact(
                "artifact-failure-same",
                source_task_id="Weighted-Risk-Assessment/task-D",
                verifier_reward=0.0,
            ),
        ],
    )

    assert [sub.artifact.artifact_id for sub in result.subscriptions] == [
        "artifact-success-cross",
        "artifact-failure-same",
    ]
    assert result.subscriptions[0].context_channel == REUSE_SUCCESS_CHANNEL
    assert result.subscriptions[1].context_channel == AVOID_RECHECK_CHANNEL


@pytest.mark.asyncio
async def test_langchain_graph_fallback_skips_cross_family_failure_graph_priors():
    previous = TaskGraphSnapshot(
        run_id="run-1",
        iteration=0,
        task_ids=["node-task-A", "node-task-C"],
        graph_policy="langchain_graph",
        edge_records=[
            TaskGraphEdgeRecord(
                source_task_id="node-task-C",
                target_task_id="node-task-A",
                relation="agent_transfer_prior",
                weight=0.9,
            )
        ],
        metadata={
            "task_nodes": {
                "node-task-A": {
                    "task_ids": ["Weighted-Risk-Assessment/task-A"],
                    "last_iteration": 0,
                },
                "node-task-C": {
                    "task_ids": ["Production-Capacity-Planning/task-C"],
                    "last_iteration": 0,
                },
            }
        },
    )
    policy = _EmptySelectionSameTaskLangChainGraphPolicy(
        model="openrouter/test/model",
        run_id="run-1",
        max_artifacts=1,
    )

    result = await policy.prepare(
        task_profile={
            "task_id": "Weighted-Risk-Assessment/task-A",
            "instruction": "same weighted workbook family",
        },
        current_iteration=1,
        previous_snapshot=previous,
        artifacts=[
            _artifact(
                "artifact-failure-cross",
                source_task_id="Production-Capacity-Planning/task-C",
                verifier_reward=0.0,
            )
        ],
    )

    assert result.subscriptions == []


@pytest.mark.asyncio
async def test_langchain_graph_fallback_keeps_same_family_failure_graph_priors():
    previous = TaskGraphSnapshot(
        run_id="run-1",
        iteration=0,
        task_ids=[
            "Inventory-&-Finance-Integration/task-A",
            "Inventory-&-Finance-Integration/task-B",
        ],
        graph_policy="langchain_graph",
        edge_records=[
            TaskGraphEdgeRecord(
                source_task_id="Inventory-&-Finance-Integration/task-B",
                target_task_id="Inventory-&-Finance-Integration/task-A",
                relation="agent_transfer_prior",
                weight=0.95,
            )
        ],
        metadata={
            "task_nodes": {
                "Inventory-&-Finance-Integration/task-A": {
                    "task_ids": ["Inventory-&-Finance-Integration/task-A"],
                    "last_iteration": 0,
                },
                "Inventory-&-Finance-Integration/task-B": {
                    "task_ids": ["Inventory-&-Finance-Integration/task-B"],
                    "last_iteration": 0,
                },
            }
        },
    )
    policy = _EmptySelectionSameTaskLangChainGraphPolicy(
        model="openrouter/test/model",
        run_id="run-1",
        max_artifacts=1,
    )

    result = await policy.prepare(
        task_profile={
            "task_id": "Inventory-&-Finance-Integration/task-A",
            "instruction": "same workbook family with incoming failure prior",
        },
        current_iteration=1,
        previous_snapshot=previous,
        artifacts=[
            _artifact(
                "artifact-failure-same-family",
                source_task_id="Inventory-&-Finance-Integration/task-B",
                verifier_reward=0.0,
            )
        ],
    )

    assert [sub.artifact.artifact_id for sub in result.subscriptions] == [
        "artifact-failure-same-family"
    ]
    assert result.subscriptions[0].relation == "same_family_failure_graph_prior"
    assert result.subscriptions[0].context_channel == AVOID_RECHECK_CHANNEL
    assert result.subscriptions[0].metadata == {"fallback": "empty_agent_selection"}


@pytest.mark.asyncio
async def test_langchain_graph_fallback_limits_same_family_failure_graph_priors_to_one_source_task():
    previous = TaskGraphSnapshot(
        run_id="run-1",
        iteration=0,
        task_ids=[
            "Inventory-&-Finance-Integration/task-A",
            "Inventory-&-Finance-Integration/task-B",
        ],
        graph_policy="langchain_graph",
        edge_records=[
            TaskGraphEdgeRecord(
                source_task_id="Inventory-&-Finance-Integration/task-B",
                target_task_id="Inventory-&-Finance-Integration/task-A",
                relation="agent_transfer_prior",
                weight=0.95,
            )
        ],
        metadata={
            "task_nodes": {
                "Inventory-&-Finance-Integration/task-A": {
                    "task_ids": ["Inventory-&-Finance-Integration/task-A"],
                    "last_iteration": 0,
                },
                "Inventory-&-Finance-Integration/task-B": {
                    "task_ids": ["Inventory-&-Finance-Integration/task-B"],
                    "last_iteration": 0,
                },
            }
        },
    )
    policy = _EmptySelectionSameTaskLangChainGraphPolicy(
        model="openrouter/test/model",
        run_id="run-1",
        max_artifacts=3,
    )

    result = await policy.prepare(
        task_profile={
            "task_id": "Inventory-&-Finance-Integration/task-A",
            "instruction": "same workbook family with only failure priors",
        },
        current_iteration=1,
        previous_snapshot=previous,
        artifacts=[
            _artifact(
                "artifact-debug-hint",
                source_task_id="Inventory-&-Finance-Integration/task-B",
                verifier_reward=0.0,
                artifact_type=DiffusionArtifactType.DEBUG_HINT,
            ),
            _artifact(
                "artifact-run-outcome",
                source_task_id="Inventory-&-Finance-Integration/task-B",
                verifier_reward=0.0,
                artifact_type=DiffusionArtifactType.RUN_OUTCOME,
            ),
            _artifact(
                "artifact-mediator-summary",
                source_task_id="Inventory-&-Finance-Integration/task-B",
                verifier_reward=0.0,
                artifact_type=DiffusionArtifactType.MEDIATOR_REPORT_SUMMARY,
            ),
        ],
    )

    assert [sub.artifact.artifact_id for sub in result.subscriptions] == [
        "artifact-mediator-summary"
    ]
    assert result.subscriptions[0].relation == "same_family_failure_graph_prior"
    assert result.subscriptions[0].context_channel == AVOID_RECHECK_CHANNEL
    assert result.subscriptions[0].metadata == {"fallback": "empty_agent_selection"}


@pytest.mark.asyncio
async def test_langchain_graph_filters_cross_family_success_outside_incoming_graph_support():
    previous = TaskGraphSnapshot(
        run_id="run-1",
        iteration=0,
        task_ids=[
            "Weighted-Risk-Assessment/task-A",
            "Weighted-Risk-Assessment/task-C",
            "HWPX-Document-Automation/task-B",
        ],
        graph_policy="langchain_graph",
        edge_records=[
            TaskGraphEdgeRecord(
                source_task_id="Weighted-Risk-Assessment/task-C",
                target_task_id="Weighted-Risk-Assessment/task-A",
                relation="agent_transfer_prior",
                weight=0.9,
            )
        ],
        metadata={
            "task_nodes": {
                "Weighted-Risk-Assessment/task-A": {
                    "task_ids": ["Weighted-Risk-Assessment/task-A"],
                    "last_iteration": 0,
                },
                "Weighted-Risk-Assessment/task-C": {
                    "task_ids": ["Weighted-Risk-Assessment/task-C"],
                    "last_iteration": 0,
                },
                "HWPX-Document-Automation/task-B": {
                    "task_ids": ["HWPX-Document-Automation/task-B"],
                    "last_iteration": 0,
                },
            }
        },
    )
    policy = _CrossFamilyOffPriorSuccessLangChainGraphPolicy(
        model="openrouter/test/model",
        run_id="run-1",
        max_artifacts=1,
    )

    result = await policy.prepare(
        task_profile={
            "task_id": "Weighted-Risk-Assessment/task-A",
            "instruction": "weighted workbook with known graph priors",
        },
        current_iteration=1,
        previous_snapshot=previous,
        artifacts=[
            _artifact(
                "artifact-success-cross",
                source_task_id="HWPX-Document-Automation/task-B",
            )
        ],
    )

    assert result.subscriptions == []


@pytest.mark.asyncio
async def test_langchain_graph_canonicalizes_duplicate_task_slug_and_drops_unknown_edges():
    previous = TaskGraphSnapshot(
        run_id="run-1",
        iteration=4,
        task_ids=[
            "Weighted-Risk-Assessment/weighted-campus-energy-balance-calc",
            "Weighted-Risk-Assessment/api-sla-at-risk-calc",
        ],
        graph_policy="langchain_graph",
        metadata={
            "task_nodes": {
                "Weighted-Risk-Assessment/weighted-campus-energy-balance-calc": {
                    "task_ids": [
                        "Weighted-Risk-Assessment/weighted-campus-energy-balance-calc"
                    ],
                    "last_iteration": 0,
                },
                "Weighted-Risk-Assessment/api-sla-at-risk-calc": {
                    "task_ids": ["Weighted-Risk-Assessment/api-sla-at-risk-calc"],
                    "last_iteration": 4,
                },
            }
        },
    )
    policy = _SuffixNodeLangChainGraphPolicy(
        model="openrouter/test/model",
        run_id="run-1",
        max_artifacts=2,
    )

    result = await policy.prepare(
        task_profile={
            "task_id": "Weighted-Risk-Assessment/weighted-campus-energy-balance-calc",
            "instruction": "same weighted workbook task",
        },
        current_iteration=5,
        previous_snapshot=previous,
        artifacts=[],
    )

    canonical_id = "Weighted-Risk-Assessment/weighted-campus-energy-balance-calc"
    assert result.snapshot.task_ids == [
        canonical_id,
        "Weighted-Risk-Assessment/api-sla-at-risk-calc",
    ]
    assert result.snapshot.metadata["current_node_id"] == canonical_id
    assert (
        result.snapshot.metadata["latest_graph_decision"]["raw_node_id"]
        == "weighted-campus-energy-balance-calc"
    )
    assert result.snapshot.metadata["latest_graph_decision"]["node_id"] == canonical_id
    assert result.snapshot.metadata["task_nodes"][canonical_id]["task_ids"] == [
        canonical_id,
        canonical_id,
    ]
    assert result.snapshot.edge_records == []


@pytest.mark.asyncio
async def test_langchain_graph_requires_agent_authored_edge_weight():
    previous = TaskGraphSnapshot(
        run_id="run-1",
        iteration=0,
        task_ids=["node-task-A", "node-task-C"],
        graph_policy="langchain_graph",
        metadata={
            "task_nodes": {
                "node-task-A": {"task_ids": ["task-A"], "last_iteration": 0},
                "node-task-C": {"task_ids": ["task-C"], "last_iteration": 0},
            }
        },
    )
    policy = _MissingWeightLangChainGraphPolicy(
        model="openrouter/test/model",
        run_id="run-1",
        max_artifacts=2,
    )

    result = await policy.prepare(
        task_profile={"task_id": "task-A", "instruction": "same task"},
        current_iteration=1,
        previous_snapshot=previous,
        artifacts=[],
    )

    assert result.snapshot.edge_records == []


def test_graph_agent_prompt_and_tools_define_weight_as_agent_output():
    prompt = langchain_graph_module._GRAPH_SYSTEM_PROMPT
    weight_schema = langchain_graph_module._GRAPH_OUTPUT_SCHEMA["edges"][0]["weight"]
    tool_docs = "\n".join(
        tool.__doc__ or "" for tool in langchain_graph_module._tools(None, [])
    )

    assert "directly assign weight as a real-number score" in prompt
    assert "does not calculate it for you" in prompt
    assert "required real-number transfer prior chosen by the graph agent" in (
        weight_schema
    )
    assert "agent-authored edge weights" in tool_docs
    assert "calibrating a graph edge weight" in tool_docs
