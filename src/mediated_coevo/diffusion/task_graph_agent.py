"""LangChain task-graph agent and graph-state materialization."""

from __future__ import annotations

import math
from typing import Any

from mediated_coevo.diffusion.langchain_runtime import (
    inspection_tools,
    normalize_openrouter_model,
    run_agent,
)
from mediated_coevo.diffusion.models import (
    DiffusionArtifact,
    TaskGraphEdgeRecord,
    TaskGraphSnapshot,
)


class LangChainTaskGraphAgent:
    """Implement the graph update ``Γ(task, bank, previous_graph)``."""

    def __init__(self, *, model: str, run_id: str) -> None:
        self.model = normalize_openrouter_model(model)
        self.run_id = run_id

    async def decide(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        previous_snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        """Ask the graph agent for a node assignment and weighted edges."""
        return await run_agent(
            model=self.model,
            system_prompt=GRAPH_SYSTEM_PROMPT,
            user_payload={
                "task_profile": task_profile,
                "current_iteration": current_iteration,
                "required_output": GRAPH_OUTPUT_SCHEMA,
            },
            tools=inspection_tools(previous_snapshot, artifacts),
        )

    def materialize_snapshot(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        previous_snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
        graph_decision: dict[str, Any],
        materialize_artifact_nodes: bool = True,
    ) -> TaskGraphSnapshot:
        """Materialize a validated immutable snapshot from an agent decision."""
        return snapshot_from_graph_decision(
            run_id=self.run_id,
            iteration=current_iteration,
            previous_snapshot=previous_snapshot,
            task_profile=task_profile,
            graph_decision=graph_decision,
            artifacts=artifacts,
            materialize_artifact_nodes=materialize_artifact_nodes,
        )

    async def update(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        previous_snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> TaskGraphSnapshot:
        """Run the graph agent and return the resulting snapshot."""
        decision = await self.decide(
            task_profile=task_profile,
            current_iteration=current_iteration,
            previous_snapshot=previous_snapshot,
            artifacts=artifacts,
        )
        return self.materialize_snapshot(
            task_profile=task_profile,
            current_iteration=current_iteration,
            previous_snapshot=previous_snapshot,
            artifacts=artifacts,
            graph_decision=decision,
        )


def snapshot_from_graph_decision(
    *,
    run_id: str,
    iteration: int,
    previous_snapshot: TaskGraphSnapshot | None,
    task_profile: dict[str, Any],
    graph_decision: dict[str, Any],
    artifacts: list[DiffusionArtifact] | None = None,
    materialize_artifact_nodes: bool = False,
) -> TaskGraphSnapshot:
    """Build graph state without mutating the previous snapshot.

    Standalone callers may materialize source tasks from the causal artifact
    bank before the current task is added. The legacy facade disables that
    addition to preserve historical seed behavior.
    """
    metadata = dict(previous_snapshot.metadata) if previous_snapshot else {}
    task_nodes = {
        str(node_id): dict(node)
        for node_id, node in dict(metadata.get("task_nodes", {})).items()
    }
    task_ids = list(previous_snapshot.task_ids) if previous_snapshot else []

    if materialize_artifact_nodes:
        _materialize_artifact_source_nodes(
            task_nodes=task_nodes,
            graph_node_ids=task_ids,
            artifacts=artifacts or [],
        )

    raw_node_id = str(graph_decision.get("node_id") or task_profile["task_id"])
    known_node_ids = set(task_nodes)
    known_node_ids.update(task_ids)
    node_id = canonical_graph_node_id(
        raw_node_id,
        current_task_id=str(task_profile["task_id"]),
        known_node_ids=known_node_ids,
    )
    node_record = dict(task_nodes.get(node_id, {}))
    node_record.setdefault("task_ids", [])
    node_record["task_ids"] = [*node_record["task_ids"], task_profile["task_id"]]
    node_record["last_iteration"] = iteration
    node_record["last_task_id"] = task_profile["task_id"]
    task_nodes[node_id] = node_record
    metadata["task_nodes"] = task_nodes
    assignments = dict(metadata.get("task_assignments", {}))
    assignments[f"{iteration}:{task_profile['task_id']}"] = node_id
    metadata["task_assignments"] = assignments
    metadata["latest_graph_decision"] = {
        **graph_decision,
        "node_id": node_id,
        "raw_node_id": raw_node_id,
    }
    metadata["current_node_id"] = node_id

    if node_id not in task_ids:
        task_ids.append(node_id)
    valid_edge_nodes = set(task_ids)

    edges = list(previous_snapshot.edge_records) if previous_snapshot else []
    keyed_edges = {
        (edge.source_task_id, edge.target_task_id, edge.relation): edge
        for edge in edges
    }
    for edge in graph_decision.get("edges", []):
        source = canonical_graph_node_id(
            str(edge.get("source_node_id") or edge.get("source_task_id") or ""),
            current_task_id=str(task_profile["task_id"]),
            known_node_ids=valid_edge_nodes,
        )
        target = canonical_graph_node_id(
            str(edge.get("target_node_id") or edge.get("target_task_id") or node_id),
            current_task_id=str(task_profile["task_id"]),
            known_node_ids=valid_edge_nodes,
        )
        try:
            weight = float(edge["weight"])
        except (KeyError, TypeError, ValueError):
            continue
        if not source or not target or not math.isfinite(weight):
            continue
        relation = str(edge.get("relation") or "agent_transfer_prior")
        keyed_edges[(source, target, relation)] = TaskGraphEdgeRecord(
            source_task_id=source,
            target_task_id=target,
            relation=relation,
            weight=weight,
            metadata={
                "reason": str(edge.get("reason") or ""),
                **dict(edge.get("metadata") or {}),
            },
        )

    return TaskGraphSnapshot(
        run_id=run_id,
        iteration=iteration,
        task_ids=task_ids,
        edge_records=list(keyed_edges.values()),
        graph_policy="langchain_graph",
        metadata=metadata,
    )


def _materialize_artifact_source_nodes(
    *,
    task_nodes: dict[str, dict[str, Any]],
    graph_node_ids: list[str],
    artifacts: list[DiffusionArtifact],
) -> None:
    """Register source tasks that are present in the causal artifact bank."""
    node_id_by_task_id = {
        str(task_id): node_id
        for node_id, node in task_nodes.items()
        for task_id in node.get("task_ids", [])
    }
    ordered = sorted(
        artifacts,
        key=lambda artifact: (
            artifact.source_iteration,
            artifact.created_at,
            artifact.artifact_id,
        ),
    )
    for artifact in ordered:
        source_task_id = artifact.source_task_id
        existing_node_id = node_id_by_task_id.get(source_task_id)
        if existing_node_id is not None:
            node_record = task_nodes[existing_node_id]
            previous_iteration = node_record.get("last_iteration")
            if (
                not isinstance(previous_iteration, int)
                or artifact.source_iteration > previous_iteration
            ):
                node_record["last_iteration"] = artifact.source_iteration
                node_record["last_task_id"] = source_task_id
            continue
        task_nodes[source_task_id] = {
            "task_ids": [source_task_id],
            "last_iteration": artifact.source_iteration,
            "last_task_id": source_task_id,
            "materialized_from_artifact_bank": True,
        }
        if source_task_id not in graph_node_ids:
            graph_node_ids.append(source_task_id)
        node_id_by_task_id[source_task_id] = source_task_id


def canonical_graph_node_id(
    raw_node_id: str,
    *,
    current_task_id: str,
    known_node_ids: set[str],
) -> str:
    """Return a known graph node ID when the agent emits an unambiguous alias."""
    node_id = raw_node_id.strip()
    if not node_id:
        return ""
    if node_id == current_task_id or node_id in known_node_ids:
        return node_id

    suffix_matches = [
        known_node_id
        for known_node_id in known_node_ids
        if known_node_id.endswith(f"/{node_id}")
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return node_id if "/" in node_id else ""


GRAPH_OUTPUT_SCHEMA = {
    "node_id": "string",
    "node_action": "reused|created",
    "edges": [
        {
            "source_node_id": "string",
            "target_node_id": "string",
            "relation": "string",
            "weight": (
                "required real-number transfer prior chosen by the graph agent; "
                "normally 0.0-1.0, where larger means stronger expected artifact "
                "usefulness but not a hard dependency"
            ),
            "reason": "string",
        }
    ],
    "reason": "string",
}

GRAPH_SYSTEM_PROMPT = (
    "You implement Γ for graph-aware experience diffusion. Decide whether the "
    "incoming task matches an existing node or creates a new node. Duplicate "
    "task text may reuse the same node. Maintain directed transfer-prior edges "
    "with weights where larger means stronger expected artifact usefulness, not "
    "a hard dependency. For every edge you create or update, directly assign "
    "weight as a real-number score; infrastructure persists your score and does "
    "not calculate it for you. Use tools to inspect previous edge weights and "
    "artifact evidence before calibrating a new score. "
    "Return only the required JSON object."
)
