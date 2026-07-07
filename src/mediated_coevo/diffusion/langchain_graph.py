"""LangChain-backed graph construction and artifact diffusion policy."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from mediated_coevo.diffusion.models import (
    DiffusionArtifact,
    TaskGraphEdgeRecord,
    TaskGraphSnapshot,
)
from mediated_coevo.diffusion.policy import (
    AVOID_RECHECK_CHANNEL,
    REUSE_SUCCESS_CHANNEL,
    DiffusionSubscription,
)
from mediated_coevo.llm.client import validate_openrouter_credentials


@dataclass(frozen=True)
class LangChainGraphPolicyResult:
    snapshot: TaskGraphSnapshot
    subscriptions: list[DiffusionSubscription]


class LangChainGraphPolicy:
    """Agent implementation of Γ and π from the paper."""

    def __init__(
        self,
        *,
        model: str,
        run_id: str,
        max_artifacts: int,
    ) -> None:
        self.model = _langchain_openrouter_model(model)
        self.run_id = run_id
        self.max_artifacts = max_artifacts

    async def prepare(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        previous_snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> LangChainGraphPolicyResult:
        graph_decision = await self._run_graph_agent(
            task_profile=task_profile,
            current_iteration=current_iteration,
            previous_snapshot=previous_snapshot,
            artifacts=artifacts,
        )
        snapshot = _snapshot_from_graph_decision(
            run_id=self.run_id,
            iteration=current_iteration,
            previous_snapshot=previous_snapshot,
            task_profile=task_profile,
            graph_decision=graph_decision,
        )
        diffusion_decision = await self._run_diffusion_agent(
            task_profile=task_profile,
            current_iteration=current_iteration,
            snapshot=snapshot,
            artifacts=artifacts,
        )
        return LangChainGraphPolicyResult(
            snapshot=snapshot,
            subscriptions=_subscriptions_from_diffusion_decision(
                diffusion_decision=diffusion_decision,
                artifacts=artifacts,
                max_artifacts=self.max_artifacts,
            ),
        )

    async def _run_graph_agent(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        previous_snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        return await _run_agent(
            model=self.model,
            system_prompt=_GRAPH_SYSTEM_PROMPT,
            user_payload={
                "task_profile": task_profile,
                "current_iteration": current_iteration,
                "required_output": _GRAPH_OUTPUT_SCHEMA,
            },
            tools=_tools(previous_snapshot, artifacts),
        )

    async def _run_diffusion_agent(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        snapshot: TaskGraphSnapshot,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        return await _run_agent(
            model=self.model,
            system_prompt=_DIFFUSION_SYSTEM_PROMPT,
            user_payload={
                "task_profile": task_profile,
                "current_iteration": current_iteration,
                "graph": snapshot.model_dump(mode="json"),
                "max_artifacts": self.max_artifacts,
                "required_output": _DIFFUSION_OUTPUT_SCHEMA,
            },
            tools=_tools(snapshot, artifacts),
        )


def _langchain_openrouter_model(model: str) -> str:
    normalized = model.removeprefix("openrouter/")
    if normalized.startswith("openrouter:"):
        return normalized
    return f"openrouter:{normalized}"


async def _run_agent(
    *,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    tools: list[Any],
) -> dict[str, Any]:
    try:
        from langchain.agents import create_agent
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise RuntimeError(
            "diffusion.policy='langchain_graph' requires langchain and "
            "langchain-openrouter. Run `uv sync` after updating dependencies."
        ) from exc

    validate_openrouter_credentials()
    agent: Any = create_agent(model=model, tools=tools, system_prompt=system_prompt)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": json.dumps(user_payload, sort_keys=True),
            }
        ]
    }

    def invoke_agent() -> Any:
        return agent.invoke(payload)

    result = await asyncio.to_thread(invoke_agent)
    return _parse_json_object(_last_message_text(result))


def _tools(
    snapshot: TaskGraphSnapshot | None,
    artifacts: list[DiffusionArtifact],
) -> list[Any]:
    artifacts_by_id = {artifact.artifact_id: artifact for artifact in artifacts}

    def read_graph() -> dict[str, Any]:
        """Return the current task graph snapshot."""
        if snapshot is None:
            return {"task_ids": [], "edge_records": [], "metadata": {}}
        return snapshot.model_dump(mode="json")

    def query_artifacts(recent: int | None = None) -> list[dict[str, Any]]:
        """Return causally available artifacts, newest first."""
        selected = artifacts if recent is None else artifacts[:recent]
        return [_artifact_summary(artifact) for artifact in selected]

    def get_artifact(artifact_id: str) -> dict[str, Any]:
        """Return one full artifact by artifact_id."""
        artifact = artifacts_by_id.get(artifact_id)
        if artifact is None:
            return {"error": f"unknown artifact_id: {artifact_id}"}
        return artifact.model_dump(mode="json")

    return [read_graph, query_artifacts, get_artifact]


def _artifact_summary(artifact: DiffusionArtifact) -> dict[str, Any]:
    return {
        "artifact_id": artifact.artifact_id,
        "source_task_id": artifact.source_task_id,
        "source_iteration": artifact.source_iteration,
        "artifact_type": artifact.artifact_type,
        "risk_level": artifact.risk_level,
        "verifier_reward": artifact.verifier_reward,
        "judge_reward": artifact.judge_reward,
        "token_cost": artifact.token_cost,
        "content_excerpt": artifact.content[:500],
        "metadata": artifact.metadata,
    }


def _snapshot_from_graph_decision(
    *,
    run_id: str,
    iteration: int,
    previous_snapshot: TaskGraphSnapshot | None,
    task_profile: dict[str, Any],
    graph_decision: dict[str, Any],
) -> TaskGraphSnapshot:
    raw_node_id = str(graph_decision.get("node_id") or task_profile["task_id"])
    metadata = dict(previous_snapshot.metadata) if previous_snapshot else {}
    task_nodes = dict(metadata.get("task_nodes", {}))
    known_node_ids = set(task_nodes)
    if previous_snapshot:
        known_node_ids.update(previous_snapshot.task_ids)
    node_id = _canonical_graph_node_id(
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

    task_ids = list(previous_snapshot.task_ids) if previous_snapshot else []
    if node_id not in task_ids:
        task_ids.append(node_id)
    valid_edge_nodes = set(task_ids)

    edges = list(previous_snapshot.edge_records) if previous_snapshot else []
    keyed_edges = {
        (edge.source_task_id, edge.target_task_id, edge.relation): edge for edge in edges
    }
    for edge in graph_decision.get("edges", []):
        source = _canonical_graph_node_id(
            str(edge.get("source_node_id") or edge.get("source_task_id") or ""),
            current_task_id=str(task_profile["task_id"]),
            known_node_ids=valid_edge_nodes,
        )
        target = _canonical_graph_node_id(
            str(edge.get("target_node_id") or edge.get("target_task_id") or node_id),
            current_task_id=str(task_profile["task_id"]),
            known_node_ids=valid_edge_nodes,
        )
        if not source or not target:
            continue
        relation = str(edge.get("relation") or "agent_transfer_prior")
        keyed_edges[(source, target, relation)] = TaskGraphEdgeRecord(
            source_task_id=source,
            target_task_id=target,
            relation=relation,
            weight=float(edge.get("weight", 1.0)),
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


def _canonical_graph_node_id(
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


def _subscriptions_from_diffusion_decision(
    *,
    diffusion_decision: dict[str, Any],
    artifacts: list[DiffusionArtifact],
    max_artifacts: int,
) -> list[DiffusionSubscription]:
    artifacts_by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    subscriptions: list[DiffusionSubscription] = []
    seen: set[str] = set()
    for item in diffusion_decision.get("selected_artifacts", []):
        artifact_id = str(item.get("artifact_id") or "")
        if artifact_id in seen:
            continue
        artifact = artifacts_by_id.get(artifact_id)
        if artifact is None:
            continue
        channel = str(item.get("context_channel") or REUSE_SUCCESS_CHANNEL)
        if channel not in {REUSE_SUCCESS_CHANNEL, AVOID_RECHECK_CHANNEL}:
            channel = REUSE_SUCCESS_CHANNEL
        subscriptions.append(
            DiffusionSubscription(
                artifact=artifact,
                policy_name="langchain_graph",
                relation=str(item.get("relation") or "agent_selected"),
                reason=str(item.get("reason") or "selected_by_langchain_graph_policy"),
                context_channel=channel,
                metadata=dict(item.get("metadata") or {}),
            )
        )
        seen.add(artifact_id)
        if len(subscriptions) >= max_artifacts:
            break
    return subscriptions


def _last_message_text(result: Any) -> str:
    if isinstance(result, dict) and result.get("messages"):
        message = result["messages"][-1]
    else:
        message = result
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", block)) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _parse_json_object(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("LangChain graph policy agent must return a JSON object")
    return parsed


_GRAPH_OUTPUT_SCHEMA = {
    "node_id": "string",
    "node_action": "reused|created",
    "edges": [
        {
            "source_node_id": "string",
            "target_node_id": "string",
            "relation": "string",
            "weight": "float transfer prior",
            "reason": "string",
        }
    ],
    "reason": "string",
}

_DIFFUSION_OUTPUT_SCHEMA = {
    "selected_artifacts": [
        {
            "artifact_id": "string",
            "relation": "string",
            "reason": "string",
            "context_channel": f"{REUSE_SUCCESS_CHANNEL}|{AVOID_RECHECK_CHANNEL}",
        }
    ]
}

_GRAPH_SYSTEM_PROMPT = (
    "You implement Γ for graph-aware experience diffusion. Decide whether the "
    "incoming task matches an existing node or creates a new node. Duplicate "
    "task text may reuse the same node. Maintain directed transfer-prior edges "
    "with weights where larger means stronger expected artifact usefulness, not "
    "a hard dependency. Use tools when you need graph or artifact details. "
    "Return only the required JSON object."
)

_DIFFUSION_SYSTEM_PROMPT = (
    "You implement π(t, k_t, G_t, B_{t-1}) for graph-aware experience "
    "diffusion. You may inspect the whole causal artifact store through tools, "
    "using the current graph node and incoming graph neighbors as transfer "
    "priors rather than hard eligibility filters. Select only artifacts likely "
    "to help under the artifact budget, and explain selections outside the "
    "strongest graph priors explicitly. "
    "Return only the required JSON object."
)
