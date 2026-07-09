"""LangChain-backed graph construction and artifact diffusion policy."""

from __future__ import annotations

import asyncio
import json
import math
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
                task_profile=task_profile,
                snapshot=snapshot,
                artifacts=artifacts,
                max_artifacts=self.max_artifacts,
            ),
        )

    async def select_with_fixed_graph(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        snapshot: TaskGraphSnapshot,
        artifacts: list[DiffusionArtifact],
    ) -> LangChainGraphPolicyResult:
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
                task_profile=task_profile,
                snapshot=snapshot,
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
        """Return the current graph snapshot, including agent-authored edge weights."""
        if snapshot is None:
            return {"task_ids": [], "edge_records": [], "metadata": {}}
        return snapshot.model_dump(mode="json")

    def query_artifacts(recent: int | None = None) -> list[dict[str, Any]]:
        """Return artifacts used as evidence for graph edges and transfer weights."""
        selected = artifacts if recent is None else artifacts[:recent]
        return [_artifact_summary(artifact) for artifact in selected]

    def get_artifact(artifact_id: str) -> dict[str, Any]:
        """Return full artifact evidence for calibrating a graph edge weight."""
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
    task_profile: dict[str, Any],
    snapshot: TaskGraphSnapshot,
    artifacts: list[DiffusionArtifact],
    max_artifacts: int,
) -> list[DiffusionSubscription]:
    if max_artifacts <= 0:
        return []
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
    if not subscriptions:
        subscriptions.extend(
            _fallback_subscriptions(
                task_profile=task_profile,
                snapshot=snapshot,
                artifacts=artifacts,
                max_artifacts=max_artifacts,
            )
        )
    else:
        subscriptions = _rebalance_subscriptions(
            subscriptions=subscriptions,
            task_profile=task_profile,
            snapshot=snapshot,
            artifacts=artifacts,
            max_artifacts=max_artifacts,
        )
    return subscriptions


def _rebalance_subscriptions(
    *,
    subscriptions: list[DiffusionSubscription],
    task_profile: dict[str, Any],
    snapshot: TaskGraphSnapshot,
    artifacts: list[DiffusionArtifact],
    max_artifacts: int,
) -> list[DiffusionSubscription]:
    if max_artifacts <= 0:
        return []
    if len(subscriptions) < max_artifacts:
        return subscriptions[:max_artifacts]
    current_task_id = str(task_profile["task_id"])
    current_family = _task_family(current_task_id)
    selected_ids = {subscription.artifact.artifact_id for subscription in subscriptions}
    same_task_selected = any(
        subscription.artifact.source_task_id == current_task_id
        for subscription in subscriptions
    )
    same_family_selected = any(
        _task_family(subscription.artifact.source_task_id) == current_family
        for subscription in subscriptions
    )

    preferred_candidates = _preferred_affinity_candidates(
        task_profile=task_profile,
        snapshot=snapshot,
        artifacts=artifacts,
        excluded_artifact_ids=selected_ids,
    )
    if not preferred_candidates:
        return subscriptions[:max_artifacts]

    balanced = list(subscriptions[:max_artifacts])
    for candidate in preferred_candidates:
        candidate_is_same_task = candidate.artifact.source_task_id == current_task_id
        candidate_is_same_family = (
            _task_family(candidate.artifact.source_task_id) == current_family
        )
        if candidate_is_same_task and same_task_selected:
            continue
        if (
            not candidate_is_same_task
            and candidate_is_same_family
            and same_family_selected
        ):
            continue

        replacement_index = _weakest_cross_family_index(
            subscriptions=balanced,
            current_family=current_family,
        )
        if replacement_index is None:
            continue
        balanced[replacement_index] = candidate

        if candidate_is_same_task:
            same_task_selected = True
        if candidate_is_same_family:
            same_family_selected = True
    return balanced[:max_artifacts]


def _preferred_affinity_candidates(
    *,
    task_profile: dict[str, Any],
    snapshot: TaskGraphSnapshot,
    artifacts: list[DiffusionArtifact],
    excluded_artifact_ids: set[str],
) -> list[DiffusionSubscription]:
    current_task_id = str(task_profile["task_id"])
    current_family = _task_family(current_task_id)
    task_nodes = dict(snapshot.metadata.get("task_nodes") or {})
    current_node_id = str(snapshot.metadata.get("current_node_id") or current_task_id)
    current_node = dict(task_nodes.get(current_node_id) or {})
    same_node_task_ids = {current_task_id}
    same_node_task_ids.update(str(task_id) for task_id in current_node.get("task_ids", []))

    ranked: list[tuple[float, str, DiffusionSubscription]] = []
    for artifact in artifacts:
        if artifact.artifact_id in excluded_artifact_ids:
            continue
        source_family = _task_family(artifact.source_task_id)
        if artifact.source_task_id == current_task_id:
            base_score = 300.0
            relation = "same_task_rebalance"
            reason = "rebalanced selection to include same-task artifact"
        elif artifact.source_task_id in same_node_task_ids:
            base_score = 250.0
            relation = "same_node_rebalance"
            reason = "rebalanced selection to include same-node artifact"
        elif source_family == current_family:
            base_score = 200.0
            relation = "same_family_rebalance"
            reason = "rebalanced selection to prefer same-family artifact"
        else:
            continue
        channel = (
            REUSE_SUCCESS_CHANNEL
            if _artifact_reward(artifact) >= 0.5
            else AVOID_RECHECK_CHANNEL
        )
        subscription = DiffusionSubscription(
            artifact=artifact,
            policy_name="langchain_graph",
            relation=relation,
            reason=reason,
            context_channel=channel,
            metadata={"fallback": "selection_rebalance"},
        )
        ranked.append(
            (
                base_score + _artifact_quality_score(artifact),
                artifact.artifact_id,
                subscription,
            )
        )
    ranked.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return [subscription for _, _, subscription in ranked]


def _weakest_cross_family_index(
    *,
    subscriptions: list[DiffusionSubscription],
    current_family: str,
) -> int | None:
    weakest_index: int | None = None
    weakest_score: float | None = None
    for index, subscription in enumerate(subscriptions):
        if _task_family(subscription.artifact.source_task_id) == current_family:
            continue
        score = _artifact_quality_score(subscription.artifact)
        if weakest_score is None or score < weakest_score:
            weakest_index = index
            weakest_score = score
    return weakest_index


def _fallback_subscriptions(
    *,
    task_profile: dict[str, Any],
    snapshot: TaskGraphSnapshot,
    artifacts: list[DiffusionArtifact],
    max_artifacts: int,
) -> list[DiffusionSubscription]:
    if max_artifacts <= 0:
        return []
    current_task_id = str(task_profile["task_id"])
    task_nodes = dict(snapshot.metadata.get("task_nodes") or {})
    current_node_id = str(snapshot.metadata.get("current_node_id") or current_task_id)
    current_node = dict(task_nodes.get(current_node_id) or {})
    same_node_task_ids = {current_task_id}
    same_node_task_ids.update(str(task_id) for task_id in current_node.get("task_ids", []))

    node_by_task_id: dict[str, str] = {}
    for node_id, node in task_nodes.items():
        for task_id in dict(node).get("task_ids", []):
            node_by_task_id[str(task_id)] = str(node_id)
    incoming_weight_by_node = {
        edge.source_task_id: edge.weight
        for edge in snapshot.edge_records
        if edge.target_task_id == current_node_id
    }

    ranked: list[tuple[float, str, str, str, str, DiffusionArtifact]] = []
    for artifact in artifacts:
        source_node_id = node_by_task_id.get(artifact.source_task_id)
        if artifact.source_task_id == current_task_id:
            base_score = 300.0
            relation = "same_task_prior"
            reason = "fallback selected same-task artifact after empty agent selection"
        elif artifact.source_task_id in same_node_task_ids:
            base_score = 250.0
            relation = "same_node_prior"
            reason = "fallback selected same-node artifact after empty agent selection"
        elif source_node_id in incoming_weight_by_node:
            base_score = 100.0 + 100.0 * incoming_weight_by_node[source_node_id]
            relation = "graph_prior_fallback"
            reason = "fallback selected incoming graph-prior artifact after empty agent selection"
        else:
            continue
        score = base_score + _artifact_quality_score(artifact)
        channel = (
            REUSE_SUCCESS_CHANNEL
            if _artifact_reward(artifact) >= 0.5
            else AVOID_RECHECK_CHANNEL
        )
        ranked.append((score, artifact.artifact_id, relation, reason, channel, artifact))

    ranked.sort(reverse=True, key=lambda item: (item[0], item[1]))
    subscriptions: list[DiffusionSubscription] = []
    seen: set[str] = set()
    for _, _, relation, reason, channel, artifact in ranked:
        if artifact.artifact_id in seen:
            continue
        subscriptions.append(
            DiffusionSubscription(
                artifact=artifact,
                policy_name="langchain_graph",
                relation=relation,
                reason=reason,
                context_channel=channel,
                metadata={"fallback": "empty_agent_selection"},
            )
        )
        seen.add(artifact.artifact_id)
        if len(subscriptions) >= max_artifacts:
            break
    return subscriptions


def _artifact_quality_score(artifact: DiffusionArtifact) -> float:
    artifact_type = getattr(artifact.artifact_type, "value", artifact.artifact_type)
    type_score = {
        "mediator_report_summary": 3.0,
        "debug_hint": 2.0,
        "run_outcome": 1.0,
    }.get(str(artifact_type), 0.0)
    return 10.0 * _artifact_reward(artifact) + type_score


def _artifact_reward(artifact: DiffusionArtifact) -> float:
    reward = artifact.verifier_reward
    if reward is None:
        reward = artifact.judge_reward
    try:
        return float(reward or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _task_family(task_id: str) -> str:
    return task_id.split("/", 1)[0]


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
    "a hard dependency. For every edge you create or update, directly assign "
    "weight as a real-number score; infrastructure persists your score and does "
    "not calculate it for you. Use tools to inspect previous edge weights and "
    "artifact evidence before calibrating a new score. "
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
