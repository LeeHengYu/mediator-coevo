"""LangChain diffusion-policy agent and subscription materialization."""

from __future__ import annotations

from typing import Any, Literal

from mediated_coevo.diffusion.langchain_runtime import (
    inspection_tools,
    normalize_openrouter_model,
    run_agent,
)
from mediated_coevo.diffusion.models import DiffusionArtifact, TaskGraphSnapshot
from mediated_coevo.diffusion.policy import (
    AVOID_RECHECK_CHANNEL,
    REUSE_SUCCESS_CHANNEL,
    DiffusionSubscription,
)

FallbackStrategy = Literal["none", "empty_agent_selection"]


class LangChainDiffusionPolicyAgent:
    """Implement artifact selection ``π(task, graph, bank)`` independently."""

    def __init__(
        self,
        *,
        model: str,
        max_artifacts: int,
        fallback_strategy: FallbackStrategy = "none",
    ) -> None:
        self.model = normalize_openrouter_model(model)
        self.max_artifacts = max_artifacts
        self.fallback_strategy = fallback_strategy

    async def decide(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        """Ask the policy agent which causal artifacts to route."""
        return await run_agent(
            model=self.model,
            system_prompt=DIFFUSION_SYSTEM_PROMPT,
            user_payload={
                "task_profile": task_profile,
                "current_iteration": current_iteration,
                "graph": snapshot.model_dump(mode="json") if snapshot else None,
                "max_artifacts": self.max_artifacts,
                "required_output": DIFFUSION_OUTPUT_SCHEMA,
            },
            tools=inspection_tools(snapshot, artifacts),
        )

    def materialize_subscriptions(
        self,
        *,
        diffusion_decision: dict[str, Any],
        task_profile: dict[str, Any],
        snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> list[DiffusionSubscription]:
        """Validate and materialize the policy agent's artifact selection."""
        return subscriptions_from_diffusion_decision(
            diffusion_decision=diffusion_decision,
            task_profile=task_profile,
            snapshot=snapshot,
            artifacts=artifacts,
            max_artifacts=self.max_artifacts,
            fallback_strategy=self.fallback_strategy,
        )

    async def select(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> list[DiffusionSubscription]:
        """Run the policy agent and return validated subscriptions."""
        decision = await self.decide(
            task_profile=task_profile,
            current_iteration=current_iteration,
            snapshot=snapshot,
            artifacts=artifacts,
        )
        return self.materialize_subscriptions(
            diffusion_decision=decision,
            task_profile=task_profile,
            snapshot=snapshot,
            artifacts=artifacts,
        )


def subscriptions_from_diffusion_decision(
    *,
    diffusion_decision: dict[str, Any],
    task_profile: dict[str, Any],
    snapshot: TaskGraphSnapshot | None,
    artifacts: list[DiffusionArtifact],
    max_artifacts: int,
    fallback_strategy: FallbackStrategy = "empty_agent_selection",
) -> list[DiffusionSubscription]:
    """Convert an agent decision to subscriptions with an explicit fallback."""
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
        if artifact_reward(artifact) < 0.5:
            channel = AVOID_RECHECK_CHANNEL
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
    if (
        not subscriptions
        and fallback_strategy == "empty_agent_selection"
        and snapshot is not None
    ):
        subscriptions.extend(
            fallback_subscriptions(
                task_profile=task_profile,
                snapshot=snapshot,
                artifacts=artifacts,
                max_artifacts=max_artifacts,
            )
        )
    return subscriptions


def fallback_subscriptions(
    *,
    task_profile: dict[str, Any],
    snapshot: TaskGraphSnapshot,
    artifacts: list[DiffusionArtifact],
    max_artifacts: int,
) -> list[DiffusionSubscription]:
    """Preserve the legacy deterministic fallback used by the facade."""
    if max_artifacts <= 0:
        return []
    current_task_id = str(task_profile["task_id"])
    task_nodes = dict(snapshot.metadata.get("task_nodes") or {})
    current_node_id = str(snapshot.metadata.get("current_node_id") or current_task_id)
    current_node = dict(task_nodes.get(current_node_id) or {})
    same_node_task_ids = {current_task_id}
    same_node_task_ids.update(
        str(task_id) for task_id in current_node.get("task_ids", [])
    )

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
            reason = (
                "fallback selected incoming graph-prior artifact after empty agent "
                "selection"
            )
        else:
            continue
        score = base_score + artifact_quality_score(artifact)
        channel = (
            REUSE_SUCCESS_CHANNEL
            if artifact_reward(artifact) >= 0.5
            else AVOID_RECHECK_CHANNEL
        )
        ranked.append(
            (score, artifact.artifact_id, relation, reason, channel, artifact)
        )

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


def artifact_quality_score(artifact: DiffusionArtifact) -> float:
    """Return the deterministic legacy fallback ranking score."""
    artifact_type = getattr(artifact.artifact_type, "value", artifact.artifact_type)
    type_score = {
        "mediator_report_summary": 3.0,
        "debug_hint": 2.0,
        "run_outcome": 1.0,
    }.get(str(artifact_type), 0.0)
    return 10.0 * artifact_reward(artifact) + type_score


def artifact_reward(artifact: DiffusionArtifact) -> float:
    """Return the verifier reward, falling back to the judge reward."""
    reward = artifact.verifier_reward
    if reward is None:
        reward = artifact.judge_reward
    try:
        return float(reward or 0.0)
    except (TypeError, ValueError):
        return 0.0


DIFFUSION_OUTPUT_SCHEMA = {
    "selected_artifacts": [
        {
            "artifact_id": "string",
            "relation": "string",
            "reason": "string",
            "context_channel": f"{REUSE_SUCCESS_CHANNEL}|{AVOID_RECHECK_CHANNEL}",
        }
    ]
}

DIFFUSION_SYSTEM_PROMPT = (
    "You implement π(t, k_t, G_t, B_{t-1}) for graph-aware experience "
    "diffusion. You may inspect the whole causal artifact store through tools, "
    "using the current graph node and incoming graph neighbors as transfer "
    "priors rather than hard eligibility filters. First identify the strongest "
    "incoming graph-prior source nodes and prefer their successful artifacts, "
    "especially same-family or same-output-format mediator_report_summary and "
    "run_outcome artifacts. For exact-schema, exact-file, spreadsheet, or JSON "
    "output tasks, prioritize artifacts that preserve the same contract shape "
    "over broad cross-domain calculation analogies. Select outside the "
    "strongest graph priors only when that artifact supplies a concrete missing "
    "method or failure-mode check not already covered by a stronger-prior "
    "artifact, and state that reason explicitly. Failed artifacts may be useful "
    "only as avoid/recheck warnings and should not displace a successful "
    "high-prior artifact. Select only artifacts likely to help under the "
    "artifact budget. "
    "Return only the required JSON object."
)
