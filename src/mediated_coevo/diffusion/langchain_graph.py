"""Compatibility facade sharing extracted LangChain graph and policy helpers.

Historical heuristic-learning harnesses replace this exact file. Keep the
facade stable while standalone graph and policy agents reuse the same runtime,
materialization, and selection helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mediated_coevo.diffusion.langchain_runtime import (
    artifact_summary as _artifact_summary,  # noqa: F401 - compatibility alias
)
from mediated_coevo.diffusion.langchain_runtime import (
    inspection_tools as _tools,
)
from mediated_coevo.diffusion.langchain_runtime import (
    last_message_text as _last_message_text,  # noqa: F401 - compatibility alias
)
from mediated_coevo.diffusion.langchain_runtime import (
    normalize_openrouter_model as _langchain_openrouter_model,
)
from mediated_coevo.diffusion.langchain_runtime import (
    parse_json_object as _parse_json_object,  # noqa: F401 - compatibility alias
)
from mediated_coevo.diffusion.langchain_runtime import (
    run_agent as _run_agent,
)
from mediated_coevo.diffusion.models import (
    DiffusionArtifact,
    TaskGraphSnapshot,
)
from mediated_coevo.diffusion.policy import DiffusionSubscription
from mediated_coevo.diffusion.policy_agent import (
    DIFFUSION_OUTPUT_SCHEMA as _DIFFUSION_OUTPUT_SCHEMA,
)
from mediated_coevo.diffusion.policy_agent import (
    DIFFUSION_SYSTEM_PROMPT as _DIFFUSION_SYSTEM_PROMPT,
)
from mediated_coevo.diffusion.policy_agent import (
    artifact_quality_score as _artifact_quality_score,  # noqa: F401
)
from mediated_coevo.diffusion.policy_agent import (
    artifact_reward as _artifact_reward,  # noqa: F401 - compatibility alias
)
from mediated_coevo.diffusion.policy_agent import (
    fallback_subscriptions as _fallback_subscriptions,  # noqa: F401
)
from mediated_coevo.diffusion.policy_agent import (
    subscriptions_from_diffusion_decision as _subscriptions_from_diffusion_decision,
)
from mediated_coevo.diffusion.task_graph_agent import (
    GRAPH_OUTPUT_SCHEMA as _GRAPH_OUTPUT_SCHEMA,
)
from mediated_coevo.diffusion.task_graph_agent import (
    GRAPH_SYSTEM_PROMPT as _GRAPH_SYSTEM_PROMPT,
)
from mediated_coevo.diffusion.task_graph_agent import (
    canonical_graph_node_id as _canonical_graph_node_id,  # noqa: F401
)
from mediated_coevo.diffusion.task_graph_agent import (
    snapshot_from_graph_decision as _snapshot_from_graph_decision,
)


@dataclass(frozen=True)
class LangChainGraphPolicyResult:
    """Legacy combined result returned by the stable facade."""

    snapshot: TaskGraphSnapshot
    subscriptions: list[DiffusionSubscription]


class LangChainGraphPolicy:
    """Compatibility implementation of ``Γ`` followed by ``π``.

    The additive :meth:`update_graph` and :meth:`select` methods expose the
    two phases separately while preserving the historical combined calls,
    mutable attributes, protected hooks, and top-level invocation and
    materialization patch points.
    """

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
        """Preserve the historical combined graph-then-policy call."""
        snapshot = await self.update_graph(
            task_profile=task_profile,
            current_iteration=current_iteration,
            previous_snapshot=previous_snapshot,
            artifacts=artifacts,
        )
        subscriptions = await self.select(
            task_profile=task_profile,
            current_iteration=current_iteration,
            snapshot=snapshot,
            artifacts=artifacts,
        )
        return LangChainGraphPolicyResult(
            snapshot=snapshot,
            subscriptions=subscriptions,
        )

    async def update_graph(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        previous_snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> TaskGraphSnapshot:
        """Run only the graph-agent half of the facade."""
        graph_decision = await self._run_graph_agent(
            task_profile=task_profile,
            current_iteration=current_iteration,
            previous_snapshot=previous_snapshot,
            artifacts=artifacts,
        )
        return _snapshot_from_graph_decision(
            run_id=self.run_id,
            iteration=current_iteration,
            previous_snapshot=previous_snapshot,
            task_profile=task_profile,
            graph_decision=graph_decision,
        )

    async def select(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        snapshot: TaskGraphSnapshot,
        artifacts: list[DiffusionArtifact],
    ) -> list[DiffusionSubscription]:
        """Run only the diffusion-policy half of the facade."""
        diffusion_decision = await self._run_diffusion_agent(
            task_profile=task_profile,
            current_iteration=current_iteration,
            snapshot=snapshot,
            artifacts=artifacts,
        )
        return _subscriptions_from_diffusion_decision(
            diffusion_decision=diffusion_decision,
            task_profile=task_profile,
            snapshot=snapshot,
            artifacts=artifacts,
            max_artifacts=self.max_artifacts,
        )

    async def select_with_fixed_graph(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        snapshot: TaskGraphSnapshot,
        artifacts: list[DiffusionArtifact],
    ) -> LangChainGraphPolicyResult:
        """Select artifacts without invoking the graph agent."""
        subscriptions = await self.select(
            task_profile=task_profile,
            current_iteration=current_iteration,
            snapshot=snapshot,
            artifacts=artifacts,
        )
        return LangChainGraphPolicyResult(
            snapshot=snapshot,
            subscriptions=subscriptions,
        )

    async def _run_graph_agent(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        previous_snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        """Compatibility hook retained for harnesses and test subclasses."""
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
        """Compatibility hook retained for harnesses and test subclasses."""
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
