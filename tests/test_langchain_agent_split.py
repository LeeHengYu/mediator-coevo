from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from mediated_coevo.diffusion import (
    REUSE_SUCCESS_CHANNEL,
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    TaskGraphSnapshot,
)
from mediated_coevo.diffusion import langchain_graph as langchain_graph_module
from mediated_coevo.diffusion import policy_agent as policy_agent_module
from mediated_coevo.diffusion import task_graph_agent as task_graph_agent_module
from mediated_coevo.diffusion.langchain_graph import LangChainGraphPolicy
from mediated_coevo.diffusion.policy_agent import LangChainDiffusionPolicyAgent
from mediated_coevo.diffusion.task_graph_agent import LangChainTaskGraphAgent


def _artifact(artifact_id: str, task_id: str, position: int) -> DiffusionArtifact:
    return DiffusionArtifact(
        artifact_id=artifact_id,
        source_task_id=task_id,
        source_iteration=position,
        artifact_type=DiffusionArtifactType.RUN_OUTCOME,
        risk_level=DiffusionRiskLevel.LOW,
        content=f"outcome for {task_id}",
        verifier_reward=1.0,
    )


class _FakeGraphAgent(LangChainTaskGraphAgent):
    async def decide(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        previous_snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        return {
            "node_id": task_profile["task_id"],
            "node_action": "created",
            "edges": [],
            "reason": "new task",
        }


class _FakePolicyAgent(LangChainDiffusionPolicyAgent):
    def __init__(self, decision: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._decision = decision

    async def decide(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        return self._decision


@pytest.mark.asyncio
async def test_first_graph_update_materializes_warmup_sources_from_artifact_bank():
    agent = _FakeGraphAgent(model="openrouter/test/model", run_id="sample-1")
    warmups = [
        _artifact("artifact-0", "warmup-a", 0),
        _artifact("artifact-1", "warmup-b", 1),
        _artifact("artifact-2", "warmup-a", 2),
    ]

    snapshot = await agent.update(
        task_profile={"task_id": "task-3", "instruction": "fourth task"},
        current_iteration=3,
        previous_snapshot=None,
        artifacts=warmups,
    )

    assert snapshot.task_ids == ["warmup-a", "warmup-b", "task-3"]
    assert snapshot.metadata["current_node_id"] == "task-3"
    assert snapshot.metadata["task_nodes"]["warmup-a"]["task_ids"] == ["warmup-a"]
    assert snapshot.metadata["task_nodes"]["warmup-a"]["last_iteration"] == 2
    assert snapshot.metadata["task_nodes"]["warmup-b"]["last_iteration"] == 1


@pytest.mark.asyncio
async def test_diffusion_policy_agent_has_explicit_no_fallback_mode():
    agent = _FakePolicyAgent(
        {"selected_artifacts": []},
        model="openrouter/test/model",
        max_artifacts=2,
        fallback_strategy="none",
    )
    artifact = _artifact("artifact-0", "task-0", 0)

    subscriptions = await agent.select(
        task_profile={"task_id": "task-1", "instruction": "next task"},
        current_iteration=1,
        snapshot=None,
        artifacts=[artifact],
    )

    assert subscriptions == []


@pytest.mark.asyncio
async def test_diffusion_policy_agent_preserves_agent_selection_without_graph():
    artifact = _artifact("artifact-0", "task-0", 0)
    agent = _FakePolicyAgent(
        {
            "selected_artifacts": [
                {
                    "artifact_id": "artifact-0",
                    "relation": "agent_selected",
                    "reason": "useful precedent",
                    "context_channel": REUSE_SUCCESS_CHANNEL,
                }
            ]
        },
        model="openrouter/test/model",
        max_artifacts=2,
        fallback_strategy="none",
    )

    subscriptions = await agent.select(
        task_profile={"task_id": "task-1", "instruction": "next task"},
        current_iteration=1,
        snapshot=None,
        artifacts=[artifact],
    )

    assert [item.artifact.artifact_id for item in subscriptions] == ["artifact-0"]
    assert subscriptions[0].relation == "agent_selected"


class _FacadeWithOverriddenHooks(LangChainGraphPolicy):
    async def _run_graph_agent(
        self,
        *,
        task_profile: dict[str, Any],
        current_iteration: int,
        previous_snapshot: TaskGraphSnapshot | None,
        artifacts: list[DiffusionArtifact],
    ) -> dict[str, Any]:
        return {
            "node_id": task_profile["task_id"],
            "node_action": "created",
            "edges": [],
            "reason": "facade hook",
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
async def test_compatibility_facade_exposes_split_calls_and_keeps_override_hooks():
    facade = _FacadeWithOverriddenHooks(
        model="openrouter/test/model",
        run_id="run-1",
        max_artifacts=1,
    )

    snapshot = await facade.update_graph(
        task_profile={"task_id": "task-1", "instruction": "task"},
        current_iteration=1,
        previous_snapshot=None,
        artifacts=[],
    )
    subscriptions = await facade.select(
        task_profile={"task_id": "task-1", "instruction": "task"},
        current_iteration=1,
        snapshot=snapshot,
        artifacts=[],
    )

    assert snapshot.metadata["latest_graph_decision"]["reason"] == "facade hook"
    assert subscriptions == []


@pytest.mark.asyncio
async def test_compatibility_facade_keeps_module_run_agent_as_patch_point(monkeypatch):
    calls: list[dict[str, Any]] = []

    async def patched_run_agent(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if kwargs["system_prompt"] == langchain_graph_module._GRAPH_SYSTEM_PROMPT:
            return {
                "node_id": "task-1",
                "node_action": "created",
                "edges": [],
                "reason": "patched graph call",
            }
        return {"selected_artifacts": []}

    async def unexpected_split_call(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("legacy facade bypassed its module patch point")

    monkeypatch.setattr(langchain_graph_module, "_run_agent", patched_run_agent)
    monkeypatch.setattr(task_graph_agent_module, "run_agent", unexpected_split_call)
    monkeypatch.setattr(policy_agent_module, "run_agent", unexpected_split_call)
    facade = LangChainGraphPolicy(
        model="openrouter/model-a",
        run_id="run-a",
        max_artifacts=3,
    )
    facade.model = "openrouter:model-b"
    facade.run_id = "run-b"
    facade.max_artifacts = 1

    result = await facade.prepare(
        task_profile={"task_id": "task-1", "instruction": "task"},
        current_iteration=1,
        previous_snapshot=None,
        artifacts=[],
    )

    assert [call["model"] for call in calls] == [
        "openrouter:model-b",
        "openrouter:model-b",
    ]
    assert calls[1]["user_payload"]["max_artifacts"] == 1
    assert result.snapshot.run_id == "run-b"


@pytest.mark.asyncio
async def test_compatibility_facade_keeps_materializers_as_patch_points(monkeypatch):
    facade = _FacadeWithOverriddenHooks(
        model="openrouter/test/model",
        run_id="run-1",
        max_artifacts=1,
    )
    calls: list[str] = []

    def patched_snapshot(**kwargs: Any) -> TaskGraphSnapshot:
        calls.append("snapshot")
        return TaskGraphSnapshot(
            run_id=kwargs["run_id"],
            iteration=kwargs["iteration"],
            task_ids=["patched-node"],
            graph_policy="patched",
        )

    def patched_subscriptions(**kwargs: Any) -> list[Any]:
        calls.append("subscriptions")
        assert kwargs["max_artifacts"] == 1
        return []

    monkeypatch.setattr(
        langchain_graph_module,
        "_snapshot_from_graph_decision",
        patched_snapshot,
    )
    monkeypatch.setattr(
        langchain_graph_module,
        "_subscriptions_from_diffusion_decision",
        patched_subscriptions,
    )

    result = await facade.prepare(
        task_profile={"task_id": "task-1", "instruction": "task"},
        current_iteration=1,
        previous_snapshot=None,
        artifacts=[],
    )

    assert calls == ["snapshot", "subscriptions"]
    assert result.snapshot.task_ids == ["patched-node"]


def test_representative_final_hl4_overlay_remains_importable_and_constructible(
    tmp_path: Path,
):
    # The final HL4 runtime archive is intentionally absent from the lean source
    # checkout. Keep its import and constructor surface as a self-contained
    # compatibility fixture instead of depending on archived experiment data.
    overlay_path = tmp_path / "langchain_graph.py"
    overlay_path.write_text(
        '''\
"""Representative final-HL4 LangChain graph-policy overlay."""

from __future__ import annotations

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
    def __init__(self, *, model: str, run_id: str, max_artifacts: int) -> None:
        self.model = _langchain_openrouter_model(model)
        self.run_id = run_id
        self.max_artifacts = max_artifacts

    async def prepare(self, **kwargs: Any) -> LangChainGraphPolicyResult:
        raise NotImplementedError

    async def select_with_fixed_graph(
        self,
        **kwargs: Any,
    ) -> LangChainGraphPolicyResult:
        raise NotImplementedError


def _langchain_openrouter_model(model: str) -> str:
    normalized = model.removeprefix("openrouter/")
    if normalized.startswith("openrouter:"):
        return normalized
    return f"openrouter:{normalized}"


# Preserve the historical overlay's module-level dependency surface.
_COMPATIBILITY_IMPORTS = (
    DiffusionArtifact,
    TaskGraphEdgeRecord,
    AVOID_RECHECK_CHANNEL,
    REUSE_SUCCESS_CHANNEL,
    validate_openrouter_credentials,
)
''',
        encoding="utf-8",
    )
    module_name = "_hl4_update_0005_langchain_graph_smoke"
    spec = importlib.util.spec_from_file_location(module_name, overlay_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        policy = module.LangChainGraphPolicy(
            model="openrouter/test/model",
            run_id="overlay-smoke",
            max_artifacts=1,
        )

        assert policy.model == "openrouter:test/model"
        assert callable(policy.prepare)
        assert callable(policy.select_with_fixed_graph)
    finally:
        sys.modules.pop(module_name, None)
