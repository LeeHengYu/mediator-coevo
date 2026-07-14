"""Shared LangChain runtime helpers for graph and policy agents.

This module intentionally contains no orchestration sequencing.  The graph and
diffusion-policy agents use the same invocation and inspection tools, while the
legacy :mod:`langchain_graph` module remains the stable harness-overlay seam.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from mediated_coevo.diffusion.models import DiffusionArtifact, TaskGraphSnapshot
from mediated_coevo.llm.client import validate_openrouter_credentials


def normalize_openrouter_model(model: str) -> str:
    """Return the LangChain OpenRouter model identifier."""
    normalized = model.removeprefix("openrouter/")
    if normalized.startswith("openrouter:"):
        return normalized
    return f"openrouter:{normalized}"


async def run_agent(
    *,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    tools: list[Any],
) -> dict[str, Any]:
    """Invoke one LangChain agent and parse its object response."""
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
    return parse_json_object(last_message_text(result))


def inspection_tools(
    snapshot: TaskGraphSnapshot | None,
    artifacts: list[DiffusionArtifact],
) -> list[Any]:
    """Build the read-only tools shared by both orchestration agents."""
    artifacts_by_id = {artifact.artifact_id: artifact for artifact in artifacts}

    def read_graph() -> dict[str, Any]:
        """Return the current graph snapshot, including agent-authored edge weights."""
        if snapshot is None:
            return {"task_ids": [], "edge_records": [], "metadata": {}}
        return snapshot.model_dump(mode="json")

    def query_artifacts(recent: int | None = None) -> list[dict[str, Any]]:
        """Return artifacts used as evidence for graph edges and transfer weights."""
        selected = artifacts if recent is None else artifacts[:recent]
        return [artifact_summary(artifact) for artifact in selected]

    def get_artifact(artifact_id: str) -> dict[str, Any]:
        """Return full artifact evidence for calibrating a graph edge weight."""
        artifact = artifacts_by_id.get(artifact_id)
        if artifact is None:
            return {"error": f"unknown artifact_id: {artifact_id}"}
        return artifact.model_dump(mode="json")

    return [read_graph, query_artifacts, get_artifact]


def artifact_summary(artifact: DiffusionArtifact) -> dict[str, Any]:
    """Return the compact artifact view exposed through agent tools."""
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


def last_message_text(result: Any) -> str:
    """Extract text from the last LangChain response message."""
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


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the JSON object emitted by a graph or policy agent."""
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
