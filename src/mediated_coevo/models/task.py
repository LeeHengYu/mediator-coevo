"""Task specification and executor-facing task envelope models."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    """A task planned by the Planner for the Executor."""

    task_id: str
    instruction: str
    skills_context: list[str] = Field(default_factory=list)
    planner_reasoning: str | None = None
    iteration: int = 0


def render_executor_envelope(
    *,
    task_instruction: str,
    executor_policy: str | None,
    task_resources: Iterable[str] = (),
    verifier_contract: str | None = None,
) -> str:
    """Render the shared executor policy envelope used by benchmark adapters."""
    resources = "\n\n".join(
        block.strip() for block in task_resources if block.strip()
    )
    sections = (
        ("Task Instruction", task_instruction, "no task instruction supplied"),
        ("Executor Policy", executor_policy, "no executor policy supplied"),
        ("Task Resources", resources, "no task resources supplied"),
        ("Verifier Contract", verifier_contract, "no verifier contract supplied"),
    )
    return "\n\n".join(
        f"# {heading}\n\n{(content or '').strip() or f'({fallback})'}"
        for heading, content, fallback in sections
    )


def executor_policy_hash(executor_policy: str | None) -> str | None:
    """Return the stable hash for an injected executor policy."""
    if not executor_policy:
        return None
    return hashlib.sha256(executor_policy.encode("utf-8")).hexdigest()


def executor_policy_metadata(
    *,
    executor_policy: str | None,
    injection_location: str,
    task_resource_names: Iterable[str] = (),
    verifier_contract_kind: str,
) -> dict[str, str]:
    """Return compact trace metadata for executor policy observability."""
    resource_names = tuple(name for name in task_resource_names if name)
    policy_hash = executor_policy_hash(executor_policy)
    return {
        "executor_policy_hash": policy_hash or "",
        "executor_policy_injected": "true" if policy_hash else "false",
        "executor_policy_injection": injection_location,
        "task_resource_count": str(len(resource_names)),
        "task_resource_names": ",".join(resource_names),
        "verifier_contract_kind": verifier_contract_kind,
    }

