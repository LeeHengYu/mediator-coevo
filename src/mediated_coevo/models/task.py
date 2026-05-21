"""Task specification and executor-facing task envelope models."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    """A task planned by the Planner for the Executor."""

    task_id: str
    instruction: str
    skills_context: list[str] = Field(default_factory=list)
    planner_reasoning: str | None = None
    iteration: int = 0


@dataclass(frozen=True, slots=True)
class ExecutorEnvelope:
    """Portable runtime context shown to the Executor across benchmarks."""

    task_instruction: str
    executor_policy: str | None
    task_resources: tuple[str, ...] = ()
    verifier_contract: str | None = None

    def render(self) -> str:
        """Return a stable Markdown envelope for executor runtimes."""
        return "\n\n".join(
            [
                _render_section(
                    "Task Instruction",
                    self.task_instruction,
                    fallback="(no task instruction supplied)",
                ),
                _render_section(
                    "Executor Policy",
                    self.executor_policy,
                    fallback="(no executor policy supplied)",
                ),
                _render_section(
                    "Task Resources",
                    _join_blocks(self.task_resources),
                    fallback="(no task resources supplied)",
                ),
                _render_section(
                    "Verifier Contract",
                    self.verifier_contract,
                    fallback="(no verifier contract supplied)",
                ),
            ]
        )


def render_executor_envelope(
    *,
    task_instruction: str,
    executor_policy: str | None,
    task_resources: Iterable[str] = (),
    verifier_contract: str | None = None,
) -> str:
    """Render the shared executor policy envelope used by benchmark adapters."""
    return ExecutorEnvelope(
        task_instruction=task_instruction,
        executor_policy=executor_policy,
        task_resources=tuple(task_resources),
        verifier_contract=verifier_contract,
    ).render()


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


def _render_section(title: str, content: str | None, *, fallback: str) -> str:
    body = (content or "").strip() or fallback
    return f"# {title}\n\n{body}"


def _join_blocks(blocks: Iterable[str]) -> str:
    return "\n\n".join(block.strip() for block in blocks if block.strip())
