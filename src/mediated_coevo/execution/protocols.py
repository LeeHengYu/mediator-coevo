"""Protocols for frozen task resolution and explicit-context execution."""

from __future__ import annotations

from typing import Protocol

from mediated_coevo.execution.models import (
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskProfile,
)


class TaskProfileProvider(Protocol):
    """Resolve a task ID into a detached profile without executing it."""

    def resolve(self, task_id: str) -> TaskProfile: ...


class TaskExecutionAgent(Protocol):
    """Execute one task using only the request's explicit context."""

    async def execute(self, request: TaskExecutionRequest) -> TaskExecutionResult: ...
