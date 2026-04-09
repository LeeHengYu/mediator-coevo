"""Task specification model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    """A task planned by the Planner for the Executor."""

    task_id: str
    instruction: str
    skills_context: list[str] = Field(default_factory=list)
    planner_reasoning: str | None = None
    iteration: int = 0
