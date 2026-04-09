"""Execution trace from the Executor agent."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ExecutionTrace(BaseModel):
    """Raw execution output from Gemini running a task in the sandbox."""

    task_id: str
    iteration: int
    timestamp: datetime = Field(default_factory=datetime.now)
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    test_results: dict | None = None
    reward: float = 0.0  # 0.0–1.0 task score
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    duration_sec: float = 0.0
