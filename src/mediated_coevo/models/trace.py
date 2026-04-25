"""Execution trace from the Executor agent."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# Status classification for an ExecutionTrace.
#   ok            — Harbor ran, verifier produced a reward, no env hiccups.
#   task_failed   — Reserved for downstream consumers (orchestrator) to mark a
#                   legitimate task failure when reward < threshold; the parser
#                   itself never assigns this — a parsed-but-low reward stays "ok".
#   env_failure   — Missing/unparseable artifact, missing trial dir, or harbor
#                   binary not found. Reward is unreliable; downstream consumers
#                   should NOT feed this into mediator/skill-update channels.
#   parse_error   — A reward source was present but its content was malformed.
#   harbor_failed — Harbor subprocess returned non-zero. Reward may or may not
#                   be present; treat as env_failure for co-evolution signal.
TraceStatus = Literal["ok", "task_failed", "env_failure", "parse_error", "harbor_failed"]


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ExecutionTrace(BaseModel):
    """Raw execution output from the executor sandbox.

    `reward` is Optional: ``None`` means "no reward could be parsed" and
    must not be confused with a legitimate score of 0.0. The orchestrator
    skips mediator/proposal/tagging when `reward is None` or `status != "ok"`.
    """

    task_id: str
    iteration: int
    timestamp: datetime = Field(default_factory=datetime.now)
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    test_results: dict | None = None
    reward: float | None = None  # 0.0–1.0; None when no reward could be parsed
    status: TraceStatus = "ok"
    error_kind: str | None = None
    error_detail: Any = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    duration_sec: float = 0.0
