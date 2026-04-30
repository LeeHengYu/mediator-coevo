"""Configuration loading and validation."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mediated_coevo.conditions import ConditionName


class ModelsConfig(BaseModel):
    planner: str
    executor: str
    mediator: str


class BudgetsConfig(BaseModel):
    max_skill_tokens: int = 4000
    trace_excerpt_tokens: int = 6000
    historical_summary_tokens: int = 3000
    mediator_report_tokens: int = 4000
    planner_context_tokens: int = 24000
    skill_update_diff_tokens: int = 6000
    mediator_prompt_tokens: int = 16000
    advisor_prompt_tokens: int = 12000
    reflector_prompt_tokens: int = 16000
    planner_completion_tokens: int = 4096
    mediator_completion_tokens: int = 2048
    advisor_completion_tokens: int = 512
    reflector_completion_tokens: int = 4096


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    num_iterations: int = 30
    coevo_interval: int = 5
    seed: int = 42
    advisor_buffer_max: int = 10
    condition_name: ConditionName = "learned_mediator"
    shared_notes: str | None = None
    allow_cross_task_feedback: bool = False


class PathsConfig(BaseModel):
    skills_dir: str = "skills"
    data_dir: str = "data"
    benchmarks_dir: str = "benchmarks/skillsbench"


class ExecutorRuntimeConfig(BaseModel):
    backend: str = "skillsbench"
    agent_name: str = "opencode"
    jobs_dir: str = "jobs"
    task_dirs: list[str] = Field(default_factory=lambda: ["tasks"])
    injected_skill_name: str = "executor-evolved"
    # Hard wall-clock cap on a single Harbor subprocess (seconds). Prevents
    # a hung run from blocking the orchestrator indefinitely.
    harbor_timeout_sec: float = 1800.0
    # When True, refuse to start the experiment if the harbor CLI is missing.
    # When False, the executor synthesizes env_failure traces on each task
    # so CI can exercise the orchestrator without harbor installed.
    harbor_required: bool = True


class Config(BaseModel):
    """Top-level configuration. Loaded from TOML."""

    models: ModelsConfig
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    executor_runtime: ExecutorRuntimeConfig = Field(default_factory=ExecutorRuntimeConfig)


def load_config(config_dir: Path) -> Config:
    """Load default.toml from config_dir."""
    default_path = config_dir / "default.toml"
    data: dict = {}

    if default_path.exists():
        with open(default_path, "rb") as f:
            data = tomllib.load(f)

    return Config(**data)
