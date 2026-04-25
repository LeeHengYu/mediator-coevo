"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from mediated_coevo.conditions import ConditionName

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

class ModelsConfig(BaseModel):
    planner: str = "anthropic/claude-opus-4"
    executor: str = "gemini-3-flash-preview" # no litellm prefix since it's used in Skillsbench
    mediator: str = "openai/gpt-5.4"


class BudgetsConfig(BaseModel):
    max_skill_tokens: int = 4000


class ExperimentConfig(BaseModel):
    num_iterations: int = 30
    coevo_interval: int = 5
    seed: int = 42
    advisor_buffer_max: int = 10
    condition_name: ConditionName = "learned_mediator"
    shared_notes: str | None = None


class PathsConfig(BaseModel):
    skills_dir: str = "skills"
    data_dir: str = "data"
    benchmarks_dir: str = "benchmarks/skillsbench"


class ExecutorRuntimeConfig(BaseModel):
    backend: str = "skillsbench"
    agent_name: str = "gemini"
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

    models: ModelsConfig = Field(default_factory=ModelsConfig)
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
