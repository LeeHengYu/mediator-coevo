"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


class ModelsConfig(BaseModel):
    planner: str = "anthropic/claude-opus-4"
    executor: str = "gemini/gemini-3-flash-preview"
    mediator: str = "openai/gpt-5.4"


class BudgetsConfig(BaseModel):
    mediator_report_tokens: int = 2000
    max_skill_tokens: int = 4000


class ExperimentConfig(BaseModel):
    condition: str = "learned_mediator"
    num_iterations: int = 30
    coevo_interval: int = 5
    seed: int = 42
    advisor_buffer_max: int = 10


class SandboxConfig(BaseModel):
    type: str = "harbor"
    timeout_sec: int = 300


class PathsConfig(BaseModel):
    skills_dir: str = "skills"
    data_dir: str = "data"
    benchmarks_dir: str = "benchmarks/skillsbench"


class ExecutorRuntimeConfig(BaseModel):
    backend: str = "skillsbench"
    agent_name: str = "codex"
    jobs_dir: str = "jobs"
    task_dirs: list[str] = Field(default_factory=lambda: ["tasks"])
    injected_skill_name: str = "executor-evolved"


class Config(BaseModel):
    """Top-level configuration. Loaded from TOML with env var overrides."""

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    executor_runtime: ExecutorRuntimeConfig = Field(default_factory=ExecutorRuntimeConfig)


def load_config(
    config_dir: Path,
    condition: str | None = None,
) -> Config:
    """Load default.toml, then overlay condition-specific config if given."""
    default_path = config_dir / "default.toml"
    data: dict = {}

    if default_path.exists():
        with open(default_path, "rb") as f:
            data = tomllib.load(f)

    if condition:
        cond_path = config_dir / "experiments" / f"{condition}.toml"
        if cond_path.exists():
            with open(cond_path, "rb") as f:
                cond_data = tomllib.load(f)
            # Deep merge: condition overrides default
            for section, values in cond_data.items():
                if section in data and isinstance(data[section], dict):
                    data[section].update(values)
                else:
                    data[section] = values

    return Config(**data)
