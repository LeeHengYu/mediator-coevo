"""Configuration loading and validation."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from mediated_coevo.experiment.conditions import ConditionName

OPENROUTER_MODEL_PREFIX = "openrouter/"
DEFAULT_SKILLSBENCH_ARCHIVE_URL = (
    "https://github.com/benchflow-ai/skillsbench/archive/refs/heads/main.zip"
)


class ModelConfigError(ValueError):
    """Raised when configured model names are invalid."""


def normalize_openrouter_model_name(model: str) -> str:
    """Return a model ID with exactly one OpenRouter prefix."""
    return f"{OPENROUTER_MODEL_PREFIX}{_provider_model_name(model)}"


def normalize_harbor_model_name(model: str) -> str:
    """Return a provider/model ID without the OpenRouter routing prefix."""
    return _provider_model_name(model)


def _provider_model_name(model: str) -> str:
    """Return the provider/model suffix after removing OpenRouter prefixes."""
    normalized = model.strip()
    if not normalized:
        raise ModelConfigError("model names must be non-empty OpenRouter model IDs")

    while normalized.startswith(OPENROUTER_MODEL_PREFIX):
        normalized = normalized.removeprefix(OPENROUTER_MODEL_PREFIX)

    parts = normalized.split("/")
    if len(parts) < 2 or any(not part for part in parts):
        raise ModelConfigError(
            "model names must include provider and model, for example "
            "'openrouter/openai/gpt-5.5'"
        )
    return normalized


class ModelsConfig(BaseModel):
    planner: str
    executor: str
    mediator: str
    judge: str


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
    judge_prompt_tokens: int = 16000
    planner_completion_tokens: int = 4096
    mediator_completion_tokens: int = 2048
    advisor_completion_tokens: int = 512
    reflector_completion_tokens: int = 4096
    judge_completion_tokens: int = 2048


class JudgeConfig(BaseModel):
    """LLM judge reward annotation settings."""

    rubric_version: str = "rar-v1"


class SkillUpdateConfig(BaseModel):
    """Independent permissions for committing each runtime skill family."""

    executor: bool = True
    planner: bool = True
    mediator: bool = True


class SkillValidationConfig(BaseModel):
    """Empirical gate settings for executor skill candidates."""

    enabled: bool = True
    min_mean_delta: float = 0.01
    reward_tolerance: float = 1e-9
    require_all_tasks_usable: bool = True
    sample_size: int = 3
    skillsbench_tasks: list[str] = Field(default_factory=list)
    swebench_instances: list[str] = Field(default_factory=list)
    allow_contributing_fallback: bool = True
    min_skillsbench_tag_overlap: int = 1
    allow_swebench_replacement_for_skillsbench: bool = False


class BenchmarkSelectionConfig(BaseModel):
    """Run-local benchmark task selection persisted with experiment config."""

    skillsbench_tasks: list[str] = Field(default_factory=list)
    swebench_instances: list[str] = Field(default_factory=list)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    num_iterations: int = 30
    coevo_interval: int = 5
    seed: int = 42
    advisor_buffer_max: int = 10
    condition_name: ConditionName = "learned_mediator"
    skill_updates: SkillUpdateConfig = Field(default_factory=SkillUpdateConfig)
    skill_validation: SkillValidationConfig = Field(
        default_factory=SkillValidationConfig
    )
    benchmark_selection: BenchmarkSelectionConfig = Field(
        default_factory=BenchmarkSelectionConfig
    )
    baseline_preset: str | None = None
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
    injected_skill_name: str = "executor"
    remote_fetch: bool = True
    archive_url: str = Field(default=DEFAULT_SKILLSBENCH_ARCHIVE_URL, min_length=1)
    archive_sha256: str | None = Field(default=None, pattern=r"^[A-Fa-f0-9]{64}$")
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
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    executor_runtime: ExecutorRuntimeConfig = Field(
        default_factory=ExecutorRuntimeConfig
    )

    def normalize_models(self) -> Self:
        """Normalize every configured model name in-place."""
        for field_name, model in self.models:
            if field_name == "executor":
                normalized = normalize_harbor_model_name(model)
            else:
                normalized = normalize_openrouter_model_name(model)
            setattr(self.models, field_name, normalized)
        return self


def load_config(config_dir: Path) -> Config:
    """Load default.toml from config_dir."""
    default_path = config_dir / "default.toml"
    data: dict = {}

    if default_path.exists():
        with open(default_path, "rb") as f:
            data = tomllib.load(f)

    return Config(**data)
