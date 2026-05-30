"""Configuration loading and validation."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from mediated_coevo.experiment.conditions import ConditionName

OPENROUTER_MODEL_PREFIX = "openrouter/"
DEFAULT_SKILLSBENCH_ARCHIVE_URL = (
    "https://github.com/benchflow-ai/skillsbench/archive/refs/heads/main.zip"
)
DiffusionPolicyName: TypeAlias = Literal[
    "none",
    "capped_broadcast",
    "random_k",
    "top_k_similarity",
]


class ModelConfigError(ValueError):
    """Raised when configured model names are invalid."""


class ConfigLoadError(ValueError):
    """Raised when configuration files are missing required settings."""


REQUIRED_CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("experiment", "num_iterations"),
    ("experiment", "coevo_interval"),
    ("experiment", "seed"),
    ("experiment", "advisor_buffer_max"),
    ("experiment", "condition_name"),
    ("experiment", "allow_cross_task_feedback"),
    ("experiment", "skill_updates", "executor"),
    ("experiment", "skill_updates", "planner"),
    ("experiment", "skill_updates", "mediator"),
    ("diffusion", "enabled"),
    ("diffusion", "policy"),
    ("diffusion", "max_artifacts"),
    ("diffusion", "top_k_neighbors"),
)

CONFIG_CLI_HINTS: dict[str, str] = {
    "experiment.num_iterations": "--iterations",
    "experiment.coevo_interval": "--coevo-interval",
    "experiment.seed": "--seed",
    "experiment.advisor_buffer_max": "--advisor-buffer-max",
    "experiment.condition_name": "--condition",
    "experiment.skill_updates.executor": "--skill-updates",
    "experiment.skill_updates.planner": "--skill-updates",
    "experiment.skill_updates.mediator": "--skill-updates",
    "diffusion.enabled": "--diffusion-enabled/--no-diffusion-enabled",
    "diffusion.policy": "--diffusion-policy",
    "diffusion.max_artifacts": "--diffusion-max-artifacts",
    "diffusion.top_k_neighbors": "--diffusion-top-k-neighbors",
}


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

    executor: bool
    planner: bool
    mediator: bool


class SkillValidationConfig(BaseModel):
    """Empirical gate settings for executor skill candidates."""

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

    num_iterations: int
    coevo_interval: int
    seed: int
    advisor_buffer_max: int
    condition_name: ConditionName
    skill_updates: SkillUpdateConfig
    skill_validation: SkillValidationConfig = Field(
        default_factory=SkillValidationConfig
    )
    benchmark_selection: BenchmarkSelectionConfig = Field(
        default_factory=BenchmarkSelectionConfig
    )
    baseline_preset: str | None = None
    shared_notes: str | None = None
    allow_cross_task_feedback: bool


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
    # Hard wall-clock cap on a single Harbor subprocess (seconds). This must
    # exceed task agent/verifier phase limits so Harbor can finish cleanly.
    harbor_timeout_sec: float = 5400.0
    # Optional Harbor multiplier for agent setup. Some tasks spend several
    # minutes installing agent dependencies before the actual run starts.
    harbor_agent_setup_timeout_multiplier: float | None = None
    # When True, refuse to start the experiment if the harbor CLI is missing.
    # When False, the executor synthesizes env_failure traces on each task
    # so CI can exercise the orchestrator without harbor installed.
    harbor_required: bool = True


class DiffusionConfig(BaseModel):
    """Config gate for future graph-aware context diffusion."""

    enabled: bool
    policy: DiffusionPolicyName
    max_artifacts: int = Field(ge=1)
    top_k_neighbors: int = Field(ge=1)


class Config(BaseModel):
    """Top-level configuration. Loaded from TOML."""

    models: ModelsConfig
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    experiment: ExperimentConfig
    diffusion: DiffusionConfig
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


def load_config(
    config_dir: Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Load default.toml from config_dir."""
    default_path = config_dir / "default.toml"
    data: dict = {}

    if default_path.exists():
        with open(default_path, "rb") as f:
            data = tomllib.load(f)

    if overrides:
        _deep_merge(data, overrides)
    missing_paths = _missing_required_config_paths(data)
    if missing_paths:
        raise ConfigLoadError(_missing_config_message(default_path, missing_paths))

    return Config(**data)


def _deep_merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Merge CLI override mappings into raw TOML data before validation."""
    for key, value in source.items():
        if (
            isinstance(value, Mapping)
            and isinstance(target.get(key), dict)
        ):
            _deep_merge(target[key], value)
            continue
        target[key] = value


def _missing_required_config_paths(data: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for path in REQUIRED_CONFIG_PATHS:
        current: Any = data
        for part in path:
            if not isinstance(current, Mapping) or part not in current:
                missing.append(".".join(path))
                break
            current = current[part]
    return missing


def _missing_config_message(default_path: Path, missing_paths: list[str]) -> str:
    details = []
    for path in missing_paths:
        cli_hint = CONFIG_CLI_HINTS.get(path)
        if cli_hint is None:
            details.append(path)
        else:
            details.append(f"{path} ({cli_hint})")
    joined = ", ".join(details)
    return (
        f"missing required config setting(s) in {default_path}: {joined}. "
        "Set them in default.toml or pass the matching CLI option when available."
    )
