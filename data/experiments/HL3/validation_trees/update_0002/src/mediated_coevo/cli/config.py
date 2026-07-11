"""CLI config loading, overrides, and selector parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, get_args

import typer

from mediated_coevo.core.config import (
    Config,
    ConfigLoadError,
    DiffusionPolicyName,
    load_config,
)
from mediated_coevo.experiment.baselines import parse_skill_updates
from mediated_coevo.experiment.conditions import ConditionName

VALID_CONDITION_NAMES = set(get_args(ConditionName))
VALID_DIFFUSION_POLICY_NAMES = set(get_args(DiffusionPolicyName))


def _task_ids_from_repeatable_cli(raw_values: list[str] | None) -> list[str]:
    """Parse repeatable comma-separated task options."""
    if not raw_values:
        return []
    task_ids: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for candidate in raw_value.split(","):
            task_id = candidate.strip()
            if task_id and task_id not in seen:
                task_ids.append(task_id)
                seen.add(task_id)
    if not task_ids:
        raise typer.BadParameter("at least one task ID is required")
    return task_ids


def _load_config_or_bad_parameter(
    config_dir: Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> Config:
    try:
        return load_config(config_dir, overrides=overrides)
    except ConfigLoadError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _run_config_overrides(
    *,
    iterations: int | None,
    seed: int | None,
    condition: str | None,
    skill_updates: str | None,
    coevo_interval: int | None,
    advisor_buffer_max: int | None,
    diffusion_enabled: bool | None,
    diffusion_policy: str | None,
    diffusion_graph: str | None,
    diffusion_max_artifacts: int | None,
    diffusion_top_k_neighbors: int | None,
    harbor_agent_setup_timeout_multiplier: float | None,
) -> dict[str, Any]:
    experiment: dict[str, Any] = {}
    if iterations is not None:
        experiment["num_iterations"] = iterations
    if seed is not None:
        experiment["seed"] = seed
    if coevo_interval is not None:
        experiment["coevo_interval"] = coevo_interval
    if advisor_buffer_max is not None:
        experiment["advisor_buffer_max"] = advisor_buffer_max
    if condition is not None:
        if condition not in VALID_CONDITION_NAMES:
            allowed = ", ".join(sorted(VALID_CONDITION_NAMES))
            raise typer.BadParameter(
                f"invalid condition {condition!r}; expected one of: {allowed}"
            )
        experiment["condition_name"] = condition
    if skill_updates is not None:
        try:
            experiment["skill_updates"] = parse_skill_updates(
                skill_updates
            ).model_dump()
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    diffusion: dict[str, Any] = {}
    if diffusion_enabled is not None:
        diffusion["enabled"] = diffusion_enabled
    if diffusion_policy is not None:
        if diffusion_policy not in VALID_DIFFUSION_POLICY_NAMES:
            allowed = ", ".join(sorted(VALID_DIFFUSION_POLICY_NAMES))
            raise typer.BadParameter(
                f"invalid diffusion policy {diffusion_policy!r}; "
                f"expected one of: {allowed}"
            )
        diffusion["policy"] = diffusion_policy
    if diffusion_graph is not None:
        diffusion["graph"] = diffusion_graph
    if diffusion_max_artifacts is not None:
        diffusion["max_artifacts"] = diffusion_max_artifacts
    if diffusion_top_k_neighbors is not None:
        diffusion["top_k_neighbors"] = diffusion_top_k_neighbors

    executor_runtime: dict[str, Any] = {}
    if harbor_agent_setup_timeout_multiplier is not None:
        executor_runtime["harbor_agent_setup_timeout_multiplier"] = (
            harbor_agent_setup_timeout_multiplier
        )

    overrides: dict[str, Any] = {}
    if experiment:
        overrides["experiment"] = experiment
    if diffusion:
        overrides["diffusion"] = diffusion
    if executor_runtime:
        overrides["executor_runtime"] = executor_runtime
    return overrides
