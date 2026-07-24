"""SkillFlow maintenance operations."""

from __future__ import annotations

from pathlib import Path

import typer

from mediated_coevo.benchmarks import SkillFlowSyncError
from mediated_coevo.cli.config import (
    _load_config_or_bad_parameter,
    _task_ids_from_repeatable_cli,
)
from mediated_coevo.cli.output import console
from mediated_coevo.experiment.runtime_factory import build_benchmark_repo


def sync_tasks(
    *,
    project_root: Path,
    tasks: list[str] | None,
    family: str | None,
    output_dir: Path | None,
    dataset: str,
    config_dir: Path,
) -> None:
    config = _load_config_or_bad_parameter(config_dir)
    config.executor_runtime.dataset = dataset
    repository = build_benchmark_repo(project_root, config)
    task_ids = _task_ids_from_repeatable_cli(tasks) if tasks else None
    if task_ids is not None and any(task_id.lower() == "all" for task_id in task_ids):
        if len(task_ids) > 1 or family is not None:
            raise typer.BadParameter(
                "--tasks all cannot be combined with task IDs or --family"
            )
        task_ids = None
    try:
        if family is not None:
            family_task_ids = repository.list_remote_task_ids(family=family)
            if not family_task_ids:
                raise typer.BadParameter(
                    f"no SkillFlow tasks found for family {family!r}"
                )
            task_ids = (task_ids or []) + family_task_ids
        destination = repository.sync_tasks(
            destination=output_dir,
            task_ids=task_ids,
        )
    except SkillFlowSyncError as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[bold]Downloaded SkillFlow tasks to:[/] {destination}")


def list_task_ids(
    *,
    project_root: Path,
    family: str | None,
    local: bool,
    dataset: str,
    config_dir: Path,
) -> list[str]:
    config = _load_config_or_bad_parameter(config_dir)
    config.executor_runtime.dataset = dataset
    repository = build_benchmark_repo(project_root, config)
    try:
        if local:
            return repository.list_local_task_ids(family=family)
        return repository.list_remote_task_ids(family=family)
    except (FileNotFoundError, SkillFlowSyncError) as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
        raise typer.Exit(code=1) from exc
