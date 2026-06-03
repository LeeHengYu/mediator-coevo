"""SkillFlow maintenance command registration."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from mediated_coevo.benchmarks import (
    DEFAULT_SKILLFLOW_DATASET,
    SkillFlowSyncError,
)
from mediated_coevo.cli.config import (
    _load_config_or_bad_parameter,
    _task_ids_from_repeatable_cli,
)
from mediated_coevo.cli.experiment import PROJECT_ROOT, setup_logging
from mediated_coevo.cli.output import console
from mediated_coevo.experiment.runtime_factory import build_benchmark_repo


def build_skillflow_base_image(
    base_image_tag: str = typer.Option(
        "skillflow/harbor-cli-base:ubuntu24.04",
        help="Docker tag to build for the SkillFlow Harbor CLI base image.",
    ),
    dry_run: bool = typer.Option(
        False,
        help="Show the base image build command without running it.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build the required SkillFlow Harbor CLI base image."""
    setup_logging(verbose)
    build_script = PROJECT_ROOT / "docker" / "harbor-cli-base" / "build.sh"

    if not build_script.is_file():
        console.print(f"[bold red]ERROR:[/] missing SkillFlow build script: {build_script}")
        raise typer.Exit(code=1)

    base_command = ["bash", str(build_script), base_image_tag]
    if dry_run:
        console.print(f"[bold]Would build base image:[/] {shlex.join(base_command)}")
        console.print("[bold green]SkillFlow base image dry run complete.[/]")
        return

    console.print(f"[bold]Build SkillFlow base image:[/] {shlex.join(base_command)}")
    try:
        completed = subprocess.run(base_command, cwd=PROJECT_ROOT, check=False)
    except OSError as exc:
        console.print(
            f"[bold red]ERROR:[/] Build SkillFlow base image failed to start: {exc}"
        )
        raise typer.Exit(code=1) from exc
    if completed.returncode != 0:
        console.print(
            "[bold red]ERROR:[/] Build SkillFlow base image failed with exit code "
            f"{completed.returncode}."
        )
        raise typer.Exit(code=completed.returncode)
    console.print("[bold green]SkillFlow base image build complete.[/]")


def sync_skillflow(
    tasks: Annotated[
        list[str] | None,
        typer.Option(
            "--tasks",
            "--task",
            "-t",
            help=(
                "Remote SkillFlow task ID(s) to download. Repeat the option, "
                "provide comma-separated IDs, or use 'all'."
            ),
        ),
    ] = None,
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Local tasks/ directory where SkillFlow task data should be downloaded.",
    ),
    dataset: str = typer.Option(
        DEFAULT_SKILLFLOW_DATASET,
        "--dataset",
        help="Hugging Face dataset ID.",
    ),
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download SkillFlow task data into the configured local cache."""
    setup_logging(verbose)
    config = _load_config_or_bad_parameter(config_dir)
    config.executor_runtime.dataset = dataset
    repository = build_benchmark_repo(PROJECT_ROOT, config)
    task_ids = _task_ids_from_repeatable_cli(tasks) if tasks else None
    if task_ids is not None and any(task_id.lower() == "all" for task_id in task_ids):
        if len(task_ids) > 1:
            raise typer.BadParameter("--tasks all cannot be combined with task IDs")
        task_ids = None
    try:
        destination = repository.sync_tasks(
            destination=output_dir,
            task_ids=task_ids,
        )
    except SkillFlowSyncError as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[bold]Downloaded SkillFlow tasks to:[/] {destination}")


def list_skillflow_tasks(
    family: str | None = typer.Option(
        None,
        "--family",
        help="Filter task IDs by SkillFlow family.",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="List cached local tasks instead of remote Hugging Face tasks.",
    ),
    dataset: str = typer.Option(
        DEFAULT_SKILLFLOW_DATASET,
        "--dataset",
        help="Hugging Face dataset ID.",
    ),
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List available SkillFlow task IDs."""
    setup_logging(verbose)
    config = _load_config_or_bad_parameter(config_dir)
    config.executor_runtime.dataset = dataset
    repository = build_benchmark_repo(PROJECT_ROOT, config)
    try:
        if local:
            task_ids = repository.list_local_task_ids(family=family)
        else:
            task_ids = repository.list_remote_task_ids(family=family)
    except (FileNotFoundError, SkillFlowSyncError) as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
        raise typer.Exit(code=1) from exc
    for task_id in task_ids:
        typer.echo(task_id)


def register_skillflow_commands(app: typer.Typer) -> None:
    app.command("build-base-image")(build_skillflow_base_image)
    app.command("sync")(sync_skillflow)
    app.command("list")(list_skillflow_tasks)
