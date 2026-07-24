"""Shared benchmark maintenance command registration."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from mediated_coevo.benchmarks import DEFAULT_SKILLFLOW_DATASET
from mediated_coevo.benchmarks.lifelong_agent_bench import KNOWN_FAMILIES
from mediated_coevo.cli import lifelong_agent_bench, skillflow
from mediated_coevo.cli.experiment import PROJECT_ROOT, setup_logging
from mediated_coevo.cli.output import console


def build_base_image(
    dry_run: bool = typer.Option(
        False,
        help="Show the shared base-image build commands without running them.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build the shared SkillFlow and OS benchmark base images."""
    setup_logging(verbose)
    commands = lifelong_agent_bench.base_image_commands(PROJECT_ROOT)

    if dry_run:
        for command in commands:
            console.print(f"[bold]Would build base image:[/] {shlex.join(command)}")
        console.print("[bold green]Shared base images dry run complete.[/]")
        return

    for command in commands:
        console.print(f"[bold]Build base image:[/] {shlex.join(command)}")
        try:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        except OSError as exc:
            console.print(
                f"[bold red]ERROR:[/] Build base image failed to start: {exc}"
            )
            raise typer.Exit(code=1) from exc
        if completed.returncode != 0:
            console.print(
                "[bold red]ERROR:[/] Build base image failed with exit code "
                f"{completed.returncode}."
            )
            raise typer.Exit(code=completed.returncode)
    console.print("[bold green]Shared base image build complete.[/]")


def sync(
    tasks: Annotated[
        list[str] | None,
        typer.Option(
            "--tasks",
            "--task",
            "-t",
            help=(
                "SkillFlow task slug(s) to download. Repeat the option, provide "
                "comma-separated slugs, or use 'all'."
            ),
        ),
    ] = None,
    family: str | None = typer.Option(
        None,
        "--family",
        help=(
            "Benchmark family to sync. Known local families infer their pinned "
            "source; other families use the SkillFlow dataset."
        ),
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Destination tasks directory; defaults to the inferred benchmark cache.",
    ),
    dataset: str = typer.Option(
        DEFAULT_SKILLFLOW_DATASET,
        "--dataset",
        help="SkillFlow Hugging Face dataset ID; not used by inferred local families.",
    ),
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Materialize benchmark tasks, inferring known sources from --family."""
    setup_logging(verbose)
    if family in KNOWN_FAMILIES:
        lifelong_agent_bench.sync_tasks(
            project_root=PROJECT_ROOT,
            family=family,
            tasks=tasks,
            output_dir=output_dir,
        )
        return
    skillflow.sync_tasks(
        project_root=PROJECT_ROOT,
        tasks=tasks,
        family=family,
        output_dir=output_dir,
        dataset=dataset,
        config_dir=config_dir,
    )


def list_tasks(
    family: str | None = typer.Option(
        None,
        "--family",
        help="Benchmark family whose cached task slugs should be listed.",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help=(
            "For SkillFlow, scan materialized tasks instead of its cached slug "
            "index. Inferred local families always use their local slug file."
        ),
    ),
    dataset: str = typer.Option(
        DEFAULT_SKILLFLOW_DATASET,
        "--dataset",
        help="SkillFlow Hugging Face dataset ID; not used by inferred local families.",
    ),
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List task slugs from the SkillFlow and local benchmark indexes."""
    setup_logging(verbose)
    if family in KNOWN_FAMILIES:
        task_ids = lifelong_agent_bench.list_task_ids(PROJECT_ROOT, family)
    else:
        task_ids = skillflow.list_task_ids(
            project_root=PROJECT_ROOT,
            family=family,
            local=local,
            dataset=dataset,
            config_dir=config_dir,
        )
        if family is None:
            task_ids.extend(lifelong_agent_bench.list_task_ids(PROJECT_ROOT))
    for task_id in sorted(dict.fromkeys(task_ids)):
        typer.echo(task_id)


def register_benchmark_commands(app: typer.Typer) -> None:
    app.command("build-base-image")(build_base_image)
    app.command("sync")(sync)
    app.command("list")(list_tasks)
