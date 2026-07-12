"""Task graph CLI command registration."""

from __future__ import annotations

from pathlib import Path

import typer

from mediated_coevo.analysis.task_similarity import (
    build_task_graph_precompute,
    write_task_graph_artifacts,
)
from mediated_coevo.benchmarks import SkillFlowRepository
from mediated_coevo.core.config import Config
from mediated_coevo.cli.experiment import PROJECT_ROOT
from mediated_coevo.cli.output import console

TASK_SIMILARITY_GRAPH_NAMES = frozenset({"task_similarity", "precomputed_similarity"})
DEFAULT_TASK_GRAPH_EDGE_THRESHOLD = 0.05


def materialize_task_graph_for_diffusion(
    *,
    config: Config,
    experiment_dir: Path,
    benchmark_repo: SkillFlowRepository,
) -> None:
    """Write task graph artifacts when graph-aware diffusion is enabled."""
    if (
        not config.diffusion.enabled
        or config.diffusion.graph not in TASK_SIMILARITY_GRAPH_NAMES
    ):
        return

    output_dir = experiment_dir / "task-graph"
    if output_dir.exists():
        return

    precompute = build_task_graph_precompute(
        benchmark_repo.default_local_cache_dir(),
        edge_score_threshold=DEFAULT_TASK_GRAPH_EDGE_THRESHOLD,
    )
    write_task_graph_artifacts(precompute, output_dir)


def create_graph(
    threshold: float = typer.Option(
        DEFAULT_TASK_GRAPH_EDGE_THRESHOLD,
        "--threshold",
        min=0.0,
        help="Minimum similarity score required to keep an edge.",
    ),
    tasks_root: Path = typer.Option(
        PROJECT_ROOT / "benchmarks" / "skillflow" / "tasks",
        "--tasks-root",
        help="Local SkillFlow task directory to analyze.",
    ),
    output_dir: Path = typer.Option(
        PROJECT_ROOT / "data" / "task_graphs" / "skillflow-local",
        "--output-dir",
        help="Directory where graph precompute JSON artifacts are written.",
    ),
) -> None:
    """Create a directed SkillFlow task graph from local task metadata."""
    try:
        precompute = build_task_graph_precompute(
            tasks_root,
            edge_score_threshold=threshold,
        )
        write_task_graph_artifacts(precompute, output_dir)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]Tasks:[/] {precompute.task_count}")
    console.print(f"[bold]Pairs:[/] {precompute.pair_count}")
    console.print(f"[bold]Threshold:[/] {precompute.active_threshold}")
    console.print(f"[bold]Kept edges:[/] {precompute.kept_edge_count}")
    console.print(f"[bold]Cut edges:[/] {precompute.cut_edge_count}")
    console.print(f"[bold]Output:[/] {output_dir}")


def register_graph_command(app: typer.Typer) -> None:
    app.command("create-graph")(create_graph)
