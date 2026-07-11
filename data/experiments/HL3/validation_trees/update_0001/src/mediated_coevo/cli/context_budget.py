"""CLI for read-only context-budget comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from mediated_coevo.analysis.context_budget_compare import (
    compare_context_budget_runs,
)
from mediated_coevo.cli.output import print_context_budget_comparison


def compare_context_budgets(
    run_a_dir: Annotated[
        Path,
        typer.Argument(help="First completed experiment directory."),
    ],
    run_b_dir: Annotated[
        Path,
        typer.Argument(help="Second completed experiment directory."),
    ],
    tolerance: Annotated[
        float,
        typer.Option(
            "--tolerance",
            min=0.0,
            help="Absolute tolerance for zero-baseline token deltas.",
        ),
    ] = 0.05,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Compare prior-context budgets across two completed runs."""
    if not run_a_dir.exists() or not run_a_dir.is_dir():
        raise typer.BadParameter(f"run directory not found: {run_a_dir}")
    if not run_b_dir.exists() or not run_b_dir.is_dir():
        raise typer.BadParameter(f"run directory not found: {run_b_dir}")
    try:
        comparison = compare_context_budget_runs(
            run_a_dir,
            run_b_dir,
            tolerance=tolerance,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        typer.echo(json.dumps(comparison.model_dump(mode="json"), indent=2))
        return
    print_context_budget_comparison(comparison)


def register_context_budget_command(app: typer.Typer) -> None:
    app.command("compare-context-budgets")(compare_context_budgets)
