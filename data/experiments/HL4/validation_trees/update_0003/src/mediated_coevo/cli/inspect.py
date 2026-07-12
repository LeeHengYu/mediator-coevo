"""Inspect CLI command registration."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from mediated_coevo.analysis.inspection import _inspection_payload
from mediated_coevo.cli.config import _load_config_or_bad_parameter
from mediated_coevo.cli.experiment import PROJECT_ROOT
from mediated_coevo.cli.output import print_inspection_payload


def latest_experiment_dir(experiments_root: Path) -> Path:
    if not experiments_root.exists():
        raise typer.BadParameter(f"experiment directory not found: {experiments_root}")
    candidates = [path for path in experiments_root.iterdir() if path.is_dir()]
    if not candidates:
        raise typer.BadParameter(
            f"no experiment outputs found under {experiments_root}"
        )
    return sorted(candidates, key=lambda path: path.name)[-1]


def inspect_experiment(
    experiment_dir: Path | None = typer.Argument(
        None,
        help="Experiment directory to inspect. Defaults to newest data/experiments run.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON.",
    ),
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
) -> None:
    """Inspect an experiment output directory."""
    target_dir = experiment_dir
    if target_dir is None:
        config = _load_config_or_bad_parameter(config_dir)
        target_dir = latest_experiment_dir(
            PROJECT_ROOT / config.paths.data_dir / "experiments"
        )
    payload = _inspection_payload(target_dir)
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    print_inspection_payload(payload)


def register_inspect_command(app: typer.Typer) -> None:
    app.command("inspect")(inspect_experiment)
