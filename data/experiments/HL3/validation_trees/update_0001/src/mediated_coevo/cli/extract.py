"""Extract saved diffusion artifact stores from completed experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mediated_coevo.cli.experiment import PROJECT_ROOT
from mediated_coevo.cli.output import console
from mediated_coevo.diffusion import DiffusionStore


def extract(
    experiment_path: Annotated[
        Path,
        typer.Option(
            "--path",
            "-p",
            help="Experiment directory to extract diffusion artifacts from.",
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for saved artifact stores.",
        ),
    ] = None,
) -> None:
    """Recover the final diffusion artifact state from an experiment."""
    experiment_path = experiment_path.expanduser()
    diffusion_dir = experiment_path / "diffusion"
    destination_root = output_dir or PROJECT_ROOT / "data" / "artifact-stores"
    destination = destination_root / experiment_path.name
    try:
        count = DiffusionStore.export_artifact_store(
            diffusion_dir,
            destination,
            store_id=experiment_path.name,
        )
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]Artifacts:[/] {count}")
    console.print(f"[bold]Saved artifact store:[/] {destination}")


def register_extract_command(app: typer.Typer) -> None:
    app.command("extract")(extract)
