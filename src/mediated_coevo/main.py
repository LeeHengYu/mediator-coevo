"""CLI entry point for fixed-skill mediated experiments."""

from __future__ import annotations

import typer

from mediated_coevo.cli.benchmark import register_benchmark_commands
from mediated_coevo.cli.hl import register_hl_command
from mediated_coevo.cli.matrix import register_matrix_command
from mediated_coevo.cli.run import register_run_command
from mediated_coevo.cli.sequence import register_sequence_command

app = typer.Typer(
    name="medcoevo",
    help="Mediated benchmark and heuristic-learning experiment runner.",
)

register_run_command(app)
register_sequence_command(app)
register_hl_command(app)
register_matrix_command(app)
register_benchmark_commands(app)

if __name__ == "__main__":
    app()
