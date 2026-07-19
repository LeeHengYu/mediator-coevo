"""CLI entry point for fixed-skill mediated experiments."""

from __future__ import annotations

import typer

from mediated_coevo.cli.matrix import register_matrix_command
from mediated_coevo.cli.run import register_run_command
from mediated_coevo.cli.sequence import register_sequence_command
from mediated_coevo.cli.skillflow import register_skillflow_commands

app = typer.Typer(name="medcoevo", help="Fixed-Skill Mediation Experiment Runner")

register_run_command(app)
register_sequence_command(app)
register_matrix_command(app)
register_skillflow_commands(app)

if __name__ == "__main__":
    app()
