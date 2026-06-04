"""CLI entry point for the mediated co-evolution system."""

from __future__ import annotations

import typer

from mediated_coevo.cli.context_budget import register_context_budget_command
from mediated_coevo.cli.graph import register_graph_command
from mediated_coevo.cli.inspect import register_inspect_command
from mediated_coevo.cli.matrix import register_matrix_command
from mediated_coevo.cli.run import register_run_command
from mediated_coevo.cli.skillflow import register_skillflow_commands

app = typer.Typer(name="medcoevo", help="Mediated Co-Evolution Experiment Runner")

register_run_command(app)
register_matrix_command(app)
register_inspect_command(app)
register_graph_command(app)
register_context_budget_command(app)
register_skillflow_commands(app)

if __name__ == "__main__":
    app()
