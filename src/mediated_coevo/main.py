"""CLI entry point for the mediated co-evolution system."""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime
from pathlib import Path

import tomli_w
import typer
from rich.console import Console
from rich.logging import RichHandler

from mediated_coevo.agents.executor import ExecutorAgent
from mediated_coevo.agents.mediator import MediatorAgent
from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.benchmarks import HarborRunner, SkillsBenchRepository
from mediated_coevo.conditions import create_condition
from mediated_coevo.config import load_config
from mediated_coevo.llm.client import LLMClient
from mediated_coevo.orchestrator import Orchestrator
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from mediated_coevo.stores.skill_store import SkillStore

app = typer.Typer(name="medcoevo", help="Mediated Co-Evolution Experiment Runner")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=True, show_path=False)],
    )


@app.command()
def run(
    condition: str = typer.Option("learned_mediator", help="Experimental condition"),
    tasks: str = typer.Option("fix-build-google-auto", help="Comma-separated task IDs"),
    iterations: int = typer.Option(30, help="Number of iterations"),
    seed: int = typer.Option(42, help="Random seed"),
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a single experiment with the specified condition."""
    _setup_logging(verbose)
    random.seed(seed)

    config = load_config(config_dir, condition=condition)
    config.experiment.condition = condition
    config.experiment.seed = seed

    task_ids = [t.strip() for t in tasks.split(",")]

    console.print(f"[bold]Condition:[/] {condition}")
    console.print(f"[bold]Tasks:[/] {task_ids}")
    console.print(f"[bold]Iterations:[/] {iterations}")
    console.print(f"[bold]Models:[/] planner={config.models.planner} executor={config.models.executor} mediator={config.models.mediator}")

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = PROJECT_ROOT / "data" / "experiments" / f"{timestamp}-{condition}-{seed}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save frozen config
    with open(experiment_dir / "config.toml", "wb") as f:
        tomli_w.dump(config.model_dump(), f)

    # Initialize stores
    skills_dir = PROJECT_ROOT / config.paths.skills_dir
    skill_store = SkillStore(skills_dir=skills_dir)
    artifact_store = ArtifactStore(base_dir=experiment_dir / "artifacts")
    history_store = HistoryStore(history_dir=experiment_dir / "history")
    benchmark_repo = SkillsBenchRepository(
        root_dir=PROJECT_ROOT / config.paths.benchmarks_dir,
        task_dirs=config.executor_runtime.task_dirs,
    )
    harbor_runner = HarborRunner(
        agent_name=config.executor_runtime.agent_name,
        jobs_dir=experiment_dir / config.executor_runtime.jobs_dir,
    )

    # Initialize LLM clients
    planner_llm = LLMClient(model=config.models.planner)
    executor_llm = LLMClient(model=config.models.executor)

    # Initialize agents
    planner = PlannerAgent(llm_client=planner_llm)
    executor = ExecutorAgent(
        llm_client=executor_llm,
        benchmark_repo=benchmark_repo,
        harbor_runner=harbor_runner,
        workspace_root=experiment_dir / "benchmarks",
        injected_skill_name=config.executor_runtime.injected_skill_name,
        sandbox_config=config.sandbox.model_dump(),
    )

    # Initialize mediator only for conditions that need it
    mediator: MediatorAgent | None = None
    if condition == "learned_mediator":
        mediator_llm = LLMClient(model=config.models.mediator)
        mediator = MediatorAgent(
            llm_client=mediator_llm,
            artifact_store=artifact_store,
            token_budget=config.budgets.mediator_report_tokens,
        )
        # Load coordination protocol skill
        protocol = skill_store.read_skill("mediator")
        if protocol:
            mediator.load_protocol(protocol)

    # Create condition
    feedback_condition = create_condition(condition)

    # Build orchestrator
    orchestrator = Orchestrator(
        planner=planner,
        executor=executor,
        mediator=mediator,
        condition=feedback_condition,
        skill_store=skill_store,
        artifact_store=artifact_store,
        history_store=history_store,
        benchmark_repo=benchmark_repo,
        config=config,
        experiment_dir=experiment_dir,
    )

    # Run
    console.print(f"\n[bold green]Starting experiment:[/] {experiment_dir}\n")
    records = asyncio.run(orchestrator.run_experiment(task_ids, iterations))

    # Summary
    rewards = [r.reward for r in records]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    total_tokens = sum(r.total_tokens for r in records)
    console.print(f"\n[bold]Results:[/]")
    console.print(f"  Iterations: {len(records)}")
    console.print(f"  Avg reward: {avg_reward:.3f}")
    console.print(f"  Total tokens: {total_tokens:,}")
    console.print(f"  Data: {experiment_dir}")


@app.command()
def sweep(
    tasks: str = typer.Option("fix-build-google-auto", help="Comma-separated task IDs"),
    iterations: int = typer.Option(30, help="Number of iterations per condition"),
    seeds: str = typer.Option("42,43,44", help="Comma-separated seeds"),
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run all 5 conditions for comparison."""
    conditions = [
        "no_feedback", "full_traces", "shared_notes",
        "static_mediator", "learned_mediator",
    ]
    seed_list = [int(s.strip()) for s in seeds.split(",")]

    for seed in seed_list:
        for cond in conditions:
            console.print(f"\n[bold yellow]>>> {cond} (seed={seed}) <<<[/]")
            run(
                condition=cond,
                tasks=tasks,
                iterations=iterations,
                seed=seed,
                config_dir=config_dir,
                verbose=verbose,
            )


if __name__ == "__main__":
    app()
