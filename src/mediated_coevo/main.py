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
from mediated_coevo.config import load_config
from mediated_coevo.evolution.skill_advisor import SkillAdvisor
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
    tasks: str = typer.Option("fix-build-google-auto", help="Comma-separated task IDs"),
    iterations: int = typer.Option(30, help="Number of iterations"),
    seed: int = typer.Option(42, help="Random seed"),
    condition: str = typer.Option(
        "learned_mediator",
        help="Experiment condition: no_feedback | full_traces | shared_notes | static_mediator | learned_mediator",
    ),
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a mediated co-evolution experiment."""
    _setup_logging(verbose)
    random.seed(seed)

    config = load_config(config_dir)
    config.experiment.seed = seed
    config.experiment.condition_name = condition

    task_ids = [t.strip() for t in tasks.split(",")]

    console.print(f"[bold]Tasks:[/] {task_ids}")
    console.print(f"[bold]Iterations:[/] {iterations}")
    console.print(f"[bold]Condition:[/] {condition}")
    console.print(f"[bold]Models:[/] planner={config.models.planner} executor={config.models.executor} mediator={config.models.mediator}")

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = PROJECT_ROOT / "data" / "experiments" / f"{timestamp}-{seed}-{condition}"
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

    # Initialize agents
    planner = PlannerAgent(llm_client=LLMClient(model=config.models.planner))
    executor = ExecutorAgent(
        model=config.models.executor,
        benchmark_repo=benchmark_repo,
        harbor_runner=harbor_runner,
        workspace_root=experiment_dir / "benchmarks",
        injected_skill_name=config.executor_runtime.injected_skill_name,
    )
    mediator = MediatorAgent(
        llm_client=LLMClient(model=config.models.mediator),
        artifact_store=artifact_store,
    )
    protocol = skill_store.read_skill("mediator")
    if protocol:
        mediator.load_protocol(protocol)

    # Build orchestrator
    skill_advisor = SkillAdvisor(llm_client=LLMClient(model=config.models.planner))
    orchestrator = Orchestrator(
        planner=planner,
        executor=executor,
        mediator=mediator,
        skill_store=skill_store,
        artifact_store=artifact_store,
        history_store=history_store,
        benchmark_repo=benchmark_repo,
        config=config,
        experiment_dir=experiment_dir,
        skill_advisor=skill_advisor,
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


if __name__ == "__main__":
    app()
