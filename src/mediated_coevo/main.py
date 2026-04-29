"""CLI entry point for the mediated co-evolution system."""

from __future__ import annotations

import asyncio
import logging
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast, get_args

import tomli_w
import typer
from rich.console import Console
from rich.logging import RichHandler

from mediated_coevo.agents.executor import ExecutorAgent
from mediated_coevo.agents.mediator import MediatorAgent
from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.benchmarks import HarborRunner, SkillsBenchRepository
from mediated_coevo.conditions import ConditionName
from mediated_coevo.config import Config, load_config
from mediated_coevo.evolution.skill_advisor import SkillAdvisor
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.orchestrator import Orchestrator
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from mediated_coevo.stores.skill_store import SkillStore

app = typer.Typer(name="medcoevo", help="Mediated Co-Evolution Experiment Runner")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

VALID_CONDITION_NAMES = set(get_args(ConditionName))


@dataclass(frozen=True)
class ExperimentRuntime:
    """Objects needed to execute one configured experiment."""

    experiment_dir: Path
    orchestrator: Orchestrator


class ExperimentFactory:
    """Build the object graph for one mediated co-evolution run."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root

    def build(
        self,
        *,
        config: Config,
        seed: int,
        condition_name: ConditionName,
    ) -> ExperimentRuntime:
        experiment_dir = self._create_experiment_dir(seed, condition_name)
        self._save_config(config, experiment_dir)

        skill_store = SkillStore(self._project_root / config.paths.skills_dir)
        skill_store.validate()
        artifact_store = ArtifactStore(base_dir=experiment_dir / "artifacts")
        history_store = HistoryStore(history_dir=experiment_dir / "history")
        benchmark_repo = SkillsBenchRepository(
            root_dir=self._project_root / config.paths.benchmarks_dir,
            task_dirs=config.executor_runtime.task_dirs,
        )
        harbor_runner = HarborRunner(
            agent_name=config.executor_runtime.agent_name,
            jobs_dir=experiment_dir / config.executor_runtime.jobs_dir,
            timeout_sec=config.executor_runtime.harbor_timeout_sec,
        )

        planner = self._build_planner(config)
        executor = ExecutorAgent(
            model=config.models.executor,
            benchmark_repo=benchmark_repo,
            harbor_runner=harbor_runner,
            workspace_root=experiment_dir / "benchmarks",
            injected_skill_name=config.executor_runtime.injected_skill_name,
        )
        mediator = self._build_mediator(config, artifact_store, skill_store)
        skill_advisor = self._build_skill_advisor(config)

        return ExperimentRuntime(
            experiment_dir=experiment_dir,
            orchestrator=Orchestrator(
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
            ),
        )

    def _create_experiment_dir(
        self,
        seed: int,
        condition_name: ConditionName,
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = (
            self._project_root
            / "data"
            / "experiments"
            / f"{timestamp}-{seed}-{condition_name}"
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir

    @staticmethod
    def _save_config(config: Config, experiment_dir: Path) -> None:
        with open(experiment_dir / "config.toml", "wb") as f:
            tomli_w.dump(config.model_dump(), f)

    @staticmethod
    def _build_planner(config: Config) -> PlannerAgent:
        from mediated_coevo.llm.client import LLMClient

        planner = PlannerAgent(llm_client=LLMClient(model=config.models.planner))
        planner.configure_token_budget(
            config.budgets,
            condition_name=config.experiment.condition_name,
        )
        return planner

    @staticmethod
    def _build_mediator(
        config: Config,
        artifact_store: ArtifactStore,
        skill_store: SkillStore,
    ) -> MediatorAgent:
        from mediated_coevo.llm.client import LLMClient

        mediator = MediatorAgent(
            llm_client=LLMClient(model=config.models.mediator),
            artifact_store=artifact_store,
        )
        mediator.configure_token_budget(
            config.budgets,
            condition_name=config.experiment.condition_name,
        )
        protocol = skill_store.read_skill("mediator")
        if protocol:
            mediator.load_protocol(protocol)
        return mediator

    @staticmethod
    def _build_skill_advisor(config: Config) -> SkillAdvisor:
        from mediated_coevo.llm.client import LLMClient

        skill_advisor = SkillAdvisor(
            llm_client=LLMClient(model=config.models.planner)
        )
        skill_advisor.configure_token_budget(
            config.budgets,
            condition_name=config.experiment.condition_name,
        )
        return skill_advisor


def _validate_condition_name(condition: str) -> ConditionName:
    """Validate CLI condition names before mutating the config object."""
    if condition not in VALID_CONDITION_NAMES:
        allowed = ", ".join(sorted(VALID_CONDITION_NAMES))
        raise typer.BadParameter(
            f"invalid condition {condition!r}; expected one of: {allowed}"
        )
    return cast(ConditionName, condition)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=True, show_path=False)],
    )


def _reward_summary(records: list[IterationRecord]) -> tuple[int, int, float]:
    """Return (scored_count, failure_count, avg_reward) for CLI reporting."""
    rewards = [
        reward for record in records
        if (reward := record.reward) is not None
    ]
    scored_count = len(rewards)
    failure_count = len(records) - scored_count
    avg_reward = sum(rewards) / scored_count if rewards else 0.0
    return scored_count, failure_count, avg_reward


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

    condition_name = _validate_condition_name(condition)

    config = load_config(config_dir)
    config.experiment.seed = seed
    config.experiment.condition_name = condition_name

    if config.executor_runtime.harbor_required and shutil.which("harbor") is None:
        console.print(
            "[bold red]ERROR:[/] harbor CLI not found on PATH. Install harbor, "
            "or set executor_runtime.harbor_required = false in config."
        )
        raise typer.Exit(code=1)

    task_ids = [t.strip() for t in tasks.split(",")]

    console.print(f"[bold]Tasks:[/] {task_ids}")
    console.print(f"[bold]Iterations:[/] {iterations}")
    console.print(f"[bold]Condition:[/] {condition_name}")
    console.print(f"[bold]Models:[/] planner={config.models.planner} executor={config.models.executor} mediator={config.models.mediator}")

    runtime = ExperimentFactory(PROJECT_ROOT).build(
        config=config,
        seed=seed,
        condition_name=condition_name,
    )

    # Run
    console.print(f"\n[bold green]Starting experiment:[/] {runtime.experiment_dir}\n")
    records = asyncio.run(runtime.orchestrator.run_experiment(task_ids, iterations))

    # Summary
    scored_count, failure_count, avg_reward = _reward_summary(records)
    total_tokens = sum(r.total_tokens for r in records)
    console.print("\n[bold]Results:[/]")
    console.print(f"  Iterations: {len(records)}")
    console.print(f"  Scored: {scored_count}")
    console.print(f"  Env failures: {failure_count}")
    console.print(f"  Avg reward (scored only): {avg_reward:.3f}")
    console.print(f"  Total tokens: {total_tokens:,}")
    console.print(f"  Data: {runtime.experiment_dir}")


if __name__ == "__main__":
    app()
