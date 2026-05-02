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
from mediated_coevo.baselines import (
    BASELINE_PRESET_NAMES,
    BaselinePreset,
    get_baseline_preset,
    parse_skill_updates,
)
from mediated_coevo.benchmarks import HarborRunner, SkillsBenchRepository
from mediated_coevo.conditions import ConditionName
from mediated_coevo.config import Config, SkillUpdateConfig, load_config
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


@dataclass(frozen=True)
class MatrixRuntime:
    """Runtime plus preset metadata for one baseline-matrix row."""

    preset_name: str
    runtime: ExperimentRuntime


@dataclass(frozen=True)
class ExperimentStores:
    """Persistent stores for one experiment runtime."""

    skill_store: SkillStore
    artifact_store: ArtifactStore
    history_store: HistoryStore


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
        experiment_dir: Path | None = None,
        isolate_skills: bool = False,
    ) -> ExperimentRuntime:
        experiment_dir = self._resolve_experiment_dir(
            config=config,
            seed=seed,
            condition_name=condition_name,
            experiment_dir=experiment_dir,
        )
        skills_dir = self._resolve_skills_dir(
            config=config,
            experiment_dir=experiment_dir,
            isolate_skills=isolate_skills,
        )
        self._save_config(config, experiment_dir)

        stores = self._build_stores(experiment_dir, skills_dir)
        benchmark_repo = self._build_benchmark_repo(config)
        harbor_runner = self._build_harbor_runner(config, experiment_dir)
        planner = self._build_planner(config)
        executor = ExecutorAgent(
            model=config.models.executor,
            benchmark_repo=benchmark_repo,
            harbor_runner=harbor_runner,
            workspace_root=experiment_dir / "benchmarks",
            injected_skill_name=config.executor_runtime.injected_skill_name,
        )
        mediator = self._build_mediator(
            config,
            stores.artifact_store,
            stores.skill_store,
        )
        skill_advisor = self._build_skill_advisor(config)

        return ExperimentRuntime(
            experiment_dir=experiment_dir,
            orchestrator=Orchestrator(
                planner=planner,
                executor=executor,
                mediator=mediator,
                skill_store=stores.skill_store,
                artifact_store=stores.artifact_store,
                history_store=stores.history_store,
                benchmark_repo=benchmark_repo,
                config=config,
                experiment_dir=experiment_dir,
                skill_advisor=skill_advisor,
            ),
        )

    def _resolve_experiment_dir(
        self,
        *,
        config: Config,
        seed: int,
        condition_name: ConditionName,
        experiment_dir: Path | None,
    ) -> Path:
        if experiment_dir is None:
            return self._create_experiment_dir(
                seed,
                condition_name,
                data_dir=config.paths.data_dir,
                baseline_preset=config.experiment.baseline_preset,
            )
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir

    def _resolve_skills_dir(
        self,
        *,
        config: Config,
        experiment_dir: Path,
        isolate_skills: bool,
    ) -> Path:
        source = self._project_root / config.paths.skills_dir
        if not isolate_skills:
            return source
        return self._copy_initial_skills(
            source=source,
            destination=experiment_dir / "skills",
        )

    @staticmethod
    def _build_stores(experiment_dir: Path, skills_dir: Path) -> ExperimentStores:
        skill_store = SkillStore(skills_dir)
        skill_store.validate()
        return ExperimentStores(
            skill_store=skill_store,
            artifact_store=ArtifactStore(base_dir=experiment_dir / "artifacts"),
            history_store=HistoryStore(history_dir=experiment_dir / "history"),
        )

    def _build_benchmark_repo(self, config: Config) -> SkillsBenchRepository:
        return SkillsBenchRepository(
            root_dir=self._project_root / config.paths.benchmarks_dir,
            task_dirs=config.executor_runtime.task_dirs,
        )

    @staticmethod
    def _build_harbor_runner(
        config: Config,
        experiment_dir: Path,
    ) -> HarborRunner:
        return HarborRunner(
            agent_name=config.executor_runtime.agent_name,
            jobs_dir=experiment_dir / config.executor_runtime.jobs_dir,
            timeout_sec=config.executor_runtime.harbor_timeout_sec,
        )

    def _create_experiment_dir(
        self,
        seed: int,
        condition_name: ConditionName,
        data_dir: str,
        baseline_preset: str | None = None,
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = baseline_preset or condition_name
        experiment_dir = (
            self._project_root
            / data_dir
            / "experiments"
            / f"{timestamp}-{seed}-{suffix}"
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir

    def create_matrix_dir(self, seed: int, data_dir: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        matrix_dir = (
            self._project_root
            / data_dir
            / "experiments"
            / f"{timestamp}-{seed}-baseline-matrix"
        )
        matrix_dir.mkdir(parents=True, exist_ok=True)
        return matrix_dir

    @staticmethod
    def _copy_initial_skills(source: Path, destination: Path) -> Path:
        shutil.copytree(source, destination)
        return destination

    @staticmethod
    def _save_config(config: Config, experiment_dir: Path) -> None:
        with open(experiment_dir / "config.toml", "wb") as f:
            tomli_w.dump(config.model_dump(exclude_none=True), f)

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


def _parse_skill_updates(raw_value: str) -> SkillUpdateConfig:
    """Parse CLI skill-update input and adapt parser errors to Typer."""
    try:
        return parse_skill_updates(raw_value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _validate_baseline_preset(preset_name: str) -> BaselinePreset:
    try:
        return get_baseline_preset(preset_name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _task_ids_from_cli(tasks: str) -> list[str]:
    task_ids = [task.strip() for task in tasks.split(",") if task.strip()]
    if not task_ids:
        raise typer.BadParameter("at least one task ID is required")
    return task_ids


def _ensure_harbor_available(config: Config) -> None:
    if config.executor_runtime.harbor_required and shutil.which("harbor") is None:
        console.print(
            "[bold red]ERROR:[/] harbor CLI not found on PATH. Install harbor, "
            "or set executor_runtime.harbor_required = false in config."
        )
        raise typer.Exit(code=1)


def _apply_experiment_settings(
    config: Config,
    *,
    iterations: int,
    seed: int,
    condition_name: ConditionName | None = None,
    skill_updates: SkillUpdateConfig | None = None,
    baseline_preset: str | None = None,
) -> Config:
    """Apply CLI experiment settings to a loaded config object."""
    config.experiment.num_iterations = iterations
    config.experiment.seed = seed
    if condition_name is not None:
        config.experiment.condition_name = condition_name
    if skill_updates is not None:
        config.experiment.skill_updates = skill_updates
    config.experiment.baseline_preset = baseline_preset
    return config


def _build_matrix_runtimes(
    *,
    factory: ExperimentFactory,
    base_config: Config,
    seed: int,
    matrix_dir: Path,
) -> list[MatrixRuntime]:
    """Build all baseline-matrix rows with isolated skill stores."""
    rows: list[MatrixRuntime] = []
    for preset_name in BASELINE_PRESET_NAMES:
        preset = _validate_baseline_preset(preset_name)
        row_config = preset.build_config(base_config, seed=seed)
        runtime = factory.build(
            config=row_config,
            seed=seed,
            condition_name=preset.condition_name,
            experiment_dir=matrix_dir / preset_name,
            isolate_skills=True,
        )
        rows.append(MatrixRuntime(preset_name=preset_name, runtime=runtime))
    return rows


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


def _print_model_summary(config: Config) -> None:
    console.print(
        "[bold]Models:[/] "
        f"planner={config.models.planner} "
        f"executor={config.models.executor} "
        f"mediator={config.models.mediator}"
    )


def _print_result_summary(
    *,
    records: list[IterationRecord],
    data_dir: Path,
    header: str,
) -> None:
    scored_count, failure_count, avg_reward = _reward_summary(records)
    total_tokens = sum(record.total_tokens for record in records)
    console.print(f"\n[bold]{header}:[/]")
    console.print(f"  Iterations: {len(records)}")
    console.print(f"  Scored: {scored_count}")
    console.print(f"  Env failures: {failure_count}")
    console.print(f"  Avg reward (scored only): {avg_reward:.3f}")
    console.print(f"  Total tokens: {total_tokens:,}")
    console.print(f"  Data: {data_dir}")


@app.command()
def run(
    tasks: str = typer.Option("fix-build-google-auto", help="Comma-separated task IDs"),
    iterations: int = typer.Option(30, help="Number of iterations"),
    seed: int = typer.Option(42, help="Random seed"),
    condition: str = typer.Option(
        "learned_mediator",
        help="Experiment condition: no_feedback | full_traces | shared_notes | static_mediator | learned_mediator",
    ),
    skill_updates: str = typer.Option(
        "all",
        "--skill-updates",
        help="Comma-separated skill updates allowed: none | executor | planner | mediator | all",
    ),
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a mediated co-evolution experiment."""
    _setup_logging(verbose)
    random.seed(seed)

    condition_name = _validate_condition_name(condition)
    skill_update_config = _parse_skill_updates(skill_updates)

    config = _apply_experiment_settings(
        load_config(config_dir),
        iterations=iterations,
        seed=seed,
        condition_name=condition_name,
        skill_updates=skill_update_config,
    )

    _ensure_harbor_available(config)

    task_ids = _task_ids_from_cli(tasks)

    console.print(f"[bold]Tasks:[/] {task_ids}")
    console.print(f"[bold]Iterations:[/] {iterations}")
    console.print(f"[bold]Condition:[/] {condition_name}")
    console.print(f"[bold]Skill updates:[/] {skill_update_config.model_dump()}")
    _print_model_summary(config)

    runtime = ExperimentFactory(PROJECT_ROOT).build(
        config=config,
        seed=seed,
        condition_name=condition_name,
    )

    # Run
    console.print(f"\n[bold green]Starting experiment:[/] {runtime.experiment_dir}\n")
    records = asyncio.run(runtime.orchestrator.run_experiment(task_ids, iterations))

    _print_result_summary(
        records=records,
        data_dir=runtime.experiment_dir,
        header="Results",
    )


@app.command()
def matrix(
    tasks: str = typer.Option("fix-build-google-auto", help="Comma-separated task IDs"),
    iterations: int = typer.Option(30, help="Number of iterations per row"),
    seed: int = typer.Option(42, help="Random seed reused for every row"),
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the seven-row baseline matrix with isolated per-row skills."""
    _setup_logging(verbose)

    task_ids = _task_ids_from_cli(tasks)
    config = _apply_experiment_settings(
        load_config(config_dir),
        iterations=iterations,
        seed=seed,
    )
    _ensure_harbor_available(config)

    factory = ExperimentFactory(PROJECT_ROOT)
    matrix_dir = factory.create_matrix_dir(seed=seed, data_dir=config.paths.data_dir)
    rows = _build_matrix_runtimes(
        factory=factory,
        base_config=config,
        seed=seed,
        matrix_dir=matrix_dir,
    )

    console.print(f"[bold]Tasks:[/] {task_ids}")
    console.print(f"[bold]Iterations per row:[/] {iterations}")
    console.print(f"[bold]Seed per row:[/] {seed}")
    console.print(f"[bold]Matrix:[/] {matrix_dir}")
    console.print(f"[bold]Rows:[/] {', '.join(BASELINE_PRESET_NAMES)}")

    for row in rows:
        row_config = row.runtime.orchestrator.config
        random.seed(seed)
        console.print(
            "\n[bold green]Starting matrix row:[/] "
            f"{row.preset_name} "
            f"(condition={row_config.experiment.condition_name}, "
            f"skill_updates={row_config.experiment.skill_updates.model_dump()})"
        )
        records = asyncio.run(
            row.runtime.orchestrator.run_experiment(task_ids, iterations)
        )
        _print_result_summary(
            records=records,
            data_dir=row.runtime.experiment_dir,
            header=f"Row results: {row.preset_name}",
        )

    console.print(f"\n[bold]Matrix data:[/] {matrix_dir}")


if __name__ == "__main__":
    app()
