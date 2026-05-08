"""CLI entry point for the mediated co-evolution system."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast, get_args

import tomli_w
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from mediated_coevo.agents.executor import ExecutorAgent
from mediated_coevo.agents.mediator import MediatorAgent
from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.experiment.baselines import (
    BASELINE_PRESET_NAMES,
    get_baseline_preset,
    parse_skill_updates,
)
from mediated_coevo.benchmarks import (
    HarborRunner,
    SkillsBenchFetchError,
    SkillsBenchRemoteConfig,
    SkillsBenchRepository,
)
from mediated_coevo.benchmarks.task_sets import (
    TASK_SETS,
    TaskSetError,
    resolve_task_selection,
)
from mediated_coevo.experiment.conditions import (
    ConditionName,
    ExperimentDesignError,
    validate_experiment_design,
)
from mediated_coevo.core.config import Config, SkillUpdateConfig, load_config
from mediated_coevo.evolution.skill_advisor import SkillAdvisor
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.experiment.orchestrator import Orchestrator
from mediated_coevo.analysis.reporting import (
    BootstrapConfidenceInterval,
    ExperimentScoreSummary,
    TaskScoreSummary,
    build_score_summary,
    write_score_summary,
)
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from mediated_coevo.stores.skill_store import SkillStore

app = typer.Typer(name="medcoevo", help="Mediated Co-Evolution Experiment Runner")
skillsbench_app = typer.Typer(help="Manage the local SkillsBench task cache")
app.add_typer(skillsbench_app, name="skillsbench")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALID_CONDITION_NAMES = set(get_args(ConditionName))
SKILLSBENCH_ALL_TASK_SET = "skillsbench-all"


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
        benchmark_repo: SkillsBenchRepository | None = None,
    ) -> ExperimentRuntime:
        from mediated_coevo.llm.client import LLMClient

        validate_experiment_design(
            condition=condition_name,
            skill_updates=config.experiment.skill_updates,
            baseline_preset=config.experiment.baseline_preset,
        )
        if experiment_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            suffix = config.experiment.baseline_preset or condition_name
            experiment_dir = (
                self._project_root
                / config.paths.data_dir
                / "experiments"
                / f"{timestamp}-{seed}-{suffix}"
            )
        experiment_dir.mkdir(parents=True, exist_ok=True)

        source_skills_dir = self._project_root / config.paths.skills_dir
        runtime_skills_dir = shutil.copytree(
            source_skills_dir,
            experiment_dir / "skills",
        )

        with open(experiment_dir / "config.toml", "wb") as f:
            tomli_w.dump(config.model_dump(exclude_none=True), f)

        skill_store = SkillStore(runtime_skills_dir)
        skill_store.validate()
        artifact_store = ArtifactStore(base_dir=experiment_dir / "artifacts")
        history_store = HistoryStore(history_dir=experiment_dir / "history")
        if benchmark_repo is None:
            benchmark_repo = _build_benchmark_repo(self._project_root, config)
        harbor_runner = HarborRunner(
            agent_name=config.executor_runtime.agent_name,
            jobs_dir=experiment_dir / config.executor_runtime.jobs_dir,
            timeout_sec=config.executor_runtime.harbor_timeout_sec,
        )
        planner = PlannerAgent(llm_client=LLMClient(model=config.models.planner))
        planner.configure_token_budget(
            config.budgets,
            condition_name=config.experiment.condition_name,
        )
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
        mediator.configure_token_budget(
            config.budgets,
            condition_name=config.experiment.condition_name,
        )
        protocol = skill_store.read_skill("mediator")
        if protocol:
            mediator.load_protocol(protocol)
        skill_advisor = SkillAdvisor(
            llm_client=LLMClient(model=config.models.planner)
        )
        skill_advisor.configure_token_budget(
            config.budgets,
            condition_name=config.experiment.condition_name,
        )

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


def _validate_condition_name(condition: str) -> ConditionName:
    """Validate CLI condition names before mutating the config object."""
    if condition not in VALID_CONDITION_NAMES:
        allowed = ", ".join(sorted(VALID_CONDITION_NAMES))
        raise typer.BadParameter(
            f"invalid condition {condition!r}; expected one of: {allowed}"
        )
    return cast(ConditionName, condition)


def _task_ids_from_cli(tasks: str | None, task_set: str | None) -> list[str]:
    try:
        return resolve_task_selection(
            tasks=tasks,
            task_set=task_set,
        )
    except TaskSetError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _build_benchmark_repo(project_root: Path, config: Config) -> SkillsBenchRepository:
    return SkillsBenchRepository(
        root_dir=project_root / config.paths.benchmarks_dir,
        task_dirs=config.executor_runtime.task_dirs,
        remote=SkillsBenchRemoteConfig(
            enabled=config.executor_runtime.remote_fetch,
            archive_url=config.executor_runtime.archive_url,
            archive_sha256=config.executor_runtime.archive_sha256,
            local_archive_base_dir=project_root,
        ),
    )


def _available_task_set_help() -> str:
    return ", ".join(sorted([*TASK_SETS, SKILLSBENCH_ALL_TASK_SET]))


def _skillsbench_all_task_ids(benchmark_repo: SkillsBenchRepository) -> list[str]:
    task_ids = benchmark_repo.list_local_task_ids()
    seen = set(task_ids)

    if benchmark_repo.remote.enabled:
        try:
            remote_task_ids = benchmark_repo.list_remote_task_ids()
        except SkillsBenchFetchError as exc:
            if task_ids:
                return task_ids
            raise typer.BadParameter(
                f"failed to resolve task set {SKILLSBENCH_ALL_TASK_SET!r}: {exc}"
            ) from exc
        for task_id in remote_task_ids:
            if task_id not in seen:
                task_ids.append(task_id)
                seen.add(task_id)

    if not task_ids:
        raise typer.BadParameter(
            f"task set {SKILLSBENCH_ALL_TASK_SET!r} resolved to no local tasks; "
            "enable executor_runtime.remote_fetch or sync selected tasks first"
        )
    return task_ids


def _task_ids_from_cli_with_repo(
    tasks: str | None,
    task_set: str | None,
    benchmark_repo: SkillsBenchRepository,
) -> list[str]:
    task_set_name = task_set.strip() if task_set is not None else None
    if tasks is not None:
        return _task_ids_from_cli(tasks, task_set)
    if task_set_name == SKILLSBENCH_ALL_TASK_SET:
        return _skillsbench_all_task_ids(benchmark_repo)
    return _task_ids_from_cli(tasks, task_set)


def _preflight_task_ids_from_cli(
    tasks: str | None,
    task_set: str | None,
) -> list[str] | None:
    """Validate task selection before Harbor checks when no repo is needed."""
    task_set_name = task_set.strip() if task_set is not None else None
    if tasks is None and task_set_name == SKILLSBENCH_ALL_TASK_SET:
        return None
    return _task_ids_from_cli(tasks, task_set)


def _sync_task_ids_from_cli(tasks: str | None, task_set: str | None) -> list[str]:
    task_set_name = task_set.strip() if task_set is not None else None
    if tasks is not None:
        return _task_ids_from_cli(tasks, task_set)
    if task_set_name is None:
        raise typer.BadParameter("provide --tasks or --task-set to sync selected tasks")
    if task_set_name == SKILLSBENCH_ALL_TASK_SET:
        raise typer.BadParameter(
            f"syncing {SKILLSBENCH_ALL_TASK_SET!r} is intentionally unsupported; "
            "use --tasks or --task-set skillsbench-10"
        )
    return _task_ids_from_cli(tasks, task_set)


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


def _validate_or_raise_bad_parameter(config: Config) -> None:
    try:
        validate_experiment_design(
            condition=config.experiment.condition_name,
            skill_updates=config.experiment.skill_updates,
            baseline_preset=config.experiment.baseline_preset,
        )
    except ExperimentDesignError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _build_matrix_runtimes(
    *,
    factory: ExperimentFactory,
    base_config: Config,
    seed: int,
    matrix_dir: Path,
    benchmark_repo: SkillsBenchRepository | None = None,
) -> list[MatrixRuntime]:
    """Build all baseline-matrix rows with isolated skill stores."""
    rows: list[MatrixRuntime] = []
    for preset_name in BASELINE_PRESET_NAMES:
        preset = get_baseline_preset(preset_name)
        row_config = preset.build_config(base_config, seed=seed)
        runtime = factory.build(
            config=row_config,
            seed=seed,
            condition_name=preset.condition_name,
            experiment_dir=matrix_dir / preset_name,
            benchmark_repo=benchmark_repo,
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


def _print_result_summary(
    *,
    summary: ExperimentScoreSummary,
    data_dir: Path,
    summary_path: Path,
    header: str,
) -> None:
    console.print(f"\n[bold]{header}:[/]")
    console.print(f"  Runs: {summary.total_runs}")
    console.print(f"  Scored: {summary.scored_count}")
    console.print(f"  Env failures: {summary.env_failure_count}")
    console.print(f"  Mean reward: {_format_score(summary.mean_reward)}")
    console.print(f"  Median reward: {_format_score(summary.median_reward)}")
    console.print(f"  Macro mean reward: {_format_score(summary.macro_mean_reward)}")
    console.print(f"  Bootstrap CI: {_format_ci(summary.bootstrap_ci)}")
    console.print(f"  Total tokens: {summary.total_tokens:,}")
    if summary.per_task:
        console.print("  Per-task:")
        for task_summary in summary.per_task:
            console.print(
                "    "
                f"{task_summary.task_id}{_format_task_metadata(task_summary)}: "
                f"mean={_format_score(task_summary.mean_reward)} "
                f"median={_format_score(task_summary.median_reward)} "
                f"scored={task_summary.scored_count}/{task_summary.total_runs} "
                f"env_failures={task_summary.env_failure_count} "
                f"ci={_format_ci(task_summary.bootstrap_ci)}"
            )
    if summary.dominance_warning and summary.dominant_task_id:
        console.print(
            "  [yellow]Dominance warning:[/] "
            f"{summary.dominant_task_id} contributed "
            f"{summary.max_task_scored_share:.1%} of scored runs "
            f"(threshold {summary.dominance_threshold:.1%})."
        )
    console.print(f"  Summary: {summary_path}")
    console.print(f"  Data: {data_dir}")


def _write_and_print_result_summary(
    *,
    records: list[IterationRecord],
    data_dir: Path,
    header: str,
) -> None:
    summary = build_score_summary(records)
    summary_path = data_dir / "summary.json"
    write_score_summary(summary, summary_path)
    _print_result_summary(
        summary=summary,
        data_dir=data_dir,
        summary_path=summary_path,
        header=header,
    )


def _experiments_root(config: Config) -> Path:
    return PROJECT_ROOT / config.paths.data_dir / "experiments"


def _latest_experiment_dir(experiments_root: Path) -> Path:
    if not experiments_root.exists():
        raise typer.BadParameter(f"experiment directory not found: {experiments_root}")
    candidates = [
        path
        for path in experiments_root.iterdir()
        if path.is_dir()
    ]
    if not candidates:
        raise typer.BadParameter(f"no experiment outputs found under {experiments_root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def _load_score_summary(summary_path: Path) -> ExperimentScoreSummary:
    return ExperimentScoreSummary.model_validate_json(summary_path.read_text())


def _artifact_dirs(experiment_dir: Path) -> list[str]:
    artifacts_dir = experiment_dir / "artifacts"
    if not artifacts_dir.exists():
        return []
    return [
        str(path)
        for path in sorted(artifacts_dir.iterdir(), key=lambda item: item.name)
        if path.is_dir()
    ]


def _single_inspection_payload(experiment_dir: Path) -> dict[str, Any]:
    summary_path = experiment_dir / "summary.json"
    metrics_path = experiment_dir / "metrics.jsonl"
    if summary_path.exists():
        return {
            "kind": "single",
            "experiment_dir": str(experiment_dir),
            "summary_path": str(summary_path),
            "metrics_path": str(metrics_path) if metrics_path.exists() else None,
            "artifact_dirs": _artifact_dirs(experiment_dir),
            "summary": _load_score_summary(summary_path).model_dump(mode="json"),
        }
    if metrics_path.exists():
        return {
            "kind": "single",
            "experiment_dir": str(experiment_dir),
            "summary_path": None,
            "metrics_path": str(metrics_path),
            "artifact_dirs": _artifact_dirs(experiment_dir),
            "warning": "summary.json is missing; inspect metrics.jsonl directly.",
        }
    raise typer.BadParameter(
        f"no summary.json or metrics.jsonl found under {experiment_dir}"
    )


def _matrix_inspection_payload(experiment_dir: Path) -> dict[str, Any] | None:
    rows = []
    for row_dir in sorted(experiment_dir.iterdir(), key=lambda item: item.name):
        if not row_dir.is_dir():
            continue
        summary_path = row_dir / "summary.json"
        metrics_path = row_dir / "metrics.jsonl"
        if not summary_path.exists() and not metrics_path.exists():
            continue
        row: dict[str, Any] = {
            "row": row_dir.name,
            "experiment_dir": str(row_dir),
            "summary_path": str(summary_path) if summary_path.exists() else None,
            "metrics_path": str(metrics_path) if metrics_path.exists() else None,
        }
        if summary_path.exists():
            row["summary"] = _load_score_summary(summary_path).model_dump(mode="json")
        else:
            row["warning"] = "summary.json is missing; inspect metrics.jsonl directly."
        rows.append(row)

    if not rows:
        return None
    return {
        "kind": "matrix",
        "experiment_dir": str(experiment_dir),
        "rows": rows,
    }


def _inspection_payload(experiment_dir: Path) -> dict[str, Any]:
    if not experiment_dir.exists() or not experiment_dir.is_dir():
        raise typer.BadParameter(f"experiment directory not found: {experiment_dir}")
    if matrix_payload := _matrix_inspection_payload(experiment_dir):
        return matrix_payload
    return _single_inspection_payload(experiment_dir)


def _print_single_inspection(payload: dict[str, Any]) -> None:
    summary_data = payload.get("summary")
    if summary_data is not None:
        summary = ExperimentScoreSummary.model_validate(summary_data)
        _print_result_summary(
            summary=summary,
            data_dir=Path(payload["experiment_dir"]),
            summary_path=Path(payload["summary_path"]),
            header="Inspection",
        )
    else:
        console.print("\n[bold]Inspection:[/]")
        console.print(f"  Data: {payload['experiment_dir']}")
        console.print(f"  [yellow]Warning:[/] {payload['warning']}")
    console.print(f"  Metrics: {payload.get('metrics_path') or 'n/a'}")
    artifact_dirs = payload.get("artifact_dirs") or []
    if artifact_dirs:
        console.print("  Artifact dirs:")
        for artifact_dir in artifact_dirs:
            console.print(f"    {artifact_dir}")


def _print_matrix_inspection(payload: dict[str, Any]) -> None:
    console.print("\n[bold]Matrix inspection:[/]")
    console.print(f"  Data: {payload['experiment_dir']}")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Row")
    table.add_column("Runs", justify="right")
    table.add_column("Scored", justify="right")
    table.add_column("Env failures", justify="right")
    table.add_column("Mean")
    table.add_column("Macro mean")
    table.add_column("Metrics")
    for row in payload["rows"]:
        summary_data = row.get("summary")
        if summary_data is None:
            table.add_row(
                row["row"],
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                row.get("metrics_path") or "n/a",
            )
            continue
        summary = ExperimentScoreSummary.model_validate(summary_data)
        table.add_row(
            row["row"],
            str(summary.total_runs),
            str(summary.scored_count),
            str(summary.env_failure_count),
            _format_score(summary.mean_reward),
            _format_score(summary.macro_mean_reward),
            row.get("metrics_path") or "n/a",
        )
    console.print(table)


def _print_inspection_payload(payload: dict[str, Any]) -> None:
    if payload["kind"] == "matrix":
        _print_matrix_inspection(payload)
    else:
        _print_single_inspection(payload)


def _format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_ci(interval: BootstrapConfidenceInterval) -> str:
    if interval.lower is None or interval.upper is None:
        return "n/a"
    confidence = round(interval.confidence_level * 100)
    return f"{confidence}% [{interval.lower:.3f}, {interval.upper:.3f}]"


def _format_task_metadata(task_summary: TaskScoreSummary) -> str:
    metadata = []
    if task_summary.task_category:
        metadata.append(task_summary.task_category)
    if task_summary.task_difficulty:
        metadata.append(task_summary.task_difficulty)
    if not metadata:
        return ""
    return f" ({', '.join(metadata)})"


def _print_task_selection(
    *,
    task_ids: list[str],
    tasks: str | None,
    task_set: str | None,
) -> None:
    console.print(f"[bold]Tasks:[/] {task_ids}")
    if tasks is None and task_set is not None:
        console.print(f"[bold]Task set:[/] {task_set}")


@app.command()
def run(
    tasks: str | None = typer.Option(
        None,
        "--tasks",
        help="Comma-separated task IDs. Overrides --task-set when provided.",
    ),
    task_set: str | None = typer.Option(
        None,
        "--task-set",
        help=(
            "Named task set to run when --tasks is omitted. "
            f"Available: {_available_task_set_help()}"
        ),
    ),
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
    try:
        skill_update_config = parse_skill_updates(skill_updates)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    config = _apply_experiment_settings(
        load_config(config_dir),
        iterations=iterations,
        seed=seed,
        condition_name=condition_name,
        skill_updates=skill_update_config,
    )

    _validate_or_raise_bad_parameter(config)
    preflight_task_ids = _preflight_task_ids_from_cli(tasks, task_set)
    _ensure_harbor_available(config)

    benchmark_repo = _build_benchmark_repo(PROJECT_ROOT, config)
    task_ids = preflight_task_ids or _task_ids_from_cli_with_repo(
        tasks,
        task_set,
        benchmark_repo,
    )

    _print_task_selection(task_ids=task_ids, tasks=tasks, task_set=task_set)
    console.print(f"[bold]Iterations:[/] {iterations}")
    console.print(f"[bold]Condition:[/] {condition_name}")
    console.print(f"[bold]Skill updates:[/] {skill_update_config.model_dump()}")
    console.print(
        "[bold]Models:[/] "
        f"planner={config.models.planner} "
        f"executor={config.models.executor} "
        f"mediator={config.models.mediator}"
    )

    runtime = ExperimentFactory(PROJECT_ROOT).build(
        config=config,
        seed=seed,
        condition_name=condition_name,
        benchmark_repo=benchmark_repo,
    )

    console.print(f"\n[bold green]Starting experiment:[/] {runtime.experiment_dir}\n")
    records = asyncio.run(runtime.orchestrator.run_experiment(task_ids, iterations))

    _write_and_print_result_summary(
        records=records,
        data_dir=runtime.experiment_dir,
        header="Results",
    )


@app.command()
def matrix(
    tasks: str | None = typer.Option(
        None,
        "--tasks",
        help="Comma-separated task IDs. Overrides --task-set when provided.",
    ),
    task_set: str | None = typer.Option(
        None,
        "--task-set",
        help=(
            "Named task set to run when --tasks is omitted. "
            f"Available: {_available_task_set_help()}"
        ),
    ),
    iterations: int = typer.Option(30, help="Number of iterations per row"),
    seed: int = typer.Option(42, help="Random seed reused for every row"),
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the six-row baseline matrix with isolated per-row skills."""
    _setup_logging(verbose)

    config = _apply_experiment_settings(
        load_config(config_dir),
        iterations=iterations,
        seed=seed,
    )
    for preset_name in BASELINE_PRESET_NAMES:
        preset = get_baseline_preset(preset_name)
        validate_experiment_design(
            condition=preset.condition_name,
            skill_updates=preset.skill_updates,
            baseline_preset=preset.name,
        )
    preflight_task_ids = _preflight_task_ids_from_cli(tasks, task_set)
    _ensure_harbor_available(config)
    benchmark_repo = _build_benchmark_repo(PROJECT_ROOT, config)
    task_ids = preflight_task_ids or _task_ids_from_cli_with_repo(
        tasks,
        task_set,
        benchmark_repo,
    )

    factory = ExperimentFactory(PROJECT_ROOT)
    matrix_dir = factory.create_matrix_dir(seed=seed, data_dir=config.paths.data_dir)
    rows = _build_matrix_runtimes(
        factory=factory,
        base_config=config,
        seed=seed,
        matrix_dir=matrix_dir,
        benchmark_repo=benchmark_repo,
    )

    _print_task_selection(task_ids=task_ids, tasks=tasks, task_set=task_set)
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
        _write_and_print_result_summary(
            records=records,
            data_dir=row.runtime.experiment_dir,
            header=f"Row results: {row.preset_name}",
        )

    console.print(f"\n[bold]Matrix data:[/] {matrix_dir}")


@app.command("inspect")
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
        target_dir = _latest_experiment_dir(_experiments_root(load_config(config_dir)))
    payload = _inspection_payload(target_dir)
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    _print_inspection_payload(payload)


@skillsbench_app.command("sync")
def sync_skillsbench(
    tasks: str | None = typer.Option(
        None,
        "--tasks",
        help="Comma-separated task IDs to sync into the local SkillsBench cache.",
    ),
    task_set: str | None = typer.Option(
        None,
        "--task-set",
        help=(
            "Named task set to sync when --tasks is omitted. "
            "Use skillsbench-10; skillsbench-all is intentionally unsupported."
        ),
    ),
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fetch selected SkillsBench tasks into the configured local cache."""
    _setup_logging(verbose)
    config = load_config(config_dir)
    benchmark_repo = _build_benchmark_repo(PROJECT_ROOT, config)
    task_ids = _sync_task_ids_from_cli(tasks, task_set)

    try:
        synced_tasks = benchmark_repo.sync_tasks(task_ids)
    except (FileNotFoundError, SkillsBenchFetchError) as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]Synced tasks:[/] {[task.task_id for task in synced_tasks]}")
    console.print(f"[bold]Cache:[/] {benchmark_repo.default_local_cache_dir()}")


if __name__ == "__main__":
    app()
