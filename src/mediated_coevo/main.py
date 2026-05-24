"""CLI entry point for the mediated co-evolution system."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import shlex
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, cast, get_args

import tomli_w
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from mediated_coevo.agents.executor import (
    ExecutorAgent,
    RoutedExecutorAgent,
    SWEbenchExecutorAgent,
)
from mediated_coevo.agents.mediator import MediatorAgent
from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.agents.swebench_patch_generator import LLMSWEbenchPatchGenerator
from mediated_coevo.analysis.judge_rewards import (
    JudgeRewardAnnotationError,
    annotate_judge_rewards,
)
from mediated_coevo.analysis.reporting import (
    BootstrapConfidenceInterval,
    ExperimentScoreSummary,
    TaskScoreSummary,
    build_score_summary,
    write_score_summary,
)
from mediated_coevo.benchmarks import (
    HarborRunner,
    SkillsBenchFetchError,
    SkillsBenchRemoteConfig,
    SkillsBenchRepository,
)
from mediated_coevo.benchmarks.mixed import (
    BenchmarkKind,
    BenchmarkTaskSelection,
    MixedBenchmarkRepository,
    build_benchmark_task_selection,
)
from mediated_coevo.benchmarks import swebench as swebench_helpers
from mediated_coevo.benchmarks.task_sets import (
    TASK_SETS,
    TaskSetError,
    resolve_task_selection,
)
from mediated_coevo.core.config import (
    Config,
    SkillUpdateConfig,
    load_config,
    normalize_openrouter_model_name,
)
from mediated_coevo.evolution.skill_advisor import SkillAdvisor
from mediated_coevo.experiment.baselines import (
    BASELINE_PRESET_NAMES,
    get_baseline_preset,
    parse_skill_updates,
)
from mediated_coevo.experiment.conditions import (
    ConditionName,
    ExperimentDesignError,
    validate_experiment_design,
)
from mediated_coevo.experiment.orchestrator import Orchestrator
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from mediated_coevo.stores.skill_store import SkillStore

app = typer.Typer(name="medcoevo", help="Mediated Co-Evolution Experiment Runner")
skillsbench_app = typer.Typer(help="Manage the local SkillsBench task cache")
swebench_app = typer.Typer(help="Explore and smoke-test SWE-bench tasks")
app.add_typer(skillsbench_app, name="skillsbench")
app.add_typer(swebench_app, name="swebench")
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
            agent_setup_timeout_multiplier=(
                config.executor_runtime.harbor_agent_setup_timeout_multiplier
            ),
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
        skill_advisor = SkillAdvisor(llm_client=LLMClient(model=config.models.planner))
        skill_advisor.configure_token_budget(
            config.budgets,
            condition_name=config.experiment.condition_name,
        )
        judge_client = LLMClient(model=config.models.judge)

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
                judge_llm_client=judge_client,
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


def _task_ids_from_repeatable_cli(
    raw_values: list[str] | None,
    *,
    label: str,
) -> list[str]:
    """Parse repeatable comma-separated task options."""
    if not raw_values:
        return []
    task_ids: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for candidate in raw_value.split(","):
            task_id = candidate.strip()
            if not task_id:
                continue
            if task_id not in seen:
                task_ids.append(task_id)
                seen.add(task_id)
    if not task_ids:
        raise typer.BadParameter(f"at least one {label} is required")
    return task_ids


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


def _skillsbench_task_ids_from_unified_cli(
    *,
    skillsbench_tasks: list[str] | None,
    skillsbench_task_set: str | None,
    legacy_tasks: str | None,
    legacy_task_set: str | None,
    benchmark_repo: SkillsBenchRepository,
) -> list[str]:
    """Resolve SkillsBench selections from new options and legacy aliases."""
    if skillsbench_tasks and legacy_tasks is not None:
        raise typer.BadParameter(
            "cannot combine --skillsbench-task with legacy --tasks"
        )
    if skillsbench_task_set is not None and legacy_task_set is not None:
        raise typer.BadParameter(
            "cannot combine --skillsbench-task-set with legacy --task-set"
        )

    explicit_tasks = _task_ids_from_repeatable_cli(
        skillsbench_tasks,
        label="SkillsBench task",
    )
    if explicit_tasks:
        return explicit_tasks
    if legacy_tasks is not None:
        return _task_ids_from_cli_with_repo(legacy_tasks, None, benchmark_repo)

    task_set = (
        skillsbench_task_set
        if skillsbench_task_set is not None
        else legacy_task_set
    )
    if task_set is not None:
        return _task_ids_from_cli_with_repo(None, task_set, benchmark_repo)
    return []


def _swebench_instance_ids_from_unified_cli(
    *,
    swebench_instances: list[str] | None,
    swebench_limit: int | None,
    dataset_name: str,
    split: str,
) -> list[str]:
    """Resolve optional SWE-bench selections without defaulting to smoke IDs."""
    if not swebench_instances and swebench_limit is None:
        return []
    try:
        return swebench_helpers.resolve_swebench_instance_ids(
            dataset_name=dataset_name,
            split=split,
            raw_instance_ids=swebench_instances,
            limit=swebench_limit,
        )
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _resolve_unified_task_selection(
    *,
    skillsbench_tasks: list[str] | None,
    skillsbench_task_set: str | None,
    legacy_tasks: str | None,
    legacy_task_set: str | None,
    swebench_instances: list[str] | None,
    swebench_limit: int | None,
    dataset_name: str,
    split: str,
    benchmark_repo: SkillsBenchRepository,
) -> BenchmarkTaskSelection:
    """Resolve all benchmark selections for a unified evolution run."""
    skillsbench_task_ids = _skillsbench_task_ids_from_unified_cli(
        skillsbench_tasks=skillsbench_tasks,
        skillsbench_task_set=skillsbench_task_set,
        legacy_tasks=legacy_tasks,
        legacy_task_set=legacy_task_set,
        benchmark_repo=benchmark_repo,
    )
    swebench_instance_ids = _swebench_instance_ids_from_unified_cli(
        swebench_instances=swebench_instances,
        swebench_limit=swebench_limit,
        dataset_name=dataset_name,
        split=split,
    )
    try:
        selection = build_benchmark_task_selection(
            skillsbench_task_ids=skillsbench_task_ids,
            swebench_instance_ids=swebench_instance_ids,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not selection.task_ids:
        raise typer.BadParameter(
            "provide at least one SkillsBench task or SWE-bench instance "
            "(--skillsbench-task/--skillsbench-task-set/--tasks/--task-set or "
            "--swebench-instance/--swebench-limit)"
        )
    return selection


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


def _prepare_llm_credentials_or_exit(config: Config) -> Config:
    from mediated_coevo.core.config import ModelConfigError
    from mediated_coevo.llm.client import (
        LLMCredentialError,
        validate_openrouter_credentials,
    )

    try:
        config.normalize_models()
        validate_openrouter_credentials()
    except (ModelConfigError, LLMCredentialError) as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
        raise typer.Exit(code=1) from exc
    return config


def _apply_experiment_settings(
    config: Config,
    *,
    iterations: int,
    seed: int,
    condition_name: ConditionName | None = None,
    skill_updates: SkillUpdateConfig | None = None,
    baseline_preset: str | None = None,
    coevo_interval: int | None = None,
    advisor_buffer_max: int | None = None,
    harbor_agent_setup_timeout_multiplier: float | None = None,
) -> Config:
    """Apply CLI experiment settings to a loaded config object."""
    config.experiment.num_iterations = iterations
    config.experiment.seed = seed
    if condition_name is not None:
        config.experiment.condition_name = condition_name
    if skill_updates is not None:
        config.experiment.skill_updates = skill_updates
    config.experiment.baseline_preset = baseline_preset
    if coevo_interval is not None:
        config.experiment.coevo_interval = coevo_interval
    if advisor_buffer_max is not None:
        config.experiment.advisor_buffer_max = advisor_buffer_max
    if harbor_agent_setup_timeout_multiplier is not None:
        config.executor_runtime.harbor_agent_setup_timeout_multiplier = (
            harbor_agent_setup_timeout_multiplier
        )
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
    if summary.judge_reward_summary is not None:
        judge_summary = summary.judge_reward_summary
        console.print(
            "  Judge mean reward: "
            f"{_format_score(judge_summary.mean_reward)} "
            f"(macro={_format_score(judge_summary.macro_mean_reward)})"
        )
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


def _annotate_judge_rewards_or_exit(
    *,
    data_dir: Path,
    config: Config,
    history_store: HistoryStore | None = None,
) -> None:
    try:
        asyncio.run(
            annotate_judge_rewards(
                data_dir=data_dir,
                config=config,
                history_store=history_store,
            )
        )
    except JudgeRewardAnnotationError as exc:
        console.print(f"[bold red]ERROR:[/] Judge reward annotation failed: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[bold red]ERROR:[/] Judge reward annotation failed: {exc}")
        raise typer.Exit(code=1) from exc


def _experiments_root(config: Config) -> Path:
    return PROJECT_ROOT / config.paths.data_dir / "experiments"


def _latest_experiment_dir(experiments_root: Path) -> Path:
    if not experiments_root.exists():
        raise typer.BadParameter(f"experiment directory not found: {experiments_root}")
    candidates = [path for path in experiments_root.iterdir() if path.is_dir()]
    if not candidates:
        raise typer.BadParameter(
            f"no experiment outputs found under {experiments_root}"
        )
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


def _print_experiment_controls(config: Config) -> None:
    """Print shared co-evolution controls after config and CLI overrides apply."""
    console.print(f"[bold]Coevo interval:[/] {config.experiment.coevo_interval}")
    console.print(
        f"[bold]Advisor buffer max:[/] {config.experiment.advisor_buffer_max}"
    )
    console.print("[bold]Skill validation:[/] required")


def _resolve_output_dir(output_dir: Path) -> Path:
    if output_dir.is_absolute():
        return output_dir
    return PROJECT_ROOT / output_dir


def _timestamped_run_id(suffix: str) -> str:
    """Return a run ID with a timestamp prefix and caller-provided suffix."""
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{suffix}"


def _print_swebench_instances(
    instances: list[swebench_helpers.SWEbenchInstanceInfo],
) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Instance ID")
    table.add_column("Repo")
    table.add_column("Version")
    table.add_column("Created")
    for instance in instances:
        table.add_row(
            instance.instance_id,
            instance.repo or "n/a",
            instance.version or "n/a",
            instance.created_at or "n/a",
        )
    console.print(table)


def _print_swebench_eval_result(
    *,
    traces_path: Path,
    summary_path: Path,
    raw_output_root: Path,
    traces_count: int,
) -> None:
    console.print("\n[bold]SWE-bench eval outputs:[/]")
    console.print(f"  Raw outputs: {raw_output_root}")
    console.print(f"  Traces: {traces_path}")
    console.print(f"  Summary: {summary_path}")
    console.print(f"  Instances: {traces_count}")


def _run_swebench_eval(
    *,
    instance_ids: list[str] | None,
    dataset_name: str,
    split: str,
    predictions_path: str,
    run_id: str | None,
    timeout: int,
    max_workers: int,
    output_dir: Path,
    limit: int | None = None,
    repo_filter: str | None = None,
    run_id_suffix: str = "swebench-run",
) -> None:
    try:
        selected_instance_ids = swebench_helpers.resolve_swebench_instance_ids(
            dataset_name=dataset_name,
            split=split,
            raw_instance_ids=instance_ids,
            limit=limit,
            repo_filter=repo_filter,
        )
        swebench_helpers.validate_modal_credentials()
    except (RuntimeError, ValueError) as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
        raise typer.Exit(code=1) from exc

    resolved_run_id = _timestamped_run_id(run_id or run_id_suffix)
    resolved_output_dir = _resolve_output_dir(output_dir)
    raw_output_root = resolved_output_dir / resolved_run_id / "raw"
    raw_output_root.mkdir(parents=True, exist_ok=True)

    harness_dataset_name = dataset_name
    dataset_path = Path(dataset_name)
    if not dataset_path.is_absolute():
        project_dataset_path = PROJECT_ROOT / dataset_path
        if project_dataset_path.exists():
            harness_dataset_name = str(project_dataset_path)

    harness_predictions_path = predictions_path
    if predictions_path != "gold":
        predictions_file = Path(predictions_path)
        if not predictions_file.is_absolute():
            predictions_file = PROJECT_ROOT / predictions_file
        harness_predictions_path = str(predictions_file)

    command = swebench_helpers.build_swebench_harness_command(
        dataset_name=harness_dataset_name,
        split=split,
        instance_ids=selected_instance_ids,
        predictions_path=harness_predictions_path,
        run_id=resolved_run_id,
        timeout=timeout,
        max_workers=max_workers,
        report_dir=".",
    )

    console.print(f"[bold]SWE-bench instances:[/] {selected_instance_ids}")
    console.print(f"[bold]Dataset:[/] {dataset_name} ({split})")
    console.print(f"[bold]Run ID:[/] {resolved_run_id}")
    console.print(f"[bold]Harness command:[/] {shlex.join(command)}")

    harness_run = swebench_helpers.run_swebench_harness(
        command=command,
        cwd=raw_output_root,
        stream_output=True,
    )
    traces = swebench_helpers.build_swebench_traces(
        instance_ids=selected_instance_ids,
        run_id=resolved_run_id,
        project_root=PROJECT_ROOT,
        harness_run=harness_run,
        raw_output_root=raw_output_root,
    )
    traces_path, summary_path = swebench_helpers.write_swebench_eval_outputs(
        traces=traces,
        output_dir=resolved_output_dir,
        run_id=resolved_run_id,
    )
    _print_swebench_eval_result(
        traces_path=traces_path,
        summary_path=summary_path,
        raw_output_root=raw_output_root,
        traces_count=len(traces),
    )

    if harness_run.returncode != 0:
        console.print(
            "[bold red]ERROR:[/] SWE-bench harness exited with "
            f"code {harness_run.returncode}."
        )
        raise typer.Exit(code=harness_run.returncode)


def _ensure_disjoint_swebench_sets(
    *,
    evolve_instance_ids: list[str],
    eval_instance_ids: list[str],
) -> None:
    overlap = sorted(set(evolve_instance_ids) & set(eval_instance_ids))
    if overlap:
        raise typer.BadParameter(
            "evolution and eval SWE-bench instances must be disjoint; "
            f"overlap: {overlap}"
        )


def _build_swebench_executor(
    *,
    config: Config,
    benchmark_repo: swebench_helpers.SWEbenchRepository,
    phase_dir: Path,
    timeout: int,
    max_workers: int,
    run_id_prefix: str,
) -> SWEbenchExecutorAgent:
    from mediated_coevo.llm.client import LLMClient

    artifact_root = phase_dir / "artifacts"
    patch_generator = LLMSWEbenchPatchGenerator(
        LLMClient(model=_openrouter_model_for_llm(config.models.executor))
    )
    return SWEbenchExecutorAgent(
        model=config.models.executor,
        benchmark_repo=benchmark_repo,
        patch_generator=patch_generator,
        artifact_root=artifact_root,
        injected_skill_name=config.executor_runtime.injected_skill_name,
        project_root=PROJECT_ROOT,
        timeout=timeout,
        max_workers=max_workers,
        run_id_prefix=run_id_prefix,
    )


def _openrouter_model_for_llm(model: str) -> str:
    """Return a LiteLLM/OpenRouter model id from a Harbor-ready model id."""
    return normalize_openrouter_model_name(model)


def _prepare_unified_experiment_root(
    *,
    config: Config,
    seed: int,
    run_id: str | None,
    suffix: str,
) -> tuple[Path, Path]:
    resolved_run_id = _timestamped_run_id(run_id or f"{seed}-{suffix}")
    experiment_dir = (
        PROJECT_ROOT / config.paths.data_dir / "experiments" / resolved_run_id
    )
    experiment_dir.mkdir(parents=True, exist_ok=False)
    runtime_skills_dir = shutil.copytree(
        PROJECT_ROOT / config.paths.skills_dir,
        experiment_dir / "skills",
    )
    with open(experiment_dir / "config.toml", "wb") as f:
        tomli_w.dump(config.model_dump(exclude_none=True), f)
    return experiment_dir, runtime_skills_dir


def _runtime_benchmark_by_task_id(
    selection: BenchmarkTaskSelection,
    config: Config,
) -> dict[str, BenchmarkKind]:
    """Include validation-only tasks in runtime routing without evolving on them."""
    mapping = dict(selection.benchmark_by_task_id)
    if _needs_validation_skillsbench(selection, config):
        for task_id in config.experiment.skill_validation.skillsbench_tasks:
            mapping.setdefault(task_id, "skillsbench")
    if _needs_validation_swebench(selection, config):
        for task_id in config.experiment.skill_validation.swebench_instances:
            mapping.setdefault(task_id, "swebench")
    return mapping


def _needs_runtime_skillsbench(
    selection: BenchmarkTaskSelection,
    config: Config,
) -> bool:
    return selection.has_skillsbench or _needs_validation_skillsbench(
        selection,
        config,
    )


def _needs_runtime_swebench(
    selection: BenchmarkTaskSelection,
    config: Config,
) -> bool:
    return selection.has_swebench or _needs_validation_swebench(selection, config)


def _needs_validation_skillsbench(
    selection: BenchmarkTaskSelection,
    config: Config,
) -> bool:
    return (
        bool(config.experiment.skill_validation.skillsbench_tasks)
        and selection.has_skillsbench
    )


def _needs_validation_swebench(
    selection: BenchmarkTaskSelection,
    config: Config,
) -> bool:
    validation = config.experiment.skill_validation
    if not validation.swebench_instances:
        return False
    if selection.has_swebench:
        return True
    return (
        selection.has_skillsbench
        and validation.allow_swebench_replacement_for_skillsbench
    )


def _build_unified_runtime(
    *,
    config: Config,
    selection: BenchmarkTaskSelection,
    condition_name: ConditionName,
    runtime_skills_dir: Path,
    experiment_dir: Path,
    dataset_name: str,
    split: str,
    timeout: int,
    max_workers: int,
) -> ExperimentRuntime:
    """Build one orchestrator that can route across selected benchmark backends."""
    from mediated_coevo.llm.client import LLMClient

    validate_experiment_design(
        condition=condition_name,
        skill_updates=config.experiment.skill_updates,
        baseline_preset=config.experiment.baseline_preset,
    )
    skill_store = SkillStore(runtime_skills_dir)
    skill_store.validate()
    artifact_store = ArtifactStore(base_dir=experiment_dir / "artifacts")
    history_store = HistoryStore(history_dir=experiment_dir / "history")

    needs_skillsbench = _needs_runtime_skillsbench(selection, config)
    needs_swebench = _needs_runtime_swebench(selection, config)
    skillsbench_repo = (
        _build_benchmark_repo(PROJECT_ROOT, config)
        if needs_skillsbench
        else None
    )
    swebench_repo = (
        swebench_helpers.SWEbenchRepository(dataset_name=dataset_name, split=split)
        if needs_swebench
        else None
    )
    runtime_benchmark_by_task_id = _runtime_benchmark_by_task_id(selection, config)
    benchmark_repo = MixedBenchmarkRepository(
        benchmark_by_task_id=runtime_benchmark_by_task_id,
        skillsbench_repo=skillsbench_repo,
        swebench_repo=swebench_repo,
    )

    skillsbench_executor = None
    if needs_skillsbench:
        assert skillsbench_repo is not None
        skillsbench_executor = ExecutorAgent(
            model=config.models.executor,
            benchmark_repo=skillsbench_repo,
            harbor_runner=HarborRunner(
                agent_name=config.executor_runtime.agent_name,
                jobs_dir=experiment_dir / config.executor_runtime.jobs_dir,
                timeout_sec=config.executor_runtime.harbor_timeout_sec,
                agent_setup_timeout_multiplier=(
                    config.executor_runtime.harbor_agent_setup_timeout_multiplier
                ),
            ),
            workspace_root=experiment_dir / "benchmarks",
            injected_skill_name=config.executor_runtime.injected_skill_name,
        )

    swebench_executor = None
    if needs_swebench:
        assert swebench_repo is not None
        swebench_executor = _build_swebench_executor(
            config=config,
            benchmark_repo=swebench_repo,
            phase_dir=experiment_dir,
            timeout=timeout,
            max_workers=max_workers,
            run_id_prefix=experiment_dir.name,
        )

    planner = PlannerAgent(llm_client=LLMClient(model=config.models.planner))
    planner.configure_token_budget(
        config.budgets,
        condition_name=config.experiment.condition_name,
    )
    executor = RoutedExecutorAgent(
        benchmark_by_task_id=runtime_benchmark_by_task_id,
        skillsbench_executor=skillsbench_executor,
        swebench_executor=swebench_executor,
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
    skill_advisor = SkillAdvisor(llm_client=LLMClient(model=config.models.planner))
    skill_advisor.configure_token_budget(
        config.budgets,
        condition_name=config.experiment.condition_name,
    )
    judge_client = LLMClient(model=config.models.judge)

    return ExperimentRuntime(
        experiment_dir=experiment_dir,
        orchestrator=Orchestrator(
            planner=planner,
            executor=executor,  # type: ignore[arg-type]
            mediator=mediator,
            skill_store=skill_store,
            artifact_store=artifact_store,
            history_store=history_store,
            benchmark_repo=benchmark_repo,  # type: ignore[arg-type]
            config=config,
            experiment_dir=experiment_dir,
            skill_advisor=skill_advisor,
            judge_llm_client=judge_client,
        ),
    )


def _print_unified_task_selection(selection: BenchmarkTaskSelection) -> None:
    if selection.skillsbench_task_ids:
        console.print(f"[bold]SkillsBench tasks:[/] {selection.skillsbench_task_ids}")
    if selection.swebench_instance_ids:
        console.print(f"[bold]SWE-bench instances:[/] {selection.swebench_instance_ids}")
    console.print(f"[bold]Executor backend:[/] {selection.backend_name}")


def _run_unified_experiment(
    *,
    config: Config,
    selection: BenchmarkTaskSelection,
    iterations: int,
    seed: int,
    condition_name: ConditionName,
    dataset_name: str,
    split: str,
    timeout: int,
    max_workers: int,
    run_id: str | None,
    swebench_eval_instance_ids: list[str],
) -> None:
    """Run SkillsBench, SWE-bench, or mixed selections in one evolution loop."""
    random.seed(seed)
    config.executor_runtime.backend = selection.backend_name
    config.experiment.benchmark_selection.skillsbench_tasks = (
        selection.skillsbench_task_ids
    )
    config.experiment.benchmark_selection.swebench_instances = (
        selection.swebench_instance_ids
    )

    _prepare_llm_credentials_or_exit(config)
    if _needs_runtime_skillsbench(selection, config):
        _ensure_harbor_available(config)
    if _needs_runtime_swebench(selection, config):
        try:
            swebench_helpers.validate_modal_credentials()
        except RuntimeError as exc:
            console.print(f"[bold red]ERROR:[/] {exc}")
            raise typer.Exit(code=1) from exc

    experiment_dir, runtime_skills_dir = _prepare_unified_experiment_root(
        config=config,
        seed=seed,
        run_id=run_id,
        suffix=selection.backend_name,
    )
    runtime = _build_unified_runtime(
        config=config,
        selection=selection,
        condition_name=condition_name,
        runtime_skills_dir=runtime_skills_dir,
        experiment_dir=experiment_dir,
        dataset_name=dataset_name,
        split=split,
        timeout=timeout,
        max_workers=max_workers,
    )

    _print_unified_task_selection(selection)
    if selection.has_swebench:
        console.print(f"[bold]SWE-bench dataset:[/] {dataset_name} ({split})")
    if swebench_eval_instance_ids:
        console.print(
            f"[bold]SWE-bench frozen eval instances:[/] {swebench_eval_instance_ids}"
        )
    console.print(f"[bold]Iterations:[/] {iterations}")
    console.print(f"[bold]Condition:[/] {condition_name}")
    console.print(
        f"[bold]Skill updates:[/] {config.experiment.skill_updates.model_dump()}"
    )
    _print_experiment_controls(config)
    console.print(
        "[bold]Models:[/] "
        f"planner={config.models.planner} "
        f"executor={config.models.executor} "
        f"mediator={config.models.mediator} "
        f"judge={config.models.judge}"
    )
    console.print(f"\n[bold green]Starting experiment:[/] {runtime.experiment_dir}\n")
    records = asyncio.run(
        runtime.orchestrator.run_experiment(selection.task_ids, iterations)
    )
    _write_and_print_result_summary(
        records=records,
        data_dir=runtime.experiment_dir,
        header="Results",
    )
    _annotate_judge_rewards_or_exit(
        data_dir=runtime.experiment_dir,
        config=config,
        history_store=runtime.orchestrator.history_store,
    )
    if not swebench_eval_instance_ids:
        return

    eval_config = config.model_copy(deep=True)
    eval_config.experiment.skill_updates = SkillUpdateConfig(
        executor=False,
        planner=False,
        mediator=False,
    )
    eval_dir = runtime.experiment_dir / "eval"
    benchmark_repo = swebench_helpers.SWEbenchRepository(
        dataset_name=dataset_name,
        split=split,
    )
    console.print("\n[bold green]Starting frozen SWE-bench eval phase[/]\n")
    eval_traces = asyncio.run(
        _run_swebench_frozen_eval(
            config=eval_config,
            benchmark_repo=benchmark_repo,
            runtime_skills_dir=runtime_skills_dir,
            eval_dir=eval_dir,
            instance_ids=swebench_eval_instance_ids,
            timeout=timeout,
            max_workers=max_workers,
            run_id_prefix=f"{runtime.experiment_dir.name}-eval",
        )
    )
    traces_path, predictions_path, summary_path, eval_summary = (
        _write_swebench_phase_outputs(
            traces=eval_traces,
            phase_dir=eval_dir,
            config=eval_config,
        )
    )
    _print_result_summary(
        summary=eval_summary,
        data_dir=eval_dir,
        summary_path=summary_path,
        header="Frozen eval results",
    )
    _annotate_judge_rewards_or_exit(data_dir=eval_dir, config=eval_config)
    console.print(f"  Predictions: {predictions_path}")
    console.print(f"  Traces: {traces_path}")


async def _run_swebench_frozen_eval(
    *,
    config: Config,
    benchmark_repo: swebench_helpers.SWEbenchRepository,
    runtime_skills_dir: Path,
    eval_dir: Path,
    instance_ids: list[str],
    timeout: int,
    max_workers: int,
    run_id_prefix: str,
) -> list[ExecutionTrace]:
    skill_store = SkillStore(runtime_skills_dir)
    skill_store.validate()
    executor = _build_swebench_executor(
        config=config,
        benchmark_repo=benchmark_repo,
        phase_dir=eval_dir,
        timeout=timeout,
        max_workers=max_workers,
        run_id_prefix=run_id_prefix,
    )
    executor_skill_text = skill_store.read_skill("executor") or ""
    traces: list[ExecutionTrace] = []
    for iteration, instance_id in enumerate(instance_ids):
        task = benchmark_repo.resolve(instance_id)
        trace = await executor.execute_task(
            TaskSpec(
                task_id=instance_id,
                instruction=task.instruction,
                iteration=iteration,
            ),
            [executor_skill_text] if executor_skill_text else [],
        )
        traces.append(trace)
    return traces


def _records_from_swebench_traces(
    *,
    traces: list[ExecutionTrace],
    config: Config,
) -> list[IterationRecord]:
    return [
        IterationRecord(
            iteration=trace.iteration,
            task_id=trace.task_id,
            execution_trace=trace,
            reward=trace.reward,
            duration_sec=trace.duration_sec,
            run_id=trace.run_id,
            condition_name=config.experiment.condition_name,
            seed=config.experiment.seed,
            models=config.models.model_dump(exclude_none=True),
            executor_agent=config.executor_runtime.agent_name,
            skill_update_policy=config.experiment.skill_updates.model_dump(),
            expected_reward_range=(0.0, 1.0),
            verifier_type=swebench_helpers.SWEBENCH_VERIFIER_TYPE,
            success=trace.is_usable_feedback_signal,
            verifier_status=trace.status,
            total_tokens=(
                trace.token_usage.input_tokens + trace.token_usage.output_tokens
            ),
            token_totals_by_agent={
                "executor": (
                    trace.token_usage.input_tokens + trace.token_usage.output_tokens
                )
            },
        )
        for trace in traces
    ]


def _write_swebench_phase_outputs(
    *,
    traces: list[ExecutionTrace],
    phase_dir: Path,
    config: Config,
) -> tuple[Path, Path, Path, ExperimentScoreSummary]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    traces_path = phase_dir / "traces.jsonl"
    with open(traces_path, "w") as f:
        for trace in traces:
            f.write(trace.model_dump_json())
            f.write("\n")

    predictions_path = phase_dir / "predictions.jsonl"
    with open(predictions_path, "w") as f:
        for trace in traces:
            prediction_path = trace.harbor_paths.get("prediction_jsonl")
            if not prediction_path:
                continue
            path = Path(prediction_path)
            if not path.exists():
                continue
            text = path.read_text().strip()
            if text:
                f.write(text)
                f.write("\n")

    records = _records_from_swebench_traces(traces=traces, config=config)
    summary = build_score_summary(records)
    summary_path = phase_dir / "summary.json"
    write_score_summary(summary, summary_path)
    return traces_path, predictions_path, summary_path, summary


@app.command()
def run(
    skillsbench_tasks: Annotated[
        list[str] | None,
        typer.Option(
            "--skillsbench-task",
            help=(
                "SkillsBench task ID. Repeat the option or provide comma-separated "
                "IDs. Overrides --skillsbench-task-set."
            ),
        ),
    ] = None,
    skillsbench_task_set: Annotated[
        str | None,
        typer.Option(
            "--skillsbench-task-set",
            help=(
                "Named SkillsBench task set. "
                f"Available: {_available_task_set_help()}"
            ),
        ),
    ] = None,
    swebench_instances: Annotated[
        list[str] | None,
        typer.Option(
            "--swebench-instance",
            help=(
                "SWE-bench instance ID. Repeat the option or provide "
                "comma-separated IDs."
            ),
        ),
    ] = None,
    swebench_limit: Annotated[
        int | None,
        typer.Option(
            "--swebench-limit",
            min=1,
            help="Use the first N instances from the selected SWE-bench split.",
        ),
    ] = None,
    swebench_eval_instances: Annotated[
        list[str] | None,
        typer.Option(
            "--swebench-eval-instance",
            help=(
                "SWE-bench instance ID to use for frozen eval after evolution. "
                "Repeat the option or provide comma-separated IDs."
            ),
        ),
    ] = None,
    swebench_eval_limit: Annotated[
        int | None,
        typer.Option(
            "--swebench-eval-limit",
            min=1,
            help=(
                "Use the first N instances from the selected SWE-bench split for "
                "frozen eval after evolution."
            ),
        ),
    ] = None,
    tasks: Annotated[
        str | None,
        typer.Option(
            "--tasks",
            help=(
                "Legacy alias for comma-separated --skillsbench-task IDs. "
                "Overrides --task-set."
            ),
        ),
    ] = None,
    task_set: Annotated[
        str | None,
        typer.Option(
            "--task-set",
            help="Legacy alias for --skillsbench-task-set.",
        ),
    ] = None,
    dataset_name: Annotated[
        str,
        typer.Option(
            "--swebench-dataset-name",
            help="SWE-bench dataset name or local dataset path.",
        ),
    ] = swebench_helpers.DEFAULT_SWEBENCH_DATASET,
    split: Annotated[
        str,
        typer.Option(
            "--swebench-split",
            help="SWE-bench dataset split to use.",
        ),
    ] = swebench_helpers.DEFAULT_SWEBENCH_SPLIT,
    iterations: Annotated[int, typer.Option(help="Number of iterations")] = 30,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    condition: Annotated[
        str,
        typer.Option(
            help=(
                "Experiment condition: no_feedback | full_traces | shared_notes | "
                "static_mediator | learned_mediator"
            ),
        ),
    ] = "learned_mediator",
    skill_updates: Annotated[
        str,
        typer.Option(
            "--skill-updates",
            help=(
                "Comma-separated skill updates allowed: none | executor | planner | "
                "mediator | all"
            ),
        ),
    ] = "all",
    coevo_interval: Annotated[
        int | None,
        typer.Option(
            "--coevo-interval",
            min=1,
            help="Override experiment.coevo_interval for this run.",
        ),
    ] = None,
    advisor_buffer_max: Annotated[
        int | None,
        typer.Option(
            "--advisor-buffer-max",
            min=1,
            help="Override experiment.advisor_buffer_max for this run.",
        ),
    ] = None,
    harbor_agent_setup_timeout_multiplier: Annotated[
        float | None,
        typer.Option(
            "--harbor-agent-setup-timeout-multiplier",
            min=0.1,
            help="Forwarded to Harbor for slow agent setup phases.",
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help=(
                "Run ID suffix. The actual experiment directory is prefixed with "
                "a timestamp."
            ),
        ),
    ] = None,
    timeout: Annotated[
        int,
        typer.Option(
            "--timeout",
            min=1,
            help="Per-SWE-bench-instance test timeout in seconds.",
        ),
    ] = 1800,
    max_workers: Annotated[
        int,
        typer.Option(
            "--max-workers",
            min=1,
            help="SWE-bench Modal harness worker count.",
        ),
    ] = 1,
    config_dir: Annotated[
        Path,
        typer.Option(help="Config directory"),
    ] = PROJECT_ROOT / "config",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run a unified SkillsBench, SWE-bench, or mixed co-evolution experiment."""
    _setup_logging(verbose)

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
        coevo_interval=coevo_interval,
        advisor_buffer_max=advisor_buffer_max,
        harbor_agent_setup_timeout_multiplier=harbor_agent_setup_timeout_multiplier,
    )

    _validate_or_raise_bad_parameter(config)
    benchmark_repo = _build_benchmark_repo(PROJECT_ROOT, config)
    selection = _resolve_unified_task_selection(
        skillsbench_tasks=skillsbench_tasks,
        skillsbench_task_set=skillsbench_task_set,
        legacy_tasks=tasks,
        legacy_task_set=task_set,
        swebench_instances=swebench_instances,
        swebench_limit=swebench_limit,
        dataset_name=dataset_name,
        split=split,
        benchmark_repo=benchmark_repo,
    )
    swebench_eval_requested = (
        bool(swebench_eval_instances) or swebench_eval_limit is not None
    )
    if swebench_eval_requested and not selection.has_swebench:
        raise typer.BadParameter(
            "SWE-bench frozen eval requires an evolution SWE-bench selection "
            "(--swebench-instance or --swebench-limit)"
        )
    swebench_eval_instance_ids: list[str] = []
    if swebench_eval_requested:
        swebench_eval_instance_ids = _swebench_instance_ids_from_unified_cli(
            swebench_instances=swebench_eval_instances,
            swebench_limit=swebench_eval_limit,
            dataset_name=dataset_name,
            split=split,
        )
        _ensure_disjoint_swebench_sets(
            evolve_instance_ids=selection.swebench_instance_ids,
            eval_instance_ids=swebench_eval_instance_ids,
        )
    _run_unified_experiment(
        config=config,
        selection=selection,
        iterations=iterations,
        seed=seed,
        condition_name=condition_name,
        dataset_name=dataset_name,
        split=split,
        timeout=timeout,
        max_workers=max_workers,
        run_id=run_id,
        swebench_eval_instance_ids=swebench_eval_instance_ids,
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
    coevo_interval: Annotated[
        int | None,
        typer.Option(
            "--coevo-interval",
            min=1,
            help="Override experiment.coevo_interval for every matrix row.",
        ),
    ] = None,
    advisor_buffer_max: Annotated[
        int | None,
        typer.Option(
            "--advisor-buffer-max",
            min=1,
            help="Override experiment.advisor_buffer_max for every matrix row.",
        ),
    ] = None,
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the six-row baseline matrix with isolated per-row skills."""
    _setup_logging(verbose)

    config = _apply_experiment_settings(
        load_config(config_dir),
        iterations=iterations,
        seed=seed,
        coevo_interval=coevo_interval,
        advisor_buffer_max=advisor_buffer_max,
    )
    for preset_name in BASELINE_PRESET_NAMES:
        preset = get_baseline_preset(preset_name)
        validate_experiment_design(
            condition=preset.condition_name,
            skill_updates=preset.skill_updates,
            baseline_preset=preset.name,
        )
    preflight_task_ids = _preflight_task_ids_from_cli(tasks, task_set)
    _prepare_llm_credentials_or_exit(config)
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
    _print_experiment_controls(config)
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
        _annotate_judge_rewards_or_exit(
            data_dir=row.runtime.experiment_dir,
            config=row_config,
            history_store=row.runtime.orchestrator.history_store,
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


@swebench_app.command("list-instances")
def list_swebench_instances(
    dataset_name: str = typer.Option(
        swebench_helpers.DEFAULT_SWEBENCH_DATASET,
        "--dataset-name",
        help="SWE-bench dataset name or local dataset path.",
    ),
    split: str = typer.Option(
        swebench_helpers.DEFAULT_SWEBENCH_SPLIT,
        "--split",
        help="Dataset split to inspect.",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        min=1,
        help="Maximum number of valid instance IDs to print.",
    ),
    repo_filter: str | None = typer.Option(
        None,
        "--repo-filter",
        help="Optional case-insensitive substring filter for the repo field.",
    ),
) -> None:
    """List valid instance IDs from a SWE-bench dataset split."""
    try:
        instances = swebench_helpers.list_swebench_instances(
            dataset_name=dataset_name,
            split=split,
            limit=limit,
            repo_filter=repo_filter,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if not instances:
        console.print("[yellow]No SWE-bench instances matched the filters.[/]")
        return
    _print_swebench_instances(instances)


@swebench_app.command("smoke")
def smoke_swebench(
    instance_ids: list[str] | None = typer.Option(
        None,
        "--instance-id",
        help=(
            "SWE-bench instance ID to run. Repeat the option or provide "
            "comma-separated IDs. Defaults to a tiny Lite smoke instance."
        ),
    ),
    dataset_name: str = typer.Option(
        swebench_helpers.DEFAULT_SWEBENCH_DATASET,
        "--dataset-name",
        help="SWE-bench dataset name or local dataset path.",
    ),
    split: str = typer.Option(
        swebench_helpers.DEFAULT_SWEBENCH_SPLIT,
        "--split",
        help="Dataset split to evaluate.",
    ),
    predictions_path: str = typer.Option(
        "gold",
        "--predictions-path",
        help="SWE-bench predictions JSON/JSONL path, or 'gold'.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help=(
            "Run ID suffix. The actual SWE-bench run ID is prefixed with "
            "a timestamp."
        ),
    ),
    timeout: int = typer.Option(
        1800,
        "--timeout",
        min=1,
        help="Per-instance test timeout in seconds.",
    ),
    max_workers: int = typer.Option(
        1,
        "--max-workers",
        min=1,
        help="Modal harness worker count. Kept at 1 for small smoke runs.",
    ),
    output_dir: Path = typer.Option(
        swebench_helpers.DEFAULT_SWEBENCH_OUTPUT_DIR,
        "--output-dir",
        help="Directory for normalized medcoevo smoke outputs.",
    ),
) -> None:
    """Run a tiny standalone SWE-bench evaluation and normalize its reports."""
    _run_swebench_eval(
        instance_ids=instance_ids,
        dataset_name=dataset_name,
        split=split,
        predictions_path=predictions_path,
        run_id=run_id,
        timeout=timeout,
        max_workers=max_workers,
        output_dir=output_dir,
        run_id_suffix="swebench-smoke",
    )


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
