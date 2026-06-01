"""CLI entry point for the mediated co-evolution system."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, cast, get_args

import tomli_w
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from mediated_coevo.agents.executor import ExecutorAgent
from mediated_coevo.agents.mediator import MediatorAgent
from mediated_coevo.agents.planner import PlannerAgent
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
from mediated_coevo.analysis.task_similarity import (
    build_task_graph_precompute,
    write_task_graph_artifacts,
)
from mediated_coevo.benchmarks import (
    DEFAULT_SKILLFLOW_DATASET,
    HERMES_AGENT_NAME,
    HarborPrebuiltImageMissingError,
    HarborRunner,
    SkillFlowRepository,
    SkillFlowSyncConfig,
    SkillFlowSyncError,
)
from mediated_coevo.cloud.vm import (
    CloudVMConfigError,
    GCPVMConfig,
    RemoteHarborRunner,
    load_vm_config,
)
from mediated_coevo.core.config import (
    Config,
    ConfigLoadError,
    DiffusionPolicyName,
    SkillUpdateConfig,
    load_config,
)
from mediated_coevo.diffusion import DiffusionStore
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
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from mediated_coevo.stores.skill_store import SkillStore

app = typer.Typer(name="medcoevo", help="Mediated Co-Evolution Experiment Runner")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALID_CONDITION_NAMES = set(get_args(ConditionName))
VALID_DIFFUSION_POLICY_NAMES = set(get_args(DiffusionPolicyName))
TASK_SIMILARITY_GRAPH_NAMES = frozenset({"task_similarity", "precomputed_similarity"})
DEFAULT_TASK_GRAPH_EDGE_THRESHOLD = 0.05


@dataclass(frozen=True)
class TaskSelection:
    """Resolved SkillFlow task selectors for one run."""

    task_ids: list[str]
    tasks: list[str]
    family: str | None
    task_set: str | None


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
        benchmark_repo: SkillFlowRepository | None = None,
        harbor_runner: HarborRunner | RemoteHarborRunner | None = None,
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
        if harbor_runner is None:
            harbor_runner = _build_harbor_runner(
                config=config,
                experiment_dir=experiment_dir,
                remote_harbor_config=None,
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


def _validate_diffusion_policy_name(policy: str) -> DiffusionPolicyName:
    """Validate CLI diffusion policy names before mutating the config object."""
    if policy not in VALID_DIFFUSION_POLICY_NAMES:
        allowed = ", ".join(sorted(VALID_DIFFUSION_POLICY_NAMES))
        raise typer.BadParameter(
            f"invalid diffusion policy {policy!r}; expected one of: {allowed}"
        )
    return cast(DiffusionPolicyName, policy)


def _task_ids_from_repeatable_cli(raw_values: list[str] | None) -> list[str]:
    """Parse repeatable comma-separated task options."""
    if not raw_values:
        return []
    task_ids: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for candidate in raw_value.split(","):
            task_id = candidate.strip()
            if task_id and task_id not in seen:
                task_ids.append(task_id)
                seen.add(task_id)
    if not task_ids:
        raise typer.BadParameter("at least one task ID is required")
    return task_ids


def _sync_task_ids_from_repeatable_cli(raw_values: list[str] | None) -> list[str] | None:
    """Parse sync task selectors. None means all remote test tasks."""
    if not raw_values:
        return None
    task_ids = _task_ids_from_repeatable_cli(raw_values)
    all_selectors = [task_id for task_id in task_ids if task_id.lower() == "all"]
    if all_selectors:
        if len(task_ids) > 1:
            raise typer.BadParameter("--tasks all cannot be combined with task IDs")
        return None
    return task_ids


def _build_benchmark_repo(project_root: Path, config: Config) -> SkillFlowRepository:
    return SkillFlowRepository(
        root_dir=project_root / config.paths.benchmarks_dir,
        task_dirs=config.executor_runtime.task_dirs,
        sync=SkillFlowSyncConfig(
            enabled=config.executor_runtime.sync_enabled,
            dataset=config.executor_runtime.dataset,
            repo_type=config.executor_runtime.dataset_repo_type,
            local_dir=config.executor_runtime.task_dirs[0],
        ),
    )


def _resolve_task_selection(
    *,
    repository: SkillFlowRepository,
    tasks: list[str] | None,
    family: str | None,
    task_set: str | None,
) -> TaskSelection:
    selected = repository.resolve_selection(
        tasks=tasks,
        family=family,
        task_set=task_set,
    )
    if not selected:
        raise typer.BadParameter("provide --task, --family, or --task-set")
    return TaskSelection(
        task_ids=selected,
        tasks=tasks or [],
        family=family,
        task_set=task_set,
    )


def _load_config_or_bad_parameter(
    config_dir: Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> Config:
    try:
        return load_config(config_dir, overrides=overrides)
    except ConfigLoadError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _nested_override(section: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        section[key] = value


def _run_config_overrides(
    *,
    iterations: int | None,
    seed: int | None,
    condition: str | None,
    skill_updates: str | None,
    coevo_interval: int | None,
    advisor_buffer_max: int | None,
    diffusion_enabled: bool | None,
    diffusion_policy: str | None,
    diffusion_graph: str | None,
    diffusion_max_artifacts: int | None,
    diffusion_top_k_neighbors: int | None,
    harbor_agent_setup_timeout_multiplier: float | None,
) -> dict[str, Any]:
    experiment: dict[str, Any] = {}
    _nested_override(experiment, "num_iterations", iterations)
    _nested_override(experiment, "seed", seed)
    _nested_override(experiment, "coevo_interval", coevo_interval)
    _nested_override(experiment, "advisor_buffer_max", advisor_buffer_max)
    if condition is not None:
        experiment["condition_name"] = _validate_condition_name(condition)
    if skill_updates is not None:
        try:
            experiment["skill_updates"] = parse_skill_updates(
                skill_updates
            ).model_dump()
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    diffusion: dict[str, Any] = {}
    _nested_override(diffusion, "enabled", diffusion_enabled)
    if diffusion_policy is not None:
        diffusion["policy"] = _validate_diffusion_policy_name(diffusion_policy)
    _nested_override(diffusion, "graph", diffusion_graph)
    _nested_override(diffusion, "max_artifacts", diffusion_max_artifacts)
    _nested_override(diffusion, "top_k_neighbors", diffusion_top_k_neighbors)

    executor_runtime: dict[str, Any] = {}
    _nested_override(
        executor_runtime,
        "harbor_agent_setup_timeout_multiplier",
        harbor_agent_setup_timeout_multiplier,
    )

    overrides: dict[str, Any] = {}
    if experiment:
        overrides["experiment"] = experiment
    if diffusion:
        overrides["diffusion"] = diffusion
    if executor_runtime:
        overrides["executor_runtime"] = executor_runtime
    return overrides


def _apply_experiment_settings(
    config: Config,
    *,
    iterations: int | None = None,
    seed: int | None = None,
    condition_name: ConditionName | None = None,
    skill_updates: SkillUpdateConfig | None = None,
    baseline_preset: str | None = None,
    coevo_interval: int | None = None,
    advisor_buffer_max: int | None = None,
    diffusion_enabled: bool | None = None,
    diffusion_policy: str | None = None,
    diffusion_graph: str | None = None,
    diffusion_max_artifacts: int | None = None,
    diffusion_top_k_neighbors: int | None = None,
    harbor_agent_setup_timeout_multiplier: float | None = None,
) -> Config:
    """Apply CLI experiment settings to a loaded config object."""
    if iterations is not None:
        config.experiment.num_iterations = iterations
    if seed is not None:
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
    if diffusion_enabled is not None:
        config.diffusion.enabled = diffusion_enabled
    if diffusion_policy is not None:
        config.diffusion.policy = _validate_diffusion_policy_name(diffusion_policy)
    if diffusion_graph is not None:
        config.diffusion.graph = diffusion_graph
    if diffusion_max_artifacts is not None:
        config.diffusion.max_artifacts = diffusion_max_artifacts
    if diffusion_top_k_neighbors is not None:
        config.diffusion.top_k_neighbors = diffusion_top_k_neighbors
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
    benchmark_repo: SkillFlowRepository,
    harbor_runner: HarborRunner | RemoteHarborRunner | None = None,
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
            harbor_runner=harbor_runner,
        )
        rows.append(MatrixRuntime(preset_name=preset_name, runtime=runtime))
    return rows


def _build_harbor_runner(
    *,
    config: Config,
    experiment_dir: Path,
    remote_harbor_config: GCPVMConfig | None,
) -> HarborRunner | RemoteHarborRunner:
    jobs_dir = experiment_dir / config.executor_runtime.jobs_dir
    timeout_sec = config.executor_runtime.harbor_timeout_sec
    setup_timeout_multiplier = (
        config.executor_runtime.harbor_agent_setup_timeout_multiplier
    )
    if remote_harbor_config is not None:
        return RemoteHarborRunner(
            config=remote_harbor_config,
            jobs_dir=jobs_dir,
            timeout_sec=timeout_sec,
            agent_setup_timeout_multiplier=setup_timeout_multiplier,
        )
    return HarborRunner(
        jobs_dir=jobs_dir,
        timeout_sec=timeout_sec,
        agent_setup_timeout_multiplier=setup_timeout_multiplier,
    )


def _load_remote_harbor_config(
    *,
    enabled: bool,
    env_file: Path,
) -> GCPVMConfig | None:
    if not enabled:
        return None
    try:
        return load_vm_config(env_file)
    except (OSError, CloudVMConfigError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=True, show_path=False)],
    )


def _ensure_harbor_available(config: Config) -> None:
    if config.executor_runtime.harbor_required and shutil.which("harbor") is None:
        console.print(
            "[bold red]ERROR:[/] Harbor CLI not found on PATH. Install Harbor, "
            "or set executor_runtime.harbor_required = false in config."
        )
        raise typer.Exit(code=1)


def _ensure_gcloud_available() -> None:
    if shutil.which("gcloud") is None:
        console.print(
            "[bold red]ERROR:[/] gcloud CLI not found on PATH. Install the "
            "Google Cloud CLI before using --cloud."
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


def _run_prebuild_step_or_exit(command: list[str], *, label: str) -> None:
    console.print(f"[bold]{label}:[/] {shlex.join(command)}")
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    except OSError as exc:
        console.print(f"[bold red]ERROR:[/] {label} failed to start: {exc}")
        raise typer.Exit(code=1) from exc
    if completed.returncode != 0:
        console.print(
            f"[bold red]ERROR:[/] {label} failed with exit code "
            f"{completed.returncode}."
        )
        raise typer.Exit(code=completed.returncode)


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


def _run_experiment_or_exit(
    runtime: ExperimentRuntime,
    task_ids: list[str],
    iterations: int,
) -> list[IterationRecord]:
    try:
        return asyncio.run(runtime.orchestrator.run_experiment(task_ids, iterations))
    except HarborPrebuiltImageMissingError as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
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


def _diffusion_inspection_payload(experiment_dir: Path) -> dict[str, Any] | None:
    diffusion_dir = experiment_dir / "diffusion"
    if not diffusion_dir.exists():
        return None
    store = DiffusionStore(diffusion_dir)
    artifacts = store.query_artifacts(recent=None)
    snapshots = store.query_graph_snapshots(recent=None)
    records = store.query_diffused_records(recent=None)
    return {
        "diffusion_dir": str(diffusion_dir),
        "artifacts_dir": str(diffusion_dir / "artifacts"),
        "graph_snapshots_dir": str(diffusion_dir / "graph_snapshots"),
        "diffused_records_path": str(diffusion_dir / "diffused_records.jsonl"),
        "artifact_count": len(artifacts),
        "graph_snapshot_count": len(snapshots),
        "diffused_record_count": len(records),
        "eligible_record_count": sum(1 for record in records if record.eligible),
        "selected_record_count": sum(1 for record in records if record.selected),
        "rendered_record_count": sum(1 for record in records if record.rendered),
        "source_task_ids": sorted({artifact.source_task_id for artifact in artifacts}),
        "graph_snapshot_ids": [snapshot.snapshot_id for snapshot in snapshots],
    }


def _single_inspection_payload(experiment_dir: Path) -> dict[str, Any]:
    summary_path = experiment_dir / "summary.json"
    metrics_path = experiment_dir / "metrics.jsonl"
    diffusion_payload = _diffusion_inspection_payload(experiment_dir)
    if summary_path.exists():
        payload = {
            "kind": "single",
            "experiment_dir": str(experiment_dir),
            "summary_path": str(summary_path),
            "metrics_path": str(metrics_path) if metrics_path.exists() else None,
            "artifact_dirs": _artifact_dirs(experiment_dir),
            "summary": _load_score_summary(summary_path).model_dump(mode="json"),
        }
        if diffusion_payload is not None:
            payload["diffusion"] = diffusion_payload
        return payload
    if metrics_path.exists():
        payload = {
            "kind": "single",
            "experiment_dir": str(experiment_dir),
            "summary_path": None,
            "metrics_path": str(metrics_path),
            "artifact_dirs": _artifact_dirs(experiment_dir),
            "warning": "summary.json is missing; inspect metrics.jsonl directly.",
        }
        if diffusion_payload is not None:
            payload["diffusion"] = diffusion_payload
        return payload
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
        if diffusion_payload := _diffusion_inspection_payload(row_dir):
            row["diffusion"] = diffusion_payload
        rows.append(row)
    if not rows:
        return None
    return {"kind": "matrix", "experiment_dir": str(experiment_dir), "rows": rows}


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
    if diffusion_payload := payload.get("diffusion"):
        console.print("  Diffusion:")
        console.print(f"    Records: {diffusion_payload['diffused_record_count']}")
        console.print(f"    Rendered: {diffusion_payload['rendered_record_count']}")
        console.print(f"    Graph snapshots: {diffusion_payload['graph_snapshot_count']}")


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
    table.add_column("Diffusion")
    table.add_column("Metrics")
    for row in payload["rows"]:
        diffusion_payload = row.get("diffusion") or {}
        diffusion_summary = (
            f"{diffusion_payload.get('rendered_record_count', 0)} rendered"
            if diffusion_payload
            else "n/a"
        )
        summary_data = row.get("summary")
        if summary_data is None:
            table.add_row(
                row["row"],
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                diffusion_summary,
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
            diffusion_summary,
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


def _print_task_selection(selection: TaskSelection) -> None:
    console.print(f"[bold]SkillFlow tasks:[/] {selection.task_ids}")
    if selection.family is not None:
        console.print(f"[bold]Family:[/] {selection.family}")
    if selection.task_set is not None:
        console.print(f"[bold]Task set:[/] {selection.task_set}")


def _print_experiment_controls(config: Config) -> None:
    """Print shared co-evolution controls after config and CLI overrides apply."""
    console.print(f"[bold]Coevo interval:[/] {config.experiment.coevo_interval}")
    console.print(
        f"[bold]Advisor buffer max:[/] {config.experiment.advisor_buffer_max}"
    )
    console.print(
        "[bold]Diffusion:[/] "
        f"enabled={config.diffusion.enabled} "
        f"policy={config.diffusion.policy} "
        f"graph={config.diffusion.graph}"
    )
    console.print("[bold]Skill validation:[/] required")
    console.print(
        "[bold]Harbor:[/] "
        f"agent={HERMES_AGENT_NAME} "
        "base_image=required "
        "task_prebuild=optional"
    )


def _timestamped_run_id(suffix: str) -> str:
    """Return a run ID with a timestamp prefix and caller-provided suffix."""
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{suffix}"


def _prepare_experiment_root(
    *,
    config: Config,
    seed: int,
    run_id: str | None,
) -> tuple[Path, Path]:
    resolved_run_id = _timestamped_run_id(run_id or f"{seed}-skillflow")
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


def _build_runtime(
    *,
    config: Config,
    condition_name: ConditionName,
    runtime_skills_dir: Path,
    experiment_dir: Path,
    benchmark_repo: SkillFlowRepository,
    remote_harbor_config: GCPVMConfig | None,
) -> ExperimentRuntime:
    """Build one SkillFlow runtime."""
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
    planner = PlannerAgent(llm_client=LLMClient(model=config.models.planner))
    planner.configure_token_budget(
        config.budgets,
        condition_name=config.experiment.condition_name,
    )
    executor = ExecutorAgent(
        model=config.models.executor,
        benchmark_repo=benchmark_repo,
        harbor_runner=_build_harbor_runner(
            config=config,
            experiment_dir=experiment_dir,
            remote_harbor_config=remote_harbor_config,
        ),
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


def _materialize_task_graph_for_diffusion(
    *,
    config: Config,
    experiment_dir: Path,
    benchmark_repo: SkillFlowRepository,
) -> None:
    """Write task graph artifacts when graph-aware diffusion is enabled."""
    if (
        not config.diffusion.enabled
        or config.diffusion.graph not in TASK_SIMILARITY_GRAPH_NAMES
    ):
        return

    output_dir = experiment_dir / "task-graph"
    if output_dir.exists():
        return

    precompute = build_task_graph_precompute(
        benchmark_repo.default_local_cache_dir(),
        edge_score_threshold=DEFAULT_TASK_GRAPH_EDGE_THRESHOLD,
    )
    write_task_graph_artifacts(precompute, output_dir)


def _run_skillflow_experiment(
    *,
    config: Config,
    selection: TaskSelection,
    iterations: int,
    seed: int,
    condition_name: ConditionName,
    run_id: str | None,
    remote_harbor_config: GCPVMConfig | None = None,
) -> None:
    """Run a SkillFlow selection in one evolution loop."""
    random.seed(seed)
    config.experiment.benchmark_selection.tasks = selection.task_ids
    config.experiment.benchmark_selection.family = selection.family
    config.experiment.benchmark_selection.task_set = selection.task_set

    _prepare_llm_credentials_or_exit(config)
    if remote_harbor_config is None:
        _ensure_harbor_available(config)
    else:
        _ensure_gcloud_available()

    experiment_dir, runtime_skills_dir = _prepare_experiment_root(
        config=config,
        seed=seed,
        run_id=run_id,
    )
    benchmark_repo = _build_benchmark_repo(PROJECT_ROOT, config)
    try:
        _materialize_task_graph_for_diffusion(
            config=config,
            experiment_dir=experiment_dir,
            benchmark_repo=benchmark_repo,
        )
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]ERROR:[/] failed to create diffusion task graph: {exc}")
        raise typer.Exit(code=1) from exc
    runtime = _build_runtime(
        config=config,
        condition_name=condition_name,
        runtime_skills_dir=runtime_skills_dir,
        experiment_dir=experiment_dir,
        benchmark_repo=benchmark_repo,
        remote_harbor_config=remote_harbor_config,
    )

    _print_task_selection(selection)
    if remote_harbor_config is not None:
        console.print(
            "[bold]Harbor runtime:[/] "
            f"GCP VM {remote_harbor_config.vm_name} ({remote_harbor_config.zone})"
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
    records = _run_experiment_or_exit(runtime, selection.task_ids, iterations)
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


@app.command()
def run(
    tasks: Annotated[
        list[str] | None,
        typer.Option(
            "--task",
            help="SkillFlow task ID. Repeat the option or provide comma-separated IDs.",
        ),
    ] = None,
    family: Annotated[
        str | None,
        typer.Option("--family", help="Run all local tasks in this SkillFlow family."),
    ] = None,
    task_set: Annotated[
        str | None,
        typer.Option("--task-set", help="Named local SkillFlow task set."),
    ] = None,
    iterations: Annotated[
        int | None,
        typer.Option(help="Number of iterations. Overrides experiment.num_iterations."),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(help="Random seed. Overrides experiment.seed."),
    ] = None,
    condition: Annotated[
        str | None,
        typer.Option(
            help=(
                "Experiment condition. Overrides experiment.condition_name. "
                "Allowed: no_feedback | full_traces | shared_notes | "
                "static_mediator | learned_mediator."
            ),
        ),
    ] = None,
    skill_updates: Annotated[
        str | None,
        typer.Option(
            "--skill-updates",
            help=(
                "Comma-separated skill updates allowed. Overrides "
                "experiment.skill_updates. Allowed: none | executor | planner | "
                "mediator | all."
            ),
        ),
    ] = None,
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
    diffusion_enabled: Annotated[
        bool | None,
        typer.Option(
            "--diffusion-enabled/--no-diffusion-enabled",
            help="Override diffusion.enabled for this run.",
        ),
    ] = None,
    diffusion_policy: Annotated[
        str | None,
        typer.Option(
            "--diffusion-policy",
            help=(
                "Override diffusion.policy. Allowed: none | capped_broadcast | "
                "random_k | top_k_similarity."
            ),
        ),
    ] = None,
    diffusion_graph: Annotated[
        str | None,
        typer.Option("--diffusion-graph", help="Override diffusion.graph."),
    ] = None,
    diffusion_max_artifacts: Annotated[
        int | None,
        typer.Option(
            "--diffusion-max-artifacts",
            min=1,
            help="Override diffusion.max_artifacts.",
        ),
    ] = None,
    diffusion_top_k_neighbors: Annotated[
        int | None,
        typer.Option(
            "--diffusion-top-k-neighbors",
            min=1,
            help="Override diffusion.top_k_neighbors.",
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
    config_dir: Annotated[
        Path,
        typer.Option(help="Config directory"),
    ] = PROJECT_ROOT / "config",
    cloud: Annotated[
        bool,
        typer.Option(
            "--cloud",
            help=(
                "Run Harbor jobs on the configured GCP VM while keeping the "
                "experiment control plane local."
            ),
        ),
    ] = False,
    cloud_env_file: Annotated[
        Path,
        typer.Option("--cloud-env-file", help="Dotenv file containing GCP VM settings."),
    ] = PROJECT_ROOT / ".env",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run a SkillFlow co-evolution experiment."""
    _setup_logging(verbose)
    config = _load_config_or_bad_parameter(
        config_dir,
        overrides=_run_config_overrides(
            iterations=iterations,
            seed=seed,
            condition=condition,
            skill_updates=skill_updates,
            coevo_interval=coevo_interval,
            advisor_buffer_max=advisor_buffer_max,
            diffusion_enabled=diffusion_enabled,
            diffusion_policy=diffusion_policy,
            diffusion_graph=diffusion_graph,
            diffusion_max_artifacts=diffusion_max_artifacts,
            diffusion_top_k_neighbors=diffusion_top_k_neighbors,
            harbor_agent_setup_timeout_multiplier=(
                harbor_agent_setup_timeout_multiplier
            ),
        ),
    )
    _validate_or_raise_bad_parameter(config)
    repository = _build_benchmark_repo(PROJECT_ROOT, config)
    selection = _resolve_task_selection(
        repository=repository,
        tasks=_task_ids_from_repeatable_cli(tasks),
        family=family,
        task_set=task_set,
    )
    remote_harbor_config = _load_remote_harbor_config(
        enabled=cloud,
        env_file=cloud_env_file,
    )
    _run_skillflow_experiment(
        config=config,
        selection=selection,
        iterations=config.experiment.num_iterations,
        seed=config.experiment.seed,
        condition_name=config.experiment.condition_name,
        run_id=run_id,
        remote_harbor_config=remote_harbor_config,
    )


@app.command()
def matrix(
    tasks: Annotated[
        list[str] | None,
        typer.Option(
            "--task",
            help="SkillFlow task ID. Repeat the option or provide comma-separated IDs.",
        ),
    ] = None,
    family: Annotated[
        str | None,
        typer.Option("--family", help="Run all local tasks in this SkillFlow family."),
    ] = None,
    task_set: Annotated[
        str | None,
        typer.Option("--task-set", help="Named local SkillFlow task set."),
    ] = None,
    iterations: int | None = typer.Option(
        None,
        help="Number of iterations per row. Overrides experiment.num_iterations.",
    ),
    seed: int | None = typer.Option(
        None,
        help="Random seed reused for every row. Overrides experiment.seed.",
    ),
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
    diffusion_enabled: Annotated[
        bool | None,
        typer.Option(
            "--diffusion-enabled/--no-diffusion-enabled",
            help="Override diffusion.enabled for every matrix row.",
        ),
    ] = None,
    diffusion_policy: Annotated[
        str | None,
        typer.Option(
            "--diffusion-policy",
            help=(
                "Override diffusion.policy for every matrix row. Allowed: none | "
                "capped_broadcast | random_k | top_k_similarity."
            ),
        ),
    ] = None,
    diffusion_graph: Annotated[
        str | None,
        typer.Option("--diffusion-graph", help="Override diffusion.graph."),
    ] = None,
    diffusion_max_artifacts: Annotated[
        int | None,
        typer.Option(
            "--diffusion-max-artifacts",
            min=1,
            help="Override diffusion.max_artifacts.",
        ),
    ] = None,
    diffusion_top_k_neighbors: Annotated[
        int | None,
        typer.Option(
            "--diffusion-top-k-neighbors",
            min=1,
            help="Override diffusion.top_k_neighbors.",
        ),
    ] = None,
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the six-row baseline matrix with isolated per-row skills."""
    _setup_logging(verbose)
    config = _load_config_or_bad_parameter(
        config_dir,
        overrides=_run_config_overrides(
            iterations=iterations,
            seed=seed,
            condition=None,
            skill_updates=None,
            coevo_interval=coevo_interval,
            advisor_buffer_max=advisor_buffer_max,
            diffusion_enabled=diffusion_enabled,
            diffusion_policy=diffusion_policy,
            diffusion_graph=diffusion_graph,
            diffusion_max_artifacts=diffusion_max_artifacts,
            diffusion_top_k_neighbors=diffusion_top_k_neighbors,
            harbor_agent_setup_timeout_multiplier=None,
        ),
    )
    for preset_name in BASELINE_PRESET_NAMES:
        preset = get_baseline_preset(preset_name)
        validate_experiment_design(
            condition=preset.condition_name,
            skill_updates=preset.skill_updates,
            baseline_preset=preset.name,
        )
    _prepare_llm_credentials_or_exit(config)
    _ensure_harbor_available(config)
    repository = _build_benchmark_repo(PROJECT_ROOT, config)
    selection = _resolve_task_selection(
        repository=repository,
        tasks=_task_ids_from_repeatable_cli(tasks),
        family=family,
        task_set=task_set,
    )
    factory = ExperimentFactory(PROJECT_ROOT)
    seed = config.experiment.seed
    iterations = config.experiment.num_iterations
    matrix_dir = factory.create_matrix_dir(seed=seed, data_dir=config.paths.data_dir)
    rows = _build_matrix_runtimes(
        factory=factory,
        base_config=config,
        seed=seed,
        matrix_dir=matrix_dir,
        benchmark_repo=repository,
    )

    _print_task_selection(selection)
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
        records = _run_experiment_or_exit(
            row.runtime,
            selection.task_ids,
            iterations,
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
        target_dir = _latest_experiment_dir(
            _experiments_root(_load_config_or_bad_parameter(config_dir))
        )
    payload = _inspection_payload(target_dir)
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    _print_inspection_payload(payload)


@app.command("create-graph")
def create_graph(
    threshold: float = typer.Option(
        DEFAULT_TASK_GRAPH_EDGE_THRESHOLD,
        "--threshold",
        min=0.0,
        help="Minimum similarity score required to keep an edge.",
    ),
    tasks_root: Path = typer.Option(
        PROJECT_ROOT / "benchmarks" / "skillflow" / "tasks",
        "--tasks-root",
        help="Local SkillFlow task directory to analyze.",
    ),
    output_dir: Path = typer.Option(
        PROJECT_ROOT / "data" / "task_graphs" / "skillflow-local",
        "--output-dir",
        help="Directory where graph precompute JSON artifacts are written.",
    ),
) -> None:
    """Create a directed SkillFlow task graph from local task metadata."""
    try:
        precompute = build_task_graph_precompute(
            tasks_root,
            edge_score_threshold=threshold,
        )
        write_task_graph_artifacts(precompute, output_dir)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]Tasks:[/] {precompute.task_count}")
    console.print(f"[bold]Pairs:[/] {precompute.pair_count}")
    console.print(f"[bold]Threshold:[/] {precompute.active_threshold}")
    console.print(f"[bold]Kept edges:[/] {precompute.kept_edge_count}")
    console.print(f"[bold]Cut edges:[/] {precompute.cut_edge_count}")
    console.print(f"[bold]Output:[/] {output_dir}")


@app.command("build-base-image")
def build_skillflow_base_image(
    base_image_tag: str = typer.Option(
        "skillflow/harbor-cli-base:ubuntu24.04",
        help="Docker tag to build for the SkillFlow Harbor CLI base image.",
    ),
    dry_run: bool = typer.Option(
        False,
        help="Show the base image build command without running it.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build the required SkillFlow Harbor CLI base image."""
    _setup_logging(verbose)
    build_script = PROJECT_ROOT / "docker" / "harbor-cli-base" / "build.sh"

    if not build_script.is_file():
        console.print(f"[bold red]ERROR:[/] missing SkillFlow build script: {build_script}")
        raise typer.Exit(code=1)

    base_command = ["bash", str(build_script), base_image_tag]
    if dry_run:
        console.print(f"[bold]Would build base image:[/] {shlex.join(base_command)}")
        console.print("[bold green]SkillFlow base image dry run complete.[/]")
    else:
        _run_prebuild_step_or_exit(base_command, label="Build SkillFlow base image")
        console.print("[bold green]SkillFlow base image build complete.[/]")


@app.command("sync")
def sync_skillflow(
    tasks: Annotated[
        list[str] | None,
        typer.Option(
            "--tasks",
            "--task",
            "-t",
            help=(
                "Remote SkillFlow task ID(s) to download. Repeat the option, "
                "provide comma-separated IDs, or use 'all'."
            ),
        ),
    ] = None,
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Local tasks/ directory where SkillFlow task data should be downloaded.",
    ),
    dataset: str = typer.Option(
        DEFAULT_SKILLFLOW_DATASET,
        "--dataset",
        help="Hugging Face dataset ID.",
    ),
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download SkillFlow task data into the configured local cache."""
    _setup_logging(verbose)
    config = _load_config_or_bad_parameter(config_dir)
    config.executor_runtime.dataset = dataset
    repository = _build_benchmark_repo(PROJECT_ROOT, config)
    try:
        destination = repository.sync_tasks(
            destination=output_dir,
            task_ids=_sync_task_ids_from_repeatable_cli(tasks),
        )
    except SkillFlowSyncError as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[bold]Downloaded SkillFlow tasks to:[/] {destination}")


@app.command("list")
def list_skillflow_tasks(
    family: str | None = typer.Option(
        None,
        "--family",
        help="Filter task IDs by SkillFlow family.",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="List cached local tasks instead of remote Hugging Face tasks.",
    ),
    dataset: str = typer.Option(
        DEFAULT_SKILLFLOW_DATASET,
        "--dataset",
        help="Hugging Face dataset ID.",
    ),
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List available SkillFlow task IDs."""
    _setup_logging(verbose)
    config = _load_config_or_bad_parameter(config_dir)
    config.executor_runtime.dataset = dataset
    repository = _build_benchmark_repo(PROJECT_ROOT, config)
    try:
        if local:
            task_ids = repository.list_local_task_ids(family=family)
        else:
            task_ids = repository.list_remote_task_ids(family=family)
    except (FileNotFoundError, SkillFlowSyncError) as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
        raise typer.Exit(code=1) from exc
    for task_id in task_ids:
        typer.echo(task_id)


if __name__ == "__main__":
    app()
