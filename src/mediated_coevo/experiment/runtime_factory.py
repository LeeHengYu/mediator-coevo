"""Runtime object graph construction for experiment execution."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tomli_w

from mediated_coevo.agents.executor import ExecutorAgent
from mediated_coevo.agents.mediator import MediatorAgent
from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.benchmarks import (
    HarborRunner,
    SkillFlowRepository,
    SkillFlowSyncConfig,
)
from mediated_coevo.core.config import Config
from mediated_coevo.experiment.baselines import (
    BASELINE_PRESET_NAMES,
    get_baseline_preset,
)
from mediated_coevo.experiment.conditions import ConditionName
from mediated_coevo.experiment.orchestrator import Orchestrator
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.skill_store import SkillStore


@dataclass(frozen=True)
class ExperimentRuntime:
    """Objects needed to execute one configured experiment."""

    experiment_dir: Path
    orchestrator: Orchestrator


@dataclass(frozen=True)
class MatrixRuntime:
    """Runtime plus preset metadata for one matrix row."""

    preset_name: str
    runtime: ExperimentRuntime


def build_experiment(
    *,
    project_root: Path,
    config: Config,
    seed: int,
    condition_name: ConditionName,
    experiment_dir: Path | None = None,
    benchmark_repo: SkillFlowRepository | None = None,
    harbor_runner: HarborRunner | None = None,
) -> ExperimentRuntime:
    if experiment_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = config.experiment.baseline_preset or condition_name
        experiment_dir = (
            project_root
            / config.paths.data_dir
            / "experiments"
            / f"{timestamp}-{seed}-{suffix}"
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    runtime_skills_dir = shutil.copytree(
        project_root / config.paths.skills_dir,
        experiment_dir / "skills",
    )

    with open(experiment_dir / "config.toml", "wb") as f:
        tomli_w.dump(config.model_dump(exclude_none=True), f)

    if benchmark_repo is None:
        benchmark_repo = build_benchmark_repo(project_root, config)
    return build_experiment_runtime(
        config=config,
        condition_name=condition_name,
        runtime_skills_dir=runtime_skills_dir,
        experiment_dir=experiment_dir,
        benchmark_repo=benchmark_repo,
        harbor_runner=harbor_runner,
    )


def create_matrix_dir(
    *,
    project_root: Path,
    seed: int,
    data_dir: str,
    run_id: str | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = run_id or f"{seed}-baseline-matrix"
    matrix_dir = project_root / data_dir / "experiments" / f"{timestamp}-{suffix}"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    return matrix_dir


def build_benchmark_repo(project_root: Path, config: Config) -> SkillFlowRepository:
    return SkillFlowRepository(
        root_dir=project_root / config.paths.benchmarks_dir,
        task_dirs=config.executor_runtime.task_dirs,
        sync=SkillFlowSyncConfig(
            enabled=config.executor_runtime.sync_enabled,
            dataset=config.executor_runtime.dataset,
            repo_type=config.executor_runtime.dataset_repo_type,
            local_dir=config.executor_runtime.task_dirs[0],
            remote_task_cache_path=project_root / "docs" / "skillflow_tasks.txt",
        ),
        harbor_base_image=config.executor_runtime.harbor_base_image,
        legacy_harbor_base_images=config.executor_runtime.legacy_harbor_base_images,
    )


def build_experiment_runtime(
    *,
    config: Config,
    condition_name: ConditionName,
    runtime_skills_dir: Path,
    experiment_dir: Path,
    benchmark_repo: SkillFlowRepository,
    harbor_runner: HarborRunner | None = None,
) -> ExperimentRuntime:
    """Build one runtime from already materialized experiment directories."""
    from mediated_coevo.llm.client import LLMClient

    skill_store = SkillStore(runtime_skills_dir)
    skill_store.validate()
    artifact_store = ArtifactStore(base_dir=experiment_dir / "artifacts")
    if harbor_runner is None:
        jobs_dir = experiment_dir / config.executor_runtime.jobs_dir
        timeout_sec = config.executor_runtime.harbor_timeout_sec
        setup_timeout_multiplier = (
            config.executor_runtime.harbor_agent_setup_timeout_multiplier
        )
        harbor_runner = HarborRunner(
            jobs_dir=jobs_dir,
            timeout_sec=timeout_sec,
            agent_name=config.executor_runtime.agent_name,
            agent_env=config.executor_runtime.agent_env,
            agent_setup_timeout_multiplier=setup_timeout_multiplier,
            harbor_base_image=config.executor_runtime.harbor_base_image,
            legacy_harbor_base_images=config.executor_runtime.legacy_harbor_base_images,
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
    judge_client = LLMClient(model=config.models.judge)
    return ExperimentRuntime(
        experiment_dir=experiment_dir,
        orchestrator=Orchestrator(
            planner=planner,
            executor=executor,
            mediator=mediator,
            skill_store=skill_store,
            artifact_store=artifact_store,
            benchmark_repo=benchmark_repo,
            config=config,
            experiment_dir=experiment_dir,
            judge_llm_client=judge_client,
        ),
    )


def build_matrix_runtimes(
    *,
    project_root: Path,
    base_config: Config,
    seed: int,
    matrix_dir: Path,
    benchmark_repo: SkillFlowRepository,
    harbor_runner: HarborRunner | None = None,
    preset_names: Sequence[str] | None = None,
    flatten_single_row: bool = False,
) -> list[MatrixRuntime]:
    """Build matrix rows with isolated skill stores."""
    rows: list[MatrixRuntime] = []
    selected_preset_names = (
        BASELINE_PRESET_NAMES if preset_names is None else preset_names
    )
    if flatten_single_row and len(selected_preset_names) != 1:
        raise ValueError("flatten_single_row requires exactly one matrix preset")
    for preset_name in selected_preset_names:
        preset = get_baseline_preset(preset_name)
        row_config = preset.build_config(base_config, seed=seed)
        experiment_dir = matrix_dir if flatten_single_row else matrix_dir / preset_name
        runtime = build_experiment(
            project_root=project_root,
            config=row_config,
            seed=seed,
            condition_name=preset.condition_name,
            experiment_dir=experiment_dir,
            benchmark_repo=benchmark_repo,
            harbor_runner=harbor_runner,
        )
        rows.append(MatrixRuntime(preset_name=preset_name, runtime=runtime))
    return rows
