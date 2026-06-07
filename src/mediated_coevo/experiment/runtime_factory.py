"""Runtime object graph construction for experiment execution."""

from __future__ import annotations

import shutil
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
from mediated_coevo.cloud.vm import GCPVMConfig, RemoteHarborRunner
from mediated_coevo.core.config import Config
from mediated_coevo.evolution.skill_advisor import SkillAdvisor
from mediated_coevo.experiment.baselines import (
    BASELINE_PRESET_NAMES,
    get_baseline_preset,
)
from mediated_coevo.experiment.conditions import (
    ConditionName,
    validate_experiment_design,
)
from mediated_coevo.experiment.orchestrator import Orchestrator
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from mediated_coevo.stores.skill_store import SkillStore


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

        if benchmark_repo is None:
            benchmark_repo = build_benchmark_repo(self._project_root, config)
        return build_experiment_runtime(
            config=config,
            condition_name=condition_name,
            runtime_skills_dir=runtime_skills_dir,
            experiment_dir=experiment_dir,
            benchmark_repo=benchmark_repo,
            remote_harbor_config=None,
            harbor_runner=harbor_runner,
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
    )


def build_experiment_runtime(
    *,
    config: Config,
    condition_name: ConditionName,
    runtime_skills_dir: Path,
    experiment_dir: Path,
    benchmark_repo: SkillFlowRepository,
    remote_harbor_config: GCPVMConfig | None,
    harbor_runner: HarborRunner | RemoteHarborRunner | None = None,
) -> ExperimentRuntime:
    """Build one runtime from already materialized experiment directories."""
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
    if harbor_runner is None:
        jobs_dir = experiment_dir / config.executor_runtime.jobs_dir
        timeout_sec = config.executor_runtime.harbor_timeout_sec
        setup_timeout_multiplier = (
            config.executor_runtime.harbor_agent_setup_timeout_multiplier
        )
        if remote_harbor_config is not None:
            harbor_runner = RemoteHarborRunner(
                config=remote_harbor_config,
                jobs_dir=jobs_dir,
                timeout_sec=timeout_sec,
                agent_setup_timeout_multiplier=setup_timeout_multiplier,
            )
        else:
            harbor_runner = HarborRunner(
                jobs_dir=jobs_dir,
                timeout_sec=timeout_sec,
                agent_setup_timeout_multiplier=setup_timeout_multiplier,
            )

    planner = PlannerAgent(llm_client=LLMClient(model=config.models.planner))
    planner.configure_token_budget(
        config.budgets,
        condition_name=config.experiment.condition_name,
    )
    planner.configure_skill_updates(config.experiment.skill_updates)
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
    mediator.configure_skill_updates(config.experiment.skill_updates)
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


def build_matrix_runtimes(
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
