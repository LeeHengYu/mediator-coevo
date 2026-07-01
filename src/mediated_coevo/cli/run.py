"""Run command registration."""

from __future__ import annotations

import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated

import tomli_w
import typer

from mediated_coevo.cloud.vm import CloudVMConfigError, GCPVMConfig, load_vm_config
from mediated_coevo.core.config import Config
from mediated_coevo.cli.config import (
    _load_config_or_bad_parameter,
    _run_config_overrides,
    _task_ids_from_repeatable_cli,
)
from mediated_coevo.cli.experiment import (
    PROJECT_ROOT,
    TaskSelection,
    annotate_judge_rewards_or_exit,
    ensure_harbor_available,
    prepare_llm_credentials_or_exit,
    resolve_task_selection,
    run_experiment_or_exit,
    setup_logging,
    write_and_print_result_summary,
)
from mediated_coevo.cli.graph import materialize_task_graph_for_diffusion
from mediated_coevo.cli.output import (
    console,
    print_experiment_controls,
    print_task_selection,
)
from mediated_coevo.experiment.conditions import (
    ConditionName,
    ExperimentDesignError,
    validate_experiment_design,
)
from mediated_coevo.experiment.runtime_factory import (
    build_benchmark_repo,
    build_experiment_runtime,
)


def run_skillflow_experiment(
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

    prepare_llm_credentials_or_exit(config)
    if remote_harbor_config is None:
        ensure_harbor_available(config)
    elif shutil.which("gcloud") is None:
        console.print(
            "[bold red]ERROR:[/] gcloud CLI not found on PATH. Install the "
            "Google Cloud CLI before using --cloud."
        )
        raise typer.Exit(code=1)

    resolved_run_id = (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
        f"{run_id or f'{seed}-skillflow'}"
    )
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

    benchmark_repo = build_benchmark_repo(PROJECT_ROOT, config)
    try:
        materialize_task_graph_for_diffusion(
            config=config,
            experiment_dir=experiment_dir,
            benchmark_repo=benchmark_repo,
        )
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]ERROR:[/] failed to create diffusion task graph: {exc}")
        raise typer.Exit(code=1) from exc
    runtime = build_experiment_runtime(
        config=config,
        condition_name=condition_name,
        runtime_skills_dir=runtime_skills_dir,
        experiment_dir=experiment_dir,
        benchmark_repo=benchmark_repo,
        remote_harbor_config=remote_harbor_config,
    )

    print_task_selection(selection)
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
    print_experiment_controls(config)
    console.print(
        "[bold]Models:[/] "
        f"planner={config.models.planner} "
        f"executor={config.models.executor} "
        f"mediator={config.models.mediator} "
        f"judge={config.models.judge}"
    )
    console.print(f"\n[bold green]Starting experiment:[/] {runtime.experiment_dir}\n")
    records = run_experiment_or_exit(runtime, selection.task_ids, iterations)
    write_and_print_result_summary(
        records=records,
        data_dir=runtime.experiment_dir,
        header="Results",
    )
    annotate_judge_rewards_or_exit(
        data_dir=runtime.experiment_dir,
        config=config,
        history_store=runtime.orchestrator.history_store,
    )


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
                "random_k | top_k_similarity | llm_router_softmax."
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
    setup_logging(verbose)
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
    try:
        validate_experiment_design(
            condition=config.experiment.condition_name,
            skill_updates=config.experiment.skill_updates,
            baseline_preset=config.experiment.baseline_preset,
        )
    except ExperimentDesignError as exc:
        raise typer.BadParameter(str(exc)) from exc
    repository = build_benchmark_repo(PROJECT_ROOT, config)
    selection = resolve_task_selection(
        repository=repository,
        tasks=_task_ids_from_repeatable_cli(tasks),
        family=family,
        task_set=task_set,
    )
    if cloud:
        try:
            remote_harbor_config = load_vm_config(cloud_env_file)
        except (OSError, CloudVMConfigError) as exc:
            raise typer.BadParameter(str(exc)) from exc
    else:
        remote_harbor_config = None
    run_skillflow_experiment(
        config=config,
        selection=selection,
        iterations=config.experiment.num_iterations,
        seed=config.experiment.seed,
        condition_name=config.experiment.condition_name,
        run_id=run_id,
        remote_harbor_config=remote_harbor_config,
    )


def register_run_command(app: typer.Typer) -> None:
    app.command()(run)
