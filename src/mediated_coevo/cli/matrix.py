"""Matrix command registration."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Annotated

import typer

from mediated_coevo.cli.config import (
    _load_config_or_bad_parameter,
    _run_config_overrides,
    _task_ids_from_repeatable_cli,
)
from mediated_coevo.cli.experiment import (
    PROJECT_ROOT,
    annotate_judge_rewards_or_exit,
    ensure_harbor_available,
    prepare_llm_credentials_or_exit,
    resolve_task_selection,
    run_experiment_or_exit,
    setup_logging,
    write_and_print_result_summary,
)
from mediated_coevo.cli.output import (
    console,
    print_experiment_controls,
    print_task_selection,
)
from mediated_coevo.cli.graph import materialize_task_graph_for_diffusion
from mediated_coevo.experiment.baselines import (
    BASELINE_PRESETS,
    BASELINE_PRESET_NAMES,
    get_baseline_preset,
)
from mediated_coevo.experiment.conditions import validate_experiment_design
from mediated_coevo.experiment.runtime_factory import (
    ExperimentFactory,
    build_benchmark_repo,
    build_matrix_runtimes,
)


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
            help="Rejected for matrix runs; diffusion.enabled is row-local.",
        ),
    ] = None,
    diffusion_policy: Annotated[
        str | None,
        typer.Option(
            "--diffusion-policy",
            help=(
                "Rejected for matrix runs; diffusion.policy is row-local."
            ),
        ),
    ] = None,
    diffusion_graph: Annotated[
        str | None,
        typer.Option(
            "--diffusion-graph",
            help="Rejected for matrix runs; diffusion.graph is row-local.",
        ),
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
    list_rows: Annotated[
        bool,
        typer.Option(
            "--list",
            "-l",
            help="List matrix row indexes and row-local config, then exit.",
        ),
    ] = False,
    row_index: Annotated[
        int | None,
        typer.Option(
            "--index",
            "-i",
            min=0,
            max=len(BASELINE_PRESET_NAMES) - 1,
            help="Run only the matrix row at this zero-based index.",
        ),
    ] = None,
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the eight-row learned-mediator diffusion matrix."""
    setup_logging(verbose)
    if list_rows:
        _print_matrix_rows()
        return

    if (
        diffusion_enabled is not None
        or diffusion_policy is not None
        or diffusion_graph is not None
    ):
        raise typer.BadParameter(
            "matrix rows set diffusion.enabled, diffusion.policy, and "
            "diffusion.graph; use --diffusion-max-artifacts or "
            "--diffusion-top-k-neighbors for shared matrix knobs"
        )
    preset_names = _selected_preset_names(row_index)
    config = _load_config_or_bad_parameter(
        config_dir,
        overrides=_run_config_overrides(
            iterations=iterations,
            seed=seed,
            condition=None,
            skill_updates=None,
            coevo_interval=coevo_interval,
            advisor_buffer_max=advisor_buffer_max,
            diffusion_enabled=None,
            diffusion_policy=None,
            diffusion_graph=None,
            diffusion_max_artifacts=diffusion_max_artifacts,
            diffusion_top_k_neighbors=diffusion_top_k_neighbors,
            harbor_agent_setup_timeout_multiplier=None,
        ),
    )
    for preset_name in preset_names:
        preset = get_baseline_preset(preset_name)
        validate_experiment_design(
            condition=preset.condition_name,
            skill_updates=preset.skill_updates,
            baseline_preset=preset.name,
        )
    prepare_llm_credentials_or_exit(config)
    ensure_harbor_available(config)
    repository = build_benchmark_repo(PROJECT_ROOT, config)
    selection = resolve_task_selection(
        repository=repository,
        tasks=_task_ids_from_repeatable_cli(tasks),
        family=family,
        task_set=task_set,
    )
    factory = ExperimentFactory(PROJECT_ROOT)
    seed = config.experiment.seed
    iterations = config.experiment.num_iterations
    matrix_dir = factory.create_matrix_dir(seed=seed, data_dir=config.paths.data_dir)
    rows = build_matrix_runtimes(
        factory=factory,
        base_config=config,
        seed=seed,
        matrix_dir=matrix_dir,
        benchmark_repo=repository,
        preset_names=preset_names,
    )

    print_task_selection(selection)
    console.print(f"[bold]Iterations per row:[/] {iterations}")
    console.print(f"[bold]Seed per row:[/] {seed}")
    print_experiment_controls(config)
    console.print(f"[bold]Matrix:[/] {matrix_dir}")
    console.print(f"[bold]Rows:[/] {', '.join(preset_names)}")
    console.print("[bold]Row-local diffusion:[/] enabled, policy, graph")

    for row in rows:
        row_config = row.runtime.orchestrator.config
        random.seed(seed)
        try:
            materialize_task_graph_for_diffusion(
                config=row_config,
                experiment_dir=row.runtime.experiment_dir,
                benchmark_repo=repository,
            )
        except (OSError, ValueError) as exc:
            console.print(
                "[bold red]ERROR:[/] failed to create diffusion task graph: "
                f"{exc}"
            )
            raise typer.Exit(code=1) from exc
        console.print(
            "\n[bold green]Starting matrix row:[/] "
            f"{row.preset_name} "
            f"(condition={row_config.experiment.condition_name}, "
            f"skill_updates={row_config.experiment.skill_updates.model_dump()}, "
            f"diffusion_enabled={row_config.diffusion.enabled}, "
            f"diffusion_policy={row_config.diffusion.policy}, "
            f"diffusion_graph={row_config.diffusion.graph})"
        )
        records = run_experiment_or_exit(
            row.runtime,
            selection.task_ids,
            iterations,
        )
        write_and_print_result_summary(
            records=records,
            data_dir=row.runtime.experiment_dir,
            header=f"Row results: {row.preset_name}",
        )
        annotate_judge_rewards_or_exit(
            data_dir=row.runtime.experiment_dir,
            config=row_config,
            history_store=row.runtime.orchestrator.history_store,
        )
    console.print(f"\n[bold]Matrix data:[/] {matrix_dir}")


def _print_matrix_rows() -> None:
    console.print("[bold]Matrix rows:[/]")
    for index, preset in enumerate(BASELINE_PRESETS):
        skill_updates = preset.skill_updates.model_dump()
        enabled_roles = [
            role for role, enabled in skill_updates.items() if enabled
        ]
        if not enabled_roles:
            skill_update_label = "none"
        elif len(enabled_roles) == len(skill_updates):
            skill_update_label = "all"
        else:
            skill_update_label = ",".join(enabled_roles)

        console.print(
            f"  {index}: {preset.name} "
            f"(skill update: {skill_update_label}, "
            f"diffusion policy: {preset.diffusion_policy}, "
            f"diffusion graph: {preset.diffusion_graph})",
            soft_wrap=True,
        )


def _selected_preset_names(row_index: int | None) -> list[str]:
    if row_index is None:
        return list(BASELINE_PRESET_NAMES)
    if row_index < 0 or row_index >= len(BASELINE_PRESET_NAMES):
        raise typer.BadParameter(
            f"matrix row index must be between 0 and {len(BASELINE_PRESET_NAMES) - 1}"
        )
    return [BASELINE_PRESET_NAMES[row_index]]


def register_matrix_command(app: typer.Typer) -> None:
    app.command()(matrix)
