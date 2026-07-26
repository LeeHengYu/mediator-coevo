"""Matrix command registration."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Annotated

import typer

from mediated_coevo.cli.config import (
    _load_config_or_bad_parameter,
    _run_config_overrides,
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
from mediated_coevo.cli.graph import materialize_task_graph_for_diffusion
from mediated_coevo.cli.output import (
    console,
    print_experiment_controls,
    print_task_selection,
)
from mediated_coevo.experiment.baselines import (
    BASELINE_PRESET_NAMES,
    BASELINE_PRESETS,
    get_baseline_preset,
)
from mediated_coevo.experiment.runtime_factory import (
    build_benchmark_repo,
    build_matrix_runtimes,
    create_matrix_dir,
)


def matrix(
    family: Annotated[
        list[str] | None,
        typer.Option(
            "--family",
            help="Benchmark family to bootstrap into a stream. Repeat for multiple.",
        ),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option(
            "--split",
            help="Optional task split to sample from: train | validation | test.",
        ),
    ] = None,
    iterations: int | None = typer.Option(
        None,
        help="Number of iterations per row. Overrides experiment.num_iterations.",
    ),
    seed: int | None = typer.Option(
        None,
        help=(
            "Experiment and deterministic task-split seed reused for every row. "
            "Overrides experiment.seed."
        ),
    ),
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
            help=("Rejected for matrix runs; diffusion.policy is row-local."),
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
    row_indexes: Annotated[
        str | None,
        typer.Option(
            "--index",
            "-i",
            help=(
                "Run only selected zero-based matrix row indexes. "
                "Use comma-separated values, e.g. 1,3."
            ),
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help=(
                "Matrix run ID suffix. The parent matrix directory is prefixed "
                "with a timestamp."
            ),
        ),
    ] = None,
    save_artifacts: Annotated[
        bool,
        typer.Option(
            "--save",
            help="Save each completed diffusion artifact store locally.",
        ),
    ] = False,
    artifact_store: Annotated[
        Path | None,
        typer.Option(
            "--artifact",
            help="Saved diffusion artifact store to preload into each row.",
        ),
    ] = None,
    freeze_artifacts: Annotated[
        bool,
        typer.Option(
            "--freeze",
            "-f",
            help="Freeze the preloaded diffusion artifact store.",
        ),
    ] = False,
    config_dir: Path = typer.Option(PROJECT_ROOT / "config", help="Config directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the four-row fixed-skill diffusion matrix."""
    setup_logging(verbose)
    if list_rows:
        console.print("[bold]Matrix rows:[/]")
        for index, preset in enumerate(BASELINE_PRESETS):
            console.print(
                f"  {index}: {preset.name} "
                f"(diffusion policy: {preset.diffusion_policy}, "
                f"diffusion graph: {preset.diffusion_graph})",
                soft_wrap=True,
            )
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
    if not family:
        raise typer.BadParameter("provide --family")
    selected_indexes = _parse_matrix_row_indexes(row_indexes)
    if selected_indexes is None:
        preset_names = list(BASELINE_PRESET_NAMES)
    else:
        preset_names = [BASELINE_PRESET_NAMES[index] for index in selected_indexes]
    _validate_artifact_store_options(
        preset_names=preset_names,
        save_artifacts=save_artifacts,
        artifact_store=artifact_store,
        freeze_artifacts=freeze_artifacts,
    )
    config = _load_config_or_bad_parameter(
        config_dir,
        overrides=_run_config_overrides(
            iterations=iterations,
            seed=seed,
            condition=None,
            diffusion_enabled=None,
            diffusion_policy=None,
            diffusion_graph=None,
            diffusion_max_artifacts=diffusion_max_artifacts,
            diffusion_top_k_neighbors=diffusion_top_k_neighbors,
            harbor_agent_setup_timeout_multiplier=None,
        ),
    )
    prepare_llm_credentials_or_exit(config)
    ensure_harbor_available(config)
    repository = build_benchmark_repo(PROJECT_ROOT, config)
    selection = resolve_task_selection(
        repository=repository,
        family=family,
        seed=config.experiment.seed,
        split=split,
    )
    config.experiment.benchmark_selection.tasks = selection.task_ids
    config.experiment.benchmark_selection.family = selection.family_label
    config.experiment.benchmark_selection.split = selection.split
    config.experiment.benchmark_selection.task_stream_seed = selection.task_stream_seed
    seed = config.experiment.seed
    iterations = config.experiment.num_iterations
    matrix_dir = create_matrix_dir(
        project_root=PROJECT_ROOT,
        seed=seed,
        data_dir=config.paths.data_dir,
        run_id=run_id,
    )
    rows = build_matrix_runtimes(
        project_root=PROJECT_ROOT,
        base_config=config,
        seed=seed,
        matrix_dir=matrix_dir,
        benchmark_repo=repository,
        preset_names=preset_names,
        flatten_single_row=selected_indexes is not None and len(selected_indexes) == 1,
    )

    print_task_selection(selection)
    if selection.families:
        console.print(f"[bold]Task stream length per row:[/] {len(selection.task_ids)}")
    else:
        console.print(f"[bold]Iterations per row:[/] {iterations}")
    console.print(f"[bold]Seed per row:[/] {seed}")
    print_experiment_controls(config)
    console.print(f"[bold]Matrix:[/] {matrix_dir}")
    console.print(f"[bold]Rows:[/] {', '.join(preset_names)}")
    console.print("[bold]Row-local diffusion:[/] enabled, policy, graph")

    artifact_store = artifact_store.expanduser() if artifact_store is not None else None
    saved_store_root = PROJECT_ROOT / config.paths.data_dir / "artifact-stores"
    for row in rows:
        row_config = row.runtime.orchestrator.config
        imported_count = 0
        if artifact_store is not None:
            try:
                imported_count = (
                    row.runtime.orchestrator._diffusion_store.import_artifact_store(
                        artifact_store,
                        frozen=freeze_artifacts,
                    )
                )
            except (OSError, ValueError) as exc:
                console.print(
                    f"[bold red]ERROR:[/] failed to preload artifact store: {exc}"
                )
                raise typer.Exit(code=1) from exc
            row.runtime.orchestrator.preloaded_diffusion_artifact_store_path = str(
                artifact_store
            )
            row.runtime.orchestrator.preloaded_diffusion_artifact_store_count = (
                imported_count
            )
            row.runtime.orchestrator.freeze_diffusion_artifact_store = freeze_artifacts
            console.print(
                f"[bold]Preloaded diffusion artifacts:[/] {imported_count} "
                f"from {artifact_store}"
            )
        _write_matrix_invocation_metadata(
            row_dir=row.runtime.experiment_dir,
            family=selection.family_label,
            split=getattr(selection, "split", None),
            selected_task_ids=selection.task_ids,
            iterations=iterations,
            seed=seed,
            selected_indexes=selected_indexes,
            row_indexes_argument=row_indexes,
            run_id=run_id,
            preset_name=row.preset_name,
            config_dir=config_dir,
            artifact_store=artifact_store,
            imported_artifact_count=imported_count,
            freeze_artifacts=freeze_artifacts,
            save_artifacts=save_artifacts,
            diffusion_max_artifacts=diffusion_max_artifacts,
            diffusion_top_k_neighbors=diffusion_top_k_neighbors,
        )
        random.seed(seed)
        try:
            materialize_task_graph_for_diffusion(
                config=row_config,
                experiment_dir=row.runtime.experiment_dir,
                benchmark_repo=repository,
            )
        except (OSError, ValueError) as exc:
            console.print(
                f"[bold red]ERROR:[/] failed to create diffusion task graph: {exc}"
            )
            raise typer.Exit(code=1) from exc
        console.print(
            "\n[bold green]Starting matrix row:[/] "
            f"{row.preset_name} "
            f"(condition={row_config.experiment.condition_name}, "
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
        )
        if save_artifacts:
            destination = saved_store_root / row.runtime.experiment_dir.name
            try:
                saved_count = (
                    row.runtime.orchestrator._diffusion_store.save_artifact_store(
                        destination,
                        store_id=row.runtime.experiment_dir.name,
                    )
                )
            except (OSError, ValueError) as exc:
                console.print(
                    f"[bold red]ERROR:[/] failed to save artifact store: {exc}"
                )
                raise typer.Exit(code=1) from exc
            console.print(
                f"[bold]Saved diffusion artifact store:[/] {destination} "
                f"({saved_count} artifacts)"
            )
    console.print(f"\n[bold]Matrix data:[/] {matrix_dir}")


def _parse_matrix_row_indexes(value: str | None) -> list[int] | None:
    if value is None:
        return None
    indexes: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            raise typer.BadParameter("matrix row indexes cannot be empty")
        try:
            index = int(token)
        except ValueError as exc:
            raise typer.BadParameter(
                "matrix row indexes must be comma-separated integers"
            ) from exc
        if index < 0 or index >= len(BASELINE_PRESET_NAMES):
            raise typer.BadParameter(
                "matrix row indexes must be between "
                f"0 and {len(BASELINE_PRESET_NAMES) - 1}"
            )
        if index in indexes:
            raise typer.BadParameter("matrix row indexes cannot repeat")
        indexes.append(index)
    return indexes


def _write_matrix_invocation_metadata(
    *,
    row_dir: Path,
    family: str,
    split: str | None,
    selected_task_ids: list[str],
    iterations: int,
    seed: int,
    selected_indexes: list[int] | None,
    row_indexes_argument: str | None,
    run_id: str | None,
    preset_name: str,
    config_dir: Path,
    artifact_store: Path | None,
    imported_artifact_count: int,
    freeze_artifacts: bool,
    save_artifacts: bool,
    diffusion_max_artifacts: int | None,
    diffusion_top_k_neighbors: int | None,
) -> None:
    """Persist matrix CLI inputs that are not fully represented in config.toml."""
    payload = {
        "family": family,
        "split": split,
        "selected_task_ids": selected_task_ids,
        "iterations": iterations,
        "seed": seed,
        "selected_indexes": selected_indexes,
        "row_indexes_argument": row_indexes_argument,
        "run_id": run_id,
        "preset_name": preset_name,
        "config_dir": str(config_dir),
        "artifact_store": str(artifact_store) if artifact_store is not None else None,
        "imported_artifact_count": imported_artifact_count,
        "freeze_artifacts": freeze_artifacts,
        "save_artifacts": save_artifacts,
        "diffusion_max_artifacts": diffusion_max_artifacts,
        "diffusion_top_k_neighbors": diffusion_top_k_neighbors,
    }
    row_dir.mkdir(parents=True, exist_ok=True)
    (row_dir / "matrix_invocation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_artifact_store_options(
    *,
    preset_names: list[str],
    save_artifacts: bool,
    artifact_store: Path | None,
    freeze_artifacts: bool,
) -> None:
    if freeze_artifacts and artifact_store is None:
        raise typer.BadParameter("--freeze requires --artifact")
    if not (save_artifacts or artifact_store is not None or freeze_artifacts):
        return

    disabled = [
        preset_name
        for preset_name in preset_names
        if not get_baseline_preset(preset_name).diffusion_enabled
    ]
    if disabled:
        raise typer.BadParameter(
            "artifact store options require diffusion-enabled rows; "
            f"disabled rows: {', '.join(disabled)}"
        )


def register_matrix_command(app: typer.Typer) -> None:
    app.command()(matrix)
