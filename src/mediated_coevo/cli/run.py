"""Run command registration."""

from __future__ import annotations

import os
import random
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated

import tomli_w
import typer

from mediated_coevo.core.config import Config
from mediated_coevo.cli.config import (
    _load_config_or_bad_parameter,
    _run_config_overrides,
)
from mediated_coevo.cli.experiment import (
    PROJECT_ROOT,
    TaskSelection,
    annotate_judge_rewards_or_exit,
    ensure_harbor_available,
    load_task_manifest_selection,
    prepare_llm_credentials_or_exit,
    resolve_task_selection,
    run_experiment_or_exit,
    setup_logging,
    write_and_print_result_summary,
)
from mediated_coevo.cli.graph import materialize_task_graph_for_diffusion
from mediated_coevo.cli.harness_registry import (
    RuntimeStateSource,
    _apply_harness_overlay_with_backup,
    _copy_explicit_state,
    _experiment_split,
    _prepare_harness_workspace,
    _publish_graph_state_ref,
    _publish_promoted_harness,
    _resolve_harness_options,
    _resolve_state_options,
    _restore_harness_overlay_backup,
)
from mediated_coevo.cli.output import (
    console,
    print_experiment_controls,
    print_task_selection,
)
from mediated_coevo.experiment.conditions import ConditionName
from mediated_coevo.diffusion.store import DiffusionStore
from mediated_coevo.experiment.runtime_factory import (
    build_benchmark_repo,
    build_experiment_runtime,
)

_HARNESS_APPLIED_ENV = "MEDCOEVO_HARNESS_APPLIED"
_HARNESS_BACKUP_ENV = "MEDCOEVO_HARNESS_BACKUP"


def run_skillflow_experiment(
    *,
    config: Config,
    selection: TaskSelection,
    iterations: int,
    seed: int,
    condition_name: ConditionName,
    run_id: str | None,
    harness_dir: Path | None = None,
    harness_ref: str | None = None,
    state_source: RuntimeStateSource | None = None,
    publish_state_ref: str | None = None,
    artifact_store_dir: Path | None = None,
) -> Path:
    """Run a SkillFlow selection locally."""
    random.seed(seed)
    config.experiment.benchmark_selection.tasks = selection.task_ids
    config.experiment.benchmark_selection.family = selection.family
    config.experiment.benchmark_selection.split = selection.split
    config.experiment.benchmark_selection.task_stream_seed = selection.task_stream_seed

    prepare_llm_credentials_or_exit(config)
    ensure_harbor_available(config)

    resolved_run_id = (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{run_id or f'{seed}-skillflow'}"
    )
    experiment_dir = (
        PROJECT_ROOT / config.paths.data_dir / "experiments" / resolved_run_id
    )
    experiment_dir.mkdir(parents=True, exist_ok=False)
    runtime_skills_dir = shutil.copytree(
        PROJECT_ROOT / config.paths.skills_dir,
        experiment_dir / "skills",
    )
    _prepare_harness_workspace(
        experiment_dir,
        harness_dir,
        harness_ref=harness_ref,
    )
    _copy_explicit_state(experiment_dir, state_source)
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
        console.print(
            f"[bold red]ERROR:[/] failed to create diffusion task graph: {exc}"
        )
        raise typer.Exit(code=1) from exc
    runtime = build_experiment_runtime(
        config=config,
        condition_name=condition_name,
        runtime_skills_dir=runtime_skills_dir,
        experiment_dir=experiment_dir,
        benchmark_repo=benchmark_repo,
    )

    print_task_selection(selection)
    if selection.family is not None:
        console.print(f"[bold]Task stream length:[/] {len(selection.task_ids)}")
    else:
        console.print(f"[bold]Iterations:[/] {iterations}")
    console.print(f"[bold]Condition:[/] {condition_name}")
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
    )
    if artifact_store_dir is not None:
        runtime.orchestrator.diffusion_store.save_artifact_store(
            artifact_store_dir,
            store_id=selection.task_ids[0],
        )
        DiffusionStore.load_artifact_store(
            artifact_store_dir,
            expected_store_id=selection.task_ids[0],
        )
    if publish_state_ref is not None:
        _publish_graph_state_ref(
            publish_state_ref,
            experiment_dir=runtime.experiment_dir,
            split=selection.split,
        )
    return runtime.experiment_dir


def _remove_base_artifact_experiment(experiment_dir: Path, *, data_dir: str) -> None:
    experiments_root = (PROJECT_ROOT / data_dir / "experiments").resolve()
    resolved = experiment_dir.resolve()
    if resolved.parent != experiments_root or "base-artifact-" not in resolved.name:
        raise ValueError(f"refusing to remove non-base-artifact run: {resolved}")
    shutil.rmtree(resolved)


def base_artifacts(
    family: Annotated[
        list[str],
        typer.Option(
            "--family",
            help="SkillFlow family to pre-run. Repeat for multiple families.",
        ),
    ],
    seed: Annotated[int, typer.Option(help="Experiment seed.")] = 0,
    config_dir: Annotated[
        Path,
        typer.Option(help="Config directory"),
    ] = PROJECT_ROOT / "config",
    output_dir: Annotated[
        Path,
        typer.Option(help="Portable artifact store root."),
    ] = PROJECT_ROOT / "data" / "base_artifacts",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Pre-run each task independently and keep only its artifact store."""
    setup_logging(verbose)
    config = _load_config_or_bad_parameter(
        config_dir,
        overrides={
            "experiment": {
                "num_iterations": 1,
                "seed": seed,
                "condition_name": "learned_mediator",
                "baseline_preset": None,
            },
            "diffusion": {
                "enabled": True,
                "policy": "random_k",
                "graph": "none",
            },
        },
    )
    repository = build_benchmark_repo(PROJECT_ROOT, config)
    families = list(dict.fromkeys(name.strip() for name in family if name.strip()))
    if not families:
        raise typer.BadParameter("provide --family")
    for family_name in families:
        task_ids = repository.list_local_task_ids(family=family_name)
        if not task_ids:
            raise typer.BadParameter(
                f"no local SkillFlow tasks found for family {family_name!r}"
            )
        for task_id in task_ids:
            destination = output_dir / task_id
            if (destination / "manifest.json").is_file():
                DiffusionStore.load_artifact_store(
                    destination,
                    expected_store_id=task_id,
                )
                console.print(f"[dim]Already exists:[/] {destination}")
                continue
            experiment_dir = run_skillflow_experiment(
                config=config.model_copy(deep=True),
                selection=TaskSelection(
                    task_ids=[task_id],
                    families=(family_name,),
                    task_stream_seed=seed,
                ),
                iterations=1,
                seed=seed,
                condition_name="learned_mediator",
                run_id=f"base-artifact-{task_id.replace('/', '-')}",
                artifact_store_dir=destination,
            )
            _remove_base_artifact_experiment(
                experiment_dir,
                data_dir=config.paths.data_dir,
            )
            console.print(f"[bold green]Saved:[/] {destination}")


def _apply_harness_overlay_and_reexec(harness_dir: Path) -> None:
    resolved = harness_dir.expanduser().resolve()
    if os.environ.get(_HARNESS_APPLIED_ENV) == str(resolved):
        return
    backup_dir = Path(tempfile.mkdtemp(prefix="mediated-coevo-harness-"))
    applied_files = _apply_harness_overlay_with_backup(
        resolved,
        PROJECT_ROOT,
        backup_dir,
    )
    typer.echo(f"Applied harness overlay: {resolved} ({len(applied_files)} files)")
    env = os.environ.copy()
    env[_HARNESS_APPLIED_ENV] = str(resolved)
    env[_HARNESS_BACKUP_ENV] = str(backup_dir)
    try:
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
    except OSError:
        _restore_harness_overlay_backup(PROJECT_ROOT, backup_dir)
        raise


def _restore_scoped_harness_overlay() -> None:
    backup_value = os.environ.pop(_HARNESS_BACKUP_ENV, None)
    os.environ.pop(_HARNESS_APPLIED_ENV, None)
    if backup_value is None:
        return
    _restore_harness_overlay_backup(PROJECT_ROOT, Path(backup_value))


def run(
    family: Annotated[
        list[str] | None,
        typer.Option(
            "--family",
            help="SkillFlow family to bootstrap into a stream. Repeat for multiple.",
        ),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option(
            "--split",
            help="Optional task split to sample from: train | validation | test.",
        ),
    ] = None,
    task_manifest: Annotated[
        Path | None,
        typer.Option(
            "--task-manifest",
            help=(
                "Frozen JSON task stream to replay exactly. Preserves order and "
                "duplicates; cannot be combined with --family."
            ),
        ),
    ] = None,
    iterations: Annotated[
        int | None,
        typer.Option(help="Number of iterations. Overrides experiment.num_iterations."),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            help=(
                "Experiment seed and deterministic task-split seed. "
                "Overrides experiment.seed."
            ),
        ),
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
                "random_k | top_k_similarity | langchain_graph."
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
    harness_dir: Annotated[
        Path | None,
        typer.Option(
            "--harness-dir",
            help=(
                "Repo-root harness overlay to apply before this run. Runtime graph "
                "state is not loaded from this path; use --state-dir or --state-ref."
            ),
        ),
    ] = None,
    harness_ref: Annotated[
        str | None,
        typer.Option(
            "--harness-ref",
            help=(
                "Harness registry reference, for example promoted:HL5 or "
                "promoted:HL5@update_0002."
            ),
        ),
    ] = None,
    state_dir: Annotated[
        Path | None,
        typer.Option(
            "--state-dir",
            help=(
                "Explicit runtime state source containing diffusion/ or "
                "state/diffusion/. Does not carry diffusion/artifacts."
            ),
        ),
    ] = None,
    state_ref: Annotated[
        str | None,
        typer.Option(
            "--state-ref",
            help=(
                "Promotion-registry state reference, for example latest-graph:HL3 "
                "or promoted:HL3."
            ),
        ),
    ] = None,
    publish_state_ref: Annotated[
        str | None,
        typer.Option(
            "--publish-state-ref",
            help=(
                "After a successful train split run, publish this run's graph state "
                "to a registry reference such as latest-graph:HL3."
            ),
        ),
    ] = None,
    config_dir: Annotated[
        Path,
        typer.Option(help="Config directory"),
    ] = PROJECT_ROOT / "config",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run a fixed-skill SkillFlow experiment."""
    resolved_harness_dir = _resolve_harness_options(
        harness_dir,
        harness_ref,
        applied_dir=os.environ.get(_HARNESS_APPLIED_ENV),
    )
    state_source = _resolve_state_options(state_dir, state_ref)
    if resolved_harness_dir is not None:
        _apply_harness_overlay_and_reexec(resolved_harness_dir)
    try:
        setup_logging(verbose)
        config = _load_config_or_bad_parameter(
            config_dir,
            overrides=_run_config_overrides(
                iterations=iterations,
                seed=seed,
                condition=condition,
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
        repository = build_benchmark_repo(PROJECT_ROOT, config)
        if task_manifest is not None:
            if family:
                raise typer.BadParameter(
                    "--task-manifest cannot be combined with --family"
                )
            selection = load_task_manifest_selection(
                repository=repository,
                manifest_path=task_manifest,
            )
            if split is not None and split.strip().lower() != selection.split:
                raise typer.BadParameter(
                    "--split must match the split declared in --task-manifest"
                )
        else:
            selection = resolve_task_selection(
                repository=repository,
                family=family,
                seed=config.experiment.seed,
                split=split,
            )
        if publish_state_ref is not None and selection.split != "train":
            raise typer.BadParameter(
                "--publish-state-ref may only publish graph state from --split train runs"
            )
        run_skillflow_experiment(
            config=config,
            selection=selection,
            iterations=config.experiment.num_iterations,
            seed=config.experiment.seed,
            condition_name=config.experiment.condition_name,
            run_id=run_id,
            harness_dir=resolved_harness_dir,
            harness_ref=harness_ref,
            state_source=state_source,
            publish_state_ref=publish_state_ref,
        )
    finally:
        _restore_scoped_harness_overlay()


def publish_harness(
    campaign: Annotated[
        str,
        typer.Option(
            "--campaign",
            help="HL campaign registry name, for example HL3.",
        ),
    ],
    harness_dir: Annotated[
        Path,
        typer.Option(
            "--harness-dir",
            help="Agent-owned data/experiments/<campaign>/update_XXXX directory.",
        ),
    ],
    source_sequence: Annotated[
        Path | None,
        typer.Option(
            "--source-sequence",
            help="Sequence run whose logs informed this harness update.",
        ),
    ] = None,
    validation_run: Annotated[
        str | None,
        typer.Option(
            "--validation-run",
            help="Validation run path or ID that justified the promotion.",
        ),
    ] = None,
    state_dir: Annotated[
        Path | None,
        typer.Option(
            "--state-dir",
            help=(
                "Optional explicit state source to record with the promoted "
                "harness. Prefer the latest-graph channel for graph carry-forward."
            ),
        ),
    ] = None,
) -> None:
    """Register an agent-owned harness update as the campaign's latest version."""
    channel_path = _publish_promoted_harness(
        campaign=campaign,
        harness_dir=harness_dir,
        validation_run=validation_run,
        state_dir=state_dir,
        source_sequence=source_sequence,
    )
    console.print(f"[bold]Published promoted harness:[/] {channel_path}")


def publish_graph_state(
    campaign: Annotated[
        str,
        typer.Option(
            "--campaign",
            help="HL campaign registry name, for example HL3.",
        ),
    ],
    experiment_dir: Annotated[
        Path,
        typer.Option(
            "--experiment-dir",
            help="Completed training experiment whose diffusion graph should move forward.",
        ),
    ],
) -> None:
    """Publish a completed training run's graph state to a campaign registry."""
    resolved_experiment_dir = experiment_dir.expanduser().resolve()
    channel_path = _publish_graph_state_ref(
        f"latest-graph:{campaign}",
        experiment_dir=resolved_experiment_dir,
        split=_experiment_split(resolved_experiment_dir),
    )
    console.print(f"[bold]Published graph state:[/] {channel_path}")


def register_run_command(app: typer.Typer) -> None:
    app.command()(run)
    app.command("base-artifacts")(base_artifacts)
    app.command("publish-harness")(publish_harness)
    app.command("publish-graph-state")(publish_graph_state)
