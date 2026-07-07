"""Run command registration."""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
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
_HARNESS_APPLIED_ENV = "MEDCOEVO_HARNESS_APPLIED"


def run_skillflow_experiment(
    *,
    config: Config,
    selection: TaskSelection,
    iterations: int,
    seed: int,
    condition_name: ConditionName,
    run_id: str | None,
    harness_dir: Path | None = None,
    remote_harbor_config: GCPVMConfig | None = None,
) -> None:
    """Run a SkillFlow selection in one evolution loop."""
    random.seed(seed)
    config.experiment.benchmark_selection.tasks = selection.task_ids
    config.experiment.benchmark_selection.family = selection.family

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
    _prepare_harness_workspace(experiment_dir, harness_dir)
    _copy_harness_state(experiment_dir, harness_dir)
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
    if selection.family is not None:
        console.print(f"[bold]Task stream length:[/] {len(selection.task_ids)}")
    else:
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


def _apply_harness_overlay_and_reexec(harness_dir: Path) -> None:
    resolved = harness_dir.expanduser().resolve()
    if os.environ.get(_HARNESS_APPLIED_ENV) == str(resolved):
        return
    applied_files = _apply_harness_overlay(resolved, PROJECT_ROOT)
    typer.echo(f"Applied harness overlay: {resolved} ({len(applied_files)} files)")
    env = os.environ.copy()
    env[_HARNESS_APPLIED_ENV] = str(resolved)
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def _apply_harness_overlay(harness_dir: Path, project_root: Path) -> list[str]:
    overlay_root = _harness_overlay_root(harness_dir)
    applied: list[str] = []
    for source in sorted(path for path in overlay_root.rglob("*") if path.is_file()):
        rel = source.relative_to(overlay_root)
        if len(rel.parts) == 1 and rel.name.startswith("manifest."):
            continue
        target = project_root / rel
        if source.resolve() == target.resolve():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        applied.append(rel.as_posix())
    if not applied:
        raise typer.BadParameter(f"harness overlay contains no source files: {harness_dir}")
    return applied


def _harness_overlay_root(harness_dir: Path) -> Path:
    if not harness_dir.is_dir():
        raise typer.BadParameter(f"harness directory not found: {harness_dir}")
    overlay = harness_dir / "overlay"
    root = overlay if overlay.is_dir() else harness_dir
    if not any((root / name).exists() for name in ("src", "config", "tests")):
        raise typer.BadParameter(
            "harness directory must be a repo-root overlay containing "
            "src/, config/, tests/, or overlay/"
        )
    return root


def _prepare_harness_workspace(
    experiment_dir: Path,
    harness_dir: Path | None,
) -> None:
    harnesses_dir = experiment_dir / "harnesses"
    harnesses_dir.mkdir(parents=True, exist_ok=True)
    (harnesses_dir / "README.md").write_text(
        "# Harnesses\n\n"
        "Use this folder for learned repo-root harness overlays. Put new "
        "validated harness snapshots in subdirectories; manifest files are "
        "metadata, and every other path is treated as repo-root overlay content.\n"
    )
    if harness_dir is None:
        return

    resolved = harness_dir.expanduser().resolve()
    overlay_root = _harness_overlay_root(resolved)
    seed_dir = harnesses_dir / "seed"
    shutil.copytree(resolved, seed_dir)
    metadata = {
        "source": str(resolved),
        "overlay_root": str(overlay_root),
        "applied_files": _overlay_file_paths(overlay_root),
    }
    state_root = resolved / "state"
    if state_root.is_dir():
        metadata["state_files"] = _state_file_paths(state_root)
    (harnesses_dir / "active_harness.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def _overlay_file_paths(overlay_root: Path) -> list[str]:
    paths: list[str] = []
    for path in sorted(item for item in overlay_root.rglob("*") if item.is_file()):
        rel = path.relative_to(overlay_root)
        if len(rel.parts) == 1 and rel.name.startswith("manifest."):
            continue
        paths.append(rel.as_posix())
    return paths


def _copy_harness_state(
    experiment_dir: Path,
    harness_dir: Path | None,
) -> dict[str, object] | None:
    """Copy runtime state from a learned harness into the new experiment."""
    if harness_dir is None:
        return None
    state_root = harness_dir.expanduser().resolve() / "state"
    if not state_root.is_dir():
        return None

    metadata: dict[str, object] = {"source": str(state_root)}
    diffusion_root = state_root / "diffusion"
    if diffusion_root.is_dir():
        metadata.update(_copy_diffusion_state(experiment_dir, diffusion_root))
    return metadata if len(metadata) > 1 else None


def _copy_diffusion_state(experiment_dir: Path, diffusion_root: Path) -> dict[str, object]:
    target_root = experiment_dir / "diffusion"
    metadata: dict[str, object] = {"diffusion_source": str(diffusion_root)}

    artifacts_dir = diffusion_root / "artifacts"
    if artifacts_dir.is_dir():
        skipped_artifacts = sorted(path.name for path in artifacts_dir.glob("*.json"))
        if skipped_artifacts:
            metadata["skipped_artifacts"] = skipped_artifacts
            metadata["artifact_store_reset"] = True

    graph_dir = diffusion_root / "graph_snapshots"
    copied_graph_snapshots = _copy_tree_files(graph_dir, target_root / "graph_snapshots")
    if copied_graph_snapshots:
        metadata["graph_snapshots"] = copied_graph_snapshots

    copied_files: list[str] = []
    for source in sorted(item for item in diffusion_root.iterdir() if item.is_file()):
        target = target_root / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied_files.append(source.name)
    if copied_files:
        metadata["files"] = copied_files
    return metadata


def _copy_tree_files(source_root: Path, target_root: Path) -> list[str]:
    if not source_root.is_dir():
        return []
    copied: list[str] = []
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        rel = source.relative_to(source_root)
        target = target_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(rel.as_posix())
    return copied


def _state_file_paths(state_root: Path) -> list[str]:
    return [
        path.relative_to(state_root).as_posix()
        for path in sorted(item for item in state_root.rglob("*") if item.is_file())
    ]


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
                "Repo-root harness overlay to apply before this run and copy into "
                "the experiment harnesses/ folder."
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
    if harness_dir is not None:
        _apply_harness_overlay_and_reexec(harness_dir)
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
        family=family,
        seed=config.experiment.seed,
        split=split,
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
        harness_dir=harness_dir,
        remote_harbor_config=remote_harbor_config,
    )


def register_run_command(app: typer.Typer) -> None:
    app.command()(run)
