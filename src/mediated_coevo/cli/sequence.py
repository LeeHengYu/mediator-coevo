"""Frozen warm-up-plus-suffix sequence command."""

from __future__ import annotations

import asyncio
import random
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from mediated_coevo.benchmarks import SkillFlowRepository
from mediated_coevo.cli.config import _load_config_or_bad_parameter
from mediated_coevo.cli.experiment import (
    PROJECT_ROOT,
    _normalize_families,
    ensure_harbor_available,
    prepare_llm_credentials_or_exit,
    setup_logging,
)
from mediated_coevo.cli.harness_registry import (
    _harness_overlay_root,
    _prepare_harness_workspace,
    _resolve_harness_options,
)
from mediated_coevo.cli.output import console
from mediated_coevo.cli.run import (
    _apply_harness_overlay_and_reexec,
    _restore_scoped_harness_overlay,
)
from mediated_coevo.core.config import Config
from mediated_coevo.execution.adapters import BenchmarkTaskProfileProvider
from mediated_coevo.experiment.runtime_factory import (
    build_benchmark_repo,
    build_experiment,
)
from mediated_coevo.experiment.sample_models import (
    PositionJournal,
    SampleResult,
    SampleSpec,
    SequenceSpec,
)
from mediated_coevo.experiment.sample_runtime import build_sample_runtime
from mediated_coevo.orchestration.arms import OrchestrationArm


_SEQUENCE_HARNESS_FILES = (
    Path("src/mediated_coevo/diffusion/task_graph_agent.py"),
    Path("src/mediated_coevo/diffusion/policy_agent.py"),
)


def _select_sequence_tasks(
    repository: SkillFlowRepository,
    families: list[str],
    seed: int,
) -> list[str]:
    candidates: list[str] = []
    for family in families:
        candidates.extend(repository.list_local_task_ids(family=family))
    candidates = sorted(dict.fromkeys(candidates))
    if len(candidates) < 10:
        raise typer.BadParameter(
            f"selected families have {len(candidates)} tasks; sequence requires 10"
        )
    by_family = {
        family: [
            task_id
            for task_id in candidates
            if repository.resolve(task_id).family == family
        ]
        for family in families
    }
    missing = [family for family, task_ids in by_family.items() if not task_ids]
    if missing:
        raise typer.BadParameter(
            f"selected families have no tasks for: {', '.join(missing)}"
        )
    rng = random.Random(seed)
    selected = [rng.choice(by_family[family]) for family in families]
    remaining = [task_id for task_id in candidates if task_id not in selected]
    selected.extend(rng.sample(remaining, 10 - len(selected)))
    rng.shuffle(selected)
    return selected


async def _run_sequence(
    *,
    config: Config,
    repository: SkillFlowRepository,
    sequence: SequenceSpec,
    arm: OrchestrationArm,
    sequence_dir: Path,
    artifact_store_root: Path,
    iteration: int,
    iterations: int,
) -> SampleResult:
    warmup_experiment = build_experiment(
        project_root=PROJECT_ROOT,
        config=config.model_copy(deep=True),
        seed=sequence.policy_seed,
        condition_name=config.experiment.condition_name,
        experiment_dir=sequence_dir / "warmup" / "warmup",
        benchmark_repo=repository,
    )
    warmup_runtime = build_sample_runtime(
        orchestrator=warmup_experiment.orchestrator,
        run_id="warmup",
        sequence_dir=sequence_dir,
        implementation_revision="workspace",
        implementation_dirty=True,
    )
    warmup = await warmup_runtime.prepare_warmup_from_stores(
        sequence,
        artifact_store_root=artifact_store_root,
    )
    console.print(
        f"[bold]Iteration {iteration}/{iterations}[/] · warmup loaded "
        f"{sequence.warmup_count}/{len(sequence.tasks)}: "
        f"{', '.join(sequence.task_ids[: sequence.warmup_count])}"
    )

    sample_id = arm.value
    experiment = build_experiment(
        project_root=PROJECT_ROOT,
        config=config.model_copy(deep=True),
        seed=sequence.policy_seed,
        condition_name=config.experiment.condition_name,
        experiment_dir=sequence_dir / "samples" / sample_id,
        benchmark_repo=repository,
    )
    runtime = build_sample_runtime(
        orchestrator=experiment.orchestrator,
        run_id=sample_id,
        sequence_dir=sequence_dir,
        implementation_revision="workspace",
        implementation_dirty=True,
    )

    def show_progress(journal: PositionJournal) -> None:
        next_run = journal.position + 2
        console.print(
            f"Iteration {iteration}/{iterations} · {arm.value} · "
            f"run {min(next_run, len(sequence.tasks))}/{len(sequence.tasks)}"
            f"{' complete' if next_run > len(sequence.tasks) else ''}"
        )

    console.print(
        f"Iteration {iteration}/{iterations} · {arm.value} · "
        f"run {sequence.warmup_count + 1}/{len(sequence.tasks)}"
    )
    return await runtime.run(
        SampleSpec(
            sample_id=sample_id,
            sequence=sequence,
            arm=arm,
            warmup_bundle_id=warmup.bundle_id,
        ),
        warmup=warmup,
        on_position_complete=show_progress,
    )


def sequence(
    family: Annotated[
        list[str],
        typer.Option(
            "--family",
            help="Exactly four SkillFlow families. Repeat this option.",
        ),
    ],
    seed: Annotated[
        int,
        typer.Option(
            help="Base task-stream and policy seed; each loop adds its zero-based index."
        ),
    ] = 0,
    k: Annotated[
        int,
        typer.Option(
            "-K",
            help="Number of sequences to run serially.",
        ),
    ] = 1,
    config_dir: Annotated[
        Path,
        typer.Option(help="Config directory."),
    ] = PROJECT_ROOT / "config",
    artifact_store_root: Annotated[
        Path,
        typer.Option(help="Per-task base artifact store root."),
    ] = PROJECT_ROOT / "data" / "base_artifacts",
    output_dir: Annotated[
        Path,
        typer.Option(help="Sequence archive root."),
    ] = PROJECT_ROOT / "data" / "sequences",
    harness_dir: Annotated[
        Path | None,
        typer.Option(
            "--harness-dir",
            help="Repo-root overlay containing both sequence agent harnesses.",
        ),
    ] = None,
    harness_ref: Annotated[
        str | None,
        typer.Option(
            "--harness-ref",
            help="Published harness reference, for example promoted:HL3.",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Repeat one config-selected arm over seeded 10-task sequences."""
    resolved_harness_dir = _resolve_harness_options(harness_dir, harness_ref)
    if resolved_harness_dir is not None:
        overlay_root = _harness_overlay_root(resolved_harness_dir)
        missing = [
            path.as_posix()
            for path in _SEQUENCE_HARNESS_FILES
            if not (overlay_root / path).is_file()
        ]
        if missing:
            raise typer.BadParameter(
                "sequence harness overlay is missing: " + ", ".join(missing)
            )
        _apply_harness_overlay_and_reexec(resolved_harness_dir)
    try:
        setup_logging(verbose)
        if k < 1:
            raise typer.BadParameter("-K must be at least 1")
        families = _normalize_families(family)
        if len(families) != 4:
            raise typer.BadParameter("sequence requires exactly four distinct families")
        config = _load_config_or_bad_parameter(
            config_dir,
            overrides={
                "experiment": {
                    "seed": seed,
                    "condition_name": "learned_mediator",
                    "skill_updates": {
                        "executor": False,
                        "planner": False,
                        "mediator": False,
                    },
                    "baseline_preset": None,
                },
            },
        )
        arm = config.experiment.orchestration_arm
        repository = build_benchmark_repo(PROJECT_ROOT, config)
        prepare_llm_credentials_or_exit(config)
        ensure_harbor_available(config)
        provider = BenchmarkTaskProfileProvider(repository)
        console.print(
            f"[bold]Sequence run:[/] {k} iteration(s), 10 tasks each "
            f"(3 warmup loaded + 7 evaluated), arm {arm.value}, "
            f"seeds {seed}..{seed + k - 1}"
        )
        run_id = f"sequence-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{seed}"
        run_dir = output_dir / run_id
        if resolved_harness_dir is not None:
            _prepare_harness_workspace(run_dir, resolved_harness_dir)
            console.print(f"[bold]Harness overlay:[/] {resolved_harness_dir}")
        for loop_index in range(k):
            sequence_seed = seed + loop_index
            config.experiment.seed = sequence_seed
            task_ids = _select_sequence_tasks(repository, families, sequence_seed)
            sequence_id = f"{run_id}-iter-{loop_index + 1}"
            spec = SequenceSpec(
                sequence_id=sequence_id,
                tasks=tuple(provider.resolve(task_id) for task_id in task_ids),
                warmup_count=3,
                policy_seed=sequence_seed,
                task_set_id=f"families:{','.join(families)}",
            )
            sequence_dir = run_dir / f"iter-{loop_index + 1}"
            result = asyncio.run(
                _run_sequence(
                    config=config,
                    repository=repository,
                    sequence=spec,
                    arm=arm,
                    sequence_dir=sequence_dir,
                    artifact_store_root=artifact_store_root,
                    iteration=loop_index + 1,
                    iterations=k,
                )
            )
            console.print(f"[bold green]Iteration complete:[/] {sequence_dir}")
            console.print(
                f"{result.spec.arm.value}: "
                f"reward={result.rewards.unweighted_mean} "
                f"valid={result.rewards.valid_for_reporting}"
            )
        console.print(f"[bold green]Sequence complete:[/] {run_dir}")
    finally:
        _restore_scoped_harness_overlay()


def register_sequence_command(app: typer.Typer) -> None:
    app.command("sequence")(sequence)
