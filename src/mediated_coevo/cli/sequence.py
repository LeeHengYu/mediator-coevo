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
    _normalize_split,
    _select_split_pool,
    ensure_harbor_available,
    prepare_llm_credentials_or_exit,
    setup_logging,
)
from mediated_coevo.cli.output import console
from mediated_coevo.core.config import Config
from mediated_coevo.execution.adapters import BenchmarkTaskProfileProvider
from mediated_coevo.experiment.runtime_factory import (
    build_benchmark_repo,
    build_experiment,
)
from mediated_coevo.experiment.sample_models import (
    SampleResult,
    SampleSpec,
    SequenceSpec,
)
from mediated_coevo.experiment.sample_runtime import build_sample_runtime
from mediated_coevo.orchestration.arms import OrchestrationArm

_ARMS = (
    OrchestrationArm.EXECUTION_ONLY,
    OrchestrationArm.RANDOM_POLICY,
    OrchestrationArm.NO_GRAPH,
    OrchestrationArm.FULL_ORCHESTRATION,
)


def _select_sequence_tasks(
    repository: SkillFlowRepository,
    families: list[str],
    split: str,
    seed: int,
) -> list[str]:
    candidates: list[str] = []
    for family in families:
        candidates.extend(repository.list_local_task_ids(family=family))
    candidates = _select_split_pool(
        sorted(dict.fromkeys(candidates)),
        seed=seed,
        split=split,
    )
    if len(candidates) < 10:
        raise typer.BadParameter(
            f"selected split has {len(candidates)} tasks; sequence requires 10"
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
            f"selected split has no tasks for families: {', '.join(missing)}"
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
    sequence_dir: Path,
    artifact_store_root: Path,
) -> tuple[SampleResult, ...]:
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

    results: list[SampleResult] = []
    for arm in _ARMS:
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
        results.append(
            await runtime.run(
                SampleSpec(
                    sample_id=sample_id,
                    sequence=sequence,
                    arm=arm,
                    warmup_bundle_id=warmup.bundle_id,
                ),
                warmup=warmup,
            )
        )
    return tuple(results)


def sequence(
    family: Annotated[
        list[str],
        typer.Option(
            "--family",
            help="Exactly four SkillFlow families. Repeat this option.",
        ),
    ],
    split: Annotated[
        str,
        typer.Option(help="Frozen task split: train, validation, or test."),
    ] = "test",
    seed: Annotated[int, typer.Option(help="Task selection and policy seed.")] = 0,
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
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run four frozen arms over one 3-warm-up plus 7-task suffix sequence."""
    setup_logging(verbose)
    families = _normalize_families(family)
    if len(families) != 4:
        raise typer.BadParameter("sequence requires exactly four distinct families")
    normalized_split = _normalize_split(split)
    assert normalized_split is not None
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
            "diffusion": {
                "enabled": True,
                "policy": "random_k",
                "graph": "none",
            },
        },
    )
    repository = build_benchmark_repo(PROJECT_ROOT, config)
    task_ids = _select_sequence_tasks(repository, families, normalized_split, seed)
    prepare_llm_credentials_or_exit(config)
    ensure_harbor_available(config)
    provider = BenchmarkTaskProfileProvider(repository)
    sequence_id = f"sequence-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{seed}"
    spec = SequenceSpec(
        sequence_id=sequence_id,
        tasks=tuple(provider.resolve(task_id) for task_id in task_ids),
        warmup_count=3,
        policy_seed=seed,
        task_set_id=f"{normalized_split}:{','.join(families)}",
    )
    sequence_dir = output_dir / sequence_id
    results = asyncio.run(
        _run_sequence(
            config=config,
            repository=repository,
            sequence=spec,
            sequence_dir=sequence_dir,
            artifact_store_root=artifact_store_root,
        )
    )
    console.print(f"[bold green]Sequence complete:[/] {sequence_dir}")
    for result in results:
        console.print(
            f"{result.spec.arm.value}: "
            f"reward={result.rewards.unweighted_mean} "
            f"valid={result.rewards.valid_for_reporting}"
        )


def register_sequence_command(app: typer.Typer) -> None:
    app.command("sequence")(sequence)
