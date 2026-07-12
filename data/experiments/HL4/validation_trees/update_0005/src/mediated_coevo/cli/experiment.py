"""Shared CLI helpers for experiment command execution."""

from __future__ import annotations

import asyncio
import logging
import random
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import typer
from rich.logging import RichHandler

from mediated_coevo.analysis.judge_rewards import (
    JudgeRewardAnnotationError,
    annotate_judge_rewards,
)
from mediated_coevo.analysis.reporting import build_score_summary, write_score_summary
from mediated_coevo.benchmarks import (
    HarborPrebuiltImageMissingError,
    SkillFlowRepository,
)
from mediated_coevo.core.config import Config
from mediated_coevo.experiment.runtime_factory import ExperimentRuntime
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.stores.history_store import HistoryStore
from mediated_coevo.cli.output import console, print_result_summary

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP_FAMILY_TASK_COUNT = 8
TaskSplitName = Literal["train", "validation", "test"]
TASK_SPLIT_NAMES = frozenset({"train", "validation", "test"})


@dataclass(frozen=True)
class TaskSelection:
    """Resolved SkillFlow task stream for one run."""

    task_ids: list[str]
    families: tuple[str, ...]
    split: TaskSplitName | None = None
    task_stream_seed: int | None = None

    @property
    def family(self) -> str:
        """Compact label for persisted metadata and old call sites."""
        return ",".join(self.families)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=True, show_path=False)],
    )


def resolve_task_selection(
    *,
    repository: SkillFlowRepository,
    family: str | Sequence[str] | None,
    seed: int | None = None,
    split: str | None = None,
) -> TaskSelection:
    families = _normalize_families(family)
    selected: list[str] = []
    for family_name in families:
        family_task_ids = repository.list_local_task_ids(family=family_name)
        if not family_task_ids:
            raise typer.BadParameter(
                f"no local SkillFlow tasks found for family {family_name!r}"
            )
        selected.extend(family_task_ids)
    selected = sorted(dict.fromkeys(selected))
    selected = _select_split_pool(selected, seed=seed, split=split)
    if not selected:
        raise typer.BadParameter("selected task split is empty")
    resolved_stream_seed = secrets.randbits(63)
    selected = _sample_task_stream(
        selected,
        stream_seed=resolved_stream_seed,
    )
    return TaskSelection(
        task_ids=selected,
        families=tuple(families),
        split=_normalize_split(split),
        task_stream_seed=resolved_stream_seed,
    )


def _sample_task_stream(
    task_ids: list[str],
    *,
    stream_seed: int,
) -> list[str]:
    """Build a fresh stream without needlessly concentrating a short pool."""
    rng = random.Random(stream_seed)
    if len(task_ids) >= BOOTSTRAP_FAMILY_TASK_COUNT:
        return rng.sample(task_ids, k=BOOTSTRAP_FAMILY_TASK_COUNT)

    full_cycles, extra_slots = divmod(BOOTSTRAP_FAMILY_TASK_COUNT, len(task_ids))
    stream = task_ids * full_cycles
    stream.extend(rng.sample(task_ids, k=extra_slots))
    rng.shuffle(stream)
    return stream


def _normalize_families(family: str | Sequence[str] | None) -> list[str]:
    if family is None:
        raise typer.BadParameter("provide --family")
    raw_families = [family] if isinstance(family, str) else list(family)
    families = list(
        dict.fromkeys(
            family_name.strip() for family_name in raw_families if family_name.strip()
        )
    )
    if not families:
        raise typer.BadParameter("provide --family")
    return families


def _normalize_split(split: str | None) -> TaskSplitName | None:
    if split is None:
        return None
    normalized = split.strip().lower()
    if normalized not in TASK_SPLIT_NAMES:
        allowed = ", ".join(sorted(TASK_SPLIT_NAMES))
        raise typer.BadParameter(f"invalid split {split!r}; expected one of: {allowed}")
    return normalized  # type: ignore[return-value]


def _select_split_pool(
    task_ids: list[str],
    *,
    seed: int | None,
    split: str | None,
) -> list[str]:
    split_name = _normalize_split(split)
    if split_name is None:
        return task_ids
    if len(task_ids) < 3:
        raise typer.BadParameter(
            "train/validation/test split requires at least 3 tasks"
        )
    shuffled = list(task_ids)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, len(shuffled) // 5)
    test_count = max(1, len(shuffled) // 5)
    train_count = len(shuffled) - validation_count - test_count
    if train_count < 1:
        raise typer.BadParameter("train/validation/test split leaves no training tasks")
    pools = {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count : train_count + validation_count],
        "test": shuffled[train_count + validation_count :],
    }
    return pools[split_name]


def ensure_harbor_available(config: Config) -> None:
    if config.executor_runtime.harbor_required and shutil.which("harbor") is None:
        console.print(
            "[bold red]ERROR:[/] Harbor CLI not found on PATH. Install Harbor, "
            "or set executor_runtime.harbor_required = false in config."
        )
        raise typer.Exit(code=1)


def prepare_llm_credentials_or_exit(config: Config) -> Config:
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


def write_and_print_result_summary(
    *,
    records: list[IterationRecord],
    data_dir: Path,
    header: str,
) -> None:
    summary = build_score_summary(records)
    summary_path = data_dir / "summary.json"
    write_score_summary(summary, summary_path)
    print_result_summary(
        summary=summary,
        data_dir=data_dir,
        summary_path=summary_path,
        header=header,
    )


def annotate_judge_rewards_or_exit(
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


def run_experiment_or_exit(
    runtime: ExperimentRuntime,
    task_ids: list[str],
    iterations: int,
) -> list[IterationRecord]:
    try:
        return asyncio.run(runtime.orchestrator.run_experiment(task_ids, iterations))
    except HarborPrebuiltImageMissingError as exc:
        console.print(f"[bold red]ERROR:[/] {exc}")
        raise typer.Exit(code=1) from exc
