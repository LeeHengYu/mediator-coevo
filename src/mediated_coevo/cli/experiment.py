"""Shared CLI helpers for experiment command execution."""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class TaskSelection:
    """Resolved SkillFlow task selectors for one run."""

    task_ids: list[str]
    tasks: list[str]
    family: str | None
    task_set: str | None


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
    tasks: list[str] | None,
    family: str | None,
    task_set: str | None,
) -> TaskSelection:
    selected = repository.resolve_selection(
        tasks=tasks,
        family=family,
        task_set=task_set,
    )
    if not selected:
        raise typer.BadParameter("provide --task, --family, or --task-set")
    return TaskSelection(
        task_ids=selected,
        tasks=tasks or [],
        family=family,
        task_set=task_set,
    )


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
