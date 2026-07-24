"""Rich console output helpers for the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from mediated_coevo.analysis.reporting import ExperimentScoreSummary
from mediated_coevo.core.config import Config

console = Console()


def print_result_summary(
    *,
    summary: ExperimentScoreSummary,
    data_dir: Path,
    summary_path: Path,
    header: str,
) -> None:
    console.print(f"\n[bold]{header}:[/]")
    console.print(f"  Runs: {summary.total_runs}")
    console.print(f"  Scored: {summary.scored_count}")
    console.print(f"  Env failures: {summary.env_failure_count}")
    console.print(f"  Mean reward: {_format_score(summary.mean_reward)}")
    console.print(f"  Median reward: {_format_score(summary.median_reward)}")
    console.print(f"  Macro mean reward: {_format_score(summary.macro_mean_reward)}")
    if summary.bootstrap_ci.lower is None or summary.bootstrap_ci.upper is None:
        bootstrap_ci = "n/a"
    else:
        confidence = round(summary.bootstrap_ci.confidence_level * 100)
        bootstrap_ci = (
            f"{confidence}% "
            f"[{summary.bootstrap_ci.lower:.3f}, {summary.bootstrap_ci.upper:.3f}]"
        )
    console.print(f"  Bootstrap CI: {bootstrap_ci}")
    if summary.judge_reward_summary is not None:
        judge_summary = summary.judge_reward_summary
        console.print(
            "  Judge mean reward: "
            f"{_format_score(judge_summary.mean_reward)} "
            f"(macro={_format_score(judge_summary.macro_mean_reward)})"
        )
    console.print(f"  Total tokens: {summary.total_tokens:,}")
    if summary.per_task:
        console.print("  Per-task:")
        for task_summary in summary.per_task:
            metadata = []
            if task_summary.task_category:
                metadata.append(task_summary.task_category)
            if task_summary.task_difficulty:
                metadata.append(task_summary.task_difficulty)
            metadata_text = f" ({', '.join(metadata)})" if metadata else ""
            if (
                task_summary.bootstrap_ci.lower is None
                or task_summary.bootstrap_ci.upper is None
            ):
                bootstrap_ci = "n/a"
            else:
                confidence = round(task_summary.bootstrap_ci.confidence_level * 100)
                bootstrap_ci = (
                    f"{confidence}% "
                    f"[{task_summary.bootstrap_ci.lower:.3f}, "
                    f"{task_summary.bootstrap_ci.upper:.3f}]"
                )
            console.print(
                "    "
                f"{task_summary.task_id}{metadata_text}: "
                f"mean={_format_score(task_summary.mean_reward)} "
                f"median={_format_score(task_summary.median_reward)} "
                f"scored={task_summary.scored_count}/{task_summary.total_runs} "
                f"env_failures={task_summary.env_failure_count} "
                f"ci={bootstrap_ci}"
            )
    if summary.dominance_warning and summary.dominant_task_id:
        console.print(
            "  [yellow]Dominance warning:[/] "
            f"{summary.dominant_task_id} contributed "
            f"{summary.max_task_scored_share:.1%} of scored runs "
            f"(threshold {summary.dominance_threshold:.1%})."
        )
    console.print(f"  Summary: {summary_path}")
    console.print(f"  Data: {data_dir}")


def print_task_selection(selection: Any) -> None:
    families = getattr(selection, "families", None)
    label = ", ".join(families) if families else selection.family
    console.print(f"[bold]Family:[/] {label}")
    if getattr(selection, "split", None):
        console.print(f"[bold]Split:[/] {selection.split}")


def print_experiment_controls(config: Config) -> None:
    console.print(
        "[bold]Diffusion:[/] "
        f"enabled={config.diffusion.enabled} "
        f"policy={config.diffusion.policy} "
        f"graph={config.diffusion.graph}"
    )
    console.print(
        "[bold]Harbor:[/] "
        f"agent={config.executor_runtime.agent_name} "
        "base_image=required "
        "task_prebuild=optional"
    )


def _format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
