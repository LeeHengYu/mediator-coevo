"""Rich console output helpers for the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from rich.console import Console
from rich.table import Table

from mediated_coevo.analysis.reporting import (
    BootstrapConfidenceInterval,
    ExperimentScoreSummary,
)
from mediated_coevo.benchmarks import HERMES_AGENT_NAME
from mediated_coevo.core.config import Config


class TaskSelectionDisplay(Protocol):
    @property
    def task_ids(self) -> list[str]: ...

    @property
    def family(self) -> str | None: ...

    @property
    def task_set(self) -> str | None: ...


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
    console.print(f"  Bootstrap CI: {_format_ci(summary.bootstrap_ci)}")
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
            console.print(
                "    "
                f"{task_summary.task_id}{metadata_text}: "
                f"mean={_format_score(task_summary.mean_reward)} "
                f"median={_format_score(task_summary.median_reward)} "
                f"scored={task_summary.scored_count}/{task_summary.total_runs} "
                f"env_failures={task_summary.env_failure_count} "
                f"ci={_format_ci(task_summary.bootstrap_ci)}"
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


def print_inspection_payload(payload: dict[str, Any]) -> None:
    if payload["kind"] == "matrix":
        console.print("\n[bold]Matrix inspection:[/]")
        console.print(f"  Data: {payload['experiment_dir']}")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Row")
        table.add_column("Runs", justify="right")
        table.add_column("Scored", justify="right")
        table.add_column("Env failures", justify="right")
        table.add_column("Mean")
        table.add_column("Macro mean")
        table.add_column("Diffusion")
        table.add_column("Metrics")
        for row in payload["rows"]:
            diffusion_payload = row.get("diffusion") or {}
            if diffusion_payload:
                rendered_count = diffusion_payload.get("rendered_record_count", 0)
                metrics_summary = diffusion_payload.get("metrics") or {}
                policy = metrics_summary.get("diffusion_policy")
                diffusion_summary = (
                    f"{policy}: {rendered_count} rendered"
                    if policy
                    else f"{rendered_count} rendered"
                )
            else:
                diffusion_summary = "n/a"
            summary_data = row.get("summary")
            if summary_data is None:
                table.add_row(
                    row["row"],
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                    diffusion_summary,
                    row.get("metrics_path") or "n/a",
                )
                continue
            summary = ExperimentScoreSummary.model_validate(summary_data)
            table.add_row(
                row["row"],
                str(summary.total_runs),
                str(summary.scored_count),
                str(summary.env_failure_count),
                _format_score(summary.mean_reward),
                _format_score(summary.macro_mean_reward),
                diffusion_summary,
                row.get("metrics_path") or "n/a",
            )
        console.print(table)
        return

    summary_data = payload.get("summary")
    if summary_data is not None:
        summary = ExperimentScoreSummary.model_validate(summary_data)
        print_result_summary(
            summary=summary,
            data_dir=Path(payload["experiment_dir"]),
            summary_path=Path(payload["summary_path"]),
            header="Inspection",
        )
    else:
        console.print("\n[bold]Inspection:[/]")
        console.print(f"  Data: {payload['experiment_dir']}")
        console.print(f"  [yellow]Warning:[/] {payload['warning']}")
    console.print(f"  Metrics: {payload.get('metrics_path') or 'n/a'}")
    artifact_dirs = payload.get("artifact_dirs") or []
    if artifact_dirs:
        console.print("  Artifact dirs:")
        for artifact_dir in artifact_dirs:
            console.print(f"    {artifact_dir}")
    diffusion_payload = payload.get("diffusion")
    if not diffusion_payload:
        return

    metrics_summary = diffusion_payload.get("metrics") or {}
    paths = diffusion_payload.get("paths") or {}
    console.print("  Diffusion:")
    console.print(
        f"    Enabled: "
        f"{_format_inspection_value(metrics_summary.get('diffusion_enabled'))}"
    )
    console.print(
        f"    Policy: "
        f"{_format_inspection_value(metrics_summary.get('diffusion_policy'))}"
    )
    console.print(
        f"    Graph: "
        f"{_format_inspection_value(metrics_summary.get('diffusion_graph'))}"
    )
    console.print(f"    Metrics: {paths.get('metrics') or 'n/a'}")
    console.print(f"    Diffusion records: {paths.get('diffused_records') or 'n/a'}")
    console.print(f"    Graph snapshots: {paths.get('graph_snapshots_dir') or 'n/a'}")
    console.print("    Records:")
    console.print(f"      Eligible: {diffusion_payload['eligible_record_count']}")
    console.print(f"      Selected: {diffusion_payload['selected_record_count']}")
    console.print(f"      Rendered: {diffusion_payload['rendered_record_count']}")

    context_summary = metrics_summary.get("context")
    if not context_summary:
        return
    token_summary = context_summary["diffusion_context_tokens"]
    reward_summary = context_summary["reward_after_diffusion_context"]
    regression_summary = context_summary["regression_after_diffusion_context"]
    source_task_ids = context_summary["source_task_ids"]
    if not source_task_ids:
        source_tasks = str(context_summary["source_task_count"])
    elif len(source_task_ids) > 8:
        source_tasks = (
            f"{context_summary['source_task_count']} "
            f"({', '.join(source_task_ids[:8])}, ...)"
        )
    else:
        source_tasks = (
            f"{context_summary['source_task_count']} "
            f"({', '.join(source_task_ids)})"
        )
    console.print("    Context:")
    console.print(
        f"      Rows with rendered context: "
        f"{context_summary['rows_with_rendered_context']}"
    )
    console.print(
        "      Context tokens: "
        f"total={_format_summary_number(token_summary['total'])} "
        f"mean={_format_summary_number(token_summary['mean'])} "
        f"max={_format_summary_number(token_summary['max'])}"
    )
    console.print(f"      Source tasks: {source_tasks}")
    console.print(
        "      Reward after context: "
        f"count={reward_summary['count']} "
        f"mean={_format_summary_number(reward_summary['mean'])} "
        f"min={_format_summary_number(reward_summary['min'])} "
        f"max={_format_summary_number(reward_summary['max'])}"
    )
    console.print(
        "      Regressions after context: "
        f"{regression_summary['count']} "
        f"rate={_format_summary_number(regression_summary['rate'])}"
    )


def print_task_selection(selection: TaskSelectionDisplay) -> None:
    console.print(f"[bold]SkillFlow tasks:[/] {selection.task_ids}")
    if selection.family is not None:
        console.print(f"[bold]Family:[/] {selection.family}")
    if selection.task_set is not None:
        console.print(f"[bold]Task set:[/] {selection.task_set}")


def print_experiment_controls(config: Config) -> None:
    console.print(f"[bold]Coevo interval:[/] {config.experiment.coevo_interval}")
    console.print(
        f"[bold]Advisor buffer max:[/] {config.experiment.advisor_buffer_max}"
    )
    console.print(
        "[bold]Diffusion:[/] "
        f"enabled={config.diffusion.enabled} "
        f"policy={config.diffusion.policy} "
        f"graph={config.diffusion.graph}"
    )
    console.print("[bold]Skill validation:[/] required")
    console.print(
        "[bold]Harbor:[/] "
        f"agent={HERMES_AGENT_NAME} "
        "base_image=required "
        "task_prebuild=optional"
    )


def _format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_ci(interval: BootstrapConfidenceInterval) -> str:
    if interval.lower is None or interval.upper is None:
        return "n/a"
    confidence = round(interval.confidence_level * 100)
    return f"{confidence}% [{interval.lower:.3f}, {interval.upper:.3f}]"


def _format_inspection_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _format_summary_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}"
