"""Rich console output helpers for the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from rich.console import Console
from rich.table import Table

from mediated_coevo.analysis.reporting import ExperimentScoreSummary
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
    for label, key in (
        ("Enabled", "diffusion_enabled"),
        ("Policy", "diffusion_policy"),
        ("Graph", "diffusion_graph"),
    ):
        value = metrics_summary.get(key)
        if value is None:
            formatted_value = "n/a"
        elif isinstance(value, bool):
            formatted_value = str(value).lower()
        else:
            formatted_value = str(value)
        console.print(f"    {label}: {formatted_value}")
    console.print(f"    Metrics: {paths.get('metrics') or 'n/a'}")
    console.print(f"    Diffusion records: {paths.get('diffused_records') or 'n/a'}")
    console.print(f"    Graph snapshots: {paths.get('graph_snapshots_dir') or 'n/a'}")
    console.print("    Records:")
    console.print(f"      Eligible: {diffusion_payload['eligible_record_count']}")
    console.print(f"      Selected: {diffusion_payload['selected_record_count']}")
    console.print(f"      Rendered: {diffusion_payload['rendered_record_count']}")

    prior_summary = metrics_summary.get("planner_prior_context")
    if prior_summary:
        total_prior_tokens = prior_summary["total_planner_prior_context_tokens"]
        console.print("    Planner prior context:")
        console.print(
            "      Total tokens: "
            f"total={_format_summary_number(total_prior_tokens['total'])} "
            f"mean={_format_summary_number(total_prior_tokens['mean'])} "
            f"max={_format_summary_number(total_prior_tokens['max'])}"
        )
        console.print(
            "      Budget events: "
            f"violations={prior_summary['context_budget_violation_count']} "
            f"compacted={prior_summary['compacted_diffusion_artifact_count']} "
            f"dropped={prior_summary['dropped_for_budget_artifact_count']}"
        )

    context_summary = metrics_summary.get("context")
    if not context_summary:
        return
    token_summary = context_summary["transfer_context_tokens"]
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
        "      Transfer tokens: "
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


def print_context_budget_comparison(comparison: Any) -> None:
    """Print a concise context-budget comparison."""
    console.print("\n[bold]Context-budget comparison:[/]")
    console.print(f"  Status: {comparison.comparability_status}")
    console.print(f"  Run A: {comparison.run_a.experiment_dir}")
    console.print(f"  Run B: {comparison.run_b.experiment_dir}")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Field")
    table.add_column("Run A mean", justify="right")
    table.add_column("Run B mean", justify="right")
    table.add_column("Delta", justify="right")
    for field, delta in comparison.token_delta_percent.items():
        table.add_row(
            field,
            _format_summary_number(comparison.run_a.token_means.get(field)),
            _format_summary_number(comparison.run_b.token_means.get(field)),
            "n/a" if delta is None else f"{delta * 100:.1f}%",
        )
    console.print(table)

    if comparison.setup_mismatches:
        console.print("  [red]Setup mismatches:[/]")
        for mismatch in comparison.setup_mismatches:
            console.print(
                f"    {mismatch.path}: "
                f"{mismatch.run_a!r} != {mismatch.run_b!r}"
            )
    if comparison.budget_differences:
        console.print("  Budget differences:")
        for difference in comparison.budget_differences:
            console.print(
                f"    {difference.path}: "
                f"{difference.run_a!r} != {difference.run_b!r}"
            )
    if comparison.artifact_validity_failures:
        console.print("  [red]Artifact validity failures:[/]")
        for failure in comparison.artifact_validity_failures[:10]:
            console.print(
                "    "
                f"{failure.run_label} {failure.artifact_id}: "
                f"{failure.description}"
            )
        remaining = len(comparison.artifact_validity_failures) - 10
        if remaining > 0:
            console.print(f"    ... {remaining} more")
    console.print(f"  Interpretation: {comparison.recommended_interpretation}")


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
        f"agent={config.executor_runtime.agent_name} "
        "base_image=required "
        "task_prebuild=optional"
    )


def _format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _format_summary_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}"
