"""Read-only comparison helpers for prior-context budget experiments."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from mediated_coevo.core.utils import string_list_values
from mediated_coevo.diffusion.models import DiffusedRecord

CONTEXT_TOKEN_FIELDS: tuple[str, ...] = (
    "same_task_prior_tokens",
    "transfer_context_tokens",
    "total_planner_prior_context_tokens",
)

COMPARABILITY_CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("models", "planner"),
    ("models", "executor"),
    ("models", "mediator"),
    ("models", "judge"),
    ("experiment", "num_iterations"),
    ("experiment", "coevo_interval"),
    ("experiment", "seed"),
    ("experiment", "advisor_buffer_max"),
    ("experiment", "condition_name"),
    ("experiment", "baseline_preset"),
    ("experiment", "benchmark_selection", "tasks"),
    ("experiment", "benchmark_selection", "family"),
    ("experiment", "benchmark_selection", "task_set"),
    ("experiment", "skill_updates", "executor"),
    ("experiment", "skill_updates", "planner"),
    ("experiment", "skill_updates", "mediator"),
    ("diffusion", "enabled"),
    ("diffusion", "policy"),
    ("diffusion", "graph"),
    ("diffusion", "max_artifacts"),
    ("diffusion", "top_k_neighbors"),
)

BUDGET_CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("budgets", "max_skill_tokens"),
    ("budgets", "max_same_task_prior_tokens"),
    ("budgets", "max_transfer_context_tokens"),
    ("budgets", "trace_excerpt_tokens"),
    ("budgets", "historical_summary_tokens"),
    ("budgets", "mediator_report_tokens"),
    ("budgets", "planner_context_tokens"),
    ("budgets", "skill_update_diff_tokens"),
    ("budgets", "mediator_prompt_tokens"),
    ("budgets", "advisor_prompt_tokens"),
    ("budgets", "reflector_prompt_tokens"),
    ("budgets", "judge_prompt_tokens"),
    ("budgets", "planner_completion_tokens"),
    ("budgets", "mediator_completion_tokens"),
    ("budgets", "advisor_completion_tokens"),
    ("budgets", "reflector_completion_tokens"),
    ("budgets", "judge_completion_tokens"),
)


class ConfigDifference(BaseModel):
    """One differing config path between compared runs."""

    path: str
    run_a: Any = None
    run_b: Any = None


class ArtifactValidityFailure(BaseModel):
    """One invalid or unverifiable diffusion audit record."""

    run_label: Literal["run_a", "run_b"]
    record_id: str | None = None
    artifact_id: str | None = None
    target_task_id: str | None = None
    target_iteration: int | None = None
    description: str


class ContextBudgetRunSummary(BaseModel):
    """Compact read-only summary for one completed run."""

    experiment_dir: str
    metric_rows: int
    task_ids: list[str]
    iterations: list[int]
    token_means: dict[str, float | None]
    token_totals: dict[str, float]
    context_budget_violation_count: int
    compacted_diffusion_artifact_count: int
    dropped_diffusion_artifact_count: int
    diffusion_record_count: int
    rendered_diffusion_record_count: int


class ContextBudgetComparison(BaseModel):
    """Comparison result for two completed context-budget runs."""

    comparability_status: Literal["pass", "warning", "fail"]
    run_a: ContextBudgetRunSummary
    run_b: ContextBudgetRunSummary
    token_delta_percent: dict[str, float | None]
    setup_mismatches: list[ConfigDifference] = Field(default_factory=list)
    budget_differences: list[ConfigDifference] = Field(default_factory=list)
    artifact_validity_failures: list[ArtifactValidityFailure] = Field(
        default_factory=list
    )
    recommended_interpretation: str


def compare_context_budget_runs(
    run_a_dir: Path,
    run_b_dir: Path,
    *,
    tolerance: float = 0.05,
) -> ContextBudgetComparison:
    """Compare two completed experiment directories without mutating them."""
    run_a_dir = run_a_dir.resolve()
    run_b_dir = run_b_dir.resolve()
    rows_a = _load_jsonl_dicts(run_a_dir / "metrics.jsonl")
    rows_b = _load_jsonl_dicts(run_b_dir / "metrics.jsonl")
    records_a = _load_diffused_records(run_a_dir / "diffusion" / "diffused_records.jsonl")
    records_b = _load_diffused_records(run_b_dir / "diffusion" / "diffused_records.jsonl")
    config_a = _load_toml(run_a_dir / "config.toml")
    config_b = _load_toml(run_b_dir / "config.toml")

    setup_mismatches = _config_differences(
        config_a,
        config_b,
        COMPARABILITY_CONFIG_PATHS,
    )
    budget_differences = _config_differences(
        config_a,
        config_b,
        BUDGET_CONFIG_PATHS,
    )
    artifact_failures = [
        *_artifact_validity_failures("run_a", run_a_dir, records_a),
        *_artifact_validity_failures("run_b", run_b_dir, records_b),
    ]
    run_a = _run_summary(run_a_dir, rows_a, records_a)
    run_b = _run_summary(run_b_dir, rows_b, records_b)
    token_delta_percent: dict[str, float | None] = {}
    for field in CONTEXT_TOKEN_FIELDS:
        value_a = run_a.token_means.get(field)
        value_b = run_b.token_means.get(field)
        if value_a is None or value_b is None:
            token_delta_percent[field] = None
            continue
        if value_a == 0:
            token_delta_percent[field] = (
                0.0 if abs(value_b) <= tolerance else None
            )
            continue
        token_delta_percent[field] = (value_b - value_a) / value_a
    if setup_mismatches or artifact_failures:
        status: Literal["pass", "warning", "fail"] = "fail"
    elif budget_differences:
        status = "warning"
    else:
        status = "pass"
    if setup_mismatches:
        recommended_interpretation = (
            "Do not interpret reward or token differences as a budget effect until "
            "non-budget setup mismatches are removed."
        )
    elif status == "fail":
        recommended_interpretation = (
            "Fix diffusion provenance or citation failures before interpreting "
            "transfer-context effects."
        )
    elif budget_differences:
        paths = ", ".join(difference.path for difference in budget_differences)
        recommended_interpretation = (
            "Runs are comparable except budget fields "
            f"({paths}); interpret token deltas as budget-sensitivity evidence."
        )
    elif any(delta not in (None, 0.0) for delta in token_delta_percent.values()):
        recommended_interpretation = (
            "Runs have matching setup; observed token deltas reflect realized "
            "prior-context use rather than declared setup differences."
        )
    else:
        recommended_interpretation = (
            "Runs have matching setup and no observed prior-context token delta."
        )
    return ContextBudgetComparison(
        comparability_status=status,
        run_a=run_a,
        run_b=run_b,
        token_delta_percent=token_delta_percent,
        setup_mismatches=setup_mismatches,
        budget_differences=budget_differences,
        artifact_validity_failures=artifact_failures,
        recommended_interpretation=recommended_interpretation,
    )


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"missing experiment config: {path}")
    with open(path, "rb") as file:
        return tomllib.load(file)


def _load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError(f"metrics row must be an object: {path}:{line_number}")
        rows.append(data)
    return rows


def _load_diffused_records(path: Path) -> list[DiffusedRecord]:
    if not path.exists():
        return []
    records: list[DiffusedRecord] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(DiffusedRecord.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"invalid diffusion record: {path}:{line_number}") from exc
    return records


def _config_differences(
    config_a: Mapping[str, Any],
    config_b: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
) -> list[ConfigDifference]:
    differences: list[ConfigDifference] = []
    for path in paths:
        value_a = _nested_value(config_a, path)
        value_b = _nested_value(config_b, path)
        if value_a != value_b:
            differences.append(
                ConfigDifference(
                    path=".".join(path),
                    run_a=value_a,
                    run_b=value_b,
                )
            )
    return differences


def _nested_value(config: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = config
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _artifact_validity_failures(
    run_label: Literal["run_a", "run_b"],
    run_dir: Path,
    records: list[DiffusedRecord],
) -> list[ArtifactValidityFailure]:
    failures: list[ArtifactValidityFailure] = []
    artifacts_dir = run_dir / "diffusion" / "artifacts"
    for record in records:
        description: str | None = None
        if record.source_task_id == record.target_task_id:
            description = "diffusion record leaks same-task context"
        elif record.source_iteration >= record.target_iteration:
            description = "source iteration is not strictly before target iteration"
        elif record.rendered and not (artifacts_dir / f"{record.artifact_id}.json").exists():
            description = "rendered artifact file is missing"
        elif record.rendered and not record.citation_text.strip():
            description = "rendered record lacks citation text"
        elif record.rendered and f"artifact_id={record.artifact_id}" not in record.citation_text:
            description = "citation text does not include the rendered artifact ID"
        elif record.selected and not record.rendered and not record.reason.strip():
            description = "selected unrendered record lacks a drop reason"

        if description is not None:
            failures.append(
                ArtifactValidityFailure(
                    run_label=run_label,
                    record_id=record.record_id,
                    artifact_id=record.artifact_id,
                    target_task_id=record.target_task_id,
                    target_iteration=record.target_iteration,
                    description=description,
                )
            )
    return failures


def _run_summary(
    run_dir: Path,
    rows: list[dict[str, Any]],
    records: list[DiffusedRecord],
) -> ContextBudgetRunSummary:
    compacted_ids = string_list_values(rows, "compacted_diffusion_artifact_ids")
    dropped_ids = string_list_values(rows, "dropped_for_budget_artifact_ids")
    token_values = {
        field: _numeric_values(rows, field) for field in CONTEXT_TOKEN_FIELDS
    }
    return ContextBudgetRunSummary(
        experiment_dir=str(run_dir),
        metric_rows=len(rows),
        task_ids=sorted(
            {
                task_id
                for task_id in (row.get("task_id") for row in rows)
                if isinstance(task_id, str)
            }
        ),
        iterations=sorted(
            {
                iteration
                for iteration in (row.get("iteration") for row in rows)
                if isinstance(iteration, int) and not isinstance(iteration, bool)
            }
        ),
        token_means={
            field: (sum(values) / len(values) if values else None)
            for field, values in token_values.items()
        },
        token_totals={field: sum(values) for field, values in token_values.items()},
        context_budget_violation_count=sum(
            1 for row in rows if row.get("context_budget_violation") is True
        ),
        compacted_diffusion_artifact_count=len(set(compacted_ids)),
        dropped_diffusion_artifact_count=len(set(dropped_ids)),
        diffusion_record_count=len(records),
        rendered_diffusion_record_count=sum(1 for record in records if record.rendered),
    )


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values.append(float(value))
    return values
