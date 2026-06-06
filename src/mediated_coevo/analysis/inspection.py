"""Experiment inspection payload construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from mediated_coevo.analysis.reporting import ExperimentScoreSummary
from mediated_coevo.core.utils import string_list_values
from mediated_coevo.diffusion import DiffusionStore


def _load_score_summary(summary_path: Path) -> ExperimentScoreSummary:
    return ExperimentScoreSummary.model_validate_json(summary_path.read_text())


def _artifact_dirs(experiment_dir: Path) -> list[str]:
    artifacts_dir = experiment_dir / "artifacts"
    if not artifacts_dir.exists():
        return []
    return [
        str(path)
        for path in sorted(artifacts_dir.iterdir(), key=lambda item: item.name)
        if path.is_dir()
    ]


def _single_or_mixed(rows: list[dict[str, Any]], key: str) -> Any:
    values = [row[key] for row in rows if key in row]
    if not values:
        return None
    first_value = values[0]
    if all(value == first_value for value in values):
        return first_value
    return "mixed"


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values.append(float(value))
    return values


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "total": 0.0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "total": sum(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _diffusion_inspection_payload(experiment_dir: Path) -> dict[str, Any] | None:
    diffusion_dir = experiment_dir / "diffusion"
    if not diffusion_dir.exists():
        return None
    metrics_path = experiment_dir / "metrics.jsonl"
    store = DiffusionStore(diffusion_dir)
    artifacts = store.query_artifacts(recent=None)
    snapshots = store.query_graph_snapshots(recent=None)
    records = store.query_diffused_records(recent=None)
    paths = {
        "metrics": str(metrics_path) if metrics_path.exists() else None,
        "diffused_records": str(diffusion_dir / "diffused_records.jsonl"),
        "artifacts_dir": str(diffusion_dir / "artifacts"),
        "graph_snapshots_dir": str(diffusion_dir / "graph_snapshots"),
    }
    record_counts = {
        "eligible_count": sum(1 for record in records if record.eligible),
        "selected_count": sum(1 for record in records if record.selected),
        "rendered_count": sum(1 for record in records if record.rendered),
    }
    payload = {
        "diffusion_dir": str(diffusion_dir),
        "artifacts_dir": str(diffusion_dir / "artifacts"),
        "graph_snapshots_dir": str(diffusion_dir / "graph_snapshots"),
        "diffused_records_path": str(diffusion_dir / "diffused_records.jsonl"),
        "paths": paths,
        "artifact_count": len(artifacts),
        "graph_snapshot_count": len(snapshots),
        "diffused_record_count": len(records),
        "eligible_record_count": record_counts["eligible_count"],
        "selected_record_count": record_counts["selected_count"],
        "rendered_record_count": record_counts["rendered_count"],
        "records": record_counts,
        "source_task_ids": sorted({artifact.source_task_id for artifact in artifacts}),
        "graph_snapshot_ids": [snapshot.snapshot_id for snapshot in snapshots],
    }
    if metrics_path.exists():
        rows = []
        for line_number, line in enumerate(
            metrics_path.read_text().splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise typer.BadParameter(
                    f"invalid JSON in metrics file {metrics_path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise typer.BadParameter(
                    f"metrics row must be a JSON object: {metrics_path}:{line_number}"
                )
            rows.append(row)

        rendered_rows = []
        for row in rows:
            rendered_count = row.get("diffusion_artifacts_rendered")
            if isinstance(rendered_count, bool) or not isinstance(rendered_count, int):
                continue
            if rendered_count > 0:
                rendered_rows.append(row)

        source_task_ids: set[str] = set()
        for row in rendered_rows:
            source_ids = row.get("source_task_ids")
            if not isinstance(source_ids, list):
                continue
            source_task_ids.update(
                task_id for task_id in source_ids if isinstance(task_id, str)
            )
        sorted_source_task_ids = sorted(source_task_ids)
        token_values = _numeric_values(rendered_rows, "diffusion_context_tokens")
        reward_values = _numeric_values(
            rendered_rows,
            "reward_after_diffusion_context",
        )
        regression_count = sum(
            1
            for row in rendered_rows
            if row.get("regression_after_diffusion_context")
        )
        prior_context_summary = {
            "same_task_prior_tokens": _numeric_summary(
                _numeric_values(rows, "same_task_prior_tokens")
            ),
            "cross_task_prior_tokens": _numeric_summary(
                _numeric_values(rows, "cross_task_prior_tokens")
            ),
            "diffusion_context_tokens": _numeric_summary(
                _numeric_values(rows, "diffusion_context_tokens")
            ),
            "total_planner_prior_context_tokens": _numeric_summary(
                _numeric_values(rows, "total_planner_prior_context_tokens")
            ),
            "context_budget_violation_count": sum(
                1 for row in rows if row.get("context_budget_violation") is True
            ),
            "compacted_diffusion_artifact_count": len(
                set(string_list_values(rows, "compacted_diffusion_artifact_ids"))
            ),
            "dropped_for_budget_artifact_count": len(
                set(string_list_values(rows, "dropped_for_budget_artifact_ids"))
            ),
        }

        payload["metrics"] = {
            "diffusion_enabled": _single_or_mixed(rows, "diffusion_enabled"),
            "diffusion_policy": _single_or_mixed(rows, "diffusion_policy"),
            "diffusion_graph": _single_or_mixed(rows, "diffusion_graph"),
            "planner_prior_context": prior_context_summary,
            "context": {
                "rows_with_rendered_context": len(rendered_rows),
                "diffusion_context_tokens": _numeric_summary(token_values),
                "source_task_count": len(sorted_source_task_ids),
                "source_task_ids": sorted_source_task_ids,
                "reward_after_diffusion_context": _numeric_summary(reward_values),
                "regression_after_diffusion_context": {
                    "count": regression_count,
                    "rate": (
                        regression_count / len(rendered_rows)
                        if rendered_rows
                        else 0.0
                    ),
                },
            },
        }
    return payload


def _inspection_payload(experiment_dir: Path) -> dict[str, Any]:
    if not experiment_dir.exists() or not experiment_dir.is_dir():
        raise typer.BadParameter(f"experiment directory not found: {experiment_dir}")

    rows = []
    for row_dir in sorted(experiment_dir.iterdir(), key=lambda item: item.name):
        if not row_dir.is_dir():
            continue
        summary_path = row_dir / "summary.json"
        metrics_path = row_dir / "metrics.jsonl"
        if not summary_path.exists() and not metrics_path.exists():
            continue
        row: dict[str, Any] = {
            "row": row_dir.name,
            "experiment_dir": str(row_dir),
            "summary_path": str(summary_path) if summary_path.exists() else None,
            "metrics_path": str(metrics_path) if metrics_path.exists() else None,
        }
        if summary_path.exists():
            row["summary"] = _load_score_summary(summary_path).model_dump(mode="json")
        else:
            row["warning"] = "summary.json is missing; inspect metrics.jsonl directly."
        if diffusion_payload := _diffusion_inspection_payload(row_dir):
            row["diffusion"] = diffusion_payload
        rows.append(row)
    if rows:
        return {"kind": "matrix", "experiment_dir": str(experiment_dir), "rows": rows}

    summary_path = experiment_dir / "summary.json"
    metrics_path = experiment_dir / "metrics.jsonl"
    diffusion_payload = _diffusion_inspection_payload(experiment_dir)
    if summary_path.exists():
        payload = {
            "kind": "single",
            "experiment_dir": str(experiment_dir),
            "summary_path": str(summary_path),
            "metrics_path": str(metrics_path) if metrics_path.exists() else None,
            "artifact_dirs": _artifact_dirs(experiment_dir),
            "summary": _load_score_summary(summary_path).model_dump(mode="json"),
        }
        if diffusion_payload is not None:
            payload["diffusion"] = diffusion_payload
        return payload
    if metrics_path.exists():
        payload = {
            "kind": "single",
            "experiment_dir": str(experiment_dir),
            "summary_path": None,
            "metrics_path": str(metrics_path),
            "artifact_dirs": _artifact_dirs(experiment_dir),
            "warning": "summary.json is missing; inspect metrics.jsonl directly.",
        }
        if diffusion_payload is not None:
            payload["diffusion"] = diffusion_payload
        return payload
    raise typer.BadParameter(
        f"no summary.json or metrics.jsonl found under {experiment_dir}"
    )
