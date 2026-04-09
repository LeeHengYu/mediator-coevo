"""Artifact store — persists execution traces and mediator reports.

File-backed JSON store indexed by task_id and iteration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.models.report import MediatorReport

logger = logging.getLogger(__name__)

_T = TypeVar("_T", bound=BaseModel)


class ArtifactStore:
    """Persists execution traces and mediator reports as JSON files."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._traces_dir = base_dir / "traces"
        self._reports_dir = base_dir / "reports"
        self._traces_dir.mkdir(parents=True, exist_ok=True)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def store_trace(self, trace: ExecutionTrace) -> Path:
        """Persist an execution trace. Returns the file path."""
        filename = f"{trace.task_id}_iter{trace.iteration:04d}.json"
        path = self._traces_dir / filename
        path.write_text(trace.model_dump_json(indent=2))
        logger.debug("Stored trace: %s", path)
        return path

    def store_report(self, report: MediatorReport) -> Path:
        """Persist a mediator report. Returns the file path."""
        filename = f"{report.task_id}_iter{report.iteration:04d}_{report.report_id}.json"
        path = self._reports_dir / filename
        path.write_text(report.model_dump_json(indent=2))
        logger.debug("Stored report: %s", path)
        return path

    def load_trace(self, task_id: str, iteration: int) -> ExecutionTrace | None:
        filename = f"{task_id}_iter{iteration:04d}.json"
        path = self._traces_dir / filename
        if not path.exists():
            return None
        return ExecutionTrace.model_validate_json(path.read_text())

    def _query_artifacts(
        self,
        directory: Path,
        model_cls: type[_T],
        task_id: str | None = None,
        recent: int = 10,
    ) -> list[_T]:
        """Generic query: load JSON artifacts, optionally filtered by task_id."""
        results: list[_T] = []
        for path in sorted(directory.glob("*.json"), reverse=True):
            if task_id and not path.name.startswith(task_id):
                continue
            try:
                results.append(model_cls.model_validate_json(path.read_text()))
            except Exception as e:
                logger.warning("Failed to load %s %s: %s", model_cls.__name__, path, e)
            if len(results) >= recent:
                break
        return results

    def query_traces(
        self,
        task_id: str | None = None,
        recent: int = 10,
    ) -> list[ExecutionTrace]:
        """Query traces, optionally filtered by task_id, most recent first."""
        return self._query_artifacts(self._traces_dir, ExecutionTrace, task_id, recent)

    def query_reports(
        self,
        task_id: str | None = None,
        recent: int = 10,
    ) -> list[MediatorReport]:
        """Query reports, optionally filtered by task_id, most recent first."""
        return self._query_artifacts(self._reports_dir, MediatorReport, task_id, recent)

    def query_summaries(
        self,
        task_id: str | None = None,
        recent: int = 5,
    ) -> list[str]:
        """Return short text summaries of recent traces for context injection."""
        traces = self.query_traces(task_id=task_id, recent=recent)
        summaries: list[str] = []
        for trace in traces:
            status = "OK" if trace.exit_code == 0 else f"FAIL(exit={trace.exit_code})"
            summary = (
                f"iter={trace.iteration} reward={trace.reward:.2f} {status}"
            )
            if trace.stderr:
                summary += f" stderr={trace.stderr[:200]}"
            summaries.append(summary)
        return summaries
