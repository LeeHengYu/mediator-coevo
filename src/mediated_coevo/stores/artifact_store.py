"""Artifact store — persists execution traces and mediator reports.

File-backed JSON store indexed by task_id and iteration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeVar

from mediated_coevo.runtime.context_compactor import (
    TARGET_EVIDENCE_CHARS,
    head_tail_text,
    trace_header_summary,
)
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.stores.json_store import (
    load_directory_models,
    load_model,
    write_model,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T", ExecutionTrace, MediatorReport)


class ArtifactStore:
    """Persists execution traces and mediator reports as JSON files."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._traces_dir = base_dir / "traces"
        self._reports_dir = base_dir / "reports"
        self._traces_dir.mkdir(parents=True, exist_ok=True)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def store_trace(self, trace: ExecutionTrace, *, overwrite: bool = False) -> Path:
        """Persist an execution trace. Returns the file path."""
        filename = f"{trace.task_id}_iter{trace.iteration:04d}.json"
        path = self._traces_dir / filename
        write_model(
            path,
            trace,
            overwrite=overwrite,
            exists_error_prefix="Trace",
        )
        logger.debug("Stored trace: %s", path)
        return path

    def store_report(self, report: MediatorReport, *, overwrite: bool = False) -> Path:
        """Persist a mediator report. Returns the file path."""
        filename = (
            f"{report.task_id}_iter{report.iteration:04d}_{report.report_id}.json"
        )
        path = self._reports_dir / filename
        write_model(
            path,
            report,
            overwrite=overwrite,
            exists_error_prefix="Report",
        )
        logger.debug("Stored report: %s", path)
        return path

    def load_trace(self, task_id: str, iteration: int) -> ExecutionTrace | None:
        filename = f"{task_id}_iter{iteration:04d}.json"
        path = self._traces_dir / filename
        return load_model(path, ExecutionTrace)

    def _query_artifacts(
        self,
        directory: Path,
        model_cls: type[_T],
        task_id: str | None = None,
        recent: int = 10,
        before_iteration: int | None = None,
    ) -> list[_T]:
        """Generic query: load JSON artifacts, filtered before recent slicing."""
        results: list[_T] = []
        for artifact in load_directory_models(directory, model_cls, logger=logger):
            if task_id is not None and artifact.task_id != task_id:
                continue
            if before_iteration is not None and artifact.iteration >= before_iteration:
                continue
            results.append(artifact)

        results.sort(
            key=lambda artifact: (
                artifact.iteration,
                artifact.timestamp,
            ),
            reverse=True,
        )
        return results[:recent]

    def query_traces(
        self,
        task_id: str | None = None,
        recent: int = 10,
        before_iteration: int | None = None,
    ) -> list[ExecutionTrace]:
        """Query traces, optionally filtered by task_id and iteration cutoff."""
        return self._query_artifacts(
            self._traces_dir,
            ExecutionTrace,
            task_id,
            recent,
            before_iteration=before_iteration,
        )

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
        before_iteration: int | None = None,
    ) -> list[str]:
        """Return short text summaries of recent traces for context injection."""
        traces = self.query_traces(
            task_id=task_id,
            recent=recent,
            before_iteration=before_iteration,
        )
        summaries: list[str] = []
        for trace in traces:
            summary = trace_header_summary(trace)
            if trace.stderr:
                summary += (
                    f" stderr={head_tail_text(trace.stderr, TARGET_EVIDENCE_CHARS)}"
                )
            summaries.append(summary)
        return summaries
