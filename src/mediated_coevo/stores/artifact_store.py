"""Artifact store — persists execution traces and mediator reports.

File-backed JSON store indexed by task_id and iteration.
"""

from __future__ import annotations

import difflib
import json
import logging
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from mediated_coevo.evolution.compactor import (
    TARGET_EVIDENCE_CHARS,
    head_tail_text,
    trace_header_summary,
)
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.skill import SkillUpdate, SkillUpdateCandidateBatch
from mediated_coevo.models.trace import ExecutionTrace

logger = logging.getLogger(__name__)

_T = TypeVar("_T", ExecutionTrace, MediatorReport)


class ArtifactStore:
    """Persists execution traces and mediator reports as JSON files."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._traces_dir = base_dir / "traces"
        self._reports_dir = base_dir / "reports"
        self._validation_dir = base_dir / "validation"
        self._candidate_batches_dir = base_dir / "candidate_batches"
        self._skill_updates_dir = base_dir / "skill_updates"
        self._traces_dir.mkdir(parents=True, exist_ok=True)
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._validation_dir.mkdir(parents=True, exist_ok=True)
        self._candidate_batches_dir.mkdir(parents=True, exist_ok=True)
        self._skill_updates_dir.mkdir(parents=True, exist_ok=True)

    def store_trace(self, trace: ExecutionTrace, *, overwrite: bool = False) -> Path:
        """Persist an execution trace. Returns the file path."""
        filename = f"{trace.task_id}_iter{trace.iteration:04d}.json"
        path = self._traces_dir / filename
        if path.exists() and not overwrite:
            raise FileExistsError(f"Trace already exists: {path}")
        path.write_text(trace.model_dump_json(indent=2))
        logger.debug("Stored trace: %s", path)
        return path

    def store_report(self, report: MediatorReport, *, overwrite: bool = False) -> Path:
        """Persist a mediator report. Returns the file path."""
        filename = (
            f"{report.task_id}_iter{report.iteration:04d}_{report.report_id}.json"
        )
        path = self._reports_dir / filename
        if path.exists() and not overwrite:
            raise FileExistsError(f"Report already exists: {path}")
        path.write_text(report.model_dump_json(indent=2))
        logger.debug("Stored report: %s", path)
        return path

    def store_validation_trace(
        self,
        validation_id: str,
        variant: str,
        trace: ExecutionTrace,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist an empirical validation trace outside normal run traces."""
        filename = f"{trace.task_id}_iter{trace.iteration:04d}.json"
        path = self._validation_dir / validation_id / variant / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Validation trace already exists: {path}")
        path.write_text(trace.model_dump_json(indent=2))
        logger.debug("Stored validation trace: %s", path)
        return path

    def store_validation_result(
        self,
        validation_id: str,
        result: BaseModel,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist empirical validation summary evidence."""
        path = self._validation_dir / validation_id / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Validation result already exists: {path}")
        path.write_text(result.model_dump_json(indent=2))
        logger.debug("Stored validation result: %s", path)
        return path

    def store_validation_variant_result(
        self,
        validation_id: str,
        variant: str,
        result: BaseModel,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist validation summary evidence for one candidate variant."""
        path = self._validation_dir / validation_id / variant / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Validation result already exists: {path}")
        path.write_text(result.model_dump_json(indent=2))
        logger.debug("Stored validation variant result: %s", path)
        return path

    def store_candidate_batch(
        self,
        batch: SkillUpdateCandidateBatch,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist a full candidate batch audit artifact."""
        path = self._candidate_batches_dir / f"{batch.batch_id}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Candidate batch already exists: {path}")
        path.write_text(batch.model_dump_json(indent=2))
        logger.debug("Stored candidate batch: %s", path)
        return path

    def store_skill_update(
        self,
        update_id: str,
        update: SkillUpdate,
        *,
        overwrite: bool = False,
    ) -> tuple[Path, Path | None]:
        """Persist a committed skill update and a readable content diff."""
        update_dir = self._skill_updates_dir / update_id
        update_dir.mkdir(parents=True, exist_ok=True)
        update_path = update_dir / "update.json"
        diff_path = update_dir / "changes.diff"
        if update_path.exists() and not overwrite:
            raise FileExistsError(f"Skill update already exists: {update_path}")

        update_path.write_text(update.model_dump_json(indent=2))
        diff_text = _skill_update_diff(update)
        written_diff_path = None
        if diff_text:
            if diff_path.exists() and not overwrite:
                raise FileExistsError(f"Skill update diff already exists: {diff_path}")
            diff_path.write_text(diff_text)
            written_diff_path = diff_path
        logger.debug("Stored skill update artifact: %s", update_path)
        return update_path, written_diff_path

    def append_skill_update_history(self, entry: BaseModel) -> Path:
        """Append one compact committed-update ledger entry."""
        path = self._skill_updates_dir / "skill_update_history.jsonl"
        with path.open("a", encoding="utf-8") as history_file:
            payload = json.dumps(entry.model_dump(mode="json"), sort_keys=True)
            history_file.write(payload)
            history_file.write("\n")
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
        before_iteration: int | None = None,
    ) -> list[_T]:
        """Generic query: load JSON artifacts, filtered before recent slicing."""
        results: list[_T] = []
        for path in directory.glob("*.json"):
            try:
                artifact = model_cls.model_validate_json(path.read_text())
            except Exception as e:
                logger.warning("Failed to load %s %s: %s", model_cls.__name__, path, e)
                continue
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


def _skill_update_diff(update: SkillUpdate) -> str:
    old_lines = update.old_content.splitlines(keepends=True)
    new_lines = update.new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"{update.skill_id}/SKILL.md@old",
        tofile=f"{update.skill_id}/SKILL.md@new",
    )
    return "".join(diff)
