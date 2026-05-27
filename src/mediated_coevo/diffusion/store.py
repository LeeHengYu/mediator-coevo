"""File-backed storage for diffusion artifacts and audit records."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from mediated_coevo.diffusion.models import (
    CandidateRecord,
    DiffusionArtifact,
    OutcomeAssociation,
    RenderRecord,
    SelectionRecord,
    TaskGraphSnapshot,
    UseCitationRecord,
)
from mediated_coevo.stores.json_store import (
    append_jsonl,
    load_directory_models,
    load_jsonl_models,
    load_model,
    write_model,
)

logger = logging.getLogger(__name__)

_TFileModel = TypeVar("_TFileModel", DiffusionArtifact, TaskGraphSnapshot)
_TRecordModel = TypeVar(
    "_TRecordModel",
    CandidateRecord,
    SelectionRecord,
    RenderRecord,
    UseCitationRecord,
    OutcomeAssociation,
)


class DiffusionStore:
    """Persists diffusion artifacts, graph snapshots, and audit ledgers."""

    _CANDIDATE_RECORDS_FILE = "candidate_records.jsonl"
    _SELECTION_RECORDS_FILE = "selection_records.jsonl"
    _RENDER_RECORDS_FILE = "render_records.jsonl"
    _USE_CITATION_RECORDS_FILE = "use_citation_records.jsonl"
    _OUTCOME_ASSOCIATIONS_FILE = "outcome_associations.jsonl"

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._artifacts_dir = base_dir / "artifacts"
        self._graph_snapshots_dir = base_dir / "graph_snapshots"
        self._candidate_records_path = base_dir / self._CANDIDATE_RECORDS_FILE
        self._selection_records_path = base_dir / self._SELECTION_RECORDS_FILE
        self._render_records_path = base_dir / self._RENDER_RECORDS_FILE
        self._use_citation_records_path = base_dir / self._USE_CITATION_RECORDS_FILE
        self._outcome_associations_path = base_dir / self._OUTCOME_ASSOCIATIONS_FILE

        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._graph_snapshots_dir.mkdir(parents=True, exist_ok=True)

    def store_artifact(
        self,
        artifact: DiffusionArtifact,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist one diffusion artifact as a JSON file."""
        path = self._artifacts_dir / f"{artifact.artifact_id}.json"
        return self._write_model(path, artifact, overwrite=overwrite)

    def store_graph_snapshot(
        self,
        snapshot: TaskGraphSnapshot,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist one graph snapshot as a JSON file."""
        path = self._graph_snapshots_dir / f"{snapshot.snapshot_id}.json"
        return self._write_model(path, snapshot, overwrite=overwrite)

    def append_candidate_record(self, record: CandidateRecord) -> Path:
        """Append one candidate eligibility audit record."""
        return self._append_record(self._candidate_records_path, record)

    def append_selection_record(self, record: SelectionRecord) -> Path:
        """Append one selection audit record."""
        return self._append_record(self._selection_records_path, record)

    def append_render_record(self, record: RenderRecord) -> Path:
        """Append one render audit record."""
        return self._append_record(self._render_records_path, record)

    def append_use_citation_record(self, record: UseCitationRecord) -> Path:
        """Append one explicit-use audit record."""
        return self._append_record(self._use_citation_records_path, record)

    def append_outcome_association(self, record: OutcomeAssociation) -> Path:
        """Append one post-render outcome association record."""
        return self._append_record(self._outcome_associations_path, record)

    def load_artifact(self, artifact_id: str) -> DiffusionArtifact | None:
        """Load one persisted artifact by stable artifact ID."""
        path = self._artifacts_dir / f"{artifact_id}.json"
        return load_model(path, DiffusionArtifact)

    def load_graph_snapshot(self, snapshot_id: str) -> TaskGraphSnapshot | None:
        """Load one persisted graph snapshot by stable snapshot ID."""
        path = self._graph_snapshots_dir / f"{snapshot_id}.json"
        return load_model(path, TaskGraphSnapshot)

    def query_artifacts(
        self,
        *,
        source_task_id: str | None = None,
        recent: int = 50,
        before_source_iteration: int | None = None,
    ) -> list[DiffusionArtifact]:
        """Query artifacts, most recent first."""
        artifacts = self._load_directory_models(
            self._artifacts_dir,
            DiffusionArtifact,
        )
        filtered = [
            artifact
            for artifact in artifacts
            if (source_task_id is None or artifact.source_task_id == source_task_id)
            and (
                before_source_iteration is None
                or artifact.source_iteration < before_source_iteration
            )
        ]
        filtered.sort(
            key=lambda artifact: (artifact.source_iteration, artifact.created_at),
            reverse=True,
        )
        return filtered[:recent]

    def query_graph_snapshots(
        self,
        *,
        run_id: str | None = None,
        recent: int = 50,
        before_iteration: int | None = None,
    ) -> list[TaskGraphSnapshot]:
        """Query graph snapshots, most recent first."""
        snapshots = self._load_directory_models(
            self._graph_snapshots_dir,
            TaskGraphSnapshot,
        )
        filtered = [
            snapshot
            for snapshot in snapshots
            if (run_id is None or snapshot.run_id == run_id)
            and (
                before_iteration is None or snapshot.iteration < before_iteration
            )
        ]
        filtered.sort(
            key=lambda snapshot: (snapshot.iteration, snapshot.created_at),
            reverse=True,
        )
        return filtered[:recent]

    def query_candidate_records(
        self,
        *,
        target_task_id: str | None = None,
        artifact_id: str | None = None,
        recent: int = 50,
        before_target_iteration: int | None = None,
    ) -> list[CandidateRecord]:
        """Query candidate records, most recent first."""
        return self._query_target_records(
            self._candidate_records_path,
            CandidateRecord,
            target_task_id=target_task_id,
            artifact_id=artifact_id,
            recent=recent,
            before_target_iteration=before_target_iteration,
        )

    def query_selection_records(
        self,
        *,
        target_task_id: str | None = None,
        artifact_id: str | None = None,
        recent: int = 50,
        before_target_iteration: int | None = None,
    ) -> list[SelectionRecord]:
        """Query selection records, most recent first."""
        return self._query_target_records(
            self._selection_records_path,
            SelectionRecord,
            target_task_id=target_task_id,
            artifact_id=artifact_id,
            recent=recent,
            before_target_iteration=before_target_iteration,
        )

    def query_render_records(
        self,
        *,
        target_task_id: str | None = None,
        artifact_id: str | None = None,
        recent: int = 50,
        before_target_iteration: int | None = None,
    ) -> list[RenderRecord]:
        """Query render records, most recent first."""
        return self._query_target_records(
            self._render_records_path,
            RenderRecord,
            target_task_id=target_task_id,
            artifact_id=artifact_id,
            recent=recent,
            before_target_iteration=before_target_iteration,
        )

    def query_use_citation_records(
        self,
        *,
        target_task_id: str | None = None,
        artifact_id: str | None = None,
        recent: int = 50,
        before_target_iteration: int | None = None,
    ) -> list[UseCitationRecord]:
        """Query explicit-use citation records, most recent first."""
        return self._query_target_records(
            self._use_citation_records_path,
            UseCitationRecord,
            target_task_id=target_task_id,
            artifact_id=artifact_id,
            recent=recent,
            before_target_iteration=before_target_iteration,
        )

    def query_outcome_associations(
        self,
        *,
        target_task_id: str | None = None,
        artifact_id: str | None = None,
        recent: int = 50,
        before_target_iteration: int | None = None,
    ) -> list[OutcomeAssociation]:
        """Query outcome associations, most recent first."""
        return self._query_target_records(
            self._outcome_associations_path,
            OutcomeAssociation,
            target_task_id=target_task_id,
            artifact_id=artifact_id,
            recent=recent,
            before_target_iteration=before_target_iteration,
        )

    def _write_model(
        self,
        path: Path,
        model: BaseModel,
        *,
        overwrite: bool,
    ) -> Path:
        write_model(
            path,
            model,
            overwrite=overwrite,
            exists_error_prefix="Diffusion model",
        )
        logger.debug("Stored diffusion model: %s", path)
        return path

    def _append_record(self, path: Path, record: BaseModel) -> Path:
        return append_jsonl(path, record)

    def _load_directory_models(
        self,
        directory: Path,
        model_cls: type[_TFileModel],
    ) -> list[_TFileModel]:
        return load_directory_models(directory, model_cls, logger=logger)

    def _load_jsonl_records(
        self,
        path: Path,
        model_cls: type[_TRecordModel],
    ) -> list[_TRecordModel]:
        return load_jsonl_models(path, model_cls, logger=logger)

    def _query_target_records(
        self,
        path: Path,
        model_cls: type[_TRecordModel],
        *,
        target_task_id: str | None,
        artifact_id: str | None,
        recent: int,
        before_target_iteration: int | None,
    ) -> list[_TRecordModel]:
        records = self._load_jsonl_records(path, model_cls)
        filtered = [
            record
            for record in records
            if (target_task_id is None or record.target_task_id == target_task_id)
            and (artifact_id is None or record.artifact_id == artifact_id)
            and (
                before_target_iteration is None
                or record.target_iteration < before_target_iteration
            )
        ]
        filtered.sort(
            key=lambda record: (record.target_iteration, record.created_at),
            reverse=True,
        )
        return filtered[:recent]
