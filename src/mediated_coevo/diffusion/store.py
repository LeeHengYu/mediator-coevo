"""File-backed storage for diffusion artifacts and unified audit records."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from mediated_coevo.diffusion.models import (
    DiffusedRecord,
    DiffusionArtifact,
    TaskGraphSnapshot,
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


class DiffusionStore:
    """Persists diffusion artifacts, graph snapshots, and audit ledgers."""

    _DIFFUSED_RECORDS_FILE = "diffused_records.jsonl"

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._artifacts_dir = base_dir / "artifacts"
        self._graph_snapshots_dir = base_dir / "graph_snapshots"
        self._diffused_records_path = base_dir / self._DIFFUSED_RECORDS_FILE

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

    def append_diffused_record(self, record: DiffusedRecord) -> Path:
        """Append one unified diffusion audit record."""
        return append_jsonl(self._diffused_records_path, record)

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
        recent: int | None = 50,
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
            key=lambda artifact: (
                artifact.source_iteration,
                artifact.created_at,
                artifact.artifact_id,
            ),
            reverse=True,
        )
        if recent is None:
            return filtered
        return filtered[:recent]

    def query_graph_snapshots(
        self,
        *,
        run_id: str | None = None,
        recent: int | None = 50,
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
            and (before_iteration is None or snapshot.iteration < before_iteration)
        ]
        filtered.sort(
            key=lambda snapshot: (snapshot.iteration, snapshot.created_at, snapshot.snapshot_id),
            reverse=True,
        )
        if recent is None:
            return filtered
        return filtered[:recent]

    def query_diffused_records(
        self,
        *,
        target_task_id: str | None = None,
        artifact_id: str | None = None,
        recent: int | None = 50,
        before_target_iteration: int | None = None,
    ) -> list[DiffusedRecord]:
        """Query unified diffusion audit records, most recent first."""
        records = load_jsonl_models(
            self._diffused_records_path,
            DiffusedRecord,
            logger=logger,
        )
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
            key=lambda record: (
                record.target_iteration,
                record.created_at,
                record.record_id,
            ),
            reverse=True,
        )
        if recent is None:
            return filtered
        return filtered[:recent]

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

    def _load_directory_models(
        self,
        directory: Path,
        model_cls: type[_TFileModel],
    ) -> list[_TFileModel]:
        return load_directory_models(directory, model_cls, logger=logger)
