"""File-backed storage for diffusion artifacts and unified audit records."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mediated_coevo.diffusion.models import (
    DiffusedRecord,
    DiffusionArtifact,
    TaskGraphSnapshot,
)
from mediated_coevo.stores.json_store import (
    append_jsonl,
    load_jsonl_dicts,
    load_directory_models,
    load_jsonl_models,
    load_model,
    write_model,
)

logger = logging.getLogger(__name__)


class DiffusionStore:
    """Persists diffusion artifacts, graph snapshots, and audit ledgers."""

    _DIFFUSED_RECORDS_FILE = "diffused_records.jsonl"
    _MANIFEST_FILE = "manifest.json"

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
        write_model(
            path,
            artifact,
            overwrite=overwrite,
            exists_error_prefix="Diffusion model",
        )
        logger.debug("Stored diffusion model: %s", path)
        return path

    def store_graph_snapshot(
        self,
        snapshot: TaskGraphSnapshot,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist one graph snapshot as a JSON file."""
        path = self._graph_snapshots_dir / f"{snapshot.snapshot_id}.json"
        write_model(
            path,
            snapshot,
            overwrite=overwrite,
            exists_error_prefix="Diffusion model",
        )
        logger.debug("Stored diffusion model: %s", path)
        return path

    def append_diffused_record(self, record: DiffusedRecord) -> Path:
        """Append one unified diffusion audit record."""
        return append_jsonl(self._diffused_records_path, record)

    def append_audit_record(self, filename: str, payload: dict) -> Path:
        """Append one JSON object row to a diffusion audit ledger."""
        if "/" in filename or filename.startswith("."):
            raise ValueError(f"invalid diffusion audit filename: {filename!r}")
        path = self._base_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(payload, sort_keys=True))
            output.write("\n")
        return path

    def load_audit_records(self, filename: str) -> list[dict]:
        """Load JSON object rows from a diffusion audit ledger."""
        if "/" in filename or filename.startswith("."):
            raise ValueError(f"invalid diffusion audit filename: {filename!r}")
        return load_jsonl_dicts(
            self._base_dir / filename,
            missing_ok=True,
            skip_non_dict=True,
        )

    def save_artifact_store(self, destination: Path, *, store_id: str) -> int:
        """Save planner-visible artifacts into a portable artifact store."""
        return self.export_artifact_store(
            self._base_dir,
            destination,
            store_id=store_id,
        )

    def import_artifact_store(
        self,
        source: Path,
        *,
        initial_source_iteration: int = -1,
        frozen: bool = False,
    ) -> int:
        """Import saved artifacts as pre-warmup artifacts."""
        artifacts = _load_saved_artifacts(source)
        for artifact in artifacts:
            metadata = dict(artifact.metadata)
            metadata["preloaded_from_artifact_store"] = str(source)
            metadata["original_source_iteration"] = artifact.source_iteration
            metadata["preloaded_artifact_store_frozen"] = frozen
            preloaded = artifact.model_copy(
                update={
                    "source_iteration": initial_source_iteration,
                    "metadata": metadata,
                }
            )
            self.store_artifact(preloaded, overwrite=True)
        return len(artifacts)

    @classmethod
    def export_artifact_store(
        cls,
        source: Path,
        destination: Path,
        *,
        store_id: str,
    ) -> int:
        """Copy the final artifact state from one diffusion store to another."""
        artifacts = _load_saved_artifacts(source)
        if (destination / cls._MANIFEST_FILE).exists():
            raise FileExistsError(f"artifact store already exists: {destination}")
        artifacts_dir = destination / "artifacts"
        if artifacts_dir.exists() and any(artifacts_dir.glob("*.json")):
            raise FileExistsError(
                f"artifact store already has artifacts: {destination}"
            )
        destination.mkdir(parents=True, exist_ok=True)

        target = cls(destination)
        for artifact in artifacts:
            target.store_artifact(artifact, overwrite=True)
        manifest = {
            "id": store_id,
            "artifact_count": len(artifacts),
            "source": str(source),
        }
        (destination / cls._MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return len(artifacts)

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
        artifacts = load_directory_models(
            self._artifacts_dir,
            DiffusionArtifact,
            logger=logger,
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
        snapshots = load_directory_models(
            self._graph_snapshots_dir,
            TaskGraphSnapshot,
            logger=logger,
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


def _load_saved_artifacts(source: Path) -> list[DiffusionArtifact]:
    artifacts_dir = source / "artifacts"
    if not artifacts_dir.is_dir():
        raise ValueError(f"artifact store has no artifacts directory: {source}")
    artifacts = load_directory_models(
        artifacts_dir,
        DiffusionArtifact,
        logger=logger,
    )
    if not artifacts:
        raise ValueError(f"artifact store contains no diffusion artifacts: {source}")
    return artifacts
