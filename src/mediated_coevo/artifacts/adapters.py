"""Projection and transactional persistence adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mediated_coevo.artifacts.models import ArtifactBankUpdate
from mediated_coevo.diffusion.emitter import DiffusionEmitter
from mediated_coevo.diffusion.models import DiffusionArtifact
from mediated_coevo.diffusion.store import DiffusionStore
from mediated_coevo.execution.models import (
    TaskExecutionResult,
    TaskProfile,
    redact_sensitive_data,
)


@dataclass(frozen=True, slots=True)
class DiffusionEmitterProjector:
    """Adapt the existing low-risk emitter without mutating durable state."""

    emitter: DiffusionEmitter

    async def project(
        self,
        *,
        task: TaskProfile,
        execution: TaskExecutionResult,
    ) -> tuple[DiffusionArtifact, ...]:
        """Return normalized transfer projections with available provenance."""
        record = execution.record
        trace = record.execution_trace
        if trace is None:
            return ()
        judge_reward = _optional_number(execution.metadata.get("judge_reward"))
        artifacts = await self.emitter.emit(
            trace=trace,
            report=record.mediator_report,
            record=record,
            task_metadata=task.normalized_metadata(),
            judge_reward=judge_reward,
        )
        normalized: list[DiffusionArtifact] = []
        for artifact in artifacts:
            metadata = dict(artifact.metadata)
            metadata.update(
                {
                    "task_profile_schema_version": task.schema_version,
                    "source_trace_status": trace.status,
                }
            )
            if trace.run_id is not None:
                metadata["source_execution_run_id"] = trace.run_id
            for key in (
                "judge_reward_record_id",
                "reward_source",
                "verifier_reward_source",
            ):
                value = execution.metadata.get(key)
                if isinstance(value, str) and value:
                    metadata[key] = value
            normalized.append(
                artifact.model_copy(
                    deep=True,
                    update={
                        "source_task_id": task.task_id,
                        "source_iteration": execution.position,
                        "source_run_id": execution.run_id,
                        "judge_reward": judge_reward,
                        "metadata": metadata,
                    },
                )
            )
        return tuple(normalized)


@dataclass(frozen=True, slots=True)
class DiffusionArtifactBankUpdater:
    """Validate, persist, and roll back one artifact append transaction."""

    store: DiffusionStore

    def prepare(
        self,
        *,
        run_id: str,
        position: int,
        task: TaskProfile,
        execution: TaskExecutionResult,
        current_bank: tuple[DiffusionArtifact, ...],
        projected_artifacts: tuple[DiffusionArtifact, ...],
    ) -> ArtifactBankUpdate:
        """Build a fully validated transition without writing any artifact."""
        if execution.run_id != run_id:
            raise ValueError("execution result belongs to a different run")
        if execution.position != position or execution.task_id != task.task_id:
            raise ValueError("execution result does not match the bank update target")
        if execution.is_infrastructure_failure:
            raise ValueError("infrastructure-failed execution cannot update the bank")
        before_ids = tuple(artifact.artifact_id for artifact in current_bank)
        if len(before_ids) != len(set(before_ids)):
            raise ValueError("current artifact bank IDs must be unique")
        if any(artifact.source_iteration >= position for artifact in current_bank):
            raise ValueError("current artifact bank contains non-causal artifacts")
        normalized_artifacts = tuple(
            DiffusionArtifact.model_validate(
                redact_sensitive_data(artifact.model_dump(mode="python"))
            )
            for artifact in projected_artifacts
        )
        added_ids = tuple(
            artifact.artifact_id for artifact in normalized_artifacts
        )
        return ArtifactBankUpdate(
            run_id=run_id,
            position=position,
            task_id=task.task_id,
            before_artifact_ids=before_ids,
            added_artifacts=normalized_artifacts,
            after_artifact_ids=(*before_ids, *added_ids),
        )

    def persist(self, update: ArtifactBankUpdate) -> tuple[Path, ...]:
        """Preflight the full batch, then persist it with all-or-none cleanup."""
        expected_paths = tuple(
            self._artifact_path(artifact.artifact_id)
            for artifact in update.added_artifacts
        )
        for artifact, path in zip(
            update.added_artifacts,
            expected_paths,
            strict=True,
        ):
            if path.exists():
                raise FileExistsError(
                    f"diffusion artifact already exists: {artifact.artifact_id}"
                )

        try:
            for artifact, expected_path in zip(
                update.added_artifacts,
                expected_paths,
                strict=True,
            ):
                path = self.store.store_artifact(artifact)
                if path != expected_path:
                    raise RuntimeError(
                        "diffusion store returned an unexpected artifact path"
                    )
        except Exception as exc:
            try:
                self.rollback(expected_paths)
            except Exception as rollback_exc:
                raise RuntimeError(
                    f"artifact persistence failed ({exc}); rollback also failed: "
                    f"{rollback_exc}"
                ) from exc
            raise
        return expected_paths

    def rollback(self, paths: tuple[Path, ...]) -> None:
        """Remove only files from this updater's artifact directory."""
        artifact_dir = self._artifact_dir().resolve()
        failures: list[str] = []
        for path in reversed(paths):
            resolved = path.resolve()
            if resolved.parent != artifact_dir:
                raise ValueError(f"refusing to roll back path outside store: {path}")
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                failures.append(f"{path}: {exc}")
        if failures:
            raise RuntimeError("artifact rollback failed: " + "; ".join(failures))

    def _artifact_dir(self) -> Path:
        directory = getattr(self.store, "_artifacts_dir", None)
        if not isinstance(directory, Path):
            raise TypeError("DiffusionStore does not expose its artifact directory")
        return directory

    def _artifact_path(self, artifact_id: str) -> Path:
        if (
            not artifact_id
            or artifact_id != artifact_id.strip()
            or artifact_id in {".", ".."}
            or "/" in artifact_id
            or "\\" in artifact_id
            or "\x00" in artifact_id
        ):
            raise ValueError("artifact_id must be a safe path component")
        return self._artifact_dir() / f"{artifact_id}.json"


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
