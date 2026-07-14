"""One-shot construction, archiving, and failure handling for one sample run."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import tomli_w

from mediated_coevo.artifacts.adapters import (
    DiffusionArtifactBankUpdater,
    DiffusionEmitterProjector,
)
from mediated_coevo.artifacts.models import ArtifactBankUpdate
from mediated_coevo.diffusion.emitter import DiffusionEmitter
from mediated_coevo.diffusion.models import DiffusionArtifact, TaskGraphSnapshot
from mediated_coevo.diffusion.policy_agent import LangChainDiffusionPolicyAgent
from mediated_coevo.diffusion.store import DiffusionStore
from mediated_coevo.diffusion.task_graph_agent import LangChainTaskGraphAgent
from mediated_coevo.execution.adapters import ExplicitContextOrchestratorExecutionAgent
from mediated_coevo.execution.models import empty_context_pack, is_sensitive_key
from mediated_coevo.experiment.sample_archive import (
    ARCHIVE_MANIFEST_FILENAME,
    SAMPLE_RESULT_FILENAME,
    WARMUP_BUNDLE_FILENAME,
    bind_terminal_payload,
    build_archive_manifest,
    credential_archive_paths,
    external_archive_refs,
    load_sample_result,
    load_warmup_bundle,
    nonportable_archive_paths,
    sanitize_archive_workspace,
    sensitive_archive_paths,
    write_archive_manifest,
    write_model_atomic,
    write_or_validate_model,
    write_position_journal,
)
from mediated_coevo.experiment.sample_models import (
    FailureRecord,
    FailureStage,
    PositionJournal,
    RunProgress,
    RuntimeProvenance,
    SampleExecution,
    SampleResult,
    SampleRunError,
    SampleSpec,
    SampleTaskRecord,
    SequenceSpec,
    TaskRecord,
    WarmupBundle,
    WarmupExecution,
    WarmupTaskRecord,
)
from mediated_coevo.experiment.sample_runner import (
    PositionCompleteCallback,
    SampleRunner,
)
from mediated_coevo.orchestration.adapters import (
    DiffusionContextPacker,
    LangChainDiffusionPolicyAdapter,
    LangChainTaskGraphAdapter,
    RandomPolicyAgent,
)

if TYPE_CHECKING:
    from mediated_coevo.experiment.orchestrator import Orchestrator


_SEQUENCE_SPEC_FILENAME = "sequence_spec.json"
_SAMPLE_SPEC_FILENAME = "sample_spec.json"
_WARMUP_REFERENCE_FILENAME = "warmup_ref.json"
_WARMUP_FAILURE_FILENAME = "warmup_failure.json"
_SAMPLE_FAILURE_FILENAME = "sample_failure.json"

_CONTROL_FILENAMES = {
    ARCHIVE_MANIFEST_FILENAME,
    SAMPLE_RESULT_FILENAME,
    WARMUP_BUNDLE_FILENAME,
    _SAMPLE_SPEC_FILENAME,
    _WARMUP_REFERENCE_FILENAME,
    _WARMUP_FAILURE_FILENAME,
    _SAMPLE_FAILURE_FILENAME,
}


@dataclass(slots=True)
class SampleRuntime:
    """A fresh run-specific workspace that permits exactly one operation."""

    orchestrator: Orchestrator
    run_id: str
    sequence_dir: Path
    implementation_revision: str
    implementation_dirty: bool
    runner: SampleRunner
    diffusion_store: DiffusionStore
    _used: bool = field(default=False, init=False, repr=False)
    _started_at: datetime | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.run_id = _validate_path_component(self.run_id, label="run_id")
        self.sequence_dir = Path(self.sequence_dir).resolve()
        if (
            not self.implementation_revision
            or self.implementation_revision != self.implementation_revision.strip()
        ):
            raise ValueError(
                "implementation_revision must be a non-empty stripped string"
            )

    async def prepare_warmup(self, sequence: SequenceSpec) -> WarmupBundle:
        """Execute and archive the arm-neutral prefix exactly once."""
        sequence, workspace, started_at = self._begin_warmup(sequence)

        async def journal_callback(journal: PositionJournal) -> str:
            return self._write_journal(workspace, journal)

        try:
            execution = await self.runner.prepare_warmup(
                sequence,
                warmup_run_id=self.run_id,
                on_position_complete=journal_callback,
            )
        except SampleRunError as run_error:
            self._persist_failure(workspace, run_error, warmup=True)
            raise
        except Exception as cause:
            wrapped_error = self._runtime_error(
                stage=FailureStage.FINALIZE,
                sequence=sequence,
                position=0,
                cause=cause,
                sample_id=None,
            )
            self._persist_failure(workspace, wrapped_error, warmup=True)
            raise wrapped_error from cause
        return self._finish_warmup(sequence, workspace, execution, started_at)

    async def prepare_warmup_from_stores(
        self,
        sequence: SequenceSpec,
        *,
        artifact_store_root: Path,
    ) -> WarmupBundle:
        """Build the arm-neutral prefix from pre-run task artifact stores."""
        sequence, workspace, started_at = self._begin_warmup(sequence)
        try:
            root = Path(artifact_store_root).resolve()
            loaded: list[tuple[int, Any, tuple[DiffusionArtifact, ...]]] = []
            seen_ids: set[str] = set()
            for position, task in enumerate(sequence.tasks[: sequence.warmup_count]):
                store_dir = (root / task.task_id).resolve()
                if not store_dir.is_relative_to(root):
                    raise ValueError(f"artifact store escapes its root: {task.task_id}")
                artifacts = DiffusionStore.load_artifact_store(
                    store_dir,
                    expected_store_id=task.task_id,
                )
                if any(
                    artifact.source_task_id != task.task_id for artifact in artifacts
                ):
                    raise ValueError(
                        f"artifact store contains another task's artifacts: {task.task_id}"
                    )
                repeated = seen_ids.intersection(
                    artifact.artifact_id for artifact in artifacts
                )
                if repeated:
                    raise ValueError(
                        "warm-up artifact stores repeat artifact IDs: "
                        f"{sorted(repeated)!r}"
                    )
                seen_ids.update(artifact.artifact_id for artifact in artifacts)
                loaded.append((position, task, artifacts))

            records: list[WarmupTaskRecord] = []
            bank: list[DiffusionArtifact] = []
            journal_paths: list[str] = []
            for position, task, artifacts in loaded:
                before = tuple(artifact.artifact_id for artifact in bank)
                normalized: list[DiffusionArtifact] = []
                for artifact in artifacts:
                    metadata = dict(artifact.metadata)
                    metadata.update(
                        {
                            "preloaded_from_artifact_store": task.task_id,
                            "original_source_iteration": artifact.source_iteration,
                            "original_source_run_id": artifact.source_run_id,
                            "preloaded_artifact_store_frozen": False,
                        }
                    )
                    normalized.append(
                        artifact.model_copy(
                            update={
                                "source_iteration": position,
                                "source_run_id": self.run_id,
                                "metadata": metadata,
                            }
                        )
                    )
                added = tuple(normalized)
                after = (*before, *(artifact.artifact_id for artifact in added))
                update = ArtifactBankUpdate(
                    run_id=self.run_id,
                    position=position,
                    task_id=task.task_id,
                    before_artifact_ids=before,
                    added_artifacts=added,
                    after_artifact_ids=after,
                )
                record = WarmupTaskRecord(
                    run_id=self.run_id,
                    sequence_id=sequence.sequence_id,
                    position=position,
                    task=task,
                    artifact_ids_before=before,
                    context=empty_context_pack(),
                    artifact_store_id=task.task_id,
                    bank_update=update,
                )
                for artifact in added:
                    self.diffusion_store.store_artifact(artifact)
                bank.extend(added)
                records.append(record)
                journal_paths.append(
                    self._write_journal(
                        workspace,
                        PositionJournal(
                            run_id=self.run_id,
                            sequence_id=sequence.sequence_id,
                            position=position,
                            task_record=record,
                            bank_artifact_ids=after,
                        ),
                    )
                )
            execution = WarmupExecution(
                sequence_id=sequence.sequence_id,
                warmup_run_id=self.run_id,
                task_records=tuple(records),
                final_artifact_bank=tuple(bank),
                completed_journal_paths=tuple(journal_paths),
            )
        except Exception as cause:
            wrapped_error = self._runtime_error(
                stage=FailureStage.PERSIST,
                sequence=sequence,
                position=0,
                cause=cause,
                sample_id=None,
            )
            self._persist_failure(workspace, wrapped_error, warmup=True)
            raise wrapped_error from cause
        return self._finish_warmup(sequence, workspace, execution, started_at)

    def _begin_warmup(
        self,
        sequence: SequenceSpec,
    ) -> tuple[SequenceSpec, Path, datetime]:
        sequence = SequenceSpec.model_validate(sequence)
        workspace = self._claim(
            expected_workspace=self.sequence_dir / "warmup" / self.run_id
        )
        self._assert_fresh(workspace)
        self._assert_no_successful_warmup()
        started_at = datetime.now(UTC)
        self._started_at = started_at
        try:
            self._sanitize_config_snapshot(workspace)
            self._persist_sequence(sequence)
        except Exception as cause:
            wrapped_error = self._runtime_error(
                stage=FailureStage.PERSIST,
                sequence=sequence,
                position=0,
                cause=cause,
                sample_id=None,
            )
            self._persist_failure(workspace, wrapped_error, warmup=True)
            raise wrapped_error from cause
        return sequence, workspace, started_at

    def _finish_warmup(
        self,
        sequence: SequenceSpec,
        workspace: Path,
        execution: WarmupExecution,
        started_at: datetime,
    ) -> WarmupBundle:

        try:
            provenance = self._provenance(started_at=started_at)
            manifest = self._manifest(workspace, records=execution.task_records)
            execution = self._reload_warmup_execution(workspace, execution)
            bundle = WarmupBundle.create(
                sequence_id=sequence.sequence_id,
                warmup_run_id=self.run_id,
                warmup_count=sequence.warmup_count,
                task_records=execution.task_records,
                final_artifact_bank=execution.final_artifact_bank,
                archive_manifest=manifest,
                provenance=provenance,
            )
            manifest = bind_terminal_payload(manifest, bundle)
            bundle = WarmupBundle.create(
                sequence_id=sequence.sequence_id,
                warmup_run_id=self.run_id,
                warmup_count=sequence.warmup_count,
                task_records=execution.task_records,
                final_artifact_bank=execution.final_artifact_bank,
                archive_manifest=manifest,
                provenance=provenance,
            )
            self._publish_success(
                workspace=workspace,
                manifest=manifest,
                terminal_path=workspace / WARMUP_BUNDLE_FILENAME,
                terminal_model=bundle,
                terminal_label="Warm-up bundle",
            )
            return bundle
        except Exception as cause:
            wrapped_error = self._finalize_error_from_warmup(
                sequence=sequence,
                execution=execution,
                cause=cause,
            )
            self._persist_failure(workspace, wrapped_error, warmup=True)
            raise wrapped_error from cause

    async def run(
        self,
        spec: SampleSpec,
        *,
        warmup: WarmupBundle | None = None,
        on_position_complete: PositionCompleteCallback | None = None,
    ) -> SampleResult:
        """Execute and archive only the arm-specific suffix."""
        spec = SampleSpec.model_validate(spec)
        if warmup is not None:
            warmup = WarmupBundle.model_validate(warmup)
        if spec.sample_id != self.run_id:
            raise ValueError("SampleRuntime run_id must equal SampleSpec.sample_id")
        workspace = self._claim(
            expected_workspace=self.sequence_dir / "samples" / spec.sample_id
        )
        self._assert_fresh(workspace)
        started_at = datetime.now(UTC)
        self._started_at = started_at

        try:
            self._sanitize_config_snapshot(workspace)
            self._persist_sequence(spec.sequence)
            write_model_atomic(
                workspace / _SAMPLE_SPEC_FILENAME,
                spec,
                exists_error_prefix="Sample spec",
            )
        except Exception as cause:
            wrapped_error = self._runtime_error(
                stage=FailureStage.PERSIST,
                sequence=spec.sequence,
                position=spec.sequence.warmup_count,
                cause=cause,
                sample_id=spec.sample_id,
            )
            self._persist_failure(workspace, wrapped_error, warmup=False)
            raise wrapped_error from cause

        try:
            reference = self._prepare_warmup_reference(
                workspace=workspace,
                spec=spec,
                warmup=warmup,
            )
        except Exception as cause:
            wrapped_error = self._runtime_error(
                stage=FailureStage.PERSIST,
                sequence=spec.sequence,
                position=spec.sequence.warmup_count,
                cause=cause,
                sample_id=spec.sample_id,
            )
            self._persist_failure(workspace, wrapped_error, warmup=False)
            raise wrapped_error from cause

        async def journal_callback(journal: PositionJournal) -> str:
            path = self._write_journal(workspace, journal)
            if on_position_complete is not None:
                result = on_position_complete(journal)
                if inspect.isawaitable(result):
                    await result
            return path

        try:
            execution = await self.runner.run(
                spec,
                warmup=warmup,
                on_position_complete=journal_callback,
            )
            if execution.warmup_reference != reference:
                raise ValueError(
                    "sample runner returned a different shared warm-up reference"
                )
        except SampleRunError as run_error:
            self._persist_failure(workspace, run_error, warmup=False)
            raise
        except Exception as cause:
            wrapped_error = self._runtime_error(
                stage=FailureStage.FINALIZE,
                sequence=spec.sequence,
                position=spec.sequence.warmup_count,
                cause=cause,
                sample_id=spec.sample_id,
            )
            self._persist_failure(workspace, wrapped_error, warmup=False)
            raise wrapped_error from cause

        try:
            provenance = self._provenance(started_at=started_at)
            manifest = self._manifest(
                workspace,
                records=execution.task_records,
            )
            execution = self._reload_sample_execution(workspace, execution)
            result = SampleResult(
                spec=execution.spec,
                warmup_reference=execution.warmup_reference,
                task_records=execution.task_records,
                rewards=execution.rewards,
                final_artifact_bank=execution.final_artifact_bank,
                final_graph=execution.final_graph,
                archive_manifest=manifest,
                provenance=provenance,
            )
            manifest = bind_terminal_payload(manifest, result)
            result = SampleResult(
                spec=execution.spec,
                warmup_reference=execution.warmup_reference,
                task_records=execution.task_records,
                rewards=execution.rewards,
                final_artifact_bank=execution.final_artifact_bank,
                final_graph=execution.final_graph,
                archive_manifest=manifest,
                provenance=provenance,
            )
            self._publish_success(
                workspace=workspace,
                manifest=manifest,
                terminal_path=workspace / SAMPLE_RESULT_FILENAME,
                terminal_model=result,
                terminal_label="Sample result",
            )
            return result
        except Exception as cause:
            wrapped_error = self._finalize_error_from_sample(
                execution=execution,
                cause=cause,
            )
            self._persist_failure(workspace, wrapped_error, warmup=False)
            raise wrapped_error from cause

    def _claim(self, *, expected_workspace: Path) -> Path:
        if self._used:
            raise RuntimeError("SampleRuntime instances are single-use")
        self._used = True
        workspace = Path(self.orchestrator.experiment_dir).resolve()
        expected = expected_workspace.resolve()
        if workspace != expected:
            raise ValueError(
                "Orchestrator workspace does not match the sample archive layout: "
                f"expected {expected}, got {workspace}"
            )
        self._validate_runtime_layout(workspace)
        return workspace

    def _validate_runtime_layout(self, workspace: Path) -> None:
        """Require durable state stores to belong to the claimed workspace."""
        expected_roots = (
            (self.diffusion_store, "_base_dir", "diffusion"),
            (
                getattr(self.orchestrator, "artifact_store", None),
                "_base_dir",
                "artifacts",
            ),
            (
                getattr(self.orchestrator, "history_store", None),
                "_history_dir",
                "history",
            ),
            (self.orchestrator, "_snapshots_dir", "skills_snapshots"),
            (self.orchestrator, "_metrics_path", "metrics.jsonl"),
            (
                getattr(self.orchestrator, "executor", None),
                "_workspace_root",
                "benchmarks",
            ),
            (getattr(self.orchestrator, "skill_store", None), "_skills_dir", "skills"),
        )
        for owner, attribute, relative in expected_roots:
            if owner is None or not hasattr(owner, attribute):
                continue
            actual_value = getattr(owner, attribute)
            if actual_value is None:
                continue
            actual = Path(actual_value).resolve()
            expected = (workspace / relative).resolve()
            if actual != expected or not actual.is_relative_to(workspace):
                raise ValueError(
                    "sample runtime state store is outside its claimed workspace: "
                    f"{attribute} expected {expected}, got {actual}"
                )

        jobs_value = _nested_attr(
            self.orchestrator,
            "config.executor_runtime.jobs_dir",
            default="jobs",
        )
        configured_jobs = Path(str(jobs_value))
        configured_jobs = (
            configured_jobs.resolve()
            if configured_jobs.is_absolute()
            else (workspace / configured_jobs).resolve()
        )
        if not configured_jobs.is_relative_to(workspace):
            raise ValueError(
                "sample runtime state store is outside its claimed workspace: "
                f"configured jobs_dir got {configured_jobs}"
            )
        harbor_runner = getattr(
            getattr(self.orchestrator, "executor", None),
            "_harbor_runner",
            None,
        )
        if harbor_runner is not None and hasattr(harbor_runner, "jobs_dir"):
            actual_jobs = Path(harbor_runner.jobs_dir).resolve()
            if actual_jobs != configured_jobs:
                raise ValueError(
                    "sample runtime state store is outside its claimed workspace: "
                    f"jobs_dir expected {configured_jobs}, got {actual_jobs}"
                )

    def _assert_fresh(self, workspace: Path) -> None:
        """Reject durable or in-memory output before any agent can be called."""
        blockers: list[str] = []
        for filename in sorted(_CONTROL_FILENAMES):
            path = workspace / filename
            if path.exists():
                blockers.append(str(path))
        for relative in (
            "journal",
            "history",
            "artifacts",
            "diffusion",
            "task-graph",
            "skills_snapshots",
            "benchmarks",
        ):
            path = workspace / relative
            if path.is_file() or (
                path.is_dir() and any(item.is_file() for item in path.rglob("*"))
            ):
                blockers.append(str(path))
        metrics = workspace / "metrics.jsonl"
        if metrics.exists():
            blockers.append(str(metrics))
        jobs_dir = _nested_attr(
            self.orchestrator,
            "config.executor_runtime.jobs_dir",
            default="jobs",
        )
        jobs = Path(str(jobs_dir))
        jobs = jobs if jobs.is_absolute() else workspace / jobs
        if jobs.is_file() or (
            jobs.is_dir() and any(item.is_file() for item in jobs.rglob("*"))
        ):
            blockers.append(str(jobs))

        for name in (
            "_proposal_buffer",
            "_previous_report_by_task",
            "_released_cross_task_reports_by_task",
            "_staged_cross_task_reports_by_task",
            "_previous_reward_by_task",
            "_prior_context_by_target",
            "_diffusion_context_by_target",
            "_diffusion_sub_board",
            "_diffusion_prepared_iterations",
            "_langchain_graph_prepared_targets",
            "_diffusion_snapshot_by_iteration",
            "_diffusion_target_task_ids",
            "_explicit_execution_provenance_by_key",
        ):
            if getattr(self.orchestrator, name, None):
                blockers.append(f"in-memory:{name}")
        history_store = getattr(self.orchestrator, "history_store", None)
        for name in (
            "_entries",
            "_rejected_proposal_batches",
            "_rejected_reflection_batches",
        ):
            if getattr(history_store, name, None):
                blockers.append(f"in-memory:history_store.{name}")
        if getattr(self.orchestrator, "freeze_diffusion_artifact_store", False):
            blockers.append("in-memory:freeze_diffusion_artifact_store")
        if getattr(
            self.orchestrator,
            "preloaded_diffusion_artifact_store_path",
            None,
        ):
            blockers.append("in-memory:preloaded_diffusion_artifact_store_path")
        if getattr(
            self.orchestrator,
            "preloaded_diffusion_artifact_store_count",
            0,
        ):
            blockers.append("in-memory:preloaded_diffusion_artifact_store_count")
        if sensitive_archive_paths(workspace):
            blockers.append("sensitive archive path")
        if credential_archive_paths(
            workspace,
            secret_values=_archive_secret_values(
                _json_value(getattr(self.orchestrator, "config", None))
            ),
        ):
            blockers.append("credential-bearing archive path")
        if any(path.is_symlink() for path in workspace.rglob("*")):
            blockers.append("archive symlink")

        if blockers:
            raise ValueError(
                "sample runtime requires a fresh Orchestrator/workspace; found: "
                + ", ".join(sorted(set(blockers)))
            )

    def _assert_no_successful_warmup(self) -> None:
        warmup_root = self.sequence_dir / "warmup"
        if not warmup_root.is_dir():
            return
        successful = tuple(
            path
            for path in warmup_root.glob(f"*/{WARMUP_BUNDLE_FILENAME}")
            if path.is_file()
        )
        if successful:
            raise ValueError(
                "sequence already has a successful shared warm-up bundle: "
                + ", ".join(str(path) for path in sorted(successful))
            )

    def _sanitize_config_snapshot(self, workspace: Path) -> None:
        """Replace runtime-factory config.toml with a credential-free snapshot."""
        path = workspace / "config.toml"
        if not path.exists():
            return
        if path.is_symlink() or not path.is_file():
            raise ValueError("runtime config snapshot must be a regular file")
        payload = _redact_sensitive_config(
            _json_value(getattr(self.orchestrator, "config", None))
        )
        if not isinstance(payload, dict):
            raise ValueError("runtime config must serialize as a TOML table")
        encoded = tomli_w.dumps(payload).encode("utf-8")
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _persist_sequence(self, sequence: SequenceSpec) -> None:
        self.sequence_dir.mkdir(parents=True, exist_ok=True)
        write_or_validate_model(
            self.sequence_dir / _SEQUENCE_SPEC_FILENAME,
            sequence,
        )

    def _prepare_warmup_reference(
        self,
        *,
        workspace: Path,
        spec: SampleSpec,
        warmup: WarmupBundle | None,
    ) -> Any:
        if spec.sequence.warmup_count == 0:
            if warmup is not None:
                raise ValueError("zero-warm-up sample cannot receive a warm-up bundle")
            return None
        if warmup is None:
            raise ValueError("sample requires the declared warm-up bundle")
        if warmup.bundle_id != spec.warmup_bundle_id:
            raise ValueError("warm-up bundle ID does not match SampleSpec")
        if warmup.sequence_id != spec.sequence.sequence_id:
            raise ValueError("warm-up bundle belongs to another sequence")
        if warmup.warmup_count != spec.sequence.warmup_count:
            raise ValueError("warm-up bundle has the wrong prefix length")
        if (
            tuple(record.task for record in warmup.task_records)
            != (spec.sequence.tasks[: spec.sequence.warmup_count])
        ):
            raise ValueError("warm-up bundle differs from the frozen task prefix")

        _validate_path_component(warmup.warmup_run_id, label="warmup_run_id")

        relative_bundle = f"warmup/{warmup.warmup_run_id}/{WARMUP_BUNDLE_FILENAME}"
        relative_manifest = f"warmup/{warmup.warmup_run_id}/{ARCHIVE_MANIFEST_FILENAME}"
        persisted = load_warmup_bundle(self.sequence_dir / relative_bundle)
        if persisted != warmup:
            raise ValueError("provided warm-up bundle differs from the shared archive")
        reference = warmup.reference(
            relative_path=relative_bundle,
            manifest_path=relative_manifest,
        )
        write_model_atomic(
            workspace / _WARMUP_REFERENCE_FILENAME,
            reference,
            exists_error_prefix="Warm-up reference",
        )
        self._materialize_transfer_bank(warmup)
        return reference

    def _materialize_transfer_bank(self, warmup: WarmupBundle) -> None:
        """Copy only compact transfer artifacts, never shared Harbor archives."""
        artifact_dir = Path(getattr(self.diffusion_store, "_artifacts_dir"))
        paths = tuple(
            artifact_dir
            / f"{_validate_path_component(artifact.artifact_id, label='artifact_id')}.json"
            for artifact in warmup.final_artifact_bank
        )
        for path in paths:
            if path.exists():
                raise FileExistsError(
                    f"sample transfer artifact already exists: {path}"
                )
        try:
            for artifact in warmup.final_artifact_bank:
                self.diffusion_store.store_artifact(artifact)
        except Exception:
            # A store implementation may publish the file and then raise before
            # returning its path. Clean the entire preflighted batch, not only
            # paths whose calls returned successfully.
            for path in reversed(paths):
                path.unlink(missing_ok=True)
            raise

    def _write_journal(self, workspace: Path, journal: PositionJournal) -> str:
        credential_paths = credential_archive_paths(
            workspace,
            secret_values=_archive_secret_values(
                _json_value(getattr(self.orchestrator, "config", None))
            ),
        )
        if credential_paths:
            raise ValueError(
                "execution produced a credential-bearing archive path before journal"
            )
        invalid_paths = nonportable_archive_paths(workspace)
        if invalid_paths:
            raise ValueError(
                "execution produced non-portable archive paths before journal: "
                + ", ".join(
                    sorted(
                        path.relative_to(workspace).as_posix() for path in invalid_paths
                    )
                )
            )
        local_path = write_position_journal(workspace, journal)
        run_relative = workspace.relative_to(self.sequence_dir).as_posix()
        return f"{run_relative}/{local_path}"

    def _manifest(self, workspace: Path, *, records: tuple[Any, ...]) -> Any:
        removed_paths = sanitize_archive_workspace(
            workspace,
            secret_values=_archive_secret_values(
                _json_value(getattr(self.orchestrator, "config", None))
            ),
        )
        declared_paths = {
            path
            for record in records
            for path in (
                record.execution.archive_paths if record.execution is not None else ()
            )
        }
        for removed in removed_paths:
            if any(
                removed == declared
                or removed.startswith(f"{declared}/")
                or declared.startswith(f"{removed}/")
                for declared in declared_paths
            ):
                raise ValueError(
                    f"archive sanitizer removed a declared execution path: {removed}"
                )
        persisted_records = self._workspace_journal_records(workspace, records)
        run_relative = workspace.relative_to(self.sequence_dir).as_posix()
        include_roots = [run_relative]
        if (self.sequence_dir / _SEQUENCE_SPEC_FILENAME).is_file():
            include_roots.insert(0, _SEQUENCE_SPEC_FILENAME)
        return build_archive_manifest(
            self.sequence_dir,
            external_refs=external_archive_refs(
                persisted_records,
                workspace=self.sequence_dir,
            ),
            include_relative_roots=tuple(include_roots),
        )

    def _publish_success(
        self,
        *,
        workspace: Path,
        manifest: Any,
        terminal_path: Path,
        terminal_model: Any,
        terminal_label: str,
    ) -> None:
        manifest_path = workspace / ARCHIVE_MANIFEST_FILENAME
        if manifest_path.exists() or terminal_path.exists():
            raise FileExistsError("success terminal or archive manifest already exists")
        try:
            write_archive_manifest(workspace, manifest)
            write_model_atomic(
                terminal_path,
                terminal_model,
                exists_error_prefix=terminal_label,
            )
            loaded = (
                load_warmup_bundle(terminal_path)
                if terminal_path.name == WARMUP_BUNDLE_FILENAME
                else load_sample_result(terminal_path)
            )
            if loaded != terminal_model:
                raise ValueError(
                    "published success terminal failed its archive round-trip"
                )
        except Exception:
            # As with artifact transfer, clean both preflighted paths even if a
            # writer published its target and then raised before returning.
            for path in (terminal_path, manifest_path):
                path.unlink(missing_ok=True)
            raise

    def _persist_failure(
        self,
        workspace: Path,
        error: SampleRunError,
        *,
        warmup: bool,
    ) -> None:
        success_path = workspace / (
            WARMUP_BUNDLE_FILENAME if warmup else SAMPLE_RESULT_FILENAME
        )
        if success_path.exists():
            raise RuntimeError("refusing to persist failure beside a success terminal")
        failure_path = workspace / (
            _WARMUP_FAILURE_FILENAME if warmup else _SAMPLE_FAILURE_FILENAME
        )
        started_at = self._started_at or datetime.now(UTC)
        failure = FailureRecord.from_error(
            error,
            provenance=self._provenance(started_at=started_at),
        )
        write_model_atomic(
            failure_path,
            failure,
            exists_error_prefix="Run failure",
        )
        journal_records = self._journal_records(error.progress.completed_journal_paths)
        try:
            manifest = self._manifest(workspace, records=journal_records)
        except Exception:
            # The original staged failure remains authoritative even if a
            # completed journal contains malformed external provenance. Seal a
            # portable local inventory instead of replacing SampleRunError with
            # a secondary manifest exception.
            sanitize_archive_workspace(
                workspace,
                secret_values=_archive_secret_values(
                    _json_value(getattr(self.orchestrator, "config", None))
                ),
            )
            run_relative = workspace.relative_to(self.sequence_dir).as_posix()
            include_roots = [run_relative]
            if (self.sequence_dir / _SEQUENCE_SPEC_FILENAME).is_file():
                include_roots.insert(0, _SEQUENCE_SPEC_FILENAME)
            manifest = build_archive_manifest(
                self.sequence_dir,
                include_relative_roots=tuple(include_roots),
            )
        write_archive_manifest(workspace, manifest)

    def _journal_records(self, paths: tuple[str, ...]) -> tuple[Any, ...]:
        records: list[Any] = []
        for value in paths:
            relative = PurePosixPath(value)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "." in relative.parts
            ):
                raise ValueError(
                    f"journal path is not normalized and relative: {value!r}"
                )
            path = self.sequence_dir.joinpath(*relative.parts)
            if path.is_file():
                journal = PositionJournal.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                records.append(journal.task_record)
        return tuple(records)

    def _workspace_journal_records(
        self,
        workspace: Path,
        records: tuple[Any, ...],
    ) -> tuple[TaskRecord, ...]:
        """Reload post-sanitizer journals without trusting stale in-memory data."""
        persisted: list[TaskRecord] = []
        for record in records:
            path = workspace / "journal" / f"position-{record.position:04d}.json"
            journal = PositionJournal.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            persisted.append(journal.task_record)
        return tuple(persisted)

    def _reload_artifact_bank(
        self,
        artifacts: tuple[DiffusionArtifact, ...],
    ) -> tuple[DiffusionArtifact, ...]:
        persisted: list[DiffusionArtifact] = []
        for artifact in artifacts:
            loaded = self.diffusion_store.load_artifact(artifact.artifact_id)
            if loaded is None:
                raise ValueError(
                    f"persisted artifact is missing: {artifact.artifact_id}"
                )
            persisted.append(loaded)
        return tuple(persisted)

    def _reload_warmup_execution(
        self,
        workspace: Path,
        execution: WarmupExecution,
    ) -> WarmupExecution:
        records = tuple(
            WarmupTaskRecord.model_validate(record)
            for record in self._workspace_journal_records(
                workspace,
                execution.task_records,
            )
        )
        return WarmupExecution(
            sequence_id=execution.sequence_id,
            warmup_run_id=execution.warmup_run_id,
            task_records=records,
            final_artifact_bank=self._reload_artifact_bank(
                execution.final_artifact_bank
            ),
            completed_journal_paths=execution.completed_journal_paths,
        )

    def _reload_sample_execution(
        self,
        workspace: Path,
        execution: SampleExecution,
    ) -> SampleExecution:
        records = tuple(
            SampleTaskRecord.model_validate(record)
            for record in self._workspace_journal_records(
                workspace,
                execution.task_records,
            )
        )
        graph: TaskGraphSnapshot | None = None
        if execution.final_graph is not None:
            graph = self.diffusion_store.load_graph_snapshot(
                execution.final_graph.snapshot_id
            )
            if graph is None:
                raise ValueError(
                    "persisted final graph is missing: "
                    f"{execution.final_graph.snapshot_id}"
                )
        return SampleExecution(
            spec=execution.spec,
            warmup_reference=execution.warmup_reference,
            task_records=records,
            rewards=execution.rewards,
            final_artifact_bank=self._reload_artifact_bank(
                execution.final_artifact_bank
            ),
            final_graph=graph,
            completed_journal_paths=execution.completed_journal_paths,
        )

    def _provenance(self, *, started_at: datetime) -> RuntimeProvenance:
        config = getattr(self.orchestrator, "config", None)
        models = _json_mapping(getattr(config, "models", {}))
        model_mapping = {
            str(key): str(value) for key, value in models.items() if value is not None
        }
        executor = getattr(self.orchestrator, "executor", None)
        backend = getattr(executor, "_harbor_runner", None) or executor
        return RuntimeProvenance(
            implementation_revision=self.implementation_revision,
            implementation_dirty=self.implementation_dirty,
            config_hash=_sha256_json(_redact_sensitive_config(_json_value(config))),
            graph_implementation_hash=_source_hash(LangChainTaskGraphAgent),
            policy_implementation_hash=_source_hash(LangChainDiffusionPolicyAgent),
            harness_hash=_optional_tree_hash(
                Path(self.orchestrator.experiment_dir) / "harnesses"
            ),
            model_mapping=model_mapping,
            executor_backend=f"{type(backend).__module__}.{type(backend).__qualname__}",
            executor_agent=str(
                _nested_attr(
                    self.orchestrator,
                    "config.executor_runtime.agent_name",
                    default=type(getattr(self.orchestrator, "executor", None)).__name__,
                )
            ),
            python_version=platform.python_version(),
            package_version=_package_version(),
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    def _runtime_error(
        self,
        *,
        stage: FailureStage,
        sequence: SequenceSpec,
        position: int,
        cause: BaseException,
        sample_id: str | None,
        completed_positions: tuple[int, ...] = (),
        completed_journal_paths: tuple[str, ...] = (),
        bank_artifact_ids: tuple[str, ...] = (),
        graph_snapshot_id: str | None = None,
    ) -> SampleRunError:
        bounded_position = min(max(position, 0), len(sequence.tasks) - 1)
        return SampleRunError(
            stage=stage,
            position=bounded_position,
            task_id=sequence.tasks[bounded_position].task_id,
            progress=RunProgress(
                run_id=self.run_id,
                sequence_id=sequence.sequence_id,
                sample_id=sample_id,
                completed_positions=completed_positions,
                completed_journal_paths=completed_journal_paths,
                bank_artifact_ids=bank_artifact_ids,
                graph_snapshot_id=graph_snapshot_id,
            ),
            cause=cause,
        )

    def _finalize_error_from_warmup(
        self,
        *,
        sequence: SequenceSpec,
        execution: WarmupExecution,
        cause: BaseException,
    ) -> SampleRunError:
        return self._runtime_error(
            stage=FailureStage.FINALIZE,
            sequence=sequence,
            position=max(sequence.warmup_count - 1, 0),
            cause=cause,
            sample_id=None,
            completed_positions=tuple(
                record.position for record in execution.task_records
            ),
            completed_journal_paths=execution.completed_journal_paths,
            bank_artifact_ids=tuple(
                artifact.artifact_id for artifact in execution.final_artifact_bank
            ),
        )

    def _finalize_error_from_sample(
        self,
        *,
        execution: SampleExecution,
        cause: BaseException,
    ) -> SampleRunError:
        return self._runtime_error(
            stage=FailureStage.FINALIZE,
            sequence=execution.spec.sequence,
            position=len(execution.spec.sequence.tasks) - 1,
            cause=cause,
            sample_id=execution.spec.sample_id,
            completed_positions=tuple(
                record.position for record in execution.task_records
            ),
            completed_journal_paths=execution.completed_journal_paths,
            bank_artifact_ids=tuple(
                artifact.artifact_id for artifact in execution.final_artifact_bank
            ),
            graph_snapshot_id=(
                execution.final_graph.snapshot_id if execution.final_graph else None
            ),
        )


def build_sample_runtime(
    *,
    orchestrator: Orchestrator,
    run_id: str,
    sequence_dir: Path,
    implementation_revision: str,
    implementation_dirty: bool = False,
) -> SampleRuntime:
    """Wire a one-shot causal sample runtime from a fresh Orchestrator."""
    run_id = _validate_path_component(run_id, label="run_id")
    if (
        not implementation_revision
        or implementation_revision != implementation_revision.strip()
    ):
        raise ValueError("implementation_revision must be a non-empty stripped string")
    updates = _nested_attr(
        orchestrator,
        "config.experiment.skill_updates",
        default=None,
    )
    if updates is not None and any(
        bool(getattr(updates, role, False))
        for role in ("executor", "planner", "mediator")
    ):
        raise ValueError("sample runtime requires all online skill updates disabled")
    if _nested_attr(
        orchestrator,
        "config.experiment.baseline_preset",
        default=None,
    ):
        raise ValueError(
            "sample runtime arms cannot be layered over a legacy baseline preset"
        )

    config = orchestrator.config
    store = orchestrator.diffusion_store
    artifact_cap = int(config.diffusion.max_artifacts)
    graph = LangChainTaskGraphAdapter(
        agent=LangChainTaskGraphAgent(
            model=config.models.mediator,
            run_id=run_id,
        ),
        store=store,
    )
    diffusion_policy = LangChainDiffusionPolicyAdapter(
        agent=LangChainDiffusionPolicyAgent(
            model=config.models.mediator,
            max_artifacts=artifact_cap,
            fallback_strategy="none",
        )
    )
    random_policy = RandomPolicyAgent(max_artifacts=artifact_cap)
    packer = DiffusionContextPacker(
        store=store,
        model=orchestrator.planner.llm_client.model,
        max_context_tokens=config.budgets.max_transfer_context_tokens,
        compact_artifact_content=orchestrator._compact_diffusion_artifact_content,
    )
    execution = ExplicitContextOrchestratorExecutionAgent(backend=orchestrator)
    projector = DiffusionEmitterProjector(
        emitter=DiffusionEmitter(
            model=orchestrator.planner.llm_client.model,
            llm_client=orchestrator.mediator.llm_client,
            budgets=config.budgets,
            condition_name=config.experiment.condition_name,
        )
    )
    bank_updater = DiffusionArtifactBankUpdater(store=store)
    runner = SampleRunner(
        graph_agent=graph,
        diffusion_policy_agent=diffusion_policy,
        random_policy_agent=random_policy,
        context_packer=packer,
        execution_agent=execution,
        artifact_projector=projector,
        artifact_bank_updater=bank_updater,
    )
    return SampleRuntime(
        orchestrator=orchestrator,
        run_id=run_id,
        sequence_dir=Path(sequence_dir).resolve(),
        implementation_revision=implementation_revision,
        implementation_dirty=implementation_dirty,
        runner=runner,
        diffusion_store=store,
    )


def _nested_attr(value: Any, path: str, *, default: Any) -> Any:
    current = value
    for part in path.split("."):
        if current is None or not hasattr(current, part):
            return default
        current = getattr(current, part)
    return current


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_value(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _json_mapping(value: Any) -> dict[str, Any]:
    normalized = _json_value(value)
    return normalized if isinstance(normalized, dict) else {}


def _redact_sensitive_config(value: Any) -> Any:
    """Remove credential values before computing portable config identity."""
    if isinstance(value, dict):
        return {
            str(key): (
                "[redacted]"
                if is_sensitive_key(str(key))
                else _redact_sensitive_config(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_config(item) for item in value]
    return value


def _credential_values(value: Any) -> tuple[str, ...]:
    discovered: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                if is_sensitive_key(str(key)):
                    if isinstance(nested, str) and nested:
                        discovered.add(nested)
                    elif isinstance(nested, dict):
                        for nested_value in nested.values():
                            if isinstance(nested_value, str) and nested_value:
                                discovered.add(nested_value)
                else:
                    visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(sorted(discovered))


def _archive_secret_values(config: Any) -> tuple[str, ...]:
    values = set(_credential_values(config))
    values.update(
        value
        for key, value in os.environ.items()
        if is_sensitive_key(key) and len(value) >= 4
    )
    return tuple(sorted(values))


def _validate_path_component(value: str, *, label: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value != value.strip()
        or value in {".", ".."}
        or len(path.parts) != 1
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be one portable path component")
    return value


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_hash(value: Any) -> str:
    return hashlib.sha256(inspect.getsource(value).encode("utf-8")).hexdigest()


def _optional_tree_hash(root: Path) -> str | None:
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    found = False
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name.startswith(".env"):
            continue
        found = True
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest() if found else None


def _package_version() -> str:
    try:
        return version("mediated-coevo")
    except PackageNotFoundError:
        return "0.1.0"
