"""Adapters for benchmark resolution and the existing task backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit, urlunsplit

from mediated_coevo.execution.models import (
    ContextPack,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskProfile,
    redact_sensitive_data,
)
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.trace import ExecutionTrace


class ExplicitContextTaskBackend(Protocol):
    """Backend seam that receives a complete pack and performs no discovery."""

    async def execute_task_with_context(
        self,
        *,
        task_id: str,
        position: int,
        context: ContextPack,
        task: TaskProfile,
    ) -> IterationRecord: ...


class BenchmarkTaskRepository(Protocol):
    """Minimal benchmark repository interface needed for task profiles."""

    def resolve(self, task_id: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class BenchmarkTaskProfileProvider:
    """Resolve repository tasks into frozen, normalized execution profiles."""

    repository: BenchmarkTaskRepository

    def resolve(self, task_id: str) -> TaskProfile:
        """Return a detached profile without exposing repository-owned values."""
        task = self.repository.resolve(task_id)
        resolved_task_id = str(getattr(task, "task_id", task_id))
        if resolved_task_id != task_id:
            raise ValueError("benchmark repository returned a different task ID")
        task_config = getattr(task, "task_config", {}) or {}
        if hasattr(task_config, "model_dump"):
            task_config = task_config.model_dump(mode="json")
        return TaskProfile(
            task_id=task_id,
            instruction=str(getattr(task, "instruction", "")),
            task_config=task_config,
        )


@dataclass(frozen=True, slots=True)
class ExplicitContextOrchestratorExecutionAgent:
    """Adapt the explicit Orchestrator seam to the execution-agent protocol."""

    backend: ExplicitContextTaskBackend

    async def execute(self, request: TaskExecutionRequest) -> TaskExecutionResult:
        """Forward the entire context pack, never merely its rendered text."""
        record = await self.backend.execute_task_with_context(
            task_id=request.task.task_id,
            position=request.position,
            context=request.context,
            task=request.task,
        )
        take_provenance = getattr(
            self.backend,
            "take_explicit_execution_provenance",
            None,
        )
        execution_provenance = (
            take_provenance(task_id=request.task.task_id, position=request.position)
            if callable(take_provenance)
            else {}
        )
        record = IterationRecord.model_validate(
            redact_sensitive_data(record.model_dump(mode="python"))
        )
        record, archive_paths, external_refs = _portable_execution_record(
            record,
            workspace=getattr(self.backend, "experiment_dir", None),
        )
        _persist_portable_trace_if_supported(self.backend, record)
        _validate_record_context(record=record, context=request.context)
        safe_provenance = (
            redact_sensitive_data(execution_provenance)
            if isinstance(execution_provenance, dict)
            else {}
        )
        metadata: dict[str, Any] = {
            key: value
            for key, value in safe_provenance.items()
            if key
            not in {"phase", "arm", "context_policy", "context_snapshot_id"}
        }
        provenance_refs = metadata.pop("external_archive_refs", ())
        combined_external_refs = tuple(external_refs)
        if isinstance(provenance_refs, (list, tuple)):
            combined_external_refs = (*combined_external_refs, *provenance_refs)
        if combined_external_refs:
            metadata["external_archive_refs"] = combined_external_refs
        metadata.update({
            "phase": request.phase,
            "context_policy": request.context.policy_name,
            "context_snapshot_id": request.context.snapshot_id,
        })
        if request.arm is not None:
            metadata["arm"] = request.arm
        return TaskExecutionResult(
            run_id=request.run_id,
            position=request.position,
            task_id=request.task.task_id,
            record=record,
            archive_paths=archive_paths,
            metadata=metadata,
        )


def _portable_execution_record(
    record: IterationRecord,
    *,
    workspace: Any,
) -> tuple[IterationRecord, tuple[str, ...], tuple[dict[str, Any], ...]]:
    """Localize Harbor paths and separate explicitly external provenance."""
    trace = record.execution_trace
    if trace is None or not trace.harbor_paths:
        return record, (), ()
    portable_trace, archive_paths, external_refs = portable_execution_trace(
        trace,
        workspace=workspace,
    )
    portable_record = record.model_copy(
        deep=True,
        update={"execution_trace": portable_trace},
    )
    return portable_record, archive_paths, external_refs


def portable_execution_trace(
    trace: ExecutionTrace,
    *,
    workspace: Any,
) -> tuple[ExecutionTrace, tuple[str, ...], tuple[dict[str, Any], ...]]:
    """Localize trace paths before any explicit-context persistence."""
    workspace_path = (
        Path(workspace).resolve() if isinstance(workspace, (str, Path)) else None
    )
    localized: dict[str, str] = {}
    archive_paths: list[str] = []
    external_refs: list[dict[str, Any]] = []
    for kind, raw_value in sorted(trace.harbor_paths.items()):
        value = str(raw_value).strip()
        if not value:
            continue
        local_path, external_uri = _local_or_external_path(
            value,
            workspace=workspace_path,
        )
        if local_path is not None:
            localized[kind] = local_path
            if local_path not in archive_paths:
                archive_paths.append(local_path)
        else:
            assert external_uri is not None
            external_refs.append(
                {
                    "kind": f"harbor_{kind}",
                    "uri": external_uri,
                    "provenance": {"localization": "external_not_materialized"},
                }
            )
    portable_trace = trace.model_copy(deep=True, update={"harbor_paths": localized})
    return portable_trace, tuple(archive_paths), tuple(external_refs)


def _local_or_external_path(
    value: str,
    *,
    workspace: Path | None,
) -> tuple[str | None, str | None]:
    parsed = urlsplit(value)
    if parsed.scheme and parsed.scheme != "file":
        return None, _sanitize_external_uri(value)
    raw_path = unquote(parsed.path) if parsed.scheme == "file" else value
    path = Path(raw_path)
    if path.is_absolute():
        resolved = path.resolve()
        if workspace is not None and resolved.is_relative_to(workspace):
            relative = resolved.relative_to(workspace).as_posix()
            return _normalized_relative_path(relative), None
        return None, resolved.as_posix()
    return _normalized_relative_path(PurePosixPath(raw_path).as_posix()), None


def _normalized_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value != value.strip()
        or path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(f"execution archive path is not normalized: {value!r}")
    return value


def _sanitize_external_uri(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme or parsed.scheme == "file":
        return value[:2048]
    hostname = parsed.hostname or ""
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))[:2048]


def _persist_portable_trace_if_supported(
    backend: ExplicitContextTaskBackend,
    record: IterationRecord,
) -> None:
    trace = record.execution_trace
    artifact_store = getattr(backend, "artifact_store", None)
    store_trace = getattr(artifact_store, "store_trace", None)
    if trace is not None and callable(store_trace):
        store_trace(trace, overwrite=True)
    report = record.mediator_report
    store_report = getattr(artifact_store, "store_report", None)
    if report is not None and callable(store_report):
        store_report(report, overwrite=True)


def _validate_record_context(*, record: IterationRecord, context: ContextPack) -> None:
    """Reject a backend record that does not reflect the supplied context."""
    expected = {
        "graph_snapshot_id": context.snapshot_id,
        "diffusion_policy": context.policy_name,
        "diffusion_artifacts_eligible": len(context.eligible_artifact_ids),
        "diffusion_artifacts_selected": len(context.selected_artifact_ids),
        "diffusion_artifacts_rendered": len(context.rendered_artifact_ids),
        "transfer_context_tokens": context.token_count,
        "max_transfer_context_tokens": context.max_context_tokens or 0,
        "context_budget_violation": context.budget_violation,
        "compacted_diffusion_artifact_ids": list(context.compacted_artifact_ids),
        "dropped_for_budget_artifact_ids": list(
            context.dropped_for_budget_artifact_ids
        ),
        "source_task_ids": list(context.source_task_ids),
    }
    for field, expected_value in expected.items():
        if getattr(record, field) != expected_value:
            raise ValueError(
                f"explicit-context backend did not stamp {field}: "
                f"expected {expected_value!r}"
            )
    expected_kind = "diffusion" if context.text is not None else "none"
    if record.transfer_context_kind != expected_kind:
        raise ValueError(
            "explicit-context backend did not stamp transfer_context_kind: "
            f"expected {expected_kind!r}"
        )
