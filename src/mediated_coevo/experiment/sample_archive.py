"""Portable, content-addressed archives for causal sample runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import TypeVar
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from pydantic import BaseModel

from mediated_coevo.diffusion.models import DiffusionArtifact, TaskGraphSnapshot
from mediated_coevo.execution.models import contains_sensitive_text
from mediated_coevo.experiment.sample_models import (
    ArchiveEntry,
    ArchiveManifest,
    ExternalArchiveRef,
    PositionJournal,
    SampleResult,
    SequenceSpec,
    TaskRecord,
    WarmupBundle,
)

_ModelT = TypeVar("_ModelT", bound=BaseModel)

ARCHIVE_MANIFEST_FILENAME = "archive_manifest.json"
SAMPLE_RESULT_FILENAME = "sample_result.json"
WARMUP_BUNDLE_FILENAME = "warmup_bundle.json"

_TERMINAL_FILENAMES = {
    ARCHIVE_MANIFEST_FILENAME,
    SAMPLE_RESULT_FILENAME,
    WARMUP_BUNDLE_FILENAME,
}
_SENSITIVE_NAMES = {
    ".env",
    "credentials",
    "credentials.json",
    "secrets",
    "secrets.json",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sensitive_archive_paths(workspace: Path) -> tuple[Path, ...]:
    """Return sensitive-name paths without following workspace symlinks."""
    workspace = workspace.resolve()
    if not workspace.is_dir():
        return ()
    return tuple(
        path
        for path in workspace.rglob("*")
        if _is_sensitive_path(path.relative_to(workspace))
    )


def nonportable_archive_paths(workspace: Path) -> tuple[Path, ...]:
    """Return filesystem entries that cannot be named in a POSIX manifest."""
    workspace = workspace.resolve()
    if not workspace.is_dir():
        return ()
    invalid: list[Path] = []
    for path in workspace.rglob("*"):
        relative = path.relative_to(workspace).as_posix()
        try:
            _normalize_relative_path(relative)
        except ValueError:
            invalid.append(path)
    return tuple(invalid)


def credential_archive_paths(
    workspace: Path,
    *,
    secret_values: Iterable[str] = (),
) -> tuple[Path, ...]:
    """Return paths whose portable names disclose credential material."""
    workspace = workspace.resolve()
    if not workspace.is_dir():
        return ()
    secrets = tuple(
        sorted(
            {
                value
                for value in secret_values
                if isinstance(value, str) and len(value) >= 4
            },
            key=len,
            reverse=True,
        )
    )
    invalid: list[Path] = []
    for path in workspace.rglob("*"):
        relative = path.relative_to(workspace).as_posix()
        if contains_sensitive_text(relative) or any(
            secret in relative for secret in secrets
        ):
            invalid.append(path)
    return tuple(invalid)


def sanitize_archive_workspace(
    workspace: Path,
    *,
    secret_values: Iterable[str] = (),
) -> tuple[str, ...]:
    """Remove secret-only paths and redact known credential values in place."""
    workspace = workspace.resolve()
    secrets = tuple(
        sorted(
            {
                value
                for value in secret_values
                if isinstance(value, str) and len(value) >= 4
            },
            key=len,
            reverse=True,
        )
    )
    removed: list[str] = []
    removable_paths = {
        *sensitive_archive_paths(workspace),
        *nonportable_archive_paths(workspace),
        *credential_archive_paths(workspace, secret_values=secrets),
        *(path for path in workspace.rglob("*") if path.is_symlink()),
    }
    for path in sorted(
        removable_paths,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        if not path.exists() and not path.is_symlink():
            continue
        removed.append(path.relative_to(workspace).as_posix())
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)

    encoded_secrets = tuple(
        value.encode("utf-8") for value in secrets
    )
    if not encoded_secrets:
        return tuple(sorted(set(removed)))
    for path in sorted(workspace.rglob("*")):
        if path.is_symlink() or not path.is_file() or _is_temporary_path(path):
            continue
        if not _file_contains_any(path, encoded_secrets):
            continue
        payload = path.read_bytes()
        for secret in encoded_secrets:
            payload = payload.replace(secret, b"[redacted]")
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return tuple(sorted(set(removed)))


def _file_contains_any(path: Path, needles: tuple[bytes, ...]) -> bool:
    overlap = max(len(value) for value in needles) - 1
    tail = b""
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            haystack = tail + chunk
            if any(needle in haystack for needle in needles):
                return True
            tail = haystack[-overlap:] if overlap else b""
    return False


def terminal_payload_sha256(model: BaseModel) -> str:
    """Hash a success terminal without its embedded manifest.

    The manifest embeds this digest while the terminal embeds the manifest, so
    removing only the digest field avoids recursion while still binding archive
    entries, external references, semantic results, and provenance.
    """
    payload = model.model_dump(mode="json")
    embedded_manifest = payload.get("archive_manifest")
    if not isinstance(embedded_manifest, dict):
        raise ValueError("success terminal must embed an archive manifest")
    embedded_manifest.pop("terminal_payload_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_terminal_payload(
    manifest: ArchiveManifest,
    terminal_model: BaseModel,
) -> ArchiveManifest:
    """Return a revalidated manifest bound to one success terminal payload."""
    return ArchiveManifest.model_validate(
        {
            **manifest.model_dump(mode="python"),
            "terminal_payload_sha256": terminal_payload_sha256(terminal_model),
        }
    )


def build_archive_manifest(
    workspace: Path,
    *,
    external_refs: Iterable[ExternalArchiveRef] = (),
    exclude_relative_paths: Iterable[str] = (),
    include_relative_roots: Iterable[str] = (),
) -> ArchiveManifest:
    """Hash every portable run file using workspace-relative POSIX paths.

    Terminal success files and the manifest itself are excluded because they
    embed the manifest and would make its digest recursive. Failure records,
    journals, executor outputs, and compact transfer artifacts remain covered.
    """
    workspace = workspace.resolve()
    excluded = {
        _normalize_relative_path(value) for value in exclude_relative_paths
    }
    entries: list[ArchiveEntry] = []
    if not workspace.is_dir():
        raise ValueError(f"archive workspace does not exist: {workspace}")

    roots = tuple(
        _normalize_relative_path(value) for value in include_relative_roots
    )
    candidate_paths: set[Path] = set()
    if roots:
        for relative_root in roots:
            root = workspace.joinpath(*PurePosixPath(relative_root).parts)
            if root.is_symlink():
                raise ValueError(f"archive include root is a symlink: {relative_root}")
            if root.is_file():
                candidate_paths.add(root)
            elif root.is_dir():
                candidate_paths.update(root.rglob("*"))
            else:
                raise ValueError(f"archive include root does not exist: {relative_root}")
    else:
        candidate_paths.update(workspace.rglob("*"))

    for path in sorted(candidate_paths):
        if path.is_symlink():
            raise ValueError(f"archive workspace contains a symlink: {path}")
        if not path.is_file():
            continue
        relative_path = path.relative_to(workspace)
        relative = relative_path.as_posix()
        if _is_sensitive_path(relative_path):
            raise ValueError(
                f"archive workspace contains a sensitive path: {relative}"
            )
        if (
            relative in excluded
            or _is_terminal_contract_path(relative)
            or _is_temporary_path(path)
        ):
            continue
        entries.append(
            ArchiveEntry(
                relative_path=relative,
                kind=_archive_kind(relative),
                sha256=sha256_file(path),
                byte_size=path.stat().st_size,
            )
        )

    refs_by_identity: dict[tuple[str, str], ExternalArchiveRef] = {}
    for ref in external_refs:
        key = (ref.kind, ref.uri)
        existing = refs_by_identity.get(key)
        if existing is not None and existing != ref:
            raise ValueError(f"conflicting external archive reference: {key!r}")
        refs_by_identity[key] = ref
    normalized_external_refs = tuple(
        refs_by_identity[key] for key in sorted(refs_by_identity)
    )
    return ArchiveManifest(
        entries=tuple(entries),
        external_refs=normalized_external_refs,
    )


def validate_archive_manifest(
    manifest: ArchiveManifest,
    *,
    workspace: Path,
    include_relative_roots: Iterable[str] = (),
) -> None:
    """Verify every manifest entry without following paths outside workspace."""
    workspace = workspace.resolve()
    discovered = build_archive_manifest(
        workspace,
        external_refs=manifest.external_refs,
        include_relative_roots=include_relative_roots,
    )
    expected_paths = {entry.relative_path for entry in manifest.entries}
    discovered_paths = {entry.relative_path for entry in discovered.entries}
    if discovered_paths != expected_paths:
        unexpected = sorted(discovered_paths - expected_paths)
        missing = sorted(expected_paths - discovered_paths)
        raise ValueError(
            "archive manifest file set mismatch: "
            f"unexpected={unexpected!r} missing={missing!r}"
        )
    discovered_by_path = {
        entry.relative_path: entry for entry in discovered.entries
    }
    seen: set[str] = set()
    for entry in manifest.entries:
        relative = _normalize_relative_path(entry.relative_path)
        if relative in seen:
            raise ValueError(f"archive manifest repeats path: {relative}")
        seen.add(relative)
        path = workspace.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"archive entry is missing or not a regular file: {relative}")
        if path.stat().st_size != entry.byte_size:
            raise ValueError(f"archive entry byte size mismatch: {relative}")
        if sha256_file(path) != entry.sha256:
            raise ValueError(f"archive entry SHA-256 mismatch: {relative}")
        if discovered_by_path[relative] != entry:
            raise ValueError(f"archive entry metadata mismatch: {relative}")


def external_archive_refs(
    records: Sequence[TaskRecord],
    *,
    workspace: Path | None = None,
) -> tuple[ExternalArchiveRef, ...]:
    """Extract external paths from arm-neutral warm-up or suffix records."""
    refs: list[ExternalArchiveRef] = []
    seen: set[tuple[str, str]] = set()
    workspace = workspace.resolve() if workspace is not None else None
    for task_record in records:
        if task_record.execution is None:
            continue
        declared_refs = task_record.execution.metadata.get("external_archive_refs", ())
        if isinstance(declared_refs, (list, tuple)):
            for value in declared_refs:
                if not isinstance(value, dict):
                    raise ValueError("external_archive_refs entries must be objects")
                kind = str(value.get("kind") or "").strip()
                raw_uri = str(value.get("uri") or "").strip()
                if not kind or not raw_uri:
                    raise ValueError("external archive reference requires kind and uri")
                uri = _sanitize_external_uri(raw_uri)
                key = (kind, uri)
                if key in seen:
                    continue
                provenance = value.get("provenance")
                refs.append(
                    ExternalArchiveRef(
                        kind=kind,
                        uri=uri,
                        provenance={
                            **(provenance if isinstance(provenance, dict) else {}),
                            "run_id": task_record.run_id,
                            "position": task_record.position,
                            "task_id": task_record.task_id,
                        },
                    )
                )
                seen.add(key)
        trace = task_record.execution.record.execution_trace
        if trace is None:
            continue
        for path_kind, raw_uri in sorted(trace.harbor_paths.items()):
            if not raw_uri:
                continue
            uri = _sanitize_external_uri(raw_uri)
            if not _looks_external(uri) or _is_localized_path(uri, workspace=workspace):
                continue
            key = (path_kind, uri)
            if key in seen:
                continue
            refs.append(
                ExternalArchiveRef(
                    kind=f"harbor_{path_kind}",
                    uri=uri,
                    provenance={
                        "run_id": task_record.run_id,
                        "position": task_record.position,
                        "task_id": task_record.task_id,
                        "localization": "external_not_materialized",
                    },
                )
            )
            seen.add(key)
    return tuple(refs)


def write_model_atomic(
    path: Path,
    model: BaseModel,
    *,
    exists_error_prefix: str = "Archive model",
) -> Path:
    """Atomically publish one immutable JSON model without overwriting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (model.model_dump_json(indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{exists_error_prefix} already exists: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_or_validate_model(path: Path, model: BaseModel) -> Path:
    """Create a shared immutable model, or require byte-semantic equality."""
    if path.exists():
        existing = type(model).model_validate_json(path.read_text(encoding="utf-8"))
        if existing != model:
            raise ValueError(f"existing shared model conflicts with requested value: {path}")
        return path
    return write_model_atomic(path, model, exists_error_prefix="Shared model")


def write_position_journal(workspace: Path, journal: PositionJournal) -> str:
    """Persist one completed position and return its workspace-relative path."""
    relative = f"journal/position-{journal.position:04d}.json"
    write_model_atomic(
        workspace / relative,
        journal,
        exists_error_prefix="Position journal",
    )
    return relative


def write_archive_manifest(workspace: Path, manifest: ArchiveManifest) -> Path:
    """Publish the immutable manifest after all covered files are durable."""
    return write_model_atomic(
        workspace / ARCHIVE_MANIFEST_FILENAME,
        manifest,
        exists_error_prefix="Archive manifest",
    )


def load_warmup_bundle(path: Path, *, validate: bool = True) -> WarmupBundle:
    """Load a warm-up bundle and optionally verify its complete local archive."""
    bundle_path = _resolve_model_path(path, WARMUP_BUNDLE_FILENAME)
    bundle = _load_required_model(bundle_path, WarmupBundle)
    if validate:
        _validate_embedded_manifest(
            bundle.archive_manifest,
            bundle_path,
            terminal_model=bundle,
        )
        _validate_warmup_sequence_contract(bundle, bundle_path)
    return bundle


def load_sample_result(path: Path, *, validate: bool = True) -> SampleResult:
    """Load a suffix result and verify both local and shared-prefix state."""
    result_path = _resolve_model_path(path, SAMPLE_RESULT_FILENAME)
    result = _load_required_model(result_path, SampleResult)
    if validate:
        _validate_embedded_manifest(
            result.archive_manifest,
            result_path,
            terminal_model=result,
        )
        _validate_sample_sequence_contract(result, result_path)
    return result


def _validate_embedded_manifest(
    embedded: ArchiveManifest,
    terminal_path: Path,
    *,
    terminal_model: BaseModel,
) -> None:
    run_workspace = terminal_path.parent
    manifest_path = run_workspace / ARCHIVE_MANIFEST_FILENAME
    persisted = _load_required_model(manifest_path, ArchiveManifest)
    if persisted != embedded:
        raise ValueError("embedded archive manifest differs from archive_manifest.json")
    if embedded.terminal_payload_sha256 is None:
        raise ValueError("success archive manifest lacks a terminal payload digest")
    if embedded.terminal_payload_sha256 != terminal_payload_sha256(terminal_model):
        raise ValueError("success terminal bank/result payload SHA-256 mismatch")
    try:
        sequence_workspace = run_workspace.parents[1]
        run_relative = run_workspace.relative_to(sequence_workspace).as_posix()
    except (IndexError, ValueError) as exc:
        raise ValueError("terminal archive is not inside a sequence workspace") from exc
    validate_archive_manifest(
        persisted,
        workspace=sequence_workspace,
        include_relative_roots=("sequence_spec.json", run_relative),
    )


def _validate_warmup_sequence_contract(
    bundle: WarmupBundle,
    bundle_path: Path,
) -> None:
    sequence_workspace = _sequence_workspace(bundle_path, kind="warmup")
    sequence = _load_required_model(
        sequence_workspace / "sequence_spec.json",
        SequenceSpec,
    )
    if bundle_path.parent.name != bundle.warmup_run_id:
        raise ValueError("warm-up archive directory differs from warmup_run_id")
    if bundle.sequence_id != sequence.sequence_id:
        raise ValueError("warm-up bundle belongs to another sequence specification")
    if bundle.warmup_count != sequence.warmup_count:
        raise ValueError("warm-up bundle has the wrong frozen prefix length")
    if tuple(record.task for record in bundle.task_records) != sequence.tasks[
        : sequence.warmup_count
    ]:
        raise ValueError("warm-up records differ from the frozen task prefix")
    _validate_persisted_run_state(
        run_workspace=bundle_path.parent,
        task_records=bundle.task_records,
        final_artifact_bank=bundle.final_artifact_bank,
        final_graph=None,
        manifest=bundle.archive_manifest,
    )
    _validate_external_refs(
        manifest=bundle.archive_manifest,
        task_records=bundle.task_records,
        sequence_workspace=sequence_workspace,
    )


def _validate_sample_sequence_contract(
    result: SampleResult,
    result_path: Path,
) -> None:
    sequence_workspace = _sequence_workspace(result_path, kind="samples")
    sequence = _load_required_model(
        sequence_workspace / "sequence_spec.json",
        SequenceSpec,
    )
    if result_path.parent.name != result.spec.sample_id:
        raise ValueError("sample archive directory differs from sample_id")
    if result.spec.sequence != sequence:
        raise ValueError("sample result differs from sequence_spec.json")

    _validate_persisted_run_state(
        run_workspace=result_path.parent,
        task_records=result.task_records,
        final_artifact_bank=result.final_artifact_bank,
        final_graph=result.final_graph,
        manifest=result.archive_manifest,
    )
    _validate_external_refs(
        manifest=result.archive_manifest,
        task_records=result.task_records,
        sequence_workspace=sequence_workspace,
    )

    reference = result.warmup_reference
    if reference is None:
        if result.task_records[0].artifact_ids_before:
            raise ValueError("zero-warm-up sample must start from an empty bank")
        return

    expected_bundle_path = (
        f"warmup/{reference.warmup_run_id}/{WARMUP_BUNDLE_FILENAME}"
    )
    expected_manifest_path = (
        f"warmup/{reference.warmup_run_id}/{ARCHIVE_MANIFEST_FILENAME}"
    )
    if reference.relative_path != expected_bundle_path:
        raise ValueError("warm-up reference does not use the canonical bundle path")
    if reference.manifest_path != expected_manifest_path:
        raise ValueError("warm-up reference does not use the canonical manifest path")

    bundle = load_warmup_bundle(
        sequence_workspace.joinpath(*PurePosixPath(reference.relative_path).parts),
        validate=True,
    )
    if reference != bundle.reference(
        relative_path=expected_bundle_path,
        manifest_path=expected_manifest_path,
    ):
        raise ValueError("warm-up reference differs from the shared bundle")

    prefix_ids = tuple(
        artifact.artifact_id for artifact in bundle.final_artifact_bank
    )
    if result.task_records[0].artifact_ids_before != prefix_ids:
        raise ValueError("sample suffix does not start from the shared warm-up bank")
    prefix_size = len(bundle.final_artifact_bank)
    if result.final_artifact_bank[:prefix_size] != bundle.final_artifact_bank:
        raise ValueError("sample final bank does not preserve the shared warm-up prefix")
    for artifact in bundle.final_artifact_bank:
        artifact_id = _validate_path_component(
            artifact.artifact_id,
            label="warm-up artifact_id",
        )
        materialized = _load_required_model(
            result_path.parent / "diffusion" / "artifacts" / f"{artifact_id}.json",
            type(artifact),
        )
        if materialized != artifact:
            raise ValueError(
                "sample transfer artifact differs from the shared warm-up bank: "
                f"{artifact_id}"
            )


def _validate_persisted_run_state(
    *,
    run_workspace: Path,
    task_records: Sequence[TaskRecord],
    final_artifact_bank: Sequence[DiffusionArtifact],
    final_graph: TaskGraphSnapshot | None,
    manifest: ArchiveManifest,
) -> None:
    """Bind terminal state to journals and concrete diffusion-store objects."""
    expected_journals = {
        f"position-{record.position:04d}.json" for record in task_records
    }
    journal_dir = run_workspace / "journal"
    actual_journals = (
        {path.name for path in journal_dir.glob("*.json") if path.is_file()}
        if journal_dir.is_dir()
        else set()
    )
    if actual_journals != expected_journals:
        raise ValueError(
            "position journal set differs from terminal task records: "
            f"expected={sorted(expected_journals)!r} "
            f"actual={sorted(actual_journals)!r}"
        )
    for record in task_records:
        journal = _load_required_model(
            journal_dir / f"position-{record.position:04d}.json",
            PositionJournal,
        )
        if journal.task_record != record:
            raise ValueError(
                f"position journal differs from terminal task record: {record.position}"
            )

    manifest_paths = {entry.relative_path for entry in manifest.entries}
    sequence_workspace = run_workspace.parents[1]
    for record in task_records:
        if record.execution is None:
            continue
        for raw_path in record.execution.archive_paths:
            relative = _normalize_relative_path(raw_path)
            local_path = run_workspace.joinpath(*PurePosixPath(relative).parts)
            if local_path.is_symlink() or not local_path.exists():
                raise ValueError(f"execution archive path is missing: {relative}")
            archived_files: tuple[Path, ...]
            if local_path.is_file():
                archived_files = (local_path,)
            elif local_path.is_dir():
                archived_files = tuple(
                    path
                    for path in local_path.rglob("*")
                    if path.is_file() and not path.is_symlink()
                )
                if not archived_files:
                    raise ValueError(
                        f"execution archive directory is empty: {relative}"
                    )
            else:
                raise ValueError(
                    f"execution archive path is not a file or directory: {relative}"
                )
            for archived_file in archived_files:
                manifest_relative = archived_file.relative_to(
                    sequence_workspace
                ).as_posix()
                if manifest_relative not in manifest_paths:
                    raise ValueError(
                        "execution archive path is not covered by the manifest: "
                        f"{manifest_relative}"
                    )

    artifact_dir = run_workspace / "diffusion" / "artifacts"
    expected_artifact_ids = {
        str(getattr(artifact, "artifact_id")) for artifact in final_artifact_bank
    }
    actual_artifact_ids = (
        {path.stem for path in artifact_dir.glob("*.json") if path.is_file()}
        if artifact_dir.is_dir()
        else set()
    )
    if actual_artifact_ids != expected_artifact_ids:
        raise ValueError(
            "persisted artifact set differs from terminal final bank: "
            f"expected={sorted(expected_artifact_ids)!r} "
            f"actual={sorted(actual_artifact_ids)!r}"
        )
    for artifact in final_artifact_bank:
        artifact_id = _validate_path_component(
            str(getattr(artifact, "artifact_id")),
            label="artifact_id",
        )
        persisted_artifact = _load_required_model(
            artifact_dir / f"{artifact_id}.json",
            DiffusionArtifact,
        )
        if persisted_artifact != artifact:
            raise ValueError(
                f"persisted artifact differs from terminal final bank: {artifact_id}"
            )

    graph_dir = run_workspace / "diffusion" / "graph_snapshots"
    persisted_graphs: dict[str, TaskGraphSnapshot] = {}
    expected_graph_ids = {
        snapshot_id
        for record in task_records
        if (snapshot_id := getattr(record, "graph_snapshot_id_after", None))
        is not None
    }
    actual_graph_ids = (
        {path.stem for path in graph_dir.glob("*.json") if path.is_file()}
        if graph_dir.is_dir()
        else set()
    )
    if actual_graph_ids != expected_graph_ids:
        raise ValueError(
            "persisted graph snapshot set differs from terminal task records: "
            f"expected={sorted(expected_graph_ids)!r} "
            f"actual={sorted(actual_graph_ids)!r}"
        )
    for record in task_records:
        snapshot_id = getattr(record, "graph_snapshot_id_after", None)
        if snapshot_id is None:
            continue
        safe_id = _validate_path_component(snapshot_id, label="graph snapshot_id")
        snapshot = _load_required_model(
            graph_dir / f"{safe_id}.json",
            TaskGraphSnapshot,
        )
        if (
            snapshot.snapshot_id != snapshot_id
            or snapshot.run_id != record.run_id
            or snapshot.iteration != record.position
        ):
            raise ValueError("persisted graph snapshot differs from its task record")
        persisted_graphs[snapshot_id] = snapshot
    if final_graph is not None:
        final_snapshot_id = _validate_path_component(
            str(getattr(final_graph, "snapshot_id")),
            label="final graph snapshot_id",
        )
        persisted_final = persisted_graphs.get(final_snapshot_id)
        if persisted_final is None:
            persisted_final = _load_required_model(
                graph_dir / f"{final_snapshot_id}.json",
                TaskGraphSnapshot,
            )
        if persisted_final != final_graph:
            raise ValueError("persisted final graph differs from terminal final graph")


def _validate_external_refs(
    *,
    manifest: ArchiveManifest,
    task_records: Sequence[TaskRecord],
    sequence_workspace: Path,
) -> None:
    expected = tuple(
        sorted(
            external_archive_refs(
                task_records,
                workspace=sequence_workspace,
            ),
            key=lambda ref: (ref.kind, ref.uri),
        )
    )
    if manifest.external_refs != expected:
        raise ValueError(
            "archive manifest external references differ from task records"
        )


def _sequence_workspace(terminal_path: Path, *, kind: str) -> Path:
    run_workspace = terminal_path.parent
    if run_workspace.parent.name != kind:
        raise ValueError(f"terminal archive is not inside a {kind} run workspace")
    try:
        return run_workspace.parents[1]
    except IndexError as exc:
        raise ValueError("terminal archive is not inside a sequence workspace") from exc


def _load_required_model(path: Path, model_cls: type[_ModelT]) -> _ModelT:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


def _resolve_model_path(path: Path, filename: str) -> Path:
    return path / filename if path.is_dir() else path


def _normalize_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value != value.strip()
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"archive path must be normalized and relative: {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError(f"archive path is not normalized: {value!r}")
    return normalized


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


def _archive_kind(relative: str) -> str:
    parts = PurePosixPath(relative).parts
    head = parts[0]
    if head in {"warmup", "samples"} and len(parts) >= 3:
        head = parts[2]
    return {
        "journal": "position_journal",
        "artifacts": "execution_artifact",
        "diffusion": "diffusion_state",
        "jobs": "harbor_job",
        "history": "execution_history",
        "harnesses": "harness",
        "skills": "skill_snapshot",
        "skills_snapshots": "skill_snapshot",
    }.get(head, "run_file")


def _is_terminal_contract_path(relative: str) -> bool:
    """Return whether a file is an exact run-root terminal contract.

    A Harbor job may legitimately emit a file named ``sample_result.json``;
    excluding by basename would silently drop that evidence from the archive.
    """
    parts = PurePosixPath(relative).parts
    if len(parts) == 1:
        return parts[0] in _TERMINAL_FILENAMES
    return (
        len(parts) == 3
        and parts[0] in {"warmup", "samples"}
        and parts[2] in _TERMINAL_FILENAMES
    )


def _is_temporary_path(path: Path) -> bool:
    """Recognize only this runtime's UUID-named atomic-write scratch files."""
    return bool(re.fullmatch(r"\..+\.[0-9a-f]{32}\.tmp", path.name))


def _is_sensitive_path(relative: Path) -> bool:
    lowered = {part.lower() for part in relative.parts}
    if lowered.intersection(_SENSITIVE_NAMES):
        return True
    return any(part.startswith(".env") for part in lowered)


def _sanitize_external_uri(value: str) -> str:
    """Drop URL credentials/query/fragment while retaining an external locator."""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value[:2048]
    if not parsed.scheme:
        return value[:2048]
    if parsed.scheme == "file":
        return urlunsplit(("file", "", parsed.path, "", ""))[:2048]
    hostname = parsed.hostname or ""
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))[:2048]


def _looks_external(value: str) -> bool:
    parsed = urlsplit(value)
    return bool(parsed.scheme) or Path(value).is_absolute()


def _is_localized_path(value: str, *, workspace: Path | None) -> bool:
    if workspace is None:
        return False
    parsed = urlsplit(value)
    if parsed.scheme not in {"", "file"}:
        return False
    raw_path = parsed.path if parsed.scheme == "file" else value
    path = Path(raw_path)
    return path.is_absolute() and path.resolve().is_relative_to(workspace)
