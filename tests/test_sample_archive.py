from __future__ import annotations

import hashlib

import pytest

from mediated_coevo.experiment.sample_archive import (
    build_archive_manifest,
    validate_archive_manifest,
    write_archive_manifest,
)
from mediated_coevo.experiment.sample_models import ArchiveManifest


def test_archive_manifest_is_relative_content_addressed_and_round_trips(tmp_path):
    workspace = tmp_path / "sample"
    journal = workspace / "journal" / "position-0001.json"
    artifact = workspace / "artifacts" / "trace.json"
    journal.parent.mkdir(parents=True)
    artifact.parent.mkdir(parents=True)
    journal.write_text('{"position": 1}\n', encoding="utf-8")
    artifact.write_bytes(b"trace bytes")

    manifest = build_archive_manifest(workspace)

    by_path = {entry.relative_path: entry for entry in manifest.entries}
    assert set(by_path) == {
        "artifacts/trace.json",
        "journal/position-0001.json",
    }
    assert by_path["artifacts/trace.json"].byte_size == len(b"trace bytes")
    assert by_path["artifacts/trace.json"].sha256 == hashlib.sha256(
        b"trace bytes"
    ).hexdigest()
    assert all(not entry.relative_path.startswith("/") for entry in manifest.entries)

    write_archive_manifest(workspace, manifest)
    validate_archive_manifest(manifest, workspace=workspace)


def test_archive_validation_detects_tampering_and_manifest_is_immutable(tmp_path):
    workspace = tmp_path / "warmup"
    artifact = workspace / "diffusion" / "artifacts" / "artifact-0.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"artifact_id": "artifact-0"}\n', encoding="utf-8")
    manifest = build_archive_manifest(workspace)
    write_archive_manifest(workspace, manifest)

    with pytest.raises(FileExistsError, match="Archive manifest already exists"):
        write_archive_manifest(workspace, manifest)

    artifact.write_text('{"artifact_id": "changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch|byte size mismatch"):
        validate_archive_manifest(manifest, workspace=workspace)


def test_archive_manifest_excludes_only_exact_terminal_contracts(tmp_path):
    workspace = tmp_path / "sample"
    workspace.mkdir()
    (workspace / "sample_result.json").write_text("terminal", encoding="utf-8")
    (workspace / "archive_manifest.json").write_text("recursive", encoding="utf-8")
    (workspace / "sample_spec.json").write_text("{}", encoding="utf-8")
    nested = workspace / "jobs" / "sample_result.json"
    nested.parent.mkdir()
    nested.write_text("executor evidence", encoding="utf-8")

    manifest = build_archive_manifest(workspace)

    assert [entry.relative_path for entry in manifest.entries] == [
        "jobs/sample_result.json",
        "sample_spec.json",
    ]


def test_archive_manifest_rejects_sensitive_paths_instead_of_hiding_them(tmp_path):
    workspace = tmp_path / "sample"
    workspace.mkdir()
    (workspace / ".env.local").write_text(
        "OPENAI_API_KEY=secret",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sensitive"):
        build_archive_manifest(workspace)


def test_sequence_archive_classifies_files_below_the_run_root(tmp_path):
    workspace = tmp_path / "sequence"
    paths = {
        "sequence_spec.json": "run_file",
        "warmup/warmup-run/journal/position-0000.json": "position_journal",
        "samples/sample-1/jobs/job-1/evidence.json": "harbor_job",
        "samples/sample-1/artifacts/traces/trace.json": "execution_artifact",
    }
    for relative in paths:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    manifest = build_archive_manifest(workspace)

    assert {entry.relative_path: entry.kind for entry in manifest.entries} == paths


def test_archive_manifest_rejects_symlinks(tmp_path):
    workspace = tmp_path / "sample"
    workspace.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    (workspace / "escape").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        build_archive_manifest(workspace)


def test_archive_validation_rejects_tampered_entry_kind(tmp_path):
    workspace = tmp_path / "sample"
    artifact = workspace / "artifacts" / "trace.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    manifest = build_archive_manifest(workspace)
    payload = manifest.model_dump(mode="python")
    payload["entries"][0]["kind"] = "forged_kind"
    forged = ArchiveManifest.model_validate(payload)

    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_archive_manifest(forged, workspace=workspace)
