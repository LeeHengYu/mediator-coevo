from __future__ import annotations

import json

import pytest
import typer

from mediated_coevo.cli.run import (
    _apply_harness_overlay,
    _copy_harness_state,
    _harness_overlay_root,
    _prepare_harness_workspace,
)
from mediated_coevo.diffusion import (
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    DiffusionStore,
    TaskGraphSnapshot,
)


def test_harness_overlay_copies_repo_root_files_and_skips_manifest(tmp_path):
    harness = tmp_path / "harness"
    (harness / "src/pkg").mkdir(parents=True)
    (harness / "config").mkdir()
    (harness / "manifest.md").write_text("metadata only")
    (harness / "src/pkg/module.py").write_text("VALUE = 1\n")
    (harness / "config/default.toml").write_text("[models]\n")

    project = tmp_path / "project"
    copied = _apply_harness_overlay(harness, project)

    assert copied == ["config/default.toml", "src/pkg/module.py"]
    assert (project / "src/pkg/module.py").read_text() == "VALUE = 1\n"
    assert not (project / "manifest.md").exists()


def test_harness_overlay_accepts_overlay_subdir(tmp_path):
    harness = tmp_path / "harness"
    overlay = harness / "overlay"
    (overlay / "tests").mkdir(parents=True)
    (overlay / "tests/test_policy.py").write_text("def test_ok(): pass\n")

    assert _harness_overlay_root(harness) == overlay


def test_harness_overlay_rejects_non_overlay_folder(tmp_path):
    harness = tmp_path / "experiment"
    harness.mkdir()
    (harness / "metrics.jsonl").write_text("{}\n")

    with pytest.raises(typer.BadParameter, match="repo-root overlay"):
        _harness_overlay_root(harness)


def test_prepare_harness_workspace_copies_seed_and_manifest(tmp_path):
    harness = tmp_path / "harness"
    (harness / "src/pkg").mkdir(parents=True)
    (harness / "manifest.md").write_text("# update\n")
    (harness / "src/pkg/module.py").write_text("VALUE = 2\n")
    experiment = tmp_path / "experiment"

    _prepare_harness_workspace(experiment, harness)

    assert (experiment / "harnesses/README.md").is_file()
    assert (
        experiment / "harnesses/seed/src/pkg/module.py"
    ).read_text() == "VALUE = 2\n"
    assert (experiment / "harnesses/seed/manifest.md").is_file()
    metadata = json.loads((experiment / "harnesses/active_harness.json").read_text())
    assert metadata["applied_files"] == ["src/pkg/module.py"]


def test_copy_harness_state_carries_graph_but_resets_artifact_store(tmp_path):
    harness = tmp_path / "harness"
    source = DiffusionStore(harness / "state/diffusion")
    source.store_artifact(
        DiffusionArtifact(
            artifact_id="artifact-1",
            source_task_id="task-A",
            source_iteration=5,
            artifact_type=DiffusionArtifactType.DEBUG_HINT,
            risk_level=DiffusionRiskLevel.LOW,
            content="use this",
            verifier_reward=1.0,
        )
    )
    source.store_graph_snapshot(
        TaskGraphSnapshot(
            snapshot_id="snapshot-1",
            run_id="old-run",
            iteration=7,
            task_ids=["node-A"],
            graph_policy="langchain_graph",
        )
    )
    (harness / "state/diffusion/diffused_records.jsonl").write_text("{}\n")
    experiment = tmp_path / "experiment"

    metadata = _copy_harness_state(experiment, harness)
    target = DiffusionStore(experiment / "diffusion")

    assert metadata is not None
    assert metadata["artifact_store_reset"] is True
    assert metadata["skipped_artifacts"] == ["artifact-1.json"]
    assert target.load_artifact("artifact-1") is None
    assert target.query_artifacts(before_source_iteration=0) == []
    assert target.load_graph_snapshot("snapshot-1") is not None
    assert (experiment / "diffusion/diffused_records.jsonl").read_text() == "{}\n"
