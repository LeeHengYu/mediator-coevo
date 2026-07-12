from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

import mediated_coevo.cli.harness_registry as harness_registry
from mediated_coevo.cli.harness_registry import (
    RuntimeStateSource,
    _apply_harness_overlay,
    _copy_explicit_state,
    _harness_overlay_root,
    _publish_graph_state_ref,
    _publish_promoted_harness,
    _prepare_harness_workspace,
    _resolve_harness_ref,
    _resolve_state_ref,
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


def test_prepare_harness_workspace_does_not_activate_bundled_state(tmp_path):
    harness = tmp_path / "harness"
    (harness / "overlay/src/pkg").mkdir(parents=True)
    (harness / "overlay/src/pkg/module.py").write_text("VALUE = 2\n")
    (harness / "state/diffusion/graph_snapshots").mkdir(parents=True)
    (harness / "state/diffusion/graph_snapshots/snapshot.json").write_text("{}\n")
    experiment = tmp_path / "experiment"

    _prepare_harness_workspace(experiment, harness)

    metadata = json.loads((experiment / "harnesses/active_harness.json").read_text())
    assert metadata["applied_files"] == ["src/pkg/module.py"]
    assert metadata["bundled_state_files"] == [
        "diffusion/graph_snapshots/snapshot.json"
    ]
    assert metadata["state_activation"] == (
        "not_loaded_by_harness_dir; use --state-dir or --state-ref"
    )
    assert not (experiment / "diffusion/graph_snapshots/snapshot.json").exists()


def test_copy_explicit_state_carries_graph_but_resets_artifact_store(tmp_path):
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

    metadata = _copy_explicit_state(
        experiment,
        RuntimeStateSource(
            state_root=harness / "state",
            source=harness,
            ref="latest-graph:HL3",
        ),
    )
    target = DiffusionStore(experiment / "diffusion")

    assert metadata is not None
    assert metadata["ref"] == "latest-graph:HL3"
    assert metadata["artifact_store_reset"] is True
    assert metadata["skipped_artifacts"] == ["artifact-1.json"]
    assert target.load_artifact("artifact-1") is None
    assert target.query_artifacts(before_source_iteration=0) == []
    assert target.load_graph_snapshot("snapshot-1") is not None
    assert (experiment / "diffusion/diffused_records.jsonl").read_text() == "{}\n"
    active_state = json.loads((experiment / "state/active_state.json").read_text())
    assert active_state["state_root"] == str(harness / "state")


def test_state_ref_resolves_latest_graph_channel(tmp_path, monkeypatch):
    monkeypatch.setattr(harness_registry, "PROJECT_ROOT", tmp_path)
    campaign = tmp_path / "data/experiments/HL3"
    state = campaign / "bundles/abc/state"
    (state / "diffusion/graph_snapshots").mkdir(parents=True)
    (campaign / "channels").mkdir()
    (campaign / "channels/graph_state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_dir": str(state),
                "bundle_id": "sha256:abc",
            }
        )
        + "\n"
    )

    source = _resolve_state_ref("latest-graph:HL3")

    assert source.state_root == state
    assert source.ref == "latest-graph:HL3"
    assert source.manifest is not None
    assert source.manifest["bundle_id"] == "sha256:abc"


def test_harness_ref_resolves_promoted_harness_channel(tmp_path, monkeypatch):
    monkeypatch.setattr(harness_registry, "PROJECT_ROOT", tmp_path)
    campaign = tmp_path / "data/experiments/HL3"
    harness = tmp_path / "data/experiments/run/harnesses/update_0002"
    (harness / "overlay/src/pkg").mkdir(parents=True)
    (campaign / "channels").mkdir(parents=True)
    (campaign / "channels/promoted_harness.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "harness_dir": str(harness),
                "validation_run": "validation-run",
            }
        )
        + "\n"
    )

    assert _resolve_harness_ref("promoted:HL3") == harness


def test_publish_promoted_harness_writes_channel_and_promotion_record(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(harness_registry, "PROJECT_ROOT", tmp_path)
    harness = tmp_path / "data/experiments/run/harnesses/update_0002"
    (harness / "overlay/src/pkg").mkdir(parents=True)
    (harness / "overlay/src/pkg/module.py").write_text("VALUE = 2\n")

    channel_path = _publish_promoted_harness(
        campaign="HL3",
        harness_dir=harness,
        validation_run="validation-run",
        state_dir=None,
    )

    channel = json.loads(channel_path.read_text())
    assert channel["channel"] == "promoted_harness"
    assert channel["harness_dir"] == str(harness)
    assert channel["applied_files"] == ["src/pkg/module.py"]
    record = json.loads(Path(channel["promotion_record"]).read_text())
    assert record["decision"] == "promoted"
    assert record["validation_run"] == "validation-run"


def test_publish_graph_state_ref_bundles_training_state_and_skips_artifacts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(harness_registry, "PROJECT_ROOT", tmp_path)
    experiment = tmp_path / "data/experiments/run"
    source = DiffusionStore(experiment / "diffusion")
    source.store_artifact(
        DiffusionArtifact(
            artifact_id="artifact-1",
            source_task_id="task-A",
            source_iteration=0,
            artifact_type=DiffusionArtifactType.DEBUG_HINT,
            risk_level=DiffusionRiskLevel.LOW,
            content="do not bundle artifacts",
        )
    )
    source.store_graph_snapshot(
        TaskGraphSnapshot(
            snapshot_id="snapshot-1",
            run_id="run",
            iteration=0,
            task_ids=["task-A"],
            graph_policy="langchain_graph",
        )
    )
    (experiment / "diffusion/diffused_records.jsonl").write_text("{}\n")

    channel_path = _publish_graph_state_ref(
        "latest-graph:HL3",
        experiment_dir=experiment,
        split="train",
    )

    channel = json.loads(channel_path.read_text())
    state_dir = tmp_path / channel["state_dir"]
    bundle_dir = tmp_path / channel["bundle_dir"]
    if not state_dir.exists():
        state_dir = Path(channel["state_dir"])
        bundle_dir = Path(channel["bundle_dir"])
    assert channel["channel"] == "graph_state"
    assert channel["created_from_split"] == "train"
    assert (state_dir / "diffusion/graph_snapshots/snapshot-1.json").is_file()
    assert (state_dir / "diffusion/diffused_records.jsonl").read_text() == "{}\n"
    assert not (state_dir / "diffusion/artifacts/artifact-1.json").exists()
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["artifact_store_policy"] == (
        "fresh_per_run; diffusion/artifacts not bundled"
    )


def test_publish_graph_state_ref_rejects_validation_split(tmp_path, monkeypatch):
    monkeypatch.setattr(harness_registry, "PROJECT_ROOT", tmp_path)
    experiment = tmp_path / "data/experiments/run"
    (experiment / "diffusion/graph_snapshots").mkdir(parents=True)

    with pytest.raises(typer.BadParameter, match="--split train"):
        _publish_graph_state_ref(
            "latest-graph:HL3",
            experiment_dir=experiment,
            split="validation",
        )
