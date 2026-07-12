from __future__ import annotations

import json

from typer.testing import CliRunner

from mediated_coevo.diffusion import (
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    DiffusionStore,
)
from mediated_coevo.main import app


def _artifact() -> DiffusionArtifact:
    return DiffusionArtifact(
        artifact_id="artifact-1",
        source_task_id="task-A",
        source_iteration=1,
        artifact_type=DiffusionArtifactType.DEBUG_HINT,
        risk_level=DiffusionRiskLevel.LOW,
        content="use this",
        verifier_reward=1.0,
    )


def test_extract_recovers_diffusion_artifact_store(tmp_path):
    experiment = tmp_path / "experiment-1"
    DiffusionStore(experiment / "diffusion").store_artifact(_artifact())
    output_dir = tmp_path / "artifact-stores"

    result = CliRunner().invoke(
        app,
        [
            "extract",
            "-p",
            str(experiment),
            "--output-dir",
            str(output_dir),
        ],
    )
    manifest = json.loads(
        (output_dir / "experiment-1" / "manifest.json").read_text()
    )

    assert result.exit_code == 0, result.output
    assert manifest["id"] == "experiment-1"
    assert manifest["artifact_count"] == 1
    assert (output_dir / "experiment-1" / "artifacts" / "artifact-1.json").is_file()


def test_extract_rejects_experiment_without_diffusion_artifacts(tmp_path):
    experiment = tmp_path / "experiment-1"
    DiffusionStore(experiment / "diffusion")
    output_dir = tmp_path / "artifact-stores"

    result = CliRunner().invoke(
        app,
        ["extract", "-p", str(experiment), "--output-dir", str(output_dir)],
    )

    assert result.exit_code == 1
    assert not (output_dir / "experiment-1").exists()
