from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import typer
from typer.testing import CliRunner

from mediated_coevo.benchmarks import SkillFlowRepository
from mediated_coevo.cli import sequence as sequence_module
from mediated_coevo.experiment.sample_models import SequenceSpec
from mediated_coevo.main import app
from mediated_coevo.orchestration.arms import OrchestrationArm

_FAMILIES = ("alpha", "beta", "gamma", "delta")
_FAMILY_ARGS = [item for family in _FAMILIES for item in ("--family", family)]


def _repository() -> SkillFlowRepository:
    tasks = {
        f"{family}-{index}": SimpleNamespace(
            task_id=f"{family}-{index}",
            family=family,
            instruction=f"execute {family} task {index}",
            task_config={},
        )
        for family, count in zip(_FAMILIES, (3, 3, 2, 2), strict=True)
        for index in range(count)
    }
    repository = MagicMock(spec=SkillFlowRepository)
    repository.list_local_task_ids.side_effect = lambda *, family: [
        task_id for task_id, task in tasks.items() if task.family == family
    ]
    repository.resolve.side_effect = tasks.__getitem__
    return repository


@pytest.mark.parametrize(
    ("agent_args", "expected_arm"),
    [
        ([], OrchestrationArm.EXECUTION_ONLY),
        (["--graph-agent"], OrchestrationArm.GRAPH_ONLY),
        (["--diffusion-agent"], OrchestrationArm.DIFFUSION_ONLY),
        (
            ["--graph-agent", "--diffusion-agent"],
            OrchestrationArm.FULL_ORCHESTRATION,
        ),
    ],
)
def test_sequence_runs_k_seeded_permutations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent_args: list[str],
    expected_arm: OrchestrationArm,
) -> None:
    repository = _repository()
    harness = tmp_path / "harness"
    target = harness / "overlay" / sequence_module._SEQUENCE_HARNESS_FILES[1]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# harness\n")
    runs: list[tuple[OrchestrationArm, int, int, int, tuple[str, ...], Path]] = []
    applied: list[Path] = []
    prepared: list[tuple[Path, Path, dict[str, Any]]] = []
    restored: list[bool] = []

    async def fake_run_sequence(**kwargs: Any) -> Any:
        spec: SequenceSpec = kwargs["sequence"]
        runs.append(
            (
                kwargs["arm"],
                kwargs["config"].experiment.seed,
                kwargs["iteration"],
                kwargs["iterations"],
                spec.task_ids,
                kwargs["sequence_dir"],
            )
        )
        return SimpleNamespace(
            spec=SimpleNamespace(arm=kwargs["arm"]),
            rewards=SimpleNamespace(unweighted_mean=1.0, valid_for_reporting=True),
        )

    def load_config(*args: Any, **kwargs: Any) -> SimpleNamespace:
        assert "orchestration_arm" not in kwargs["overrides"]["experiment"]
        return SimpleNamespace(
            experiment=SimpleNamespace(
                seed=0,
            )
        )

    monkeypatch.setattr(sequence_module, "_load_config_or_bad_parameter", load_config)
    monkeypatch.setattr(
        sequence_module, "build_benchmark_repo", lambda *args: repository
    )
    monkeypatch.setattr(
        sequence_module, "prepare_llm_credentials_or_exit", lambda config: None
    )
    monkeypatch.setattr(sequence_module, "ensure_harbor_available", lambda config: None)
    monkeypatch.setattr(sequence_module, "_run_sequence", fake_run_sequence)
    monkeypatch.setattr(
        sequence_module,
        "_apply_harness_overlay_and_reexec",
        applied.append,
    )
    monkeypatch.setattr(
        sequence_module,
        "_prepare_harness_workspace",
        lambda run_dir, harness_dir, **kwargs: prepared.append(
            (run_dir, harness_dir, kwargs)
        ),
    )
    monkeypatch.setattr(
        sequence_module,
        "_restore_scoped_harness_overlay",
        lambda: restored.append(True),
    )

    result = CliRunner().invoke(
        app,
        [
            "sequence",
            *_FAMILY_ARGS,
            "--seed",
            "7",
            "-K",
            "3",
            *agent_args,
            "--output-dir",
            str(tmp_path),
            "--harness-dir",
            str(harness),
        ],
    )

    assert result.exit_code == 0, result.output
    assert [
        (arm, seed, iteration, total) for arm, seed, iteration, total, _, _ in runs
    ] == [
        (expected_arm, 7, 1, 3),
        (expected_arm, 8, 2, 3),
        (expected_arm, 9, 3, 3),
    ]
    assert len({task_ids for *_, task_ids, _ in runs}) == 3
    assert len({frozenset(task_ids) for *_, task_ids, _ in runs}) == 1
    assert [path.name for *_, path in runs] == ["iter-1", "iter-2", "iter-3"]
    assert len({path.parent for *_, path in runs}) == 1
    assert runs[0][-1].parent.name.endswith("-7")
    assert applied == [harness]
    assert prepared == [
        (
            runs[0][-1].parent,
            harness,
            {"harness_ref": None, "archive_snapshot": False},
        )
    ]
    assert restored == [True]


def test_sequence_rejects_legacy_facade_only_harness(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    facade = harness / "overlay/src/mediated_coevo/diffusion/langchain_graph.py"
    facade.parent.mkdir(parents=True)
    facade.write_text("# legacy facade\n")

    with pytest.raises(typer.BadParameter):
        sequence_module.sequence(family=list(_FAMILIES), harness_dir=harness)


def test_sequence_rejects_non_positive_k() -> None:
    with pytest.raises(typer.BadParameter):
        sequence_module.sequence(family=list(_FAMILIES), k=0)
