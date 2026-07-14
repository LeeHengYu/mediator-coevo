from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
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


def test_sequence_runs_k_seeded_permutations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = _repository()
    runs: list[tuple[OrchestrationArm, int, int, int, tuple[str, ...], Path]] = []

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

    monkeypatch.setattr(
        sequence_module,
        "_load_config_or_bad_parameter",
        lambda *args, **kwargs: SimpleNamespace(
            experiment=SimpleNamespace(
                seed=0,
                orchestration_arm=OrchestrationArm.RANDOM_POLICY,
            )
        ),
    )
    monkeypatch.setattr(
        sequence_module, "build_benchmark_repo", lambda *args: repository
    )
    monkeypatch.setattr(
        sequence_module, "prepare_llm_credentials_or_exit", lambda config: None
    )
    monkeypatch.setattr(sequence_module, "ensure_harbor_available", lambda config: None)
    monkeypatch.setattr(sequence_module, "_run_sequence", fake_run_sequence)

    result = CliRunner().invoke(
        app,
        [
            "sequence",
            *_FAMILY_ARGS,
            "--seed",
            "7",
            "-K",
            "3",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert [
        (arm, seed, iteration, total) for arm, seed, iteration, total, _, _ in runs
    ] == [
        (OrchestrationArm.RANDOM_POLICY, 7, 1, 3),
        (OrchestrationArm.RANDOM_POLICY, 8, 2, 3),
        (OrchestrationArm.RANDOM_POLICY, 9, 3, 3),
    ]
    assert len({task_ids for *_, task_ids, _ in runs}) == 3
    assert len({frozenset(task_ids) for *_, task_ids, _ in runs}) == 1
    assert [path.name for *_, path in runs] == ["iter-1", "iter-2", "iter-3"]
    assert len({path.parent for *_, path in runs}) == 1
    assert runs[0][-1].parent.name.endswith("-7")
    output = " ".join(result.stdout.split())
    assert "Sequence run: 3 iteration(s), 10 tasks each" in output
    assert "arm random_policy" in output

    help_result = CliRunner().invoke(app, ["sequence", "--help"])
    assert help_result.exit_code == 0
    assert "-K" in help_result.stdout
    assert "default: 1" in help_result.stdout
    assert "--split" not in help_result.stdout


def test_sequence_rejects_non_positive_k() -> None:
    result = CliRunner().invoke(app, ["sequence", *_FAMILY_ARGS, "-K", "0"])

    assert result.exit_code == 2
    assert "-K must be at least 1" in result.stderr
