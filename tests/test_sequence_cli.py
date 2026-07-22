from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import typer
from pydantic import ValidationError
from typer.testing import CliRunner

from mediated_coevo.benchmarks import SkillFlowRepository
from mediated_coevo.cli import sequence as sequence_module
from mediated_coevo.core.config import SequenceConfig
from mediated_coevo.experiment.sample_models import SequenceSpec
from mediated_coevo.execution.models import TaskProfile
from mediated_coevo.main import app
from mediated_coevo.orchestration.arms import OrchestrationArm

_FAMILY = "alpha"
_FAMILY_ARGS = ["--family", _FAMILY]


def _repository(
    family_counts: dict[str, int] | None = None,
) -> SkillFlowRepository:
    family_counts = family_counts or {_FAMILY: 8}
    tasks = {
        f"{family}-{index}": SimpleNamespace(
            task_id=f"{family}-{index}",
            family=family,
            instruction=f"execute {family} task {index}",
            task_config={},
        )
        for family, count in family_counts.items()
        for index in range(count)
    }
    repository = MagicMock(spec=SkillFlowRepository)
    repository.list_local_task_ids.side_effect = lambda *, family: [
        task_id for task_id, task in tasks.items() if task.family == family
    ]
    repository.resolve.side_effect = tasks.__getitem__
    return repository


@pytest.mark.parametrize(
    ("task_count", "expected_multiplicities"),
    [
        (8, [1, 1, 1, 1, 1, 1, 2, 2]),
        (9, [1, 1, 1, 1, 1, 1, 1, 1, 2]),
    ],
)
def test_sequence_sampler_balances_repeats_with_distinct_warmup(
    task_count: int,
    expected_multiplicities: list[int],
) -> None:
    repository = _repository({_FAMILY: task_count})

    selected = sequence_module._select_sequence_tasks(
        repository,
        _FAMILY,
        seed=7,
        length=10,
        warmup_count=3,
    )

    assert len(selected) == 10
    assert len(set(selected[:3])) == 3
    assert sorted(Counter(selected).values()) == expected_multiplicities
    assert selected == sequence_module._select_sequence_tasks(
        repository,
        _FAMILY,
        seed=7,
        length=10,
        warmup_count=3,
    )


def test_sequence_sampler_rejects_missing_family() -> None:
    with pytest.raises(typer.BadParameter, match="no local SkillFlow tasks"):
        sequence_module._select_sequence_tasks(
            _repository({_FAMILY: 0}),
            _FAMILY,
            seed=0,
            length=10,
            warmup_count=3,
        )


def test_sequence_sampler_can_repeat_one_task_without_warmup() -> None:
    selected = sequence_module._select_sequence_tasks(
        _repository({_FAMILY: 1}),
        _FAMILY,
        seed=0,
        length=5,
        warmup_count=0,
    )

    assert selected == [f"{_FAMILY}-0"] * 5


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
    runs: list[tuple[OrchestrationArm, int, int, int, tuple[str, ...], int, Path]] = []
    applied: list[Path] = []
    prepared: list[tuple[Path, Path, dict[str, Any]]] = []
    restored: list[bool] = []
    task_set_ids: list[str | None] = []

    async def fake_run_sequence(**kwargs: Any) -> Any:
        spec: SequenceSpec = kwargs["sequence"]
        task_set_ids.append(spec.task_set_id)
        runs.append(
            (
                kwargs["arm"],
                kwargs["config"].experiment.seed,
                kwargs["iteration"],
                kwargs["iterations"],
                spec.task_ids,
                spec.warmup_count,
                kwargs["sequence_dir"],
            )
        )
        return SimpleNamespace(
            spec=SimpleNamespace(arm=kwargs["arm"]),
            rewards=SimpleNamespace(unweighted_mean=1.0, valid_for_reporting=True),
        )

    def load_config(*args: Any, **kwargs: Any) -> SimpleNamespace:
        assert "orchestration_arm" not in kwargs["overrides"]["experiment"]
        assert kwargs["overrides"]["sequence"] == {"length": 6, "warmup": 2}
        return SimpleNamespace(
            experiment=SimpleNamespace(
                seed=0,
            ),
            sequence=SimpleNamespace(length=6, warmup=2),
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
            "-n",
            "6",
            "--warmup",
            "2",
            *agent_args,
            "--output-dir",
            str(tmp_path),
            "--harness-dir",
            str(harness),
        ],
    )

    assert result.exit_code == 0, result.output
    assert [
        (arm, seed, iteration, total) for arm, seed, iteration, total, *_ in runs
    ] == [
        (expected_arm, 7, 1, 3),
        (expected_arm, 8, 2, 3),
        (expected_arm, 9, 3, 3),
    ]
    assert all(len(run[4]) == 6 for run in runs)
    assert all(run[5] == 2 for run in runs)
    assert len({run[4] for run in runs}) == 3
    assert task_set_ids == [f"families:{_FAMILY}"] * 3
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
        sequence_module.sequence(family=[_FAMILY], harness_dir=harness)


def test_sequence_rejects_non_positive_k() -> None:
    with pytest.raises(typer.BadParameter):
        sequence_module.sequence(family=[_FAMILY], k=0)


def test_sequence_rejects_multiple_families_before_loading_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_config = MagicMock()
    monkeypatch.setattr(sequence_module, "_load_config_or_bad_parameter", load_config)

    with pytest.raises(typer.BadParameter, match="exactly one family"):
        sequence_module.sequence(family=[_FAMILY, "beta"])

    load_config.assert_not_called()


def test_sequence_validates_distinct_warmup_before_runtime_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = MagicMock()
    harbor = MagicMock()
    monkeypatch.setattr(
        sequence_module,
        "build_benchmark_repo",
        lambda *args: _repository({_FAMILY: 2}),
    )
    monkeypatch.setattr(sequence_module, "prepare_llm_credentials_or_exit", credentials)
    monkeypatch.setattr(sequence_module, "ensure_harbor_available", harbor)

    with pytest.raises(typer.BadParameter, match="has 2 distinct tasks"):
        sequence_module.sequence(family=[_FAMILY], length=5, warmup=3)

    credentials.assert_not_called()
    harbor.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"length": 3},
        {"warmup": -1},
        {"length": 4, "warmup": 4},
    ),
)
def test_sequence_rejects_invalid_cli_task_counts(kwargs: dict[str, int]) -> None:
    with pytest.raises(typer.BadParameter):
        sequence_module.sequence(
            family=[_FAMILY],
            length=kwargs.get("length"),
            warmup=kwargs.get("warmup"),
        )


def test_sequence_config_defaults_and_requires_a_suffix() -> None:
    assert SequenceConfig() == SequenceConfig(length=10, warmup=3)
    with pytest.raises(ValidationError, match="leave at least one suffix task"):
        SequenceConfig(length=4, warmup=4)


@pytest.mark.asyncio
async def test_run_sequence_skips_warmup_loading_when_count_is_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = object()
    runtime = SimpleNamespace(run=AsyncMock(return_value=expected))
    config = MagicMock()
    config.model_copy.return_value = config
    config.experiment.condition_name = "learned_mediator"
    monkeypatch.setattr(
        sequence_module,
        "build_experiment",
        lambda **kwargs: SimpleNamespace(orchestrator=object()),
    )
    monkeypatch.setattr(
        sequence_module,
        "build_sample_runtime",
        lambda **kwargs: runtime,
    )
    spec = SequenceSpec(
        sequence_id="zero-warmup",
        tasks=(TaskProfile(task_id="task-0", instruction="execute"),),
        warmup_count=0,
        policy_seed=7,
    )

    result = await sequence_module._run_sequence(
        config=config,
        repository=_repository(),
        sequence=spec,
        arm=OrchestrationArm.EXECUTION_ONLY,
        sequence_dir=tmp_path,
        artifact_store_root=tmp_path / "stores",
        iteration=1,
        iterations=1,
    )

    assert result is expected
    sample_spec = runtime.run.await_args.args[0]
    assert sample_spec.warmup_bundle_id is None
    assert runtime.run.await_args.kwargs["warmup"] is None
