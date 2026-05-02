from __future__ import annotations

import json
import tomllib

import pytest

from mediated_coevo.baselines import (
    BASELINE_PRESET_NAMES,
    BASELINE_PRESETS_BY_NAME,
    SkillUpdateParseError,
    parse_skill_updates,
)
from mediated_coevo.config import Config, SkillUpdateConfig
from mediated_coevo.main import (
    ExperimentFactory,
    _build_matrix_runtimes,
)
from mediated_coevo.models.skill import SkillProposal
from mediated_coevo.orchestrator import Orchestrator
from mediated_coevo.stores.history_store import HistoryStore


def _config() -> Config:
    return Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
        }
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("none", {"executor": False, "planner": False, "mediator": False}),
        ("executor", {"executor": True, "planner": False, "mediator": False}),
        ("planner,mediator", {"executor": False, "planner": True, "mediator": True}),
        ("executor, planner", {"executor": True, "planner": True, "mediator": False}),
        ("all", {"executor": True, "planner": True, "mediator": True}),
    ],
)
def test_parse_skill_updates(raw, expected):
    assert parse_skill_updates(raw).model_dump() == expected


@pytest.mark.parametrize("raw", ["", "bad", "none,executor", "all,mediator"])
def test_parse_skill_updates_rejects_invalid_values(raw):
    with pytest.raises(SkillUpdateParseError):
        parse_skill_updates(raw)


def test_baseline_preset_mapping_matches_matrix_plan():
    expected = {
        "no_feedback": (
            "no_feedback",
            {"executor": False, "planner": False, "mediator": False},
        ),
        "full_trace_same_task": (
            "full_traces",
            {"executor": False, "planner": False, "mediator": False},
        ),
        "static_mediator_same_task": (
            "static_mediator",
            {"executor": False, "planner": False, "mediator": False},
        ),
        "learned_mediator_same_task": (
            "learned_mediator",
            {"executor": False, "planner": False, "mediator": True},
        ),
        "planner_only_skill_evolution": (
            "learned_mediator",
            {"executor": False, "planner": True, "mediator": False},
        ),
        "mediator_only_protocol_evolution": (
            "learned_mediator",
            {"executor": False, "planner": False, "mediator": True},
        ),
        "full_coevolution": (
            "learned_mediator",
            {"executor": True, "planner": True, "mediator": True},
        ),
    }

    assert list(BASELINE_PRESETS_BY_NAME) == BASELINE_PRESET_NAMES
    for preset_name, (condition, skill_updates) in expected.items():
        preset = BASELINE_PRESETS_BY_NAME[preset_name]
        assert preset.condition_name == condition
        assert preset.skill_updates.model_dump() == skill_updates


def _write_skill(root, skill_name: str, content: str) -> None:
    skill_dir = root / "skills" / skill_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(content)


def test_matrix_runtimes_use_isolated_skill_copies_and_shared_config(tmp_path):
    _write_skill(tmp_path, "executor", "# Executor\n")
    _write_skill(tmp_path, "planner", "# Planner\n")
    _write_skill(tmp_path, "mediator", "# Mediator\n")
    config = _config()
    config.experiment.num_iterations = 8
    config.paths.skills_dir = "skills"
    config.paths.data_dir = "data"
    config.paths.benchmarks_dir = "benchmarks/skillsbench"
    matrix_dir = tmp_path / "data" / "experiments" / "matrix"

    rows = _build_matrix_runtimes(
        factory=ExperimentFactory(tmp_path),
        base_config=config,
        seed=123,
        matrix_dir=matrix_dir,
    )

    assert [row.preset_name for row in rows] == BASELINE_PRESET_NAMES
    row_skill_dirs = []
    for row in rows:
        row_config = row.runtime.orchestrator.config
        skill_dir = row.runtime.orchestrator.skill_store._skills_dir
        benchmark_repo = row.runtime.orchestrator.benchmark_repo
        row_skill_dirs.append(skill_dir)
        assert skill_dir == matrix_dir / row.preset_name / "skills"
        assert (skill_dir / "executor" / "SKILL.md").read_text() == "# Executor\n"
        assert benchmark_repo.root_dir == tmp_path / "benchmarks" / "skillsbench"
        assert (
            benchmark_repo.default_local_cache_dir()
            == tmp_path / "benchmarks" / "skillsbench" / "tasks"
        )
        assert row_config.experiment.seed == 123
        assert row_config.experiment.num_iterations == 8
        assert row_config.experiment.baseline_preset == row.preset_name
        assert row_config.models.model_dump() == config.models.model_dump()
        assert row_config.budgets.model_dump() == config.budgets.model_dump()

        saved = tomllib.loads((row.runtime.experiment_dir / "config.toml").read_text())
        preset = BASELINE_PRESETS_BY_NAME[row.preset_name]
        assert saved["experiment"]["baseline_preset"] == row.preset_name
        assert saved["experiment"]["condition_name"] == preset.condition_name
        assert saved["experiment"]["seed"] == 123
        assert saved["experiment"]["num_iterations"] == 8
        assert saved["experiment"]["skill_updates"] == preset.skill_updates.model_dump()

    assert len(set(row_skill_dirs)) == len(row_skill_dirs)
    (row_skill_dirs[0] / "executor" / "SKILL.md").write_text("# Changed\n")
    assert (tmp_path / "skills" / "executor" / "SKILL.md").read_text() == "# Executor\n"
    assert (row_skill_dirs[1] / "executor" / "SKILL.md").read_text() == "# Executor\n"


class _NoCallPlanner:
    async def suggest_skill_revision(self, *args, **kwargs):
        raise AssertionError("planner should not propose or patch executor skills")


class _NoCallAdvisor:
    async def review(self, *args, **kwargs):
        raise AssertionError("advisor should not review when executor updates are disabled")


@pytest.mark.asyncio
async def test_disabled_executor_updates_skip_proposal_and_advisor(tmp_path):
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = _config()
    orch.config.experiment.skill_updates.executor = False
    orch.config.experiment.advisor_buffer_max = 1
    orch.planner = _NoCallPlanner()
    orch.skill_advisor = _NoCallAdvisor()
    orch.history_store = HistoryStore(history_dir=tmp_path / "history")
    orch._proposal_buffer = []

    await orch._ask_planner_for_skill_proposal(
        task_id="task-A",
        iteration=0,
        executor_skill="# Executor\n",
        feedback="useful feedback",
    )

    assert orch._proposal_buffer == []

    orch._proposal_buffer = [
        SkillProposal(
            iteration=0,
            task_id="task-A",
            old_content="# Executor\n",
            new_content="# New Executor\n",
        )
    ]
    update = await orch._review_proposals_and_patch_skill(iteration=1)

    assert update is None
    assert orch._proposal_buffer == []


class _LLM:
    model = "test-model"

    def drain_token_events(self):
        return []


class _Planner:
    llm_client = _LLM()


class _Mediator:
    llm_client = _LLM()

    def load_protocol(self, content: str) -> None:
        raise AssertionError("no reflection result should be loaded in this test")


class _Advisor:
    llm_client = _LLM()


@pytest.mark.asyncio
async def test_planner_and_mediator_reflection_are_independently_gated(monkeypatch):
    calls: list[str] = []

    class _RecordingReflector:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def reflect(self, agent_role, *args, **kwargs):
            calls.append(agent_role)
            return None

    import mediated_coevo.evolution.reflector as reflector_module

    monkeypatch.setattr(reflector_module, "Reflector", _RecordingReflector)

    orch = Orchestrator.__new__(Orchestrator)
    orch.config = _config()
    orch.history_store = object()
    orch.skill_store = object()
    orch.planner = _Planner()
    orch.mediator = _Mediator()
    orch.skill_advisor = _Advisor()

    orch.config.experiment.skill_updates = SkillUpdateConfig(
        executor=False,
        planner=True,
        mediator=False,
    )
    assert await orch._coevolve(4, "no_feedback") is None
    assert calls == ["planner"]

    calls.clear()
    orch.config.experiment.skill_updates = SkillUpdateConfig(
        executor=False,
        planner=False,
        mediator=True,
    )
    assert await orch._coevolve(4, "no_feedback") is None
    assert calls == ["mediator"]


def test_metrics_rows_include_baseline_and_skill_update_policy():
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = _config()
    orch.config.experiment.baseline_preset = "full_coevolution"
    orch.config.experiment.skill_updates = SkillUpdateConfig(
        executor=False,
        planner=True,
        mediator=False,
    )
    orch.skill_store = object()

    record = orch._build_coevolution_record(
        iteration=4,
        condition="learned_mediator",
        start=0.0,
        llm_token_events=[],
    )
    dumped = json.loads(record.model_dump_json())

    assert dumped["baseline_preset"] == "full_coevolution"
    assert dumped["skill_update_policy"] == {
        "executor": False,
        "planner": True,
        "mediator": False,
    }
