from __future__ import annotations

import json
import tomllib

import pytest
import typer

from mediated_coevo import main as main_module
from mediated_coevo.experiment.baselines import (
    BASELINE_PRESET_NAMES,
    BASELINE_PRESETS_BY_NAME,
    SkillUpdateParseError,
    parse_skill_updates,
)
from mediated_coevo.experiment.conditions import (
    ExperimentDesignError,
    validate_experiment_design,
)
from mediated_coevo.core.config import Config, SkillUpdateConfig
from mediated_coevo.main import (
    ExperimentFactory,
    _apply_experiment_settings,
    _build_matrix_runtimes,
)
from mediated_coevo.experiment.records import build_coevolution_record
from mediated_coevo.evolution.executor_skill_gate import ExecutorSkillGate
from mediated_coevo.models.skill import SkillProposal
from mediated_coevo.experiment.orchestrator import Orchestrator
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore


def _config() -> Config:
    return Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
            "judge": "test-judge",
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


def test_apply_experiment_settings_supports_shared_runtime_knobs():
    config = _config()

    updated = _apply_experiment_settings(
        config,
        iterations=4,
        seed=123,
        coevo_interval=2,
        advisor_buffer_max=1,
        skill_validation_enabled=False,
    )

    assert updated is config
    assert config.experiment.num_iterations == 4
    assert config.experiment.seed == 123
    assert config.experiment.coevo_interval == 2
    assert config.experiment.advisor_buffer_max == 1
    assert config.experiment.skill_validation.enabled is False


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
    assert len(BASELINE_PRESET_NAMES) == 6
    assert "learned_mediator_same_task" not in BASELINE_PRESETS_BY_NAME
    for preset_name, (condition, skill_updates) in expected.items():
        preset = BASELINE_PRESETS_BY_NAME[preset_name]
        assert preset.condition_name == condition
        assert preset.skill_updates.model_dump() == skill_updates


def test_baseline_presets_have_unique_condition_and_update_semantics():
    semantics = [
        (
            preset.condition_name,
            tuple(
                role
                for role, enabled in preset.skill_updates.model_dump().items()
                if enabled
            ),
        )
        for preset in BASELINE_PRESETS_BY_NAME.values()
    ]

    assert len(semantics) == len(set(semantics))


def test_all_baseline_presets_validate():
    for preset in BASELINE_PRESETS_BY_NAME.values():
        validate_experiment_design(
            condition=preset.condition_name,
            skill_updates=preset.skill_updates,
            baseline_preset=preset.name,
        )


@pytest.mark.parametrize(
    ("condition", "skill_updates", "match"),
    [
        ("no_feedback", SkillUpdateConfig(planner=True), "no_feedback"),
        ("full_traces", SkillUpdateConfig(mediator=True), "mediator"),
        (
            "shared_notes",
            SkillUpdateConfig(executor=True, planner=False, mediator=False),
            "shared_notes",
        ),
        ("static_mediator", SkillUpdateConfig(mediator=True), "static_mediator"),
    ],
)
def test_invalid_condition_update_designs_fail_before_runtime(
    condition,
    skill_updates,
    match,
):
    with pytest.raises(ExperimentDesignError, match=match):
        validate_experiment_design(condition=condition, skill_updates=skill_updates)


def _write_skill(root, skill_name: str, content: str) -> None:
    skill_dir = root / "skills" / skill_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(content)


def _write_minimal_config(config_dir) -> None:
    config_dir.mkdir()
    (config_dir / "default.toml").write_text(
        """
        [models]
        planner = "test-planner"
        executor = "test-executor"
        mediator = "test-mediator"
        judge = "test-judge"
        """
    )


def test_run_command_validates_design_before_harbor(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    _write_minimal_config(config_dir)

    def fail_if_called(config):
        raise AssertionError("harbor check should happen after design validation")

    monkeypatch.setattr(main_module, "_ensure_harbor_available", fail_if_called)

    with pytest.raises(typer.BadParameter, match="no_feedback"):
        main_module.run(
            skillsbench_tasks=["task-A"],
            iterations=1,
            seed=42,
            condition="no_feedback",
            skill_updates="executor",
            config_dir=config_dir,
        )


def test_run_command_requires_task_selection_before_harbor(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    _write_minimal_config(config_dir)

    def fail_if_called(config):
        raise AssertionError("harbor check should happen after task validation")

    monkeypatch.setattr(main_module, "_ensure_harbor_available", fail_if_called)

    with pytest.raises(typer.BadParameter, match="provide at least one SkillsBench"):
        main_module.run(
            iterations=1,
            seed=42,
            condition="learned_mediator",
            skill_updates="all",
            config_dir=config_dir,
        )


def test_factory_build_validates_design_before_creating_experiment_dir(tmp_path):
    for skill_name in ("executor", "planner", "mediator"):
        _write_skill(tmp_path, skill_name, f"# {skill_name}\n")
    config = _config()
    config.paths.skills_dir = "skills"
    config.experiment.condition_name = "no_feedback"
    config.experiment.skill_updates = SkillUpdateConfig(
        executor=True,
        planner=False,
        mediator=False,
    )
    experiment_dir = tmp_path / "experiment"

    with pytest.raises(ExperimentDesignError, match="no_feedback"):
        ExperimentFactory(tmp_path).build(
            config=config,
            seed=42,
            condition_name="no_feedback",
            experiment_dir=experiment_dir,
        )

    assert not experiment_dir.exists()


def test_factory_build_uses_experiment_local_skill_store_by_default(tmp_path):
    _write_skill(tmp_path, "executor", "# Executor\n")
    _write_skill(tmp_path, "planner", "# Planner\n")
    _write_skill(tmp_path, "mediator", "# Mediator\n")
    config = _config()
    config.paths.skills_dir = "skills"
    experiment_dir = tmp_path / "experiment"

    runtime = ExperimentFactory(tmp_path).build(
        config=config,
        seed=42,
        condition_name=config.experiment.condition_name,
        experiment_dir=experiment_dir,
    )

    runtime_skill_dir = runtime.orchestrator.skill_store._skills_dir
    assert runtime_skill_dir == experiment_dir / "skills"
    assert runtime_skill_dir != tmp_path / "skills"
    assert (runtime_skill_dir / "executor" / "SKILL.md").read_text() == "# Executor\n"

    runtime.orchestrator.skill_store.write_skill("executor", "# Runtime Executor\n")

    assert (runtime_skill_dir / "executor" / "SKILL.md").read_text() == (
        "# Runtime Executor\n"
    )
    assert (tmp_path / "skills" / "executor" / "SKILL.md").read_text() == (
        "# Executor\n"
    )


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
        raise AssertionError(
            "advisor should not review when executor updates are disabled"
        )


@pytest.mark.asyncio
async def test_disabled_executor_updates_skip_proposal_and_advisor(tmp_path):
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = _config()
    orch.config.experiment.skill_updates.executor = False
    orch.config.experiment.advisor_buffer_max = 1
    orch.planner = _NoCallPlanner()
    orch.skill_advisor = _NoCallAdvisor()
    orch.history_store = HistoryStore(history_dir=tmp_path / "history")
    orch.artifact_store = ArtifactStore(base_dir=tmp_path / "artifacts")
    orch.skill_store = object()
    orch.executor = object()
    orch.benchmark_repo = object()
    orch.executor_skill_gate = ExecutorSkillGate(
        config=orch.config,
        skill_store=orch.skill_store,
        history_store=orch.history_store,
        planner=orch.planner,
        skill_advisor=orch.skill_advisor,
        executor=orch.executor,
        benchmark_repo=orch.benchmark_repo,
        artifact_store=orch.artifact_store,
    )
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
    update = await orch.executor_skill_gate.review_and_patch(
        iteration=1,
        proposal_buffer=orch._proposal_buffer,
    )

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


class _SkillStore:
    def skill_hashes(self) -> dict[str, str]:
        return {}


@pytest.mark.asyncio
async def test_planner_and_mediator_reflection_are_independently_gated(monkeypatch):
    calls: list[str] = []

    class _RecordingReflector:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def reflect(self, agent_role, *args, **kwargs):
            calls.append(agent_role)
            return None

    import mediated_coevo.experiment.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "Reflector", _RecordingReflector)

    orch = Orchestrator.__new__(Orchestrator)
    orch.config = _config()
    orch.history_store = object()
    orch.skill_store = _SkillStore()
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
    orch.skill_store = _SkillStore()

    record = build_coevolution_record(
        iteration=4,
        condition="learned_mediator",
        duration_sec=0.0,
        llm_token_events=[],
        skill_updates=None,
        config=orch.config,
        skill_hashes=orch.skill_store.skill_hashes(),
    )
    dumped = json.loads(record.model_dump_json())

    assert dumped["baseline_preset"] == "full_coevolution"
    assert dumped["skill_update_policy"] == {
        "executor": False,
        "planner": True,
        "mediator": False,
    }
