from __future__ import annotations

import json
from types import SimpleNamespace
import tomllib

import pytest
import typer
from typer.testing import CliRunner

import mediated_coevo.cli.matrix as matrix_module
import mediated_coevo.cli.run as run_module
from mediated_coevo.main import app
from mediated_coevo.core.config import (
    Config,
    ConfigLoadError,
    ModelsConfig,
    SkillUpdateConfig,
    load_config,
)
from mediated_coevo.benchmarks import SkillFlowRepository
from mediated_coevo.experiment.baselines import (
    BASELINE_PRESET_NAMES,
    BASELINE_PRESETS_BY_NAME,
    SkillUpdateParseError,
    get_baseline_preset,
    parse_skill_updates,
)
from mediated_coevo.experiment.conditions import (
    ExperimentDesignError,
    validate_experiment_design,
)
from mediated_coevo.experiment.runtime_factory import (
    build_benchmark_repo,
    build_experiment,
    build_matrix_runtimes,
    create_matrix_dir,
)
from mediated_coevo.cli.config import _run_config_overrides
from mediated_coevo.cli.graph import materialize_task_graph_for_diffusion
from mediated_coevo.experiment.records import build_coevolution_record
from mediated_coevo.evolution.executor_skill_gate import ExecutorSkillGate
from mediated_coevo.models.skill import SkillProposal
from mediated_coevo.experiment.orchestrator import Orchestrator
from mediated_coevo.runtime.token_budget import TokenBudgetEvent
from mediated_coevo.diffusion import (
    DiffusionArtifact,
    DiffusionArtifactType,
    DiffusionRiskLevel,
    DiffusionStore,
)
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from tests.config_helpers import budgets_config, diffusion_config, experiment_config


def _config() -> Config:
    return Config(
        models=ModelsConfig(
            planner="test-planner",
            executor="test-executor",
            mediator="test-mediator",
            judge="test-judge",
        ),
        budgets=budgets_config(),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )


def _write_graph_task(task_dir, *, family: str, tags: list[str]) -> None:
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text(
        "Read workbook.xlsx and write result.xlsx with spreadsheet formulas."
    )
    (task_dir / "task.toml").write_text(
        "\n".join(
            [
                'schema_version = "1.2"',
                "",
                "[task]",
                f'name = "{family}/{task_dir.name}"',
                "",
                "[metadata]",
                f'family = "{family}"',
                'category = "spreadsheet-formula-reuse"',
                "tags = [" + ", ".join(f'"{tag}"' for tag in tags) + "]",
                "",
                "[environment]",
                "build_timeout_sec = 600.0",
            ]
        )
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


def test_run_config_overrides_supports_shared_runtime_knobs():
    overrides = _run_config_overrides(
        iterations=4,
        seed=123,
        condition=None,
        skill_updates=None,
        coevo_interval=2,
        advisor_buffer_max=1,
        diffusion_enabled=True,
        diffusion_policy="top_k_similarity",
        diffusion_graph="task_similarity",
        diffusion_max_artifacts=5,
        diffusion_top_k_neighbors=2,
        harbor_agent_setup_timeout_multiplier=2.5,
    )

    assert overrides == {
        "experiment": {
            "num_iterations": 4,
            "seed": 123,
            "coevo_interval": 2,
            "advisor_buffer_max": 1,
        },
        "diffusion": {
            "enabled": True,
            "policy": "top_k_similarity",
            "graph": "task_similarity",
            "max_artifacts": 5,
            "top_k_neighbors": 2,
        },
        "executor_runtime": {
            "harbor_agent_setup_timeout_multiplier": 2.5,
        },
    }


def test_materialize_task_graph_for_similarity_diffusion(tmp_path):
    tasks_root = tmp_path / "benchmarks" / "tasks"
    _write_graph_task(
        tasks_root / "family-a" / "task-one",
        family="family-a",
        tags=["excel", "formulas"],
    )
    _write_graph_task(
        tasks_root / "family-a" / "task-two",
        family="family-a",
        tags=["excel", "statistics"],
    )
    config = _config()
    config.diffusion.enabled = True
    config.diffusion.policy = "top_k_similarity"
    config.diffusion.graph = "task_similarity"
    repo = SkillFlowRepository(root_dir=tmp_path / "benchmarks", task_dirs=["tasks"])

    materialize_task_graph_for_diffusion(
        config=config,
        experiment_dir=tmp_path / "experiment",
        benchmark_repo=repo,
    )

    graph_dir = tmp_path / "experiment" / "task-graph"
    assert (graph_dir / "task_profiles.json").is_file()
    summary = json.loads((graph_dir / "graph_summary.json").read_text())
    assert summary["task_count"] == 2
    assert summary["active_threshold"] == 0.05


def test_baseline_preset_mapping_matches_matrix_plan():
    expected = {
        "skill_none_diffusion_none": (
            "learned_mediator",
            {"executor": False, "planner": False, "mediator": False},
            False,
            "none",
            "none",
        ),
        "skill_none_capped_broadcast": (
            "learned_mediator",
            {"executor": False, "planner": False, "mediator": False},
            True,
            "capped_broadcast",
            "none",
        ),
        "skill_none_random_k": (
            "learned_mediator",
            {"executor": False, "planner": False, "mediator": False},
            True,
            "random_k",
            "none",
        ),
        "skill_none_top_k_similarity": (
            "learned_mediator",
            {"executor": False, "planner": False, "mediator": False},
            True,
            "top_k_similarity",
            "task_similarity",
        ),
        "skill_all_diffusion_none": (
            "learned_mediator",
            {"executor": True, "planner": True, "mediator": True},
            False,
            "none",
            "none",
        ),
        "skill_all_capped_broadcast": (
            "learned_mediator",
            {"executor": True, "planner": True, "mediator": True},
            True,
            "capped_broadcast",
            "none",
        ),
        "skill_all_random_k": (
            "learned_mediator",
            {"executor": True, "planner": True, "mediator": True},
            True,
            "random_k",
            "none",
        ),
        "skill_all_top_k_similarity": (
            "learned_mediator",
            {"executor": True, "planner": True, "mediator": True},
            True,
            "top_k_similarity",
            "task_similarity",
        ),
    }

    assert list(BASELINE_PRESETS_BY_NAME) == BASELINE_PRESET_NAMES
    assert len(BASELINE_PRESET_NAMES) == 8
    assert "full_coevolution" not in BASELINE_PRESETS_BY_NAME
    for preset_name, (
        condition,
        skill_updates,
        diffusion_enabled,
        diffusion_policy,
        diffusion_graph,
    ) in expected.items():
        preset = BASELINE_PRESETS_BY_NAME[preset_name]
        assert preset.condition_name == condition
        assert preset.skill_updates.model_dump() == skill_updates
        assert preset.diffusion_enabled is diffusion_enabled
        assert preset.diffusion_policy == diffusion_policy
        assert preset.diffusion_graph == diffusion_graph


def test_baseline_presets_have_unique_row_semantics():
    semantics = [
        (
            preset.condition_name,
            tuple(
                role
                for role, enabled in preset.skill_updates.model_dump().items()
                if enabled
            ),
            preset.diffusion_enabled,
            preset.diffusion_policy,
            preset.diffusion_graph,
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


def test_matrix_command_rejects_row_local_diffusion_overrides(tmp_path):
    with pytest.raises(typer.BadParameter, match="matrix rows set diffusion"):
        matrix_module.matrix(
            diffusion_policy="random_k",
            config_dir=tmp_path / "config",
        )


@pytest.mark.parametrize("flag", ["--list", "-l"])
def test_matrix_list_prints_indexed_rows_without_runtime_setup(monkeypatch, flag):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("listing rows should not load runtime config")

    monkeypatch.setattr(matrix_module, "_load_config_or_bad_parameter", fail_if_called)

    result = CliRunner().invoke(app, ["matrix", flag])

    assert result.exit_code == 0, result.output


def _stub_matrix_runtime_build(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    repository = object()

    class Selection:
        task_ids = ["task-A"]
        family = "family-a"
        split = None
        task_stream_seed = 9001

    def capture_matrix_dir(*, project_root, seed, data_dir, run_id=None):
        captured["matrix_project_root"] = project_root
        captured["matrix_seed"] = seed
        captured["matrix_data_dir"] = data_dir
        captured["matrix_run_id"] = run_id
        return tmp_path / "matrix"

    def capture_matrix_build(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        matrix_module,
        "prepare_llm_credentials_or_exit",
        lambda config: config,
    )
    monkeypatch.setattr(matrix_module, "ensure_harbor_available", lambda config: None)
    monkeypatch.setattr(
        matrix_module,
        "build_benchmark_repo",
        lambda project_root, config: repository,
    )
    monkeypatch.setattr(
        matrix_module,
        "resolve_task_selection",
        lambda **kwargs: Selection(),
    )
    monkeypatch.setattr(matrix_module, "create_matrix_dir", capture_matrix_dir)
    monkeypatch.setattr(
        matrix_module,
        "build_matrix_runtimes",
        capture_matrix_build,
    )

    return captured, repository


def test_matrix_index_runs_only_selected_row(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    _write_minimal_config(config_dir)
    captured, repository = _stub_matrix_runtime_build(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "matrix",
            "--family",
            "family-a",
            "--index",
            "3",
            "--run-id",
            "custom-matrix-run",
            "--config-dir",
            str(config_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["preset_names"] == [BASELINE_PRESET_NAMES[3]]
    assert captured["flatten_single_row"] is True
    assert captured["benchmark_repo"] is repository
    assert captured["matrix_seed"] == 42
    assert captured["matrix_run_id"] == "custom-matrix-run"
    base_config = captured["base_config"]
    assert base_config.experiment.benchmark_selection.tasks == ["task-A"]
    assert base_config.experiment.benchmark_selection.family == "family-a"


def test_matrix_index_accepts_comma_separated_rows(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    _write_minimal_config(config_dir)
    captured, repository = _stub_matrix_runtime_build(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "matrix",
            "--family",
            "family-a",
            "--index",
            "1,3",
            "--config-dir",
            str(config_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["preset_names"] == [
        BASELINE_PRESET_NAMES[1],
        BASELINE_PRESET_NAMES[3],
    ]
    assert captured["flatten_single_row"] is False
    assert captured["benchmark_repo"] is repository
    assert captured["matrix_seed"] == 42


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("1,,3", "cannot be empty"),
        ("not-an-index", "must be comma-separated integers"),
        ("-1", "must be between"),
        (str(len(BASELINE_PRESET_NAMES)), "must be between"),
        ("1,1", "cannot repeat"),
    ],
)
def test_matrix_index_rejects_invalid_comma_separated_rows(value, match):
    with pytest.raises(typer.BadParameter, match=match):
        matrix_module._parse_matrix_row_indexes(value)


def test_matrix_save_rejects_no_diffusion_row():
    with pytest.raises(typer.BadParameter, match="diffusion-enabled rows"):
        matrix_module._validate_artifact_store_options(
            preset_names=[BASELINE_PRESET_NAMES[0]],
            save_artifacts=True,
            artifact_store=None,
            freeze_artifacts=False,
        )


def test_matrix_freeze_requires_artifact():
    with pytest.raises(typer.BadParameter, match="--freeze requires --artifact"):
        matrix_module._validate_artifact_store_options(
            preset_names=[BASELINE_PRESET_NAMES[1]],
            save_artifacts=False,
            artifact_store=None,
            freeze_artifacts=True,
        )


def test_matrix_preloads_artifact_store_and_freezes_runtime(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    _write_minimal_config(config_dir)
    source = DiffusionStore(tmp_path / "source")
    source.store_artifact(
        DiffusionArtifact(
            artifact_id="artifact-1",
            source_task_id="task-B",
            source_iteration=2,
            artifact_type=DiffusionArtifactType.DEBUG_HINT,
            risk_level=DiffusionRiskLevel.LOW,
            content="reuse this",
            verifier_reward=1.0,
        )
    )
    saved = tmp_path / "saved"
    source.save_artifact_store(saved, store_id="warmup")
    repository = object()
    selection = SimpleNamespace(
        task_ids=["task-A"],
        family="family-a",
        split=None,
        task_stream_seed=9001,
    )
    orch = SimpleNamespace(
        config=None,
        experiment_dir=tmp_path / "row-experiment",
        _diffusion_store=DiffusionStore(tmp_path / "row-experiment" / "diffusion"),
        history_store=object(),
        freeze_diffusion_artifact_store=False,
        preloaded_diffusion_artifact_store_path=None,
        preloaded_diffusion_artifact_store_count=0,
    )

    def fake_matrix_dir(*, project_root, seed, data_dir, run_id=None):
        return tmp_path / "matrix"

    def build_rows(**kwargs):
        row_config = get_baseline_preset(BASELINE_PRESET_NAMES[1]).build_config(
            kwargs["base_config"],
            seed=kwargs["seed"],
        )
        orch.config = row_config
        return [
            SimpleNamespace(
                preset_name=BASELINE_PRESET_NAMES[1],
                runtime=SimpleNamespace(
                    experiment_dir=orch.experiment_dir,
                    orchestrator=orch,
                ),
            )
        ]

    monkeypatch.setattr(
        matrix_module,
        "prepare_llm_credentials_or_exit",
        lambda config: config,
    )
    monkeypatch.setattr(matrix_module, "ensure_harbor_available", lambda config: None)
    monkeypatch.setattr(
        matrix_module,
        "build_benchmark_repo",
        lambda project_root, config: repository,
    )
    monkeypatch.setattr(
        matrix_module,
        "resolve_task_selection",
        lambda **kwargs: selection,
    )
    monkeypatch.setattr(matrix_module, "create_matrix_dir", fake_matrix_dir)
    monkeypatch.setattr(matrix_module, "build_matrix_runtimes", build_rows)
    monkeypatch.setattr(
        matrix_module,
        "materialize_task_graph_for_diffusion",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        matrix_module,
        "run_experiment_or_exit",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        matrix_module,
        "write_and_print_result_summary",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        matrix_module,
        "annotate_judge_rewards_or_exit",
        lambda **kwargs: None,
    )

    result = CliRunner().invoke(
        app,
        [
            "matrix",
            "--family",
            "family-a",
            "--index",
            "1",
            "--artifact",
            str(saved),
            "--freeze",
            "--config-dir",
            str(config_dir),
        ],
    )
    loaded = orch._diffusion_store.load_artifact("artifact-1")
    invocation = json.loads(
        (orch.experiment_dir / "matrix_invocation.json").read_text()
    )

    assert result.exit_code == 0, result.output
    assert loaded is not None
    assert loaded.source_iteration == -1
    assert loaded.metadata["preloaded_from_artifact_store"] == str(saved)
    assert loaded.metadata["preloaded_artifact_store_frozen"] is True
    assert orch.freeze_diffusion_artifact_store is True
    assert orch.preloaded_diffusion_artifact_store_path == str(saved)
    assert orch.preloaded_diffusion_artifact_store_count == 1
    assert "command" not in invocation
    assert invocation["family"] == "family-a"
    assert invocation["selected_task_ids"] == ["task-A"]
    assert invocation["row_indexes_argument"] == "1"
    assert invocation["selected_indexes"] == [1]
    assert invocation["artifact_store"] == str(saved)
    assert invocation["imported_artifact_count"] == 1
    assert invocation["freeze_artifacts"] is True


@pytest.mark.asyncio
async def test_frozen_diffusion_store_skips_artifact_emission(tmp_path):
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = _config()
    orch.config.diffusion.enabled = True
    orch.config.diffusion.policy = "capped_broadcast"
    orch.experiment_dir = tmp_path
    orch.freeze_diffusion_artifact_store = True

    await orch._emit_diffusion_artifacts(
        trace=None,
        report=None,
        record=None,
        task_metadata={},
        judge_reward=None,
    )

    assert orch._diffusion_store.query_artifacts(recent=None) == []


@pytest.mark.parametrize(
    ("condition", "skill_updates", "match"),
    [
        (
            "no_feedback",
            SkillUpdateConfig(executor=False, planner=True, mediator=False),
            "no_feedback",
        ),
        (
            "full_traces",
            SkillUpdateConfig(executor=False, planner=False, mediator=True),
            "mediator",
        ),
        (
            "shared_notes",
            SkillUpdateConfig(executor=True, planner=False, mediator=False),
            "shared_notes",
        ),
        (
            "static_mediator",
            SkillUpdateConfig(executor=False, planner=False, mediator=True),
            "static_mediator",
        ),
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

        [budgets]
        max_skill_tokens = 4000
        max_same_task_prior_tokens = 300
        max_transfer_context_tokens = 900
        trace_excerpt_tokens = 6000
        historical_summary_tokens = 3000
        mediator_report_tokens = 4000
        planner_context_tokens = 24000
        skill_update_diff_tokens = 6000
        mediator_prompt_tokens = 16000
        advisor_prompt_tokens = 12000
        reflector_prompt_tokens = 16000
        judge_prompt_tokens = 16000
        planner_completion_tokens = 4096
        mediator_completion_tokens = 2048
        advisor_completion_tokens = 512
        reflector_completion_tokens = 4096
        judge_completion_tokens = 2048

        [experiment]
        num_iterations = 2
        coevo_interval = 2
        advisor_buffer_max = 2
        seed = 42
        condition_name = "learned_mediator"

        [experiment.skill_updates]
        executor = true
        planner = true
        mediator = true

        [diffusion]
        enabled = false
        policy = "none"
        max_artifacts = 3
        top_k_neighbors = 3
        """
    )


def test_load_config_requires_runtime_settings_from_toml_or_overrides(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.toml").write_text(
        """
        [models]
        planner = "test-planner"
        executor = "test-executor"
        mediator = "test-mediator"
        judge = "test-judge"

        [budgets]
        max_skill_tokens = 4000
        max_same_task_prior_tokens = 300
        max_transfer_context_tokens = 900
        trace_excerpt_tokens = 6000
        historical_summary_tokens = 3000
        mediator_report_tokens = 4000
        planner_context_tokens = 24000
        skill_update_diff_tokens = 6000
        mediator_prompt_tokens = 16000
        advisor_prompt_tokens = 12000
        reflector_prompt_tokens = 16000
        judge_prompt_tokens = 16000
        planner_completion_tokens = 4096
        mediator_completion_tokens = 2048
        advisor_completion_tokens = 512
        reflector_completion_tokens = 4096
        judge_completion_tokens = 2048
        """
    )

    with pytest.raises(ConfigLoadError, match="experiment.num_iterations"):
        load_config(config_dir)

    config = load_config(
        config_dir,
        overrides={
            "experiment": {
                "num_iterations": 3,
                "coevo_interval": 2,
                "advisor_buffer_max": 2,
                "seed": 7,
                "condition_name": "learned_mediator",
                "skill_updates": {
                    "executor": True,
                    "planner": True,
                    "mediator": True,
                },
            },
            "diffusion": {
                "enabled": False,
                "policy": "none",
                "max_artifacts": 3,
                "top_k_neighbors": 3,
            },
        },
    )

    assert config.experiment.num_iterations == 3
    assert config.experiment.seed == 7


def test_load_config_requires_budget_settings_from_toml(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.toml").write_text(
        """
        [models]
        planner = "test-planner"
        executor = "test-executor"
        mediator = "test-mediator"
        judge = "test-judge"

        [experiment]
        num_iterations = 2
        coevo_interval = 2
        advisor_buffer_max = 2
        seed = 42
        condition_name = "learned_mediator"

        [experiment.skill_updates]
        executor = true
        planner = true
        mediator = true

        [diffusion]
        enabled = false
        policy = "none"
        max_artifacts = 3
        top_k_neighbors = 3
        """
    )

    with pytest.raises(ConfigLoadError, match="budgets.max_same_task_prior_tokens"):
        load_config(config_dir)


def test_load_config_requires_diffusion_policy_from_toml_or_overrides(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.toml").write_text(
        """
        [models]
        planner = "test-planner"
        executor = "test-executor"
        mediator = "test-mediator"
        judge = "test-judge"

        [budgets]
        max_skill_tokens = 4000
        max_same_task_prior_tokens = 300
        max_transfer_context_tokens = 900
        trace_excerpt_tokens = 6000
        historical_summary_tokens = 3000
        mediator_report_tokens = 4000
        planner_context_tokens = 24000
        skill_update_diff_tokens = 6000
        mediator_prompt_tokens = 16000
        advisor_prompt_tokens = 12000
        reflector_prompt_tokens = 16000
        judge_prompt_tokens = 16000
        planner_completion_tokens = 4096
        mediator_completion_tokens = 2048
        advisor_completion_tokens = 512
        reflector_completion_tokens = 4096
        judge_completion_tokens = 2048

        [experiment]
        num_iterations = 2
        coevo_interval = 2
        advisor_buffer_max = 2
        seed = 42
        condition_name = "learned_mediator"

        [experiment.skill_updates]
        executor = true
        planner = true
        mediator = true

        [diffusion]
        enabled = false
        max_artifacts = 3
        top_k_neighbors = 3
        """
    )

    with pytest.raises(ConfigLoadError, match="diffusion.policy"):
        load_config(config_dir)

    config = load_config(
        config_dir,
        overrides={"diffusion": {"policy": "none"}},
    )

    assert config.diffusion.policy == "none"


def test_load_config_requires_diffusion_max_artifacts_from_toml_or_overrides(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.toml").write_text(
        """
        [models]
        planner = "test-planner"
        executor = "test-executor"
        mediator = "test-mediator"
        judge = "test-judge"

        [budgets]
        max_skill_tokens = 4000
        max_same_task_prior_tokens = 300
        max_transfer_context_tokens = 900
        trace_excerpt_tokens = 6000
        historical_summary_tokens = 3000
        mediator_report_tokens = 4000
        planner_context_tokens = 24000
        skill_update_diff_tokens = 6000
        mediator_prompt_tokens = 16000
        advisor_prompt_tokens = 12000
        reflector_prompt_tokens = 16000
        judge_prompt_tokens = 16000
        planner_completion_tokens = 4096
        mediator_completion_tokens = 2048
        advisor_completion_tokens = 512
        reflector_completion_tokens = 4096
        judge_completion_tokens = 2048

        [experiment]
        num_iterations = 2
        coevo_interval = 2
        advisor_buffer_max = 2
        seed = 42
        condition_name = "learned_mediator"

        [experiment.skill_updates]
        executor = true
        planner = true
        mediator = true

        [diffusion]
        enabled = false
        policy = "none"
        top_k_neighbors = 3
        """
    )

    with pytest.raises(ConfigLoadError, match="diffusion.max_artifacts"):
        load_config(config_dir)

    config = load_config(
        config_dir,
        overrides={"diffusion": {"max_artifacts": 5}},
    )

    assert config.diffusion.max_artifacts == 5


def test_load_config_requires_diffusion_top_k_neighbors_from_toml_or_overrides(
    tmp_path,
):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.toml").write_text(
        """
        [models]
        planner = "test-planner"
        executor = "test-executor"
        mediator = "test-mediator"
        judge = "test-judge"

        [budgets]
        max_skill_tokens = 4000
        max_same_task_prior_tokens = 300
        max_transfer_context_tokens = 900
        trace_excerpt_tokens = 6000
        historical_summary_tokens = 3000
        mediator_report_tokens = 4000
        planner_context_tokens = 24000
        skill_update_diff_tokens = 6000
        mediator_prompt_tokens = 16000
        advisor_prompt_tokens = 12000
        reflector_prompt_tokens = 16000
        judge_prompt_tokens = 16000
        planner_completion_tokens = 4096
        mediator_completion_tokens = 2048
        advisor_completion_tokens = 512
        reflector_completion_tokens = 4096
        judge_completion_tokens = 2048

        [experiment]
        num_iterations = 2
        coevo_interval = 2
        advisor_buffer_max = 2
        seed = 42
        condition_name = "learned_mediator"

        [experiment.skill_updates]
        executor = true
        planner = true
        mediator = true

        [diffusion]
        enabled = false
        policy = "none"
        max_artifacts = 3
        """
    )

    with pytest.raises(ConfigLoadError, match="diffusion.top_k_neighbors"):
        load_config(config_dir)

    config = load_config(
        config_dir,
        overrides={"diffusion": {"top_k_neighbors": 2}},
    )

    assert config.diffusion.top_k_neighbors == 2


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"enabled": False, "policy": "capped_broadcast"},
            "diffusion.enabled=false requires diffusion.policy='none'",
        ),
        (
            {"enabled": False, "graph": "task_similarity"},
            "diffusion.enabled=false requires diffusion.graph='none'",
        ),
        (
            {"enabled": True, "policy": "none"},
            "diffusion.enabled=true requires diffusion.policy",
        ),
        (
            {
                "enabled": True,
                "policy": "top_k_similarity",
                "graph": "none",
            },
            "top_k_similarity.*diffusion.graph",
        ),
        (
            {
                "enabled": True,
                "policy": "random_k",
                "graph": "task_similarity",
            },
            "random_k.*diffusion.graph='none'",
        ),
        (
            {
                "enabled": True,
                "policy": "langchain_graph",
                "graph": "task_similarity",
            },
            "langchain_graph.*diffusion.graph='none'",
        ),
    ],
)
def test_load_config_rejects_invalid_diffusion_combinations(
    tmp_path,
    overrides,
    match,
):
    config_dir = tmp_path / "config"
    _write_minimal_config(config_dir)

    with pytest.raises(ConfigLoadError, match=match):
        load_config(config_dir, overrides={"diffusion": overrides})


@pytest.mark.parametrize(
    "overrides",
    [
        {"enabled": False, "policy": "none", "graph": "none"},
        {"enabled": True, "policy": "capped_broadcast", "graph": "none"},
        {"enabled": True, "policy": "random_k", "graph": "none"},
        {
            "enabled": True,
            "policy": "top_k_similarity",
            "graph": "task_similarity",
        },
        {
            "enabled": True,
            "policy": "top_k_similarity",
            "graph": "precomputed_similarity",
        },
        {"enabled": True, "policy": "langchain_graph", "graph": "none"},
    ],
)
def test_load_config_accepts_valid_diffusion_combinations(tmp_path, overrides):
    config_dir = tmp_path / "config"
    _write_minimal_config(config_dir)

    config = load_config(config_dir, overrides={"diffusion": overrides})

    assert config.diffusion.enabled is overrides["enabled"]
    assert config.diffusion.policy == overrides["policy"]
    assert config.diffusion.graph == overrides["graph"]


def test_run_command_validates_design_before_harbor(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    _write_minimal_config(config_dir)

    def fail_if_called(config):
        raise AssertionError("harbor check should happen after design validation")

    monkeypatch.setattr(run_module, "ensure_harbor_available", fail_if_called)

    with pytest.raises(typer.BadParameter, match="no_feedback"):
        run_module.run(
            family="family-a",
            iterations=1,
            seed=42,
            condition="no_feedback",
            skill_updates="executor",
            config_dir=config_dir,
        )


def test_run_command_requires_family_before_harbor(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    _write_minimal_config(config_dir)

    def fail_if_called(config):
        raise AssertionError("harbor check should happen after task validation")

    monkeypatch.setattr(run_module, "ensure_harbor_available", fail_if_called)

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--iterations",
            "1",
            "--seed",
            "42",
            "--condition",
            "learned_mediator",
            "--skill-updates",
            "all",
            "--config-dir",
            str(config_dir),
        ],
    )

    assert result.exit_code != 0
    assert "--family" in result.output


def test_run_command_uses_toml_defaults_when_cli_overrides_are_absent(
    monkeypatch,
    tmp_path,
):
    config_dir = tmp_path / "config"
    _write_minimal_config(config_dir)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        run_module,
        "prepare_llm_credentials_or_exit",
        lambda config: config,
    )
    monkeypatch.setattr(run_module, "ensure_harbor_available", lambda config: None)
    monkeypatch.setattr(
        run_module,
        "resolve_task_selection",
        lambda **kwargs: SimpleNamespace(
            task_ids=["family-a/task-one"],
            family="family-a",
            families=("family-a",),
            split=kwargs.get("split"),
        ),
    )

    def capture_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_module, "run_skillflow_experiment", capture_run)

    run_module.run(
        family=["family-a"],
        config_dir=config_dir,
    )

    assert captured["iterations"] == 2
    assert captured["seed"] == 42
    assert captured["condition_name"] == "learned_mediator"


def test_run_command_uses_task_manifest_without_resampling(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    _write_minimal_config(config_dir)
    manifest = tmp_path / "stream.json"
    manifest.write_text("{}\n")
    captured: dict[str, object] = {}
    selection = run_module.TaskSelection(
        task_ids=["family-a/task-one", "family-a/task-one"],
        families=("family-a",),
        split="test",
        task_stream_seed=123,
    )

    monkeypatch.setattr(run_module, "build_benchmark_repo", lambda *args: object())
    monkeypatch.setattr(
        run_module,
        "load_task_manifest_selection",
        lambda **kwargs: selection,
    )
    monkeypatch.setattr(
        run_module,
        "resolve_task_selection",
        lambda **kwargs: pytest.fail("manifest runs must not resample tasks"),
    )
    monkeypatch.setattr(
        run_module,
        "run_skillflow_experiment",
        lambda **kwargs: captured.update(kwargs),
    )

    run_module.run(task_manifest=manifest, config_dir=config_dir)

    assert captured["selection"] is selection


def test_run_command_forwards_repeated_families_and_split(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    _write_minimal_config(config_dir)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        run_module,
        "prepare_llm_credentials_or_exit",
        lambda config: config,
    )
    monkeypatch.setattr(run_module, "ensure_harbor_available", lambda config: None)
    monkeypatch.setattr(run_module, "build_benchmark_repo", lambda *args: object())

    def capture_selection(**kwargs):
        captured["family"] = kwargs["family"]
        captured["split"] = kwargs["split"]
        return SimpleNamespace(
            task_ids=["family-a/task-one"],
            family="family-a,family-b",
            families=("family-a", "family-b"),
            split=kwargs["split"],
        )

    monkeypatch.setattr(run_module, "resolve_task_selection", capture_selection)
    monkeypatch.setattr(
        run_module,
        "run_skillflow_experiment",
        lambda **kwargs: captured.update({"run_called": True}),
    )

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--family",
            "family-a",
            "--family",
            "family-b",
            "--split",
            "validation",
            "--config-dir",
            str(config_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["family"] == ["family-a", "family-b"]
    assert captured["split"] == "validation"
    assert captured["run_called"] is True


def test_run_skillflow_experiment_persists_selection_split(monkeypatch, tmp_path):
    config = _config()
    config.paths.data_dir = "data"
    config.paths.skills_dir = "skills"
    (tmp_path / "skills" / "executor").mkdir(parents=True)
    (tmp_path / "skills" / "executor" / "SKILL.md").write_text("# executor\n")
    monkeypatch.setattr(run_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        run_module,
        "prepare_llm_credentials_or_exit",
        lambda config: config,
    )
    monkeypatch.setattr(run_module, "ensure_harbor_available", lambda config: None)
    monkeypatch.setattr(run_module, "build_benchmark_repo", lambda *args: object())
    monkeypatch.setattr(
        run_module,
        "materialize_task_graph_for_diffusion",
        lambda **kwargs: None,
    )
    runtime = SimpleNamespace(
        experiment_dir=tmp_path / "runtime",
        orchestrator=SimpleNamespace(history_store=object()),
    )
    monkeypatch.setattr(
        run_module,
        "build_experiment_runtime",
        lambda **kwargs: runtime,
    )
    monkeypatch.setattr(run_module, "run_experiment_or_exit", lambda *args: [])
    monkeypatch.setattr(
        run_module, "write_and_print_result_summary", lambda **kwargs: None
    )
    monkeypatch.setattr(
        run_module, "annotate_judge_rewards_or_exit", lambda **kwargs: None
    )
    selection = run_module.TaskSelection(
        task_ids=["family-a/task-one"],
        families=("family-a",),
        split="validation",
        task_stream_seed=9001,
    )

    run_module.run_skillflow_experiment(
        config=config,
        selection=selection,
        iterations=1,
        seed=42,
        condition_name="learned_mediator",
        run_id="split-persist",
    )

    config_paths = list((tmp_path / "data" / "experiments").glob("*/config.toml"))
    assert config.experiment.benchmark_selection.split == "validation"
    assert config.experiment.benchmark_selection.task_stream_seed == 9001
    assert len(config_paths) == 1
    saved = tomllib.loads(config_paths[0].read_text())
    assert saved["experiment"]["benchmark_selection"]["split"] == "validation"
    assert saved["experiment"]["benchmark_selection"]["task_stream_seed"] == 9001


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
        build_experiment(
            project_root=tmp_path,
            config=config,
            seed=42,
            condition_name="no_feedback",
            experiment_dir=experiment_dir,
        )

    assert not experiment_dir.exists()


def test_build_experiment_uses_experiment_local_skill_store_by_default(tmp_path):
    _write_skill(tmp_path, "executor", "# Executor\n")
    _write_skill(tmp_path, "planner", "# Planner\n")
    _write_skill(tmp_path, "mediator", "# Mediator\n")
    config = _config()
    config.paths.skills_dir = "skills"
    experiment_dir = tmp_path / "experiment"

    runtime = build_experiment(
        project_root=tmp_path,
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


def test_create_matrix_dir_accepts_run_id_suffix(tmp_path):
    matrix_dir = create_matrix_dir(
        project_root=tmp_path,
        seed=42,
        data_dir="data",
        run_id="csm-matrix-skill-none-diffusion-none",
    )

    assert matrix_dir.parent == tmp_path / "data" / "experiments"
    assert matrix_dir.name.endswith("-csm-matrix-skill-none-diffusion-none")
    assert matrix_dir.is_dir()


def test_matrix_runtimes_use_isolated_skill_copies_and_shared_config(tmp_path):
    _write_skill(tmp_path, "executor", "# Executor\n")
    _write_skill(tmp_path, "planner", "# Planner\n")
    _write_skill(tmp_path, "mediator", "# Mediator\n")
    config = _config()
    config.experiment.num_iterations = 8
    config.paths.skills_dir = "skills"
    config.paths.data_dir = "data"
    config.paths.benchmarks_dir = "benchmarks/skillflow"
    matrix_dir = tmp_path / "data" / "experiments" / "matrix"
    benchmark_repo = SkillFlowRepository(
        root_dir=tmp_path / "benchmarks" / "skillflow",
        task_dirs=["tasks"],
    )

    rows = build_matrix_runtimes(
        project_root=tmp_path,
        base_config=config,
        seed=123,
        matrix_dir=matrix_dir,
        benchmark_repo=benchmark_repo,
    )

    assert [row.preset_name for row in rows] == BASELINE_PRESET_NAMES
    row_skill_dirs = []
    for row in rows:
        preset = BASELINE_PRESETS_BY_NAME[row.preset_name]
        row_config = row.runtime.orchestrator.config
        skill_dir = row.runtime.orchestrator.skill_store._skills_dir
        benchmark_repo = row.runtime.orchestrator.benchmark_repo
        row_skill_dirs.append(skill_dir)
        assert skill_dir == matrix_dir / row.preset_name / "skills"
        assert (skill_dir / "executor" / "SKILL.md").read_text() == "# Executor\n"
        assert benchmark_repo.root_dir == tmp_path / "benchmarks" / "skillflow"
        assert (
            benchmark_repo.default_local_cache_dir()
            == tmp_path / "benchmarks" / "skillflow" / "tasks"
        )
        assert row_config.experiment.seed == 123
        assert row_config.experiment.num_iterations == 8
        assert row_config.experiment.baseline_preset == row.preset_name
        assert row_config.models.model_dump() == config.models.model_dump()
        assert row_config.budgets.model_dump() == config.budgets.model_dump()
        assert row_config.diffusion.enabled == preset.diffusion_enabled
        assert row_config.diffusion.policy == preset.diffusion_policy
        assert row_config.diffusion.graph == preset.diffusion_graph

        saved = tomllib.loads((row.runtime.experiment_dir / "config.toml").read_text())
        assert saved["experiment"]["baseline_preset"] == row.preset_name
        assert saved["experiment"]["condition_name"] == preset.condition_name
        assert saved["experiment"]["seed"] == 123
        assert saved["experiment"]["num_iterations"] == 8
        assert saved["experiment"]["skill_updates"] == preset.skill_updates.model_dump()
        assert saved["diffusion"]["enabled"] == preset.diffusion_enabled
        assert saved["diffusion"]["policy"] == preset.diffusion_policy
        assert saved["diffusion"]["graph"] == preset.diffusion_graph

    assert len(set(row_skill_dirs)) == len(row_skill_dirs)
    (row_skill_dirs[0] / "executor" / "SKILL.md").write_text("# Changed\n")
    assert (tmp_path / "skills" / "executor" / "SKILL.md").read_text() == "# Executor\n"
    assert (row_skill_dirs[1] / "executor" / "SKILL.md").read_text() == "# Executor\n"


def test_build_benchmark_repo_uses_configured_harbor_base_images(tmp_path):
    config = _config()
    config.paths.benchmarks_dir = "benchmarks/skillflow"
    config.executor_runtime.harbor_base_image = "local/harbor-base:test"
    config.executor_runtime.legacy_harbor_base_images = ["legacy/harbor-base:test"]

    repo = build_benchmark_repo(tmp_path, config)

    assert repo.harbor_base_image == "local/harbor-base:test"
    assert repo.legacy_harbor_base_images == ("legacy/harbor-base:test",)


def test_matrix_runtimes_can_build_only_selected_presets(tmp_path):
    _write_skill(tmp_path, "executor", "# Executor\n")
    _write_skill(tmp_path, "planner", "# Planner\n")
    _write_skill(tmp_path, "mediator", "# Mediator\n")
    config = _config()
    config.paths.skills_dir = "skills"
    config.paths.data_dir = "data"
    config.paths.benchmarks_dir = "benchmarks/skillflow"
    matrix_dir = tmp_path / "data" / "experiments" / "matrix"
    benchmark_repo = SkillFlowRepository(
        root_dir=tmp_path / "benchmarks" / "skillflow",
        task_dirs=["tasks"],
    )
    selected_preset = BASELINE_PRESET_NAMES[3]

    rows = build_matrix_runtimes(
        project_root=tmp_path,
        base_config=config,
        seed=123,
        matrix_dir=matrix_dir,
        benchmark_repo=benchmark_repo,
        preset_names=[selected_preset],
    )

    assert [row.preset_name for row in rows] == [selected_preset]
    assert rows[0].runtime.experiment_dir == matrix_dir / selected_preset
    assert (matrix_dir / selected_preset / "config.toml").is_file()
    assert not (matrix_dir / BASELINE_PRESET_NAMES[0]).exists()


def test_matrix_runtimes_can_flatten_single_selected_preset(tmp_path):
    _write_skill(tmp_path, "executor", "# Executor\n")
    _write_skill(tmp_path, "planner", "# Planner\n")
    _write_skill(tmp_path, "mediator", "# Mediator\n")
    config = _config()
    config.paths.skills_dir = "skills"
    config.paths.data_dir = "data"
    config.paths.benchmarks_dir = "benchmarks/skillflow"
    matrix_dir = tmp_path / "data" / "experiments" / "matrix"
    benchmark_repo = SkillFlowRepository(
        root_dir=tmp_path / "benchmarks" / "skillflow",
        task_dirs=["tasks"],
    )
    selected_preset = BASELINE_PRESET_NAMES[3]

    rows = build_matrix_runtimes(
        project_root=tmp_path,
        base_config=config,
        seed=123,
        matrix_dir=matrix_dir,
        benchmark_repo=benchmark_repo,
        preset_names=[selected_preset],
        flatten_single_row=True,
    )

    assert [row.preset_name for row in rows] == [selected_preset]
    assert rows[0].runtime.experiment_dir == matrix_dir
    assert (matrix_dir / "config.toml").is_file()
    assert (matrix_dir / "skills" / "executor" / "SKILL.md").read_text() == (
        "# Executor\n"
    )
    assert not (matrix_dir / selected_preset).exists()

    saved = tomllib.loads((matrix_dir / "config.toml").read_text())
    assert saved["experiment"]["baseline_preset"] == selected_preset


def test_matrix_runtimes_reject_flattened_multi_row_matrix(tmp_path):
    _write_skill(tmp_path, "executor", "# Executor\n")
    _write_skill(tmp_path, "planner", "# Planner\n")
    _write_skill(tmp_path, "mediator", "# Mediator\n")
    config = _config()
    config.paths.skills_dir = "skills"
    matrix_dir = tmp_path / "data" / "experiments" / "matrix"
    benchmark_repo = SkillFlowRepository(
        root_dir=tmp_path / "benchmarks" / "skillflow",
        task_dirs=["tasks"],
    )

    with pytest.raises(ValueError, match="requires exactly one matrix preset"):
        build_matrix_runtimes(
            project_root=tmp_path,
            base_config=config,
            seed=123,
            matrix_dir=matrix_dir,
            benchmark_repo=benchmark_repo,
            preset_names=BASELINE_PRESET_NAMES[:2],
            flatten_single_row=True,
        )


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

    def __init__(self, token_events: list[TokenBudgetEvent] | None = None) -> None:
        self._token_events = list(token_events or [])

    def drain_token_events(self) -> list[TokenBudgetEvent]:
        events = list(self._token_events)
        self._token_events.clear()
        return events


class _Planner:
    llm_client = _LLM()


class _Mediator:
    llm_client = _LLM()

    def load_protocol(self, content: str) -> None:
        raise AssertionError("no reflection result should be loaded in this test")


class _Advisor:
    llm_client = _LLM()


class _AgentWithLLM:
    def __init__(self, llm_client: _LLM) -> None:
        self.llm_client = llm_client


class _SkillStore:
    def skill_hashes(self) -> dict[str, str]:
        return {}


class _PairableHistoryStore:
    def tagged_task_counts(self, agent_role: str) -> dict[str, int]:
        return {"task-A": 2}


@pytest.mark.asyncio
async def test_coevolve_skips_metric_when_all_skill_updates_disabled(
    caplog,
    monkeypatch,
):
    def _raise_if_reflector_is_created(*args, **kwargs) -> None:
        raise AssertionError("reflector should not run when all updates are disabled")

    import mediated_coevo.experiment.orchestrator as orchestrator_module

    monkeypatch.setattr(
        orchestrator_module,
        "Reflector",
        _raise_if_reflector_is_created,
    )

    token_event = TokenBudgetEvent(
        label="compactor.context",
        model="test-model",
        prompt_tokens=10,
        completion_tokens=3,
        total_tokens=13,
    )
    planner_llm = _LLM([token_event])

    orch = Orchestrator.__new__(Orchestrator)
    orch.config = _config()
    orch.config.experiment.skill_updates = SkillUpdateConfig(
        executor=False,
        planner=False,
        mediator=False,
    )
    orch.planner = _AgentWithLLM(planner_llm)
    orch.mediator = _AgentWithLLM(_LLM())
    orch.skill_advisor = _AgentWithLLM(_LLM())
    orch.judge_llm_client = None
    caplog.set_level("INFO")

    assert await orch._coevolve(4, "no_feedback") is None
    assert planner_llm.drain_token_events() == []
    assert "all skill updates are disabled" in caplog.text
    assert "discarded 1 pending token telemetry events" in caplog.text


@pytest.mark.asyncio
async def test_planner_and_mediator_reflection_are_independently_gated(monkeypatch):
    calls: list[str] = []

    class _RecordingReflector:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def reflect(self, agent_role, *args, **kwargs):
            calls.append(agent_role)
            return None

        async def draft_reflection(self, *, agent_role, **kwargs):
            calls.append(agent_role)
            return None

    import mediated_coevo.experiment.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "Reflector", _RecordingReflector)

    orch = Orchestrator.__new__(Orchestrator)
    orch.config = _config()
    orch.history_store = _PairableHistoryStore()
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


def test_reflection_defers_until_same_task_history_is_pairable(caplog):
    class _SparseHistoryStore:
        def tagged_task_counts(self, agent_role: str) -> dict[str, int]:
            return {"task-A": 1, "task-B": 1}

    orch = Orchestrator.__new__(Orchestrator)
    orch.history_store = _SparseHistoryStore()
    caplog.set_level("INFO")

    assert orch._reflection_has_pairable_history("mediator") is False
    assert "Mediator reflection deferred" in caplog.text
    assert "need at least two tagged same-task history entries" in caplog.text


def test_metrics_rows_include_baseline_and_skill_update_policy():
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = _config()
    orch.config.experiment.baseline_preset = "skill_all_diffusion_none"
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

    assert dumped["baseline_preset"] == "skill_all_diffusion_none"
    assert dumped["skill_update_policy"] == {
        "executor": False,
        "planner": True,
        "mediator": False,
    }
