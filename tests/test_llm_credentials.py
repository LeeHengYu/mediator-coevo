from __future__ import annotations

import tomllib

import pytest
from typer.testing import CliRunner

from mediated_coevo import main as main_module
from mediated_coevo.core.config import Config
from mediated_coevo.llm.client import (
    LLMCredentialError,
    normalize_openrouter_model,
    normalize_openrouter_models,
    validate_openrouter_credentials,
)
from mediated_coevo.main import ExperimentFactory, app


def test_normalize_openrouter_model_adds_missing_prefix():
    assert (
        normalize_openrouter_model("anthropic/claude-sonnet-4.6")
        == "openrouter/anthropic/claude-sonnet-4.6"
    )


def test_normalize_openrouter_model_preserves_existing_prefix():
    model = "openrouter/google/gemini-3-flash-preview"

    assert normalize_openrouter_model(model) == model


def test_normalize_openrouter_models_updates_all_agent_models():
    config = Config(
        models={
            "planner": "anthropic/claude-sonnet-4.6",
            "executor": "openrouter/google/gemini-3-flash-preview",
            "mediator": "openai/gpt-5.5",
        }
    )

    normalize_openrouter_models(config)

    assert config.models.planner == "openrouter/anthropic/claude-sonnet-4.6"
    assert config.models.executor == "openrouter/google/gemini-3-flash-preview"
    assert config.models.mediator == "openrouter/openai/gpt-5.5"


def test_validate_openrouter_credentials_accepts_nonblank_key():
    validate_openrouter_credentials({"OPENROUTER_API_KEY": "sk-test"})


def test_validate_openrouter_credentials_fails_for_missing_or_blank_key():
    for environ in ({}, {"OPENROUTER_API_KEY": "   "}):
        with pytest.raises(LLMCredentialError) as exc_info:
            validate_openrouter_credentials(environ)
        message = str(exc_info.value)
        assert "OPENROUTER_API_KEY" in message
        assert "ANTHROPIC_API_KEY" not in message
        assert "OPENAI_API_KEY" not in message
        assert "GOOGLE_API_KEY" not in message


def test_run_cli_fails_fast_when_openrouter_key_is_missing(monkeypatch):
    def fail_harbor_check(_: Config) -> None:
        raise AssertionError("Harbor should not be checked before credentials")

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(main_module, "_ensure_harbor_available", fail_harbor_check)

    result = CliRunner().invoke(app, ["run", "--tasks", "demo"])

    assert result.exit_code == 1
    assert "OPENROUTER_API_KEY" in result.output


def test_matrix_cli_fails_fast_when_openrouter_key_is_missing(monkeypatch):
    def fail_harbor_check(_: Config) -> None:
        raise AssertionError("Harbor should not be checked before credentials")

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(main_module, "_ensure_harbor_available", fail_harbor_check)

    result = CliRunner().invoke(app, ["matrix", "--tasks", "demo"])

    assert result.exit_code == 1
    assert "OPENROUTER_API_KEY" in result.output


def test_swebench_experiment_fails_before_directory_creation_without_key(
    monkeypatch,
    tmp_path,
):
    def fail_prepare_experiment_root(**_: object) -> None:
        raise AssertionError("experiment directory should not be created")

    config = Config(
        models={
            "planner": "anthropic/claude-sonnet-4.6",
            "executor": "google/gemini-3-flash-preview",
            "mediator": "openai/gpt-5.5",
        }
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(main_module, "load_config", lambda _: config)
    monkeypatch.setattr(
        main_module,
        "_prepare_swebench_experiment_root",
        fail_prepare_experiment_root,
    )

    with pytest.raises(main_module.typer.Exit) as exc_info:
        main_module._run_swebench_experiment(
            evolve_instance_ids=["django__django-11910"],
            eval_instance_ids=["django__django-11099"],
            dataset_name="SWE-bench/SWE-bench_Lite",
            split="test",
            iterations=1,
            seed=7,
            condition_name="learned_mediator",
            skill_updates=config.experiment.skill_updates,
            config_dir=tmp_path / "config",
            timeout=30,
            max_workers=1,
            run_id="swebench-test",
        )
    assert exc_info.value.exit_code == 1


def test_normalized_models_are_persisted_in_run_config(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    for skill_name in ("executor", "planner", "mediator"):
        skill_dir = tmp_path / "skills" / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"# {skill_name}\n")

    config = Config(
        models={
            "planner": "anthropic/claude-sonnet-4.6",
            "executor": "google/gemini-3-flash-preview",
            "mediator": "openrouter/openai/gpt-5.5",
        }
    )
    config.paths.skills_dir = "skills"
    main_module._prepare_llm_credentials_or_exit(config)

    runtime = ExperimentFactory(tmp_path).build(
        config=config,
        seed=42,
        condition_name=config.experiment.condition_name,
        experiment_dir=tmp_path / "experiment",
    )

    saved = tomllib.loads((runtime.experiment_dir / "config.toml").read_text())
    assert saved["models"] == {
        "planner": "openrouter/anthropic/claude-sonnet-4.6",
        "executor": "openrouter/google/gemini-3-flash-preview",
        "mediator": "openrouter/openai/gpt-5.5",
    }
