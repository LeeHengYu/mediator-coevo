from __future__ import annotations

import tomllib
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import mediated_coevo.cli.run as run_module
from mediated_coevo.core.config import Config, ModelConfigError, ModelsConfig
from mediated_coevo.cli.experiment import prepare_llm_credentials_or_exit
from mediated_coevo.experiment.runtime_factory import build_experiment
from mediated_coevo.llm.client import LLMCredentialError, validate_openrouter_credentials
from mediated_coevo.main import app
from tests.config_helpers import budgets_config, diffusion_config, experiment_config


def test_config_normalize_models_keeps_executor_model_harbor_ready():
    config = Config(
        models={
            "planner": "anthropic/claude-sonnet-4.6",
            "executor": "openrouter/google/gemini-3-flash-preview",
            "mediator": "openai/gpt-5.5",
            "judge": "openai/gpt-5.5",
        },
        budgets=budgets_config(),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )

    config.normalize_models()

    assert config.models.planner == "openrouter/anthropic/claude-sonnet-4.6"
    assert config.models.executor == "google/gemini-3-flash-preview"
    assert config.models.mediator == "openrouter/openai/gpt-5.5"
    assert config.models.judge == "openrouter/openai/gpt-5.5"


def test_config_normalize_models_collapses_duplicate_openrouter_prefixes():
    config = Config(
        models={
            "planner": "openrouter/openrouter/anthropic/claude-sonnet-4.6",
            "executor": "openrouter/openrouter/google/gemini-3-flash-preview",
            "mediator": "openrouter/openrouter/openai/gpt-5.5",
            "judge": "openrouter/openrouter/qwen/qwen3.6-flash",
        },
        budgets=budgets_config(),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )

    config.normalize_models()

    assert config.models.planner == "openrouter/anthropic/claude-sonnet-4.6"
    assert config.models.executor == "google/gemini-3-flash-preview"
    assert config.models.mediator == "openrouter/openai/gpt-5.5"
    assert config.models.judge == "openrouter/qwen/qwen3.6-flash"


def test_config_normalize_models_rejects_blank_model_name():
    config = Config(
        models={
            "planner": "anthropic/claude-sonnet-4.6",
            "executor": "google/gemini-3-flash-preview",
            "mediator": "openai/gpt-5.5",
            "judge": "   ",
        },
        budgets=budgets_config(),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )

    with pytest.raises(ModelConfigError, match="model names must be non-empty"):
        config.normalize_models()


def test_executor_agent_defaults_to_hermes_and_allows_configured_agent_env():
    config = Config(
        models={
            "planner": "anthropic/claude-sonnet-4.6",
            "executor": "openrouter/openai/gpt-5.5",
            "mediator": "openai/gpt-5.5",
            "judge": "openai/gpt-5.5",
        },
        budgets=budgets_config(),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )

    assert config.executor_runtime.agent_name == "hermes"
    assert config.executor_runtime.agent_env == {}

    config = Config(
        models={
            "planner": "anthropic/claude-sonnet-4.6",
            "executor": "openrouter/openai/gpt-5.5",
            "mediator": "openai/gpt-5.5",
            "judge": "openai/gpt-5.5",
        },
        budgets=budgets_config(),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
        executor_runtime={
            "agent_name": "claude-code",
            "agent_env": {
                "ANTHROPIC_API_KEY": "",
                "ANTHROPIC_AUTH_TOKEN": "${OPENROUTER_API_KEY}",
                "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
            },
        },
    )

    assert config.executor_runtime.agent_name == "claude-code"
    assert config.executor_runtime.agent_env == {
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_AUTH_TOKEN": "${OPENROUTER_API_KEY}",
        "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
    }


@pytest.mark.parametrize("model", ["openrouter", "openrouter/", "openai/"])
def test_config_normalize_models_rejects_prefix_only_model_names(model: str):
    config = Config(
        models=ModelsConfig(
            planner="anthropic/claude-sonnet-4.6",
            executor="google/gemini-3-flash-preview",
            mediator="openai/gpt-5.5",
            judge=model,
        ),
        budgets=budgets_config(),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )

    with pytest.raises(ModelConfigError, match="provider and model"):
        config.normalize_models()


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
    monkeypatch.setattr(run_module, "ensure_harbor_available", fail_harbor_check)
    monkeypatch.setattr(
        run_module,
        "resolve_task_selection",
        lambda **kwargs: SimpleNamespace(
            task_ids=["demo/task-one"],
            family="demo",
        ),
    )

    result = CliRunner().invoke(
        app,
        ["run", "--family", "demo", "--run-id", "credential-check"],
    )

    assert result.exit_code == 1


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
            "judge": "openai/gpt-5.5",
        },
        budgets=budgets_config(),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )
    config.paths.skills_dir = "skills"
    prepare_llm_credentials_or_exit(config)

    runtime = build_experiment(
        project_root=tmp_path,
        config=config,
        seed=42,
        condition_name=config.experiment.condition_name,
        experiment_dir=tmp_path / "experiment",
    )

    saved = tomllib.loads((runtime.experiment_dir / "config.toml").read_text())
    assert saved["models"] == {
        "planner": "openrouter/anthropic/claude-sonnet-4.6",
        "executor": "google/gemini-3-flash-preview",
        "mediator": "openrouter/openai/gpt-5.5",
        "judge": "openrouter/openai/gpt-5.5",
    }
