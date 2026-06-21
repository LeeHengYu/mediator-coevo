from __future__ import annotations

from mediated_coevo.cloud.vm import GCPVMConfig, build_remote_harbor_script


def test_remote_harbor_script_uses_prebuilt_images() -> None:
    script = build_remote_harbor_script(
        config=GCPVMConfig(
            project_id="demo-project",
            region="us-central1",
            zone="us-central1-a",
            vm_name="harbor-vm",
            remote_dir="/tmp/mediator-coevo",
            openrouter_secret="openrouter-api-key",
        ),
        remote_run_dir="/tmp/mediator-coevo/run-1",
        model="provider/model",
        harbor_timeout_sec=120.0,
        agent_setup_timeout_multiplier=None,
    )

    assert "harbor run" in script
    assert "harbor run -p $TASK_DIR -a hermes -m provider/model -o $JOBS_DIR" in script
    assert "unset OPENAI_API_KEY" in script
    assert 'timeout --kill-after=30s "${HARBOR_TIMEOUT_SEC}s" harbor run' in script


def test_remote_harbor_script_uses_configured_agent_and_env() -> None:
    script = build_remote_harbor_script(
        config=GCPVMConfig(
            project_id="demo-project",
            region="us-central1",
            zone="us-central1-a",
            vm_name="harbor-vm",
            remote_dir="/tmp/mediator-coevo",
            openrouter_secret="openrouter-api-key",
        ),
        remote_run_dir="/tmp/mediator-coevo/run-1",
        model="openai/gpt-5.5",
        harbor_timeout_sec=120.0,
        agent_name="claude-code",
        agent_env={
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "${OPENROUTER_API_KEY}",
            "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
        },
        agent_setup_timeout_multiplier=None,
    )

    assert "harbor run -p $TASK_DIR -a claude-code -m openai/gpt-5.5" in script
    assert "--agent-env ANTHROPIC_API_KEY=" in script
    assert "--agent-env 'ANTHROPIC_AUTH_TOKEN=${OPENROUTER_API_KEY}'" in script
    assert "--agent-env ANTHROPIC_BASE_URL=https://openrouter.ai/api" in script
    assert "--agent-env ANTHROPIC_MODEL=openai/gpt-5.5" in script
    assert "--agent-env ANTHROPIC_DEFAULT_SONNET_MODEL=openai/gpt-5.5" in script
    assert "--agent-env CLAUDE_CODE_SUBAGENT_MODEL=openai/gpt-5.5" in script
