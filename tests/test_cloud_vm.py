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
