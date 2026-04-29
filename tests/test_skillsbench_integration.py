"""Opt-in integration test: drive a real Harbor run on a vendored task.

Skipped automatically when the harbor CLI is not on PATH so the unit
suite remains hermetic. This test is **strict**: the trace must come
back ``status="ok"`` with a real reward in [0,1]. A silent env_failure
no longer passes — that path is already covered by the unit tests.

Run explicitly:

    uv run pytest tests/test_skillsbench_integration.py -m integration -v -s

Use ``uv`` as the test entrypoint; it manages and reuses the local
``.venv`` for this project.

Set ``MEDIATED_COEVO_INTEGRATION_MODEL`` (default: google/gemini-2.5-flash) to
override the executor model. Skip with ``MEDIATED_COEVO_SKIP_INTEGRATION=1``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from mediated_coevo.benchmarks.skillsbench import (
    HarborRunner,
    SkillsBenchRepository,
    parse_execution_trace,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_ROOT = PROJECT_ROOT / "benchmarks" / "skillsbench"
TASK_ID = "fix-build-google-auto"

INTEGRATION_MODEL = os.environ.get(
    "MEDIATED_COEVO_INTEGRATION_MODEL", "google/gemini-2.5-flash"
)
SKIP_FLAG = os.environ.get("MEDIATED_COEVO_SKIP_INTEGRATION") == "1"


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("harbor") is None,
        reason="harbor CLI not on PATH",
    ),
    pytest.mark.skipif(
        not (BENCH_ROOT / "tasks" / TASK_ID).exists(),
        reason=f"vendored task {TASK_ID} not present",
    ),
    pytest.mark.skipif(
        SKIP_FLAG,
        reason="MEDIATED_COEVO_SKIP_INTEGRATION=1",
    ),
]


@pytest.mark.asyncio
async def test_harbor_run_produces_classified_trace(tmp_path, capsys):
    repo = SkillsBenchRepository(root_dir=BENCH_ROOT, task_dirs=["tasks"])
    task = repo.resolve(TASK_ID)

    run_dir = repo.prepare_run_workspace(
        task=task,
        destination_root=tmp_path / "ws",
        planner_instruction=task.instruction,
        injected_skill_text=None,
        injected_skill_name="executor-evolved",
    )

    runner = HarborRunner(
        agent_name="gemini-cli",
        jobs_dir=tmp_path / "jobs",
        timeout_sec=900.0,
    )
    result = await runner.run(task_dir=run_dir, model=INTEGRATION_MODEL)

    trace = parse_execution_trace(
        run_result=result,
        task_id=TASK_ID,
        iteration=0,
        duration_sec=0.0,
    )

    # Print diagnostics so a failure is debuggable without re-running.
    with capsys.disabled():
        print(
            f"\n[integration] status={trace.status} "
            f"error_kind={trace.error_kind} reward={trace.reward} "
            f"returncode={result.returncode} trial_dir={result.trial_dir}"
        )
        if trace.error_detail:
            print(f"[integration] error_detail={trace.error_detail!r}")
        if trace.stderr:
            print(f"[integration] stderr (head)={trace.stderr[:500]}")

    # Strict: this test exists to validate the happy-path end-to-end. An
    # env_failure here means the test environment is broken (no docker, no
    # creds, etc.), not that the parser is doing its job — that's the unit
    # tests. Either fix the env or skip with MEDIATED_COEVO_SKIP_INTEGRATION=1.
    assert trace.status == "ok", (
        f"integration run did not produce status=ok "
        f"(status={trace.status} error_kind={trace.error_kind}). "
        "If this is an env issue, set MEDIATED_COEVO_SKIP_INTEGRATION=1."
    )
    assert trace.reward is not None
    assert 0.0 <= trace.reward <= 1.0
    assert (result.trial_dir / "result.json").exists()
