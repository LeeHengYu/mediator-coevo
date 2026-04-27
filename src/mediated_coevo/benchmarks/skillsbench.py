"""Local SkillsBench task loading and Harbor execution helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tomllib
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mediated_coevo.models.trace import ExecutionTrace, TokenUsage, TraceStatus

logger = logging.getLogger(__name__)


class HarborNotFoundError(RuntimeError):
    """Raised when the harbor CLI cannot be located on PATH or is not executable."""


class HarborTimeoutError(RuntimeError):
    """Raised when a Harbor subprocess exceeds the configured timeout."""


@dataclass(slots=True)
class SkillsBenchTask:
    """A locally vendored SkillsBench task."""

    task_id: str
    task_dir: Path
    instruction_path: Path
    instruction: str
    task_config: dict[str, Any]


@dataclass(slots=True)
class HarborRunResult:
    """Artifacts produced by one Harbor task run."""

    job_dir: Path | None
    trial_dir: Path | None
    returncode: int
    stdout: str
    stderr: str


class SkillsBenchRepository:
    """Resolve local SkillsBench-style tasks and materialize run workspaces."""

    def __init__(self, root_dir: Path, task_dirs: list[str]) -> None:
        self.root_dir = root_dir
        self.task_dirs = task_dirs

    def resolve(self, task_id: str) -> SkillsBenchTask:
        for task_dir_name in self.task_dirs:
            candidate = self.root_dir / task_dir_name / task_id
            if candidate.exists():
                return self._load_task(candidate, task_id)
        searched = [str(self.root_dir / name / task_id) for name in self.task_dirs]
        raise FileNotFoundError(
            f"Task '{task_id}' not found under local benchmark root. Searched: {searched}"
        )

    def prepare_run_workspace(
        self,
        task: SkillsBenchTask,
        destination_root: Path,
        planner_instruction: str,
        injected_skill_text: str | None,
        injected_skill_name: str,
    ) -> Path:
        run_dir = destination_root / task.task_id / f"run-{uuid.uuid4().hex[:8]}"
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(task.task_dir, run_dir)

        instruction_path = run_dir / "instruction.md"
        instruction_path.write_text(planner_instruction)

        if injected_skill_text:
            skill_dir = run_dir / "environment" / "skills" / injected_skill_name
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(injected_skill_text)

        return run_dir

    @staticmethod
    def _load_task(task_dir: Path, task_id: str) -> SkillsBenchTask:
        instruction_path = task_dir / "instruction.md"
        task_toml_path = task_dir / "task.toml"
        if not instruction_path.exists():
            raise FileNotFoundError(f"Missing instruction.md for task '{task_id}' at {task_dir}")
        if not task_toml_path.exists():
            raise FileNotFoundError(f"Missing task.toml for task '{task_id}' at {task_dir}")

        with open(task_toml_path, "rb") as f:
            task_config = tomllib.load(f)

        return SkillsBenchTask(
            task_id=task_id,
            task_dir=task_dir,
            instruction_path=instruction_path,
            instruction=instruction_path.read_text(),
            task_config=task_config,
        )


class HarborRunner:
    """Run a local SkillsBench task via Harbor and locate its artifacts."""

    def __init__(
        self,
        agent_name: str,
        jobs_dir: Path,
        timeout_sec: float = 1800.0,
    ) -> None:
        self.agent_name = agent_name
        self.jobs_dir = jobs_dir
        self.timeout_sec = timeout_sec
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._harbor_path: str | None = None

    def _resolve_harbor(self) -> str:
        if self._harbor_path is None:
            path = shutil.which("harbor")
            if path is None:
                raise HarborNotFoundError(
                    "harbor CLI not found on PATH. Install harbor, or set "
                    "executor_runtime.harbor_required=False to allow synthesized "
                    "env-failure traces in CI."
                )
            self._harbor_path = path
        return self._harbor_path

    async def run(self, task_dir: Path, model: str) -> HarborRunResult:
        harbor = self._resolve_harbor()
        before = {p.resolve() for p in self.jobs_dir.iterdir() if p.is_dir()}
        cmd = [
            harbor,
            "run",
            "-p",
            str(task_dir),
            "-a",
            self.agent_name,
            "-m",
            model,
            "-o",
            str(self.jobs_dir),
        ]
        logger.info("Running Harbor task: %s", " ".join(cmd))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise HarborNotFoundError(f"harbor CLI not executable: {e}") from e

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_sec,
            )
        except asyncio.TimeoutError as e:
            with suppress(ProcessLookupError):
                proc.kill()
            with suppress(Exception):
                await proc.wait()
            raise HarborTimeoutError(
                f"harbor run exceeded {self.timeout_sec}s timeout for {task_dir}"
            ) from e

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        # Only attribute directories created by THIS run, so stale artifacts
        # from prior runs cannot be mistaken for the current one.
        after = [p.resolve() for p in self.jobs_dir.iterdir() if p.is_dir()]
        job_dir = _latest_path([p for p in after if p not in before])
        trial_dir = _find_trial_dir(job_dir) if job_dir else None

        returncode = proc.returncode if proc.returncode is not None else -1
        if returncode != 0:
            logger.warning(
                "harbor exited with code %d (job_dir=%s, task_dir=%s)",
                returncode,
                job_dir,
                task_dir,
            )

        return HarborRunResult(
            job_dir=job_dir,
            trial_dir=trial_dir,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )


def parse_execution_trace(
    run_result: HarborRunResult,
    task_id: str,
    iteration: int,
    duration_sec: float,
) -> ExecutionTrace:
    """Convert Harbor artifacts into a classified ExecutionTrace.

    Missing or malformed artifacts surface as ``status != "ok"`` traces
    (rather than silently zero-padded ones) so the orchestrator can skip
    feeding them into mediator/skill-update channels.

    Status / error_kind matrix:
      env_failure    : missing_trial_dir | missing_result_json |
                       malformed_result_json | no_reward
      parse_error    : malformed_reward
      harbor_failed  : harbor_nonzero | harbor_nonzero_no_reward
      ok (warnings)  : reward_ctrf_mismatch | harbor_exception
    """
    base: dict[str, Any] = dict(
        task_id=task_id,
        iteration=iteration,
        duration_sec=duration_sec,
        exit_code=run_result.returncode,
        stdout=_format_harbor_stdout(run_result.stdout, ""),
        stderr=_format_harbor_stderr(run_result.stderr, None),
    )

    if run_result.trial_dir is None:
        logger.warning(
            "parse_execution_trace: no trial_dir for task=%s iter=%d (returncode=%d)",
            task_id, iteration, run_result.returncode,
        )
        return ExecutionTrace(
            **base,
            status="env_failure",
            error_kind="missing_trial_dir",
            error_detail=run_result.stderr.strip() or None,
        )

    result_path = run_result.trial_dir / "result.json"
    if not result_path.exists():
        logger.warning(
            "parse_execution_trace: missing result.json at %s (task=%s iter=%d)",
            result_path, task_id, iteration,
        )
        return ExecutionTrace(
            **base,
            status="env_failure",
            error_kind="missing_result_json",
            error_detail=f"result.json not found at {result_path}",
        )

    try:
        result_json: dict[str, Any] = json.loads(result_path.read_text())
    except json.JSONDecodeError as e:
        logger.warning(
            "parse_execution_trace: malformed result.json at %s: %s", result_path, e,
        )
        return ExecutionTrace(
            **base,
            status="env_failure",
            error_kind="malformed_result_json",
            error_detail=str(e),
        )

    ctrf_json: dict[str, Any] | None = None
    ctrf_path = run_result.trial_dir / "verifier" / "ctrf.json"
    if ctrf_path.exists():
        try:
            ctrf_json = json.loads(ctrf_path.read_text())
        except json.JSONDecodeError as e:
            logger.warning(
                "parse_execution_trace: malformed ctrf.json at %s: %s", ctrf_path, e,
            )

    # Prefer reward.txt; fall back to result.json only when absent. A
    # legitimate 0.0 in reward.txt must not be overwritten by the nested key.
    agent_summary = _read_agent_summary(run_result.trial_dir)
    exception_info = result_json.get("exception_info")
    reward_path = run_result.trial_dir / "verifier" / "reward.txt"
    test_results = _summarize_ctrf(ctrf_json)
    agent_result = _mapping_or_empty(result_json.get("agent_result"))
    token_usage = TokenUsage(
        input_tokens=_safe_int(
            agent_result.get("n_input_tokens", 0),
            field="agent_result.n_input_tokens",
            task_id=task_id, iteration=iteration,
        ),
        output_tokens=_safe_int(
            agent_result.get("n_output_tokens", 0),
            field="agent_result.n_output_tokens",
            task_id=task_id, iteration=iteration,
        ),
    )

    full_base: dict[str, Any] = dict(
        base,
        stdout=_format_harbor_stdout(run_result.stdout, agent_summary),
        stderr=_format_harbor_stderr(run_result.stderr, exception_info),
        test_results=test_results,
        token_usage=token_usage,
    )

    reward: float | None = None
    if reward_path.exists():
        reward_text = reward_path.read_text().strip()
        try:
            reward = float(reward_text)
        except ValueError:
            logger.warning(
                "parse_execution_trace: malformed reward.txt at %s (content=%r)",
                reward_path, reward_text,
            )
            return ExecutionTrace(
                **full_base,
                status="parse_error",
                error_kind="malformed_reward",
                error_detail=(
                    f"reward.txt at {reward_path} contained non-numeric value: "
                    f"{reward_text!r}"
                ),
            )
    else:
        nested = _mapping_or_empty(
            _mapping_or_empty(result_json.get("verifier_result")).get("rewards")
        ).get("reward")
        if nested is not None:
            try:
                reward = float(nested)
            except (TypeError, ValueError):
                reward = None

    if reward is None:
        logger.warning(
            "parse_execution_trace: no reward parsed for task=%s iter=%d (returncode=%d)",
            task_id, iteration, run_result.returncode,
        )
        if run_result.returncode != 0:
            missing_reward_status: TraceStatus = "harbor_failed"
            missing_reward_error_kind = "harbor_nonzero_no_reward"
        else:
            missing_reward_status = "env_failure"
            missing_reward_error_kind = "no_reward"
        return ExecutionTrace(
            **full_base,
            status=missing_reward_status,
            error_kind=missing_reward_error_kind,
            error_detail=exception_info or run_result.stderr.strip() or None,
        )

    error_kind, error_detail = _ctrf_mismatch(
        ctrf_json, reward, task_id=task_id, iteration=iteration,
    )

    if exception_info and error_kind is None:
        error_kind = "harbor_exception"
        error_detail = exception_info

    status: TraceStatus = "harbor_failed" if run_result.returncode != 0 else "ok"
    if run_result.returncode != 0 and error_kind is None:
        error_kind = "harbor_nonzero"
        error_detail = (
            run_result.stderr.strip()
            or f"harbor returncode={run_result.returncode}"
        )

    # Bump exit_code 0→1 to mark exception_info in legacy consumers.
    exit_code = 1 if run_result.returncode == 0 and exception_info else run_result.returncode

    return ExecutionTrace(
        **{**full_base, "exit_code": exit_code},
        reward=reward,
        status=status,
        error_kind=error_kind,
        error_detail=error_detail,
    )


def _ctrf_mismatch(
    ctrf_json: dict[str, Any] | None,
    reward: float,
    *,
    task_id: str,
    iteration: int,
) -> tuple[str | None, str | None]:
    """Sanity-check reward against CTRF summary; return (kind, detail)."""
    if ctrf_json is None:
        return None, None
    summary = _mapping_or_empty(
        _mapping_or_empty(ctrf_json.get("results")).get("summary")
    )
    failed = _safe_int(
        summary.get("failed", 0),
        field="ctrf.results.summary.failed",
        task_id=task_id, iteration=iteration,
    )
    passed = _safe_int(
        summary.get("passed", 0),
        field="ctrf.results.summary.passed",
        task_id=task_id, iteration=iteration,
    )
    if failed > 0 and reward >= 0.999:
        detail = f"reward={reward} but CTRF reports {failed} failed tests"
    elif passed > 0 and failed == 0 and reward <= 0.001:
        detail = f"reward={reward} but CTRF reports all {passed} tests passed"
    else:
        return None, None
    logger.warning(
        "parse_execution_trace: %s (task=%s iter=%d)", detail, task_id, iteration,
    )
    return "reward_ctrf_mismatch", detail


def _format_harbor_stdout(harbor_stdout: str, agent_summary: str) -> str:
    sections: list[str] = []
    if harbor_stdout.strip():
        sections.append("# Harbor Output\n\n" + harbor_stdout.strip())
    if agent_summary.strip():
        sections.append("# Agent Summary\n\n" + agent_summary.strip())
    return "\n\n".join(sections)


def _format_harbor_stderr(
    harbor_stderr: str,
    exception_info: dict[str, Any] | None,
) -> str:
    sections: list[str] = []
    if harbor_stderr.strip():
        sections.append(harbor_stderr.strip())
    if exception_info:
        sections.append(json.dumps(exception_info, indent=2))
    return "\n\n".join(sections)


def _read_agent_summary(trial_dir: Path) -> str:
    agent_dir = trial_dir / "agent"
    if not agent_dir.exists():
        return ""
    txt_files = sorted(agent_dir.glob("*.txt"))
    if not txt_files:
        return ""
    try:
        return txt_files[0].read_text()
    except OSError:
        return ""


def _summarize_ctrf(ctrf_json: dict[str, Any] | None) -> dict[str, Any] | None:
    if not ctrf_json:
        return None
    results = _mapping_or_empty(ctrf_json.get("results"))
    summary = _mapping_or_empty(results.get("summary"))
    tests = results.get("tests", [])
    if not isinstance(tests, list):
        tests = []
    return {
        "summary": summary,
        "failed_tests": [
            {
                "name": test.get("name"),
                "status": test.get("status"),
                "message": test.get("message"),
            }
            for test in tests
            if isinstance(test, dict) and test.get("status") != "passed"
        ],
    }


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _safe_int(value: Any, *, field: str, task_id: str, iteration: int) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        logger.warning(
            "parse_execution_trace: malformed integer field %s=%r "
            "(task=%s iter=%d); defaulting to 0",
            field,
            value,
            task_id,
            iteration,
        )
        return 0


def _latest_path(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def _find_trial_dir(job_dir: Path | None) -> Path | None:
    if job_dir is None:
        return None
    candidates = [
        path for path in job_dir.iterdir()
        if path.is_dir() and (path / "result.json").exists()
    ]
    return _latest_path(candidates)
