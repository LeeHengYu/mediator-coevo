"""Local SkillsBench task loading and Harbor execution helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mediated_coevo.models.trace import ExecutionTrace, TokenUsage

logger = logging.getLogger(__name__)


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

    def __init__(self, agent_name: str, jobs_dir: Path) -> None:
        self.agent_name = agent_name
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, task_dir: Path, model: str) -> HarborRunResult:
        before = {p.resolve() for p in self.jobs_dir.iterdir() if p.is_dir()}
        cmd = [
            "harbor",
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
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        after = [p.resolve() for p in self.jobs_dir.iterdir() if p.is_dir()]
        new_jobs = [p for p in after if p not in before]
        job_dir = _latest_path(new_jobs) or _latest_path(after)
        trial_dir = _find_trial_dir(job_dir) if job_dir else None
        return HarborRunResult(
            job_dir=job_dir,
            trial_dir=trial_dir,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )


def parse_execution_trace(
    run_result: HarborRunResult,
    task_id: str,
    iteration: int,
    duration_sec: float,
) -> ExecutionTrace:
    """Convert Harbor artifacts into this repo's ExecutionTrace."""
    result_json: dict[str, Any] = {}
    ctrf_json: dict[str, Any] | None = None
    reward = 0.0
    input_tokens = 0
    output_tokens = 0
    agent_summary = ""

    if run_result.trial_dir:
        result_path = run_result.trial_dir / "result.json"
        if result_path.exists():
            result_json = json.loads(result_path.read_text())

        ctrf_path = run_result.trial_dir / "verifier" / "ctrf.json"
        if ctrf_path.exists():
            ctrf_json = json.loads(ctrf_path.read_text())

        reward_path = run_result.trial_dir / "verifier" / "reward.txt"
        if reward_path.exists():
            reward_text = reward_path.read_text().strip()
            reward = float(reward_text)

        if reward == 0.0:
            reward = (
                result_json.get("verifier_result", {})
                .get("rewards", {})
                .get("reward", 0.0)
            )

        agent_result = result_json.get("agent_result", {})
        input_tokens = int(agent_result.get("n_input_tokens", 0) or 0)
        output_tokens = int(agent_result.get("n_output_tokens", 0) or 0)

        agent_dir = run_result.trial_dir / "agent"
        if agent_dir.exists():
            txt_files = sorted(agent_dir.glob("*.txt"))
            if txt_files:
                agent_summary = txt_files[0].read_text()

    stdout_sections = []
    if run_result.stdout.strip():
        stdout_sections.append("# Harbor Output\n\n" + run_result.stdout.strip())
    if agent_summary.strip():
        stdout_sections.append("# Agent Summary\n\n" + agent_summary.strip())
    stdout = "\n\n".join(stdout_sections)

    stderr_sections = []
    if run_result.stderr.strip():
        stderr_sections.append(run_result.stderr.strip())
    exception_info = result_json.get("exception_info")
    if exception_info:
        stderr_sections.append(json.dumps(exception_info, indent=2))
    stderr = "\n\n".join(stderr_sections)

    exit_code = run_result.returncode
    if exit_code == 0 and exception_info:
        exit_code = 1

    return ExecutionTrace(
        task_id=task_id,
        iteration=iteration,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        test_results=_summarize_ctrf(ctrf_json),
        reward=float(reward),
        token_usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
        duration_sec=duration_sec,
    )


def _summarize_ctrf(ctrf_json: dict[str, Any] | None) -> dict[str, Any] | None:
    if not ctrf_json:
        return None
    results = ctrf_json.get("results", {})
    summary = results.get("summary", {})
    tests = results.get("tests", [])
    return {
        "summary": summary,
        "failed_tests": [
            {
                "name": test.get("name"),
                "status": test.get("status"),
                "message": test.get("message"),
            }
            for test in tests
            if test.get("status") != "passed"
        ],
    }


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
