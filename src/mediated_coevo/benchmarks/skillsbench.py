"""Local SkillsBench task loading and Harbor execution helpers."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import shutil
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from mediated_coevo.models.trace import ExecutionTrace, TokenUsage, TraceStatus

logger = logging.getLogger(__name__)

SKILLSBENCH_ARCHIVE_URL = (
    "https://github.com/benchflow-ai/skillsbench/archive/refs/heads/main.zip"
)


class HarborNotFoundError(RuntimeError):
    """Raised when the harbor CLI cannot be located on PATH or is not executable."""


class HarborTimeoutError(RuntimeError):
    """Raised when a Harbor subprocess exceeds the configured timeout."""


class SkillsBenchFetchError(FileNotFoundError):
    """Raised when a remote SkillsBench task cannot be fetched or materialized."""


@dataclass(frozen=True, slots=True)
class SkillsBenchRemoteConfig:
    """Remote archive settings for on-demand SkillsBench task fetching."""

    enabled: bool = False
    timeout_sec: float = 60.0


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

    def __init__(
        self,
        root_dir: Path,
        task_dirs: list[str],
        remote: SkillsBenchRemoteConfig | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.task_dirs = task_dirs
        self.remote = remote or SkillsBenchRemoteConfig()
        self._archive_cache: dict[str, bytes] = {}

    def default_local_cache_dir(self) -> Path:
        """Return the local directory where task folders are cached."""
        return self.root_dir / self.task_dirs[0]

    def resolve(self, task_id: str) -> SkillsBenchTask:
        local_task_dir = self._resolve_local_task_dir(task_id)
        if local_task_dir is not None:
            return self._load_task(local_task_dir, task_id)

        if self.remote.enabled:
            fetched_task_dir = self._fetch_task(task_id)
            return self._load_task(fetched_task_dir, task_id)

        searched = [str(self.default_local_cache_dir() / task_id)]
        raise FileNotFoundError(
            f"Task '{task_id}' not found under local benchmark root. Searched: {searched}"
        )

    def list_local_task_ids(self) -> list[str]:
        """Return safe task IDs already present in local task dirs."""
        task_ids: list[str] = []
        seen: set[str] = set()
        task_dir = self.default_local_cache_dir()
        if not task_dir.exists():
            return task_ids
        for candidate in sorted(task_dir.iterdir(), key=lambda path: path.name):
            task_id = candidate.name
            if (
                candidate.is_dir()
                and task_id not in seen
                and self._is_safe_task_id(task_id)
            ):
                task_ids.append(task_id)
                seen.add(task_id)
        return task_ids

    def list_remote_task_ids(self) -> list[str]:
        """Return safe task IDs advertised by the provided remote archive."""
        if not self.remote.enabled:
            raise SkillsBenchFetchError(
                "Remote SkillsBench task discovery is disabled; set "
                "executor_runtime.remote_fetch = true to use skillsbench-all."
            )

        archive_url = SKILLSBENCH_ARCHIVE_URL
        archive_bytes = self._archive_bytes(archive_url)
        try:
            archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
        except zipfile.BadZipFile as exc:
            raise SkillsBenchFetchError(
                f"Remote SkillsBench archive {archive_url!r} is not a valid zip file"
            ) from exc

        task_ids: list[str] = []
        seen: set[str] = set()
        with archive:
            task_dir_name = self.task_dirs[0]
            for task_id in self._task_ids_from_archive(
                archive.namelist(),
                task_dir_name,
            ):
                if task_id not in seen:
                    task_ids.append(task_id)
                    seen.add(task_id)
        return task_ids

    def sync_tasks(self, task_ids: list[str]) -> list[SkillsBenchTask]:
        """Ensure selected tasks are present locally, fetching missing ones."""
        if not task_ids:
            raise SkillsBenchFetchError("At least one SkillsBench task ID is required")
        return [self.resolve(task_id) for task_id in task_ids]

    def _resolve_local_task_dir(self, task_id: str) -> Path | None:
        candidate = self.default_local_cache_dir() / task_id
        if candidate.exists():
            return candidate
        return None

    def _fetch_task(self, task_id: str) -> Path:
        self._validate_task_id(task_id)
        archive_url = SKILLSBENCH_ARCHIVE_URL
        try:
            archive_bytes = self._archive_bytes(archive_url)
        except SkillsBenchFetchError as exc:
            searched = [str(self.default_local_cache_dir() / task_id)]
            raise SkillsBenchFetchError(
                f"Task '{task_id}' is not available locally and remote fetch failed. "
                f"Searched local paths: {searched}. Remote archive: {archive_url!r}. "
                f"Cause: {exc}"
            ) from exc

        task_dir_name = self.task_dirs[0]
        target = self.default_local_cache_dir() / task_id
        remote_paths = [f"{task_dir_name}/{task_id}"]
        if target.exists():
            return target
        materialized = self._extract_task_from_archive(
            archive_bytes=archive_bytes,
            task_dir_name=task_dir_name,
            task_id=task_id,
            target=target,
            archive_url=archive_url,
        )
        if materialized is not None:
            return materialized

        raise SkillsBenchFetchError(
            f"Task '{task_id}' not found in remote SkillsBench archive "
            f"{archive_url!r}. Searched remote paths: {remote_paths}"
        )

    def _download_archive(self, archive_url: str) -> bytes:
        parsed = urllib.parse.urlparse(archive_url)
        try:
            if parsed.scheme == "":
                return Path(archive_url).read_bytes()
            with urllib.request.urlopen(archive_url, timeout=self.remote.timeout_sec) as response:
                return response.read()
        except (OSError, urllib.error.URLError) as exc:
            raise SkillsBenchFetchError(
                f"Failed to fetch SkillsBench archive {archive_url!r}: {exc}"
            ) from exc

    def _archive_bytes(self, archive_url: str) -> bytes:
        cached = self._archive_cache.get(archive_url)
        if cached is not None:
            return cached
        archive_bytes = self._download_archive(archive_url)
        self._archive_cache[archive_url] = archive_bytes
        return archive_bytes

    def _extract_task_from_archive(
        self,
        *,
        archive_bytes: bytes,
        task_dir_name: str,
        task_id: str,
        target: Path,
        archive_url: str,
    ) -> Path | None:
        try:
            archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
        except zipfile.BadZipFile as exc:
            raise SkillsBenchFetchError(
                f"Remote SkillsBench archive {archive_url!r} is not a valid zip file"
            ) from exc

        with archive:
            prefix = self._find_task_prefix(archive.namelist(), task_dir_name, task_id)
            if prefix is None:
                return None

            temp_dir = self._new_fetch_temp_dir(task_dir_name, task_id)
            for member in archive.infolist():
                member_name = member.filename
                if not member_name.startswith(prefix) or member_name == prefix:
                    continue
                relative = PurePosixPath(member_name[len(prefix) :])
                self._validate_relative_archive_path(relative, member_name)
                destination = temp_dir.joinpath(*relative.parts)
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, open(destination, "wb") as sink:
                    shutil.copyfileobj(source, sink)

        self._load_task(temp_dir, task_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return target
        temp_dir.replace(target)
        logger.info("Fetched SkillsBench task %s into %s", task_id, target)
        return target

    def _new_fetch_temp_dir(self, task_dir_name: str, task_id: str) -> Path:
        temp_dir = (
            self.root_dir
            / ".fetch-tmp"
            / task_dir_name
            / f"{task_id}-{uuid.uuid4().hex[:8]}"
        )
        temp_dir.mkdir(parents=True, exist_ok=False)
        return temp_dir

    @staticmethod
    def _find_task_prefix(
        member_names: list[str],
        task_dir_name: str,
        task_id: str,
    ) -> str | None:
        for raw_name in member_names:
            parts = PurePosixPath(raw_name).parts
            for offset in (0, 1):
                if (
                    len(parts) >= offset + 2
                    and parts[offset] == task_dir_name
                    and parts[offset + 1] == task_id
                ):
                    return "/".join(parts[: offset + 2]) + "/"
        return None

    @classmethod
    def _task_ids_from_archive(
        cls,
        member_names: list[str],
        task_dir_name: str,
    ) -> list[str]:
        task_ids: list[str] = []
        seen: set[str] = set()
        for raw_name in member_names:
            parts = PurePosixPath(raw_name).parts
            for offset in (0, 1):
                if len(parts) < offset + 3 or parts[offset] != task_dir_name:
                    continue
                task_id = parts[offset + 1]
                if not cls._is_safe_task_id(task_id) or task_id in seen:
                    continue
                task_ids.append(task_id)
                seen.add(task_id)
        return sorted(task_ids)

    @staticmethod
    def _is_safe_task_id(task_id: str) -> bool:
        return (
            bool(task_id)
            and task_id not in {".", ".."}
            and not task_id.startswith(".")
            and "/" not in task_id
            and "\\" not in task_id
        )

    @staticmethod
    def _validate_task_id(task_id: str) -> None:
        if not SkillsBenchRepository._is_safe_task_id(task_id):
            raise SkillsBenchFetchError(f"Unsafe SkillsBench task_id: {task_id!r}")

    @staticmethod
    def _validate_relative_archive_path(
        relative: PurePosixPath,
        member_name: str,
    ) -> None:
        if (
            not relative.parts
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise SkillsBenchFetchError(
                f"Unsafe path in SkillsBench archive member {member_name!r}"
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


class _MalformedReward(ValueError):
    """Raised when reward.txt exists but cannot be parsed as a number."""


class HarborTraceParser:
    """Convert one Harbor run result into a classified execution trace."""

    def __init__(
        self,
        run_result: HarborRunResult,
        task_id: str,
        iteration: int,
        duration_sec: float,
    ) -> None:
        self.run_result = run_result
        self.task_id = task_id
        self.iteration = iteration
        self.duration_sec = duration_sec
        self.base: dict[str, Any] = dict(
            task_id=task_id,
            iteration=iteration,
            duration_sec=duration_sec,
            exit_code=run_result.returncode,
            stdout=_format_harbor_stdout(run_result.stdout, ""),
            stderr=_format_harbor_stderr(run_result.stderr, None),
        )

    def parse(self) -> ExecutionTrace:
        """Parse Harbor artifacts and classify boundary failures."""
        trial_dir = self.run_result.trial_dir
        if trial_dir is None:
            return self._missing_trial_dir()

        result_path = trial_dir / "result.json"
        try:
            result_json = self._load_result_json(result_path)
        except FileNotFoundError:
            return self._missing_result_json(result_path)
        except json.JSONDecodeError as e:
            return self._malformed_result_json(result_path, e)

        ctrf_json = self._load_ctrf_json(trial_dir)
        exception_info = result_json.get("exception_info")
        full_base = self._build_full_base(trial_dir, result_json, ctrf_json)

        try:
            reward = self._parse_reward(trial_dir, result_json)
        except _MalformedReward as e:
            return self._trace_from(
                full_base,
                status="parse_error",
                error_kind="malformed_reward",
                error_detail=str(e),
            )

        if reward is None:
            return self._missing_reward_trace(full_base, exception_info)

        return self._rewarded_trace(full_base, ctrf_json, reward, exception_info)

    def _load_result_json(self, result_path: Path) -> dict[str, Any]:
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        return json.loads(result_path.read_text())

    def _load_ctrf_json(self, trial_dir: Path) -> dict[str, Any] | None:
        ctrf_path = trial_dir / "verifier" / "ctrf.json"
        if not ctrf_path.exists():
            return None
        try:
            return json.loads(ctrf_path.read_text())
        except json.JSONDecodeError as e:
            logger.warning(
                "parse_execution_trace: malformed ctrf.json at %s: %s",
                ctrf_path,
                e,
            )
            return None

    def _build_full_base(
        self,
        trial_dir: Path,
        result_json: dict[str, Any],
        ctrf_json: dict[str, Any] | None,
    ) -> dict[str, Any]:
        agent_result = _mapping_or_empty(result_json.get("agent_result"))
        return dict(
            self.base,
            stdout=_format_harbor_stdout(
                self.run_result.stdout,
                _read_agent_summary(trial_dir),
            ),
            stderr=_format_harbor_stderr(
                self.run_result.stderr,
                result_json.get("exception_info"),
            ),
            test_results=_summarize_ctrf(ctrf_json),
            token_usage=TokenUsage(
                input_tokens=_safe_int(
                    agent_result.get("n_input_tokens", 0),
                    field="agent_result.n_input_tokens",
                    task_id=self.task_id,
                    iteration=self.iteration,
                ),
                output_tokens=_safe_int(
                    agent_result.get("n_output_tokens", 0),
                    field="agent_result.n_output_tokens",
                    task_id=self.task_id,
                    iteration=self.iteration,
                ),
            ),
        )

    def _trace(self, **overrides: Any) -> ExecutionTrace:
        return self._trace_from(self.base, **overrides)

    @staticmethod
    def _trace_from(base: dict[str, Any], **overrides: Any) -> ExecutionTrace:
        return ExecutionTrace(**{**base, **overrides})

    def _parse_reward(
        self,
        trial_dir: Path,
        result_json: dict[str, Any],
    ) -> float | None:
        """Parse reward.txt first; fall back to result.json only when absent."""
        reward_path = trial_dir / "verifier" / "reward.txt"
        if reward_path.exists():
            reward_text = reward_path.read_text().strip()
            try:
                return float(reward_text)
            except ValueError as e:
                logger.warning(
                    "parse_execution_trace: malformed reward.txt at %s (content=%r)",
                    reward_path,
                    reward_text,
                )
                raise _MalformedReward(
                    f"reward.txt at {reward_path} contained non-numeric value: "
                    f"{reward_text!r}"
                ) from e

        nested = _mapping_or_empty(
            _mapping_or_empty(result_json.get("verifier_result")).get("rewards")
        ).get("reward")
        if nested is None:
            return None
        try:
            return float(nested)
        except (TypeError, ValueError):
            return None

    def _missing_trial_dir(self) -> ExecutionTrace:
        logger.warning(
            "parse_execution_trace: no trial_dir for task=%s iter=%d (returncode=%d)",
            self.task_id,
            self.iteration,
            self.run_result.returncode,
        )
        return self._trace(
            status="env_failure",
            error_kind="missing_trial_dir",
            error_detail=self.run_result.stderr.strip() or None,
        )

    def _missing_result_json(self, result_path: Path) -> ExecutionTrace:
        logger.warning(
            "parse_execution_trace: missing result.json at %s (task=%s iter=%d)",
            result_path,
            self.task_id,
            self.iteration,
        )
        return self._trace(
            status="env_failure",
            error_kind="missing_result_json",
            error_detail=f"result.json not found at {result_path}",
        )

    def _malformed_result_json(
        self,
        result_path: Path,
        exc: json.JSONDecodeError,
    ) -> ExecutionTrace:
        logger.warning(
            "parse_execution_trace: malformed result.json at %s: %s",
            result_path,
            exc,
        )
        return self._trace(
            status="env_failure",
            error_kind="malformed_result_json",
            error_detail=str(exc),
        )

    def _missing_reward_trace(
        self,
        full_base: dict[str, Any],
        exception_info: Any,
    ) -> ExecutionTrace:
        logger.warning(
            "parse_execution_trace: no reward parsed for task=%s iter=%d (returncode=%d)",
            self.task_id,
            self.iteration,
            self.run_result.returncode,
        )
        if self.run_result.returncode != 0:
            status: TraceStatus = "harbor_failed"
            error_kind = "harbor_nonzero_no_reward"
        else:
            status = "env_failure"
            error_kind = "no_reward"
        return self._trace_from(
            full_base,
            status=status,
            error_kind=error_kind,
            error_detail=exception_info or self.run_result.stderr.strip() or None,
        )

    def _rewarded_trace(
        self,
        full_base: dict[str, Any],
        ctrf_json: dict[str, Any] | None,
        reward: float,
        exception_info: Any,
    ) -> ExecutionTrace:
        error_kind, error_detail = _ctrf_mismatch(
            ctrf_json,
            reward,
            task_id=self.task_id,
            iteration=self.iteration,
        )
        if exception_info and error_kind is None:
            error_kind = "harbor_exception"
            error_detail = exception_info

        status: TraceStatus = "harbor_failed" if self.run_result.returncode != 0 else "ok"
        if self.run_result.returncode != 0 and error_kind is None:
            error_kind = "harbor_nonzero"
            error_detail = (
                self.run_result.stderr.strip()
                or f"harbor returncode={self.run_result.returncode}"
            )

        # Bump exit_code 0→1 to mark exception_info in legacy consumers.
        exit_code = (
            1
            if self.run_result.returncode == 0 and exception_info
            else self.run_result.returncode
        )
        return self._trace_from(
            full_base,
            exit_code=exit_code,
            reward=reward,
            status=status,
            error_kind=error_kind,
            error_detail=error_detail,
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
    return HarborTraceParser(run_result, task_id, iteration, duration_sec).parse()


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
