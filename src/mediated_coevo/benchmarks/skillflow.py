"""SkillFlow task loading and Harbor execution helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import shutil
import subprocess
import tempfile
import tomllib
import uuid
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mediated_coevo.core.utils import as_mapping, as_nonempty_string
from mediated_coevo.models.task import (
    executor_policy_metadata,
    render_executor_envelope,
)
from mediated_coevo.models.trace import ExecutionTrace, TokenUsage

logger = logging.getLogger(__name__)

DEFAULT_SKILLFLOW_DATASET = "zhang-ziao/SkillFlow-Task"
SKILLFLOW_VERIFIER_TYPE = "skillflow_harbor"
HERMES_AGENT_NAME = "hermes"
HF_TEST_TASKS_ROOT = "test_tasks"
HF_FAMILY_RANKING_FILENAME = "ALL_TASK_DIFFICULTY_RANKING.json"
HARBOR_PREBUILT_IMAGE_PREFIX = "harbor-prebuilt:"
LEGACY_HARBOR_BASE_IMAGE = "skillevlove/harbor-cli-openhands:ubuntu24.04"
SKILLFLOW_HARBOR_BASE_IMAGE = "skillflow/harbor-cli-base:ubuntu24.04"
DOCKERFILE_FROM_IMAGE_RE = re.compile(r"^\s*FROM\s+(?P<image>[^\s]+)", re.IGNORECASE)


class HarborNotFoundError(RuntimeError):
    """Raised when the Harbor CLI cannot be located on PATH or is not executable."""


class HarborTimeoutError(RuntimeError):
    """Raised when a Harbor subprocess exceeds the configured timeout."""


class HarborPrebuiltImageMissingError(RuntimeError):
    """Raised when a task expects a prebuilt Harbor image that is not local."""


class SkillFlowSyncError(RuntimeError):
    """Raised when SkillFlow task synchronization fails."""


@dataclass(frozen=True, slots=True)
class SkillFlowSyncConfig:
    """Hugging Face task-data synchronization settings."""

    enabled: bool = False
    dataset: str = DEFAULT_SKILLFLOW_DATASET
    repo_type: str = "dataset"
    local_dir: str = "tasks"
    remote_task_cache_path: Path | None = None


@dataclass(slots=True)
class SkillFlowTask:
    """A locally available SkillFlow task."""

    task_id: str
    task_dir: Path
    instruction_path: Path
    instruction: str
    task_config: dict[str, Any]
    family: str | None = None
    difficulty: str | None = None
    benchmark_kind: str = "skillflow"


@dataclass(slots=True)
class HarborRunResult:
    """Artifacts produced by one Harbor task run."""

    job_dir: Path | None
    trial_dir: Path | None
    returncode: int
    stdout: str
    stderr: str


class SkillFlowRepository:
    """Resolve local SkillFlow tasks and materialize Harbor workspaces."""

    def __init__(
        self,
        root_dir: Path,
        task_dirs: list[str],
        sync: SkillFlowSyncConfig | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.task_dirs = task_dirs
        self.sync = sync or SkillFlowSyncConfig()

    def default_local_cache_dir(self) -> Path:
        """Return the local directory where task folders are cached."""
        return self.root_dir / self.task_dirs[0]

    def resolve(self, task_id: str) -> SkillFlowTask:
        task_dir = self._resolve_local_task_dir(task_id)
        if task_dir is None:
            searched = [
                str(task_root / task_id)
                for task_root in self._existing_task_roots(include_missing=True)
            ]
            raise FileNotFoundError(
                f"SkillFlow task {task_id!r} not found. Searched: {searched}"
            )
        return self._load_task(task_dir, task_id)

    def list_local_task_ids(self, *, family: str | None = None) -> list[str]:
        """Return available local SkillFlow task IDs."""
        task_ids = []
        for task_dir in self._iter_task_dirs():
            task = self._load_task(task_dir, self._task_id_for_dir(task_dir))
            if family is None or task.family == family:
                task_ids.append(task.task_id)
        return sorted(dict.fromkeys(task_ids))

    def resolve_selection(
        self,
        *,
        tasks: list[str] | None,
        family: str | None,
        task_set: str | None,
    ) -> list[str]:
        """Resolve CLI task, family, and task-set selectors."""
        selected: list[str] = []
        if tasks:
            selected.extend(tasks)
        if family:
            selected.extend(self.list_local_task_ids(family=family))
        if task_set:
            selected.extend(self._resolve_task_set(task_set))
        return _dedupe(selected)

    def list_remote_task_ids(self, *, family: str | None = None) -> list[str]:
        """Return task IDs advertised by the configured remote SkillFlow dataset."""
        task_ids = self._cached_remote_task_ids()
        if task_ids is None:
            task_ids = self._list_remote_task_ids_from_hf()
        if family is None:
            return task_ids
        return [
            task_id
            for task_id in task_ids
            if task_id.split("/", 1)[0] == family
        ]

    def _cached_remote_task_ids(self) -> list[str] | None:
        if self.sync.remote_task_cache_path is None:
            return None
        if not self.sync.remote_task_cache_path.is_file():
            return None
        task_ids: set[str] = set()
        for line in self.sync.remote_task_cache_path.read_text().splitlines():
            normalized = line.strip().strip('"').strip("'")
            if not normalized or normalized.startswith("#"):
                continue
            if "/" in normalized and _is_safe_task_id(normalized):
                task_ids.add(normalized)
        return sorted(task_ids)

    def _list_remote_task_ids_from_hf(self) -> list[str]:
        command = [
            "hf",
            "datasets",
            "ls",
            self.sync.dataset,
            "-R",
            "--format",
            "quiet",
        ]
        completed = self._run_hf_command(
            command,
            failure_message="SkillFlow task list failed",
        )
        task_ids: set[str] = set()
        marker = f"{HF_TEST_TASKS_ROOT}/"
        for line in completed.stdout.splitlines():
            for token in line.split():
                normalized = token.strip().strip('"').strip("'")
                if marker not in normalized or not normalized.endswith("/task.toml"):
                    continue
                task_id = normalized.split(marker, 1)[1].removesuffix("/task.toml")
                if "/" in task_id and _is_safe_task_id(task_id):
                    task_ids.add(task_id)
        return sorted(task_ids)

    def sync_tasks(
        self,
        destination: Path | None = None,
        *,
        task_ids: list[str] | None = None,
    ) -> Path:
        """Download SkillFlow task data into the configured task cache."""
        destination = destination or self.default_local_cache_dir()
        destination.mkdir(parents=True, exist_ok=True)
        selected_task_ids = _validated_remote_task_ids(task_ids)
        with tempfile.TemporaryDirectory(
            prefix="skillflow-sync-",
            dir=destination.parent,
        ) as staging_dir_name:
            staging_dir = Path(staging_dir_name)
            command = [
                "hf",
                "download",
                self.sync.dataset,
                "--repo-type",
                self.sync.repo_type,
            ]
            if selected_task_ids is None:
                include_patterns = [f"{HF_TEST_TASKS_ROOT}/**"]
            else:
                include_patterns = []
                for task_id in selected_task_ids:
                    family_id = task_id.split("/", 1)[0]
                    include_patterns.append(f"{HF_TEST_TASKS_ROOT}/{task_id}/**")
                    include_patterns.append(
                        f"{HF_TEST_TASKS_ROOT}/{family_id}/"
                        f"{HF_FAMILY_RANKING_FILENAME}"
                    )
                include_patterns = _dedupe(include_patterns)
            for include_pattern in include_patterns:
                command.extend(["--include", include_pattern])
            command.extend(["--local-dir", str(staging_dir)])
            self._run_hf_command(
                command,
                failure_message="SkillFlow task sync failed",
            )
            _merge_downloaded_test_tasks(
                source_root=staging_dir / HF_TEST_TASKS_ROOT,
                destination=destination,
            )
        return destination

    @staticmethod
    def _run_hf_command(
        command: list[str],
        *,
        failure_message: str,
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                command,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise SkillFlowSyncError(
                "Hugging Face CLI `hf` was not found. Install it separately or "
                "pre-populate the configured SkillFlow task directory."
            ) from exc
        if completed.returncode != 0:
            raise SkillFlowSyncError(
                f"{failure_message}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        return completed

    def prepare_run_workspace(
        self,
        task: SkillFlowTask,
        destination_root: Path,
        planner_instruction: str,
        injected_skill_text: str | None,
        injected_skill_name: str,
    ) -> Path:
        del injected_skill_name
        run_dir = (
            destination_root
            / task.task_id.replace("/", "_").replace("\\", "_")
            / f"run-{uuid.uuid4().hex[:8]}"
        )
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(task.task_dir, run_dir)

        instruction_path = run_dir / "instruction.md"
        instruction_path.write_text(
            render_executor_envelope(
                task_instruction=planner_instruction,
                executor_policy=injected_skill_text,
                task_resources=_skillflow_task_resources(run_dir),
                verifier_contract=_skillflow_verifier_contract(task),
            )
        )
        return run_dir

    @staticmethod
    def executor_envelope_metadata(
        *,
        run_dir: Path,
        executor_policy: str | None,
    ) -> dict[str, str]:
        """Return trace metadata for the prepared SkillFlow executor envelope."""
        return executor_policy_metadata(
            executor_policy=executor_policy,
            injection_location="instruction_envelope",
            task_resource_names=_skillflow_task_resource_names(run_dir),
            verifier_contract_kind=SKILLFLOW_VERIFIER_TYPE,
        )

    def _resolve_local_task_dir(self, task_id: str) -> Path | None:
        if not _is_safe_task_id(task_id):
            raise FileNotFoundError(f"Unsafe SkillFlow task ID: {task_id!r}")
        for task_root in self._existing_task_roots():
            direct = task_root / task_id
            if _is_task_dir(direct):
                return direct
            normalized = task_root / task_id.replace("/", os.sep)
            if _is_task_dir(normalized):
                return normalized
        return None

    def _iter_task_dirs(self) -> Iterable[Path]:
        for task_root in self._existing_task_roots():
            if _is_task_dir(task_root):
                yield task_root
            for candidate in sorted(task_root.rglob("task.toml")):
                task_dir = candidate.parent
                if _is_task_dir(task_dir):
                    yield task_dir

    def _existing_task_roots(self, *, include_missing: bool = False) -> list[Path]:
        roots = [self.root_dir / task_dir for task_dir in self.task_dirs]
        if include_missing:
            return roots
        return [root for root in roots if root.exists()]

    def _task_id_for_dir(self, task_dir: Path) -> str:
        for task_root in self._existing_task_roots():
            with suppress(ValueError):
                return task_dir.relative_to(task_root).as_posix()
        return task_dir.name

    def _resolve_task_set(self, task_set: str) -> list[str]:
        task_set_path = self.root_dir / "task_sets" / f"{task_set}.txt"
        if task_set_path.exists():
            return [
                line.strip()
                for line in task_set_path.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        task_ids = self.list_local_task_ids(family=task_set)
        if task_ids:
            return task_ids
        raise FileNotFoundError(
            f"SkillFlow task set {task_set!r} was not found under "
            f"{task_set_path.parent}"
        )

    def _load_task(self, task_dir: Path, task_id: str) -> SkillFlowTask:
        instruction_path = task_dir / "instruction.md"
        task_toml_path = task_dir / "task.toml"
        if not instruction_path.exists():
            raise FileNotFoundError(
                f"Missing instruction.md for SkillFlow task {task_id!r} at {task_dir}"
            )
        if not task_toml_path.exists():
            raise FileNotFoundError(
                f"Missing task.toml for SkillFlow task {task_id!r} at {task_dir}"
            )
        with open(task_toml_path, "rb") as f:
            task_config = tomllib.load(f)
        metadata = as_mapping(task_config.get("metadata"))
        task_section = as_mapping(task_config.get("task"))
        family = as_nonempty_string(metadata.get("family"))
        if family is None and "/" in task_id:
            family = task_id.split("/", 1)[0]
        return SkillFlowTask(
            task_id=task_id,
            task_dir=task_dir,
            instruction_path=instruction_path,
            instruction=instruction_path.read_text(),
            task_config=task_config,
            family=family,
            difficulty=as_nonempty_string(
                metadata.get("difficulty") or task_section.get("difficulty")
            ),
        )


class HarborRunner:
    """Run a local SkillFlow task via Harbor and locate its artifacts."""

    def __init__(
        self,
        jobs_dir: Path,
        timeout_sec: float = 1800.0,
        agent_setup_timeout_multiplier: float | None = None,
    ) -> None:
        self.jobs_dir = jobs_dir
        self.timeout_sec = timeout_sec
        self.agent_setup_timeout_multiplier = agent_setup_timeout_multiplier
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._harbor_path: str | None = None

    def _resolve_harbor(self) -> str:
        if self._harbor_path is None:
            path = shutil.which("harbor")
            if path is None:
                raise HarborNotFoundError(
                    "harbor CLI not found on PATH. Install Harbor, or set "
                    "executor_runtime.harbor_required=false to allow synthesized "
                    "environment-failure traces in CI."
                )
            self._harbor_path = path
        return self._harbor_path

    async def run(self, task_dir: Path, model: str) -> HarborRunResult:
        await asyncio.to_thread(_ensure_declared_prebuilt_image, task_dir)
        result = await self._run_once(task_dir, model)
        if harbor_run_missing_prebuilt_image(result):
            raise HarborPrebuiltImageMissingError(
                harbor_missing_prebuilt_image_message(result, task_dir)
            )
        return result

    async def _run_once(
        self,
        task_dir: Path,
        model: str,
    ) -> HarborRunResult:
        harbor = self._resolve_harbor()
        before = {p.resolve() for p in self.jobs_dir.iterdir() if p.is_dir()}
        command = [
            harbor,
            "run",
            "-p",
            str(task_dir),
            "-a",
            HERMES_AGENT_NAME,
            "-m",
            model,
            "-o",
            str(self.jobs_dir),
            "--yes",
        ]
        if self.agent_setup_timeout_multiplier is not None:
            command.extend(
                [
                    "--agent-setup-timeout-multiplier",
                    str(self.agent_setup_timeout_multiplier),
                ]
            )
        logger.info("Running SkillFlow Harbor task: %s", " ".join(command))
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)

        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise HarborNotFoundError(f"harbor CLI not executable: {exc}") from exc

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_sec,
            )
        except TimeoutError as exc:
            await _terminate_process_tree(proc)
            raise HarborTimeoutError(
                f"harbor run exceeded {self.timeout_sec}s timeout for {task_dir}"
            ) from exc

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
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


class SkillFlowTraceParser:
    """Convert one SkillFlow Harbor run into a normalized execution trace."""

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

    def parse(self) -> ExecutionTrace:
        if self.run_result.trial_dir is None:
            return ExecutionTrace(
                task_id=self.task_id,
                iteration=self.iteration,
                duration_sec=self.duration_sec,
                exit_code=self.run_result.returncode,
                stdout=self.run_result.stdout,
                stderr=self.run_result.stderr,
                harbor_paths=_harbor_paths(self.run_result),
                status="env_failure",
                error_kind="missing_trial_dir",
                error_detail="Harbor did not produce a trial directory.",
            )

        trial_result_json = _load_json_or_empty(
            self.run_result.trial_dir / "result.json"
        )
        job_result_json = _load_json_or_empty(
            self.run_result.job_dir / "result.json"
            if self.run_result.job_dir is not None
            else None
        )
        exception_info = as_mapping(trial_result_json.get("exception_info"))
        if exception_info:
            return ExecutionTrace(
                task_id=self.task_id,
                iteration=self.iteration,
                duration_sec=self.duration_sec,
                exit_code=self.run_result.returncode,
                stdout=self.run_result.stdout,
                stderr=self.run_result.stderr,
                harbor_paths=_harbor_paths(self.run_result),
                status="env_failure",
                error_kind=_trial_exception_error_kind(exception_info),
                error_detail=dict(exception_info),
                harbor_trial_id=as_nonempty_string(trial_result_json.get("id")),
                run_id=as_nonempty_string(job_result_json.get("id")),
                harbor_metadata=_harbor_metadata(trial_result_json),
            )
        reward_values: list[Any] = []
        stats = as_mapping(job_result_json.get("stats"))
        evals = as_mapping(stats.get("evals"))
        for eval_result in evals.values():
            metrics = as_mapping(eval_result).get("metrics")
            if isinstance(metrics, list):
                for metric in metrics:
                    reward_values.append(as_mapping(metric).get("mean"))
        reward_values.append(trial_result_json.get("reward"))
        verifier_result = as_mapping(trial_result_json.get("verifier_result"))
        reward_values.append(verifier_result.get("reward"))
        reward_values.append(as_mapping(verifier_result.get("rewards")).get("reward"))
        reward_path = self.run_result.trial_dir / "verifier" / "reward.txt"
        if reward_path.exists():
            reward_values.append(reward_path.read_text().strip())

        reward = None
        for value in reward_values:
            try:
                reward = float(value)
            except (TypeError, ValueError):
                continue
            break
        if reward is None:
            return ExecutionTrace(
                task_id=self.task_id,
                iteration=self.iteration,
                duration_sec=self.duration_sec,
                exit_code=self.run_result.returncode,
                stdout=self.run_result.stdout,
                stderr=self.run_result.stderr,
                harbor_paths=_harbor_paths(self.run_result),
                status="env_failure",
                error_kind="missing_reward",
                error_detail="No SkillFlow reward was found in Harbor artifacts.",
                harbor_trial_id=as_nonempty_string(trial_result_json.get("id")),
                run_id=as_nonempty_string(job_result_json.get("id")),
                harbor_metadata=_harbor_metadata(trial_result_json),
            )

        agent_result = as_mapping(trial_result_json.get("agent_result"))
        try:
            input_tokens = int(agent_result.get("n_input_tokens", 0))
        except (TypeError, ValueError):
            input_tokens = 0
        try:
            output_tokens = int(agent_result.get("n_output_tokens", 0))
        except (TypeError, ValueError):
            output_tokens = 0
        if evals:
            reward_source = "job_stats"
        elif "reward" in trial_result_json:
            reward_source = "trial_result"
        elif verifier_result.get("reward") is not None:
            reward_source = "trial_verifier_result"
        elif as_mapping(verifier_result.get("rewards")).get("reward") is not None:
            reward_source = "trial_verifier_rewards"
        else:
            reward_source = "verifier_reward_file"
        ctrf_path = self.run_result.trial_dir / "verifier" / "ctrf.json"
        test_results = (
            _load_json_or_empty(ctrf_path)
            if ctrf_path.exists()
            else None
        )
        return ExecutionTrace(
            task_id=self.task_id,
            iteration=self.iteration,
            duration_sec=self.duration_sec,
            exit_code=self.run_result.returncode,
            stdout=self.run_result.stdout,
            stderr=self.run_result.stderr,
            harbor_paths=_harbor_paths(self.run_result),
            status="ok",
            reward=reward,
            run_id=as_nonempty_string(job_result_json.get("id")),
            harbor_trial_id=as_nonempty_string(trial_result_json.get("id")),
            harbor_metadata={
                **_harbor_metadata(trial_result_json),
                "verifier_type": SKILLFLOW_VERIFIER_TYPE,
                "reward_source": reward_source,
            },
            token_usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            test_results=test_results,
        )


def parse_skillflow_execution_trace(
    *,
    run_result: HarborRunResult,
    task_id: str,
    iteration: int,
    duration_sec: float,
) -> ExecutionTrace:
    """Parse one Harbor run result into the project trace contract."""
    return SkillFlowTraceParser(
        run_result=run_result,
        task_id=task_id,
        iteration=iteration,
        duration_sec=duration_sec,
    ).parse()


def _skillflow_task_resources(run_dir: Path) -> tuple[str, ...]:
    resource_names = _skillflow_task_resource_names(run_dir)
    if resource_names:
        return (
            "Task-local resources are available under `environment/skills`: "
            f"{', '.join(resource_names)}.",
        )
    return ("Inspect the task files, environment, tests, and expected outputs directly.",)


def _skillflow_task_resource_names(run_dir: Path) -> tuple[str, ...]:
    skills_dir = run_dir / "environment" / "skills"
    if not skills_dir.exists():
        return ()
    return tuple(
        skill_dir.name
        for skill_dir in sorted(skills_dir.iterdir(), key=lambda path: path.name)
        if skill_dir.is_dir() and (skill_dir / "SKILL.md").is_file()
    )


def _ensure_declared_prebuilt_image(task_dir: Path) -> None:
    image_name = _declared_task_docker_image(task_dir)
    if image_name is None or not image_name.startswith(HARBOR_PREBUILT_IMAGE_PREFIX):
        return
    if _docker_image_exists(image_name):
        return

    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        logger.warning(
            "Prebuilt image %s is missing, but task Dockerfile is absent: %s",
            image_name,
            dockerfile,
        )
        return

    logger.info(
        "Prebuilt image %s is missing; rebuilding from %s",
        image_name,
        dockerfile,
    )
    _ensure_legacy_base_image_alias(dockerfile)
    _run_docker_command(
        [
            "docker",
            "build",
            "--progress=plain",
            "-f",
            str(dockerfile),
            "-t",
            image_name,
            str(dockerfile.parent),
        ],
        failure_message=(
            f"Prebuilt Harbor image `{image_name}` is missing and automatic "
            f"rebuild failed for SkillFlow task workspace {task_dir}"
        ),
    )


def _declared_task_docker_image(task_dir: Path) -> str | None:
    task_toml_path = task_dir / "task.toml"
    if not task_toml_path.is_file():
        return None
    with open(task_toml_path, "rb") as f:
        task_config = tomllib.load(f)
    environment = as_mapping(task_config.get("environment"))
    return as_nonempty_string(environment.get("docker_image"))


def _docker_image_exists(image_name: str) -> bool:
    completed = subprocess.run(
        ["docker", "image", "inspect", image_name],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.returncode == 0


def _ensure_legacy_base_image_alias(dockerfile: Path) -> None:
    if LEGACY_HARBOR_BASE_IMAGE not in _dockerfile_base_images(dockerfile):
        return
    if _docker_image_exists(LEGACY_HARBOR_BASE_IMAGE):
        return
    if not _docker_image_exists(SKILLFLOW_HARBOR_BASE_IMAGE):
        logger.warning(
            "Task Dockerfile uses legacy Harbor base image %s, but local base "
            "image %s is missing. Run `uv run medcoevo build-base-image` first.",
            LEGACY_HARBOR_BASE_IMAGE,
            SKILLFLOW_HARBOR_BASE_IMAGE,
        )
        return
    _run_docker_command(
        [
            "docker",
            "tag",
            SKILLFLOW_HARBOR_BASE_IMAGE,
            LEGACY_HARBOR_BASE_IMAGE,
        ],
        failure_message=(
            f"Failed to tag {SKILLFLOW_HARBOR_BASE_IMAGE} as "
            f"{LEGACY_HARBOR_BASE_IMAGE}"
        ),
    )


def _dockerfile_base_images(dockerfile: Path) -> set[str]:
    images: set[str] = set()
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        match = DOCKERFILE_FROM_IMAGE_RE.match(line)
        if match:
            images.add(match.group("image"))
    return images


def _run_docker_command(command: list[str], *, failure_message: str) -> None:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode == 0:
        if completed.stdout.strip():
            logger.debug("Docker command output: %s", completed.stdout.strip())
        return
    details = completed.stderr.strip() or completed.stdout.strip()
    raise HarborPrebuiltImageMissingError(
        f"{failure_message}. Command: {' '.join(command)}. {details}"
    )


def _skillflow_verifier_contract(task: SkillFlowTask) -> str:
    metadata = as_mapping(task.task_config.get("metadata"))
    verifier = as_mapping(task.task_config.get("verifier"))
    lines = [
        "Success is judged by the SkillFlow verifier for this task.",
        "Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or "
        "expected-output checks.",
        "Run the provided tests or verifier command when practical before finalizing.",
    ]
    if metadata:
        metadata_parts: list[str] = []
        for key, value in sorted(metadata.items()):
            if isinstance(value, list):
                rendered_value = "[" + ", ".join(str(item) for item in value) + "]"
            else:
                rendered_value = str(value)
            metadata_parts.append(f"{key}={rendered_value}")
        lines.append(f"Task metadata: {', '.join(metadata_parts)}.")
    if verifier:
        verifier_parts: list[str] = []
        for key, value in sorted(verifier.items()):
            if isinstance(value, list):
                rendered_value = "[" + ", ".join(str(item) for item in value) + "]"
            else:
                rendered_value = str(value)
            verifier_parts.append(f"{key}={rendered_value}")
        lines.append(f"Verifier config: {', '.join(verifier_parts)}.")
    return "\n".join(lines)


def _is_task_dir(path: Path) -> bool:
    return (path / "instruction.md").is_file() and (path / "task.toml").is_file()


def _is_safe_task_id(task_id: str) -> bool:
    return bool(task_id) and not any(part in {"", ".", ".."} for part in task_id.split("/"))


def _validated_remote_task_ids(task_ids: list[str] | None) -> list[str] | None:
    if task_ids is None:
        return None
    selected_task_ids = _dedupe(task_ids)
    if not selected_task_ids:
        raise SkillFlowSyncError("at least one SkillFlow task ID is required")
    unsafe_task_ids = [
        task_id for task_id in selected_task_ids if not _is_safe_task_id(task_id)
    ]
    if unsafe_task_ids:
        raise SkillFlowSyncError(f"unsafe SkillFlow task IDs: {unsafe_task_ids}")
    missing_family_task_ids = [
        task_id for task_id in selected_task_ids if "/" not in task_id
    ]
    if missing_family_task_ids:
        raise SkillFlowSyncError(
            "remote SkillFlow task IDs must use '<family>/<task>' format: "
            f"{missing_family_task_ids}"
        )
    return selected_task_ids


def _merge_downloaded_test_tasks(*, source_root: Path, destination: Path) -> None:
    if not source_root.exists():
        raise SkillFlowSyncError(
            "SkillFlow task sync did not produce a test_tasks/ directory. "
            "Check the dataset layout or selected task IDs."
        )
    for source_path in sorted(source_root.iterdir(), key=lambda path: path.name):
        target_path = destination / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)


def _dedupe(task_ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for task_id in task_ids:
        cleaned = task_id.strip()
        if cleaned and cleaned not in seen:
            unique.append(cleaned)
            seen.add(cleaned)
    return unique


def _latest_path(paths: Iterable[Path]) -> Path | None:
    candidates = list(paths)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _find_trial_dir(job_dir: Path | None) -> Path | None:
    if job_dir is None:
        return None
    candidates = [
        path
        for path in job_dir.rglob("result.json")
        if path.parent != job_dir and path.parent.is_dir()
    ]
    latest = _latest_path([path.parent for path in candidates])
    if latest is not None:
        return latest
    trial_dirs = [
        path for path in job_dir.rglob("*") if path.is_dir() and path.name == "trial"
    ]
    return _latest_path(trial_dirs)


def _harbor_paths(run_result: HarborRunResult) -> dict[str, str]:
    paths: dict[str, str] = {}
    if run_result.job_dir is not None:
        paths["job"] = str(run_result.job_dir)
    if run_result.trial_dir is not None:
        paths["trial"] = str(run_result.trial_dir)
    return paths


def _load_json_or_empty(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Malformed Harbor JSON artifact: %s", path)
        return {}


def _trial_exception_error_kind(exception_info: Mapping[str, Any]) -> str:
    message = str(exception_info.get("exception_message") or "").lower()
    traceback = str(exception_info.get("exception_traceback") or "").lower()
    detail = f"{message}\n{traceback}"
    if "pull access denied" in detail and "prebuilt" in detail:
        return "missing_prebuilt_image"
    if "docker compose command failed" in detail:
        return "docker_compose_failed"
    return "harbor_trial_exception"


def harbor_run_missing_prebuilt_image(run_result: HarborRunResult) -> bool:
    """Return True when Harbor failed only because a prebuilt image is missing."""
    if run_result.trial_dir is None:
        return False
    trial_result_json = _load_json_or_empty(run_result.trial_dir / "result.json")
    exception_info = as_mapping(trial_result_json.get("exception_info"))
    if exception_info:
        return _trial_exception_error_kind(exception_info) == "missing_prebuilt_image"
    combined_output = f"{run_result.stdout}\n{run_result.stderr}".lower()
    return "pull access denied" in combined_output and "prebuilt" in combined_output


def harbor_missing_prebuilt_image_message(
    run_result: HarborRunResult,
    task_dir: Path,
) -> str:
    """Return an actionable error for missing prebuilt Harbor images."""
    trial_result_json = {}
    if run_result.trial_dir is not None:
        trial_result_json = _load_json_or_empty(run_result.trial_dir / "result.json")
    exception_info = as_mapping(trial_result_json.get("exception_info"))
    image_name = None
    for text in (
        str(exception_info.get("exception_message") or ""),
        str(exception_info.get("exception_traceback") or ""),
        run_result.stdout,
        run_result.stderr,
    ):
        match = _MISSING_PREBUILT_IMAGE_RE.search(text)
        if match is not None:
            image_name = match.group("image").rstrip(",.")
            break
    image_text = f" `{image_name}`" if image_name is not None else ""
    return (
        f"Prebuilt Harbor image{image_text} is missing for SkillFlow task workspace "
        f"{task_dir}. The official SkillFlow quick start requires the base image "
        "`./docker/harbor-cli-base/build.sh`; task-image prebuild is optional. "
        "Run `uv run medcoevo build-base-image` for the required base image. "
        "If this task intentionally declares `[environment].docker_image`, either "
        "remove a stale `docker_image` field so Harbor builds from the task "
        "Dockerfile, or run SkillFlow's optional task prebuilder manually and "
        "re-run after `docker image inspect <image>` succeeds."
    )


_MISSING_PREBUILT_IMAGE_RE = re.compile(
    r"(?:Image\s+|pull access denied for\s+)"
    r"(?P<image>[\w./:-]*prebuilt[\w./:-]*)",
    re.IGNORECASE,
)


def _harbor_metadata(result_json: dict[str, Any]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    agent_info = as_mapping(result_json.get("agent_info"))
    model_info = as_mapping(agent_info.get("model_info"))
    if agent_info.get("name") is not None:
        metadata["agent_info.name"] = str(agent_info["name"])
    if model_info.get("provider") is not None:
        metadata["agent_info.model_provider"] = str(model_info["provider"])
    if model_info.get("name") is not None:
        metadata["agent_info.model_name"] = str(model_info["name"])
    agent_result = as_mapping(result_json.get("agent_result"))
    for key in ("agent", "model", "duration_sec"):
        value = agent_result.get(key)
        if value is not None:
            metadata[f"agent_result.{key}"] = str(value)
    return metadata


async def _terminate_process_tree(proc: asyncio.subprocess.Process) -> None:
    pid = proc.pid
    if pid is None:
        return
    with suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except TimeoutError:
        with suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)
        await proc.wait()
