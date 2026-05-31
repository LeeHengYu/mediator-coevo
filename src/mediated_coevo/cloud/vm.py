"""GCP VM support for running SkillFlow Harbor jobs remotely.

The local experiment remains the control plane. For each SkillFlow execution,
only the prepared task workspace is copied to the VM, Harbor runs there, and the
Harbor job artifacts are copied back for normal local parsing.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import shlex
import tarfile
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

from mediated_coevo.benchmarks import (
    HarborRunResult,
    HarborTimeoutError,
)


DEFAULT_REMOTE_DIR = "/tmp/mediator-coevo"
REMOTE_HARBOR_TIMEOUT_BUFFER_SEC = 120

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SECRET_RESOURCE_RE = re.compile(
    r"^projects/(?P<project>[^/]+)/secrets/(?P<secret>[^/]+)/versions/"
    r"(?P<version>[^/]+)$"
)


class CloudVMConfigError(ValueError):
    """Raised when local cloud VM configuration is incomplete."""


@dataclass(frozen=True)
class SecretResource:
    """A resolved Google Secret Manager version resource."""

    project: str
    secret: str
    version: str


@dataclass(frozen=True)
class GCPVMConfig:
    """GCP settings needed to run Harbor on an existing VM."""

    project_id: str
    region: str
    zone: str
    vm_name: str
    remote_dir: str
    openrouter_secret: str
    service_account: str | None = None


@dataclass(frozen=True)
class CompletedRemoteCommand:
    """Captured result from one local gcloud command."""

    returncode: int
    stdout: str
    stderr: str


AsyncCommandRunner = Callable[
    [Sequence[str], float],
    Awaitable[CompletedRemoteCommand],
]


def parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple dotenv file without consulting the process environment."""
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise CloudVMConfigError(
                f"{path}:{line_number}: expected KEY=VALUE dotenv syntax"
            )
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not _ENV_KEY_RE.fullmatch(key):
            raise CloudVMConfigError(f"{path}:{line_number}: invalid key {key!r}")

        value = _parse_env_value(raw_value.strip())
        values[key] = _ENV_REF_RE.sub(
            lambda match: values.get(match.group(1), match.group(0)),
            value,
        )
    return values


def load_vm_config(env_path: Path) -> GCPVMConfig:
    """Load direct-VM Harbor settings from a dotenv file."""
    values = parse_dotenv(env_path)
    missing: list[str] = []

    project_id = _require_value(
        values,
        ("GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT"),
        missing,
    )
    zone = _require_value(
        values,
        ("GCP_ZONE", "GOOGLE_CLOUD_ZONE"),
        missing,
    )
    vm_name = _require_value(
        values,
        ("GCP_VM_NAME", "GCP_INSTANCE_NAME", "GCP_COMPUTE_INSTANCE"),
        missing,
    )
    region = _first_value(values, ("GCP_REGION", "GOOGLE_CLOUD_REGION"))
    if region is None and zone is not None:
        region = zone.rsplit("-", 1)[0]

    openrouter_secret = _require_value(
        values,
        (
            "GCP_OPENROUTER_SECRET",
            "GCP_OPENROUTER_SECRET_RESOURCE",
            "OPENROUTER_API_KEY_SECRET",
            "OPENROUTER_SECRET",
        ),
        missing,
    )

    if missing:
        joined = ", ".join(missing)
        raise CloudVMConfigError(f"missing required cloud VM env keys: {joined}")

    assert project_id is not None
    assert region is not None
    assert zone is not None
    assert vm_name is not None
    assert openrouter_secret is not None

    return GCPVMConfig(
        project_id=project_id,
        region=region,
        zone=zone,
        vm_name=vm_name,
        remote_dir=_first_value(values, ("GCP_REMOTE_DIR",)) or DEFAULT_REMOTE_DIR,
        openrouter_secret=openrouter_secret,
        service_account=_first_value(
            values,
            ("GCP_SERVICE_ACCOUNT", "GCP_SA", "GCP_VM_SERVICE_ACCOUNT"),
        ),
    )


def parse_secret_resource(raw: str, *, default_project: str) -> SecretResource:
    """Resolve a secret resource path or bare secret name."""
    value = raw.strip()
    if not value:
        raise CloudVMConfigError("OpenRouter secret resource is empty")

    match = _SECRET_RESOURCE_RE.fullmatch(value)
    if match is not None:
        return SecretResource(
            project=match.group("project"),
            secret=match.group("secret"),
            version=match.group("version"),
        )
    return SecretResource(project=default_project, secret=value, version="latest")


def create_task_archive(task_dir: Path, archive_path: Path) -> Path:
    """Create a gzipped archive containing only one prepared task workspace."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    task_dir = task_dir.resolve()
    archive_path = archive_path.resolve()

    with tarfile.open(archive_path, "w:gz") as tar:
        for source_path, relative_path in _iter_task_files(task_dir):
            if source_path.resolve() == archive_path:
                continue
            tar.add(source_path, arcname=relative_path.as_posix(), recursive=False)
    return archive_path


def build_remote_harbor_script(
    *,
    config: GCPVMConfig,
    remote_run_dir: str,
    agent_name: str,
    model: str,
    harbor_timeout_sec: float,
    agent_setup_timeout_multiplier: float | None,
) -> str:
    """Build the bash script executed on the VM for one Harbor run."""
    secret = parse_secret_resource(
        config.openrouter_secret,
        default_project=config.project_id,
    )
    secret_command = shlex.join(
        [
            "gcloud",
            "secrets",
            "versions",
            "access",
            secret.version,
            "--secret",
            secret.secret,
            "--project",
            secret.project,
        ]
    )
    harbor_command = [
        "harbor",
        "run",
        "-p",
        "$TASK_DIR",
        "-a",
        agent_name,
        "-m",
        model,
        "-o",
        "$JOBS_DIR",
    ]
    if agent_setup_timeout_multiplier is not None:
        harbor_command.extend(
            [
                "--agent-setup-timeout-multiplier",
                str(agent_setup_timeout_multiplier),
            ]
        )
    quoted_harbor_command = _quote_command_preserving_vars(
        harbor_command,
        variable_args={"$TASK_DIR", "$JOBS_DIR"},
    )

    return "\n".join(
        [
            "set -euo pipefail",
            f"RUN_DIR={_remote_shell_path(remote_run_dir)}",
            'TASK_TAR="$RUN_DIR/task.tar.gz"',
            'TASK_DIR="$RUN_DIR/task"',
            'JOBS_DIR="$RUN_DIR/jobs"',
            'RESULTS_TAR="$RUN_DIR/results.tar.gz"',
            f"HARBOR_TIMEOUT_SEC={math.ceil(harbor_timeout_sec)}",
            'mkdir -p "$TASK_DIR" "$JOBS_DIR"',
            'tar -xzf "$TASK_TAR" -C "$TASK_DIR"',
            f'export OPENROUTER_API_KEY="$({secret_command})"',
            "set +e",
            (
                'timeout --kill-after=30s "${HARBOR_TIMEOUT_SEC}s" '
                f'{quoted_harbor_command} > "$RUN_DIR/stdout.txt" '
                '2> "$RUN_DIR/stderr.txt"'
            ),
            "returncode=$?",
            "set -e",
            'printf "%s\\n" "$returncode" > "$RUN_DIR/returncode.txt"',
            (
                'tar -C "$RUN_DIR" -czf "$RESULTS_TAR" '
                "stdout.txt stderr.txt returncode.txt jobs"
            ),
        ]
    )


class RemoteHarborRunner:
    """Run SkillFlow Harbor jobs on a configured GCP VM."""

    def __init__(
        self,
        *,
        config: GCPVMConfig,
        agent_name: str,
        jobs_dir: Path,
        timeout_sec: float = 1800.0,
        agent_setup_timeout_multiplier: float | None = None,
        command_runner: AsyncCommandRunner | None = None,
    ) -> None:
        self.config = config
        self.agent_name = agent_name
        self.jobs_dir = jobs_dir
        self.timeout_sec = timeout_sec
        self.agent_setup_timeout_multiplier = agent_setup_timeout_multiplier
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._command_runner = command_runner or _run_local_command

    async def run(self, task_dir: Path, model: str) -> HarborRunResult:
        run_id = _new_remote_run_id()
        local_result_dir = self.jobs_dir / run_id
        local_result_dir.mkdir(parents=True, exist_ok=True)
        remote_run_dir = _remote_join(self.config.remote_dir, "harbor-runs", run_id)
        remote_task_tar = _remote_join(remote_run_dir, "task.tar.gz")
        remote_results_tar = _remote_join(remote_run_dir, "results.tar.gz")

        with tempfile.TemporaryDirectory(prefix="medcoevo-harbor-") as temp_dir:
            temp_root = Path(temp_dir)
            task_archive = create_task_archive(task_dir, temp_root / "task.tar.gz")
            results_archive = temp_root / "results.tar.gz"

            await self._run_checked(
                _ssh_command(
                    self.config,
                    "mkdir -p " + _remote_shell_path(remote_run_dir),
                )
            )
            await self._run_checked(
                _scp_command(
                    self.config,
                    str(task_archive),
                    f"{self.config.vm_name}:{remote_task_tar}",
                )
            )
            await self._run_checked(
                _ssh_command(
                    self.config,
                    "bash -lc "
                    + shlex.quote(
                        build_remote_harbor_script(
                            config=self.config,
                            remote_run_dir=remote_run_dir,
                            agent_name=self.agent_name,
                            model=model,
                            harbor_timeout_sec=self.timeout_sec,
                            agent_setup_timeout_multiplier=(
                                self.agent_setup_timeout_multiplier
                            ),
                        )
                    ),
                ),
                timeout_sec=self.timeout_sec + REMOTE_HARBOR_TIMEOUT_BUFFER_SEC,
            )
            await self._run_checked(
                _scp_command(
                    self.config,
                    f"{self.config.vm_name}:{remote_results_tar}",
                    str(results_archive),
                )
            )
            _extract_results_archive(results_archive, local_result_dir)

        return _harbor_run_result_from_remote_dir(local_result_dir)

    async def _run_checked(
        self,
        command: Sequence[str],
        *,
        timeout_sec: float | None = None,
    ) -> CompletedRemoteCommand:
        result = await self._command_runner(command, timeout_sec or self.timeout_sec)
        if result.returncode != 0:
            detail = _command_failure_detail(command, result)
            raise OSError(detail)
        return result


async def _run_local_command(
    command: Sequence[str],
    timeout_sec: float,
) -> CompletedRemoteCommand:
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise OSError(f"command not found: {command[0]}") from exc

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_sec,
        )
    except TimeoutError as exc:
        proc.terminate()
        await proc.wait()
        raise HarborTimeoutError(
            f"remote Harbor command exceeded {timeout_sec}s: {shlex.join(command)}"
        ) from exc

    return CompletedRemoteCommand(
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
    )


def _parse_env_value(raw_value: str) -> str:
    if not raw_value:
        return ""
    lexer = shlex.shlex(raw_value, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    parts = list(lexer)
    if not parts:
        return ""
    return " ".join(parts)


def _first_value(values: Mapping[str, str], names: Sequence[str]) -> str | None:
    for name in names:
        value = values.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _require_value(
    values: Mapping[str, str],
    names: Sequence[str],
    missing: list[str],
) -> str | None:
    value = _first_value(values, names)
    if value is None:
        missing.append(" or ".join(names))
    return value


def _iter_task_files(task_dir: Path) -> Iterator[tuple[Path, PurePosixPath]]:
    for dirpath, dirnames, filenames in os.walk(task_dir):
        directory = Path(dirpath)
        relative_dir = _relative_posix(directory, task_dir)
        dirnames[:] = sorted(
            dirname
            for dirname in dirnames
            if not _exclude_task_path(relative_dir / dirname)
        )
        for filename in sorted(filenames):
            path = directory / filename
            relative_path = _relative_posix(path, task_dir)
            if _exclude_task_path(relative_path):
                continue
            yield path, relative_path


def _exclude_task_path(relative_path: PurePosixPath) -> bool:
    parts = relative_path.parts
    return any(part == "__pycache__" for part in parts) or (
        bool(parts)
        and (parts[-1] == ".DS_Store" or parts[-1].endswith(".pyc"))
    )


def _relative_posix(path: Path, root: Path) -> PurePosixPath:
    relative = path.relative_to(root)
    if str(relative) == ".":
        return PurePosixPath()
    return PurePosixPath(relative.as_posix())


def _quote_command_preserving_vars(
    command: Sequence[str],
    *,
    variable_args: set[str],
) -> str:
    return " ".join(
        arg if arg in variable_args else shlex.quote(arg)
        for arg in command
    )


def _ssh_command(config: GCPVMConfig, remote_command: str) -> list[str]:
    return _gcloud_compute_command(
        config,
        "ssh",
        config.vm_name,
        "--command",
        remote_command,
    )


def _scp_command(config: GCPVMConfig, source: str, destination: str) -> list[str]:
    return _gcloud_compute_command(config, "scp", source, destination)


def _gcloud_compute_command(
    config: GCPVMConfig,
    subcommand: str,
    *args: str,
) -> list[str]:
    return [
        "gcloud",
        "compute",
        subcommand,
        *args,
        "--project",
        config.project_id,
        "--zone",
        config.zone,
    ]


def _remote_join(*parts: str) -> str:
    first, *rest = parts
    path = first.rstrip("/")
    for part in rest:
        path += "/" + part.strip("/")
    return path


def _remote_shell_path(path: str) -> str:
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def _new_remote_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"remote-harbor-{timestamp}-{uuid.uuid4().hex[:8]}"


def _extract_results_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(destination, filter="data")


def _harbor_run_result_from_remote_dir(remote_result_dir: Path) -> HarborRunResult:
    stdout = _read_text(remote_result_dir / "stdout.txt")
    stderr = _read_text(remote_result_dir / "stderr.txt")
    returncode = _read_returncode(remote_result_dir / "returncode.txt")
    jobs_dir = remote_result_dir / "jobs"
    job_dirs: list[Path] = []
    if jobs_dir.exists():
        job_dirs = [path for path in jobs_dir.iterdir() if path.is_dir()]
    job_dir = _latest_path(job_dirs)
    trial_dir = _find_trial_dir(job_dir)
    return HarborRunResult(
        job_dir=job_dir,
        trial_dir=trial_dir,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _read_returncode(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return -1


def _latest_path(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def _find_trial_dir(job_dir: Path | None) -> Path | None:
    if job_dir is None:
        return None
    candidates = [
        path
        for path in job_dir.iterdir()
        if path.is_dir() and (path / "result.json").exists()
    ]
    return _latest_path(candidates)


def _command_failure_detail(
    command: Sequence[str],
    result: CompletedRemoteCommand,
) -> str:
    output = result.stderr.strip() or result.stdout.strip()
    if len(output) > 1000:
        output = output[-1000:]
    detail = f"command exited {result.returncode}: {shlex.join(command)}"
    if output:
        detail += f"\n{output}"
    return detail


__all__ = [
    "CloudVMConfigError",
    "CompletedRemoteCommand",
    "DEFAULT_REMOTE_DIR",
    "GCPVMConfig",
    "REMOTE_HARBOR_TIMEOUT_BUFFER_SEC",
    "RemoteHarborRunner",
    "SecretResource",
    "build_remote_harbor_script",
    "create_task_archive",
    "load_vm_config",
    "parse_dotenv",
    "parse_secret_resource",
]
