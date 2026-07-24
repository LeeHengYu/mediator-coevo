"""Materialize LifelongAgentBench rows as local Harbor task packages."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from mediated_coevo.benchmarks.lifelong_agent_bench_templates import (
    DOCKERFILE_TEMPLATE,
    TASK_TOML_TEMPLATE,
    VERIFIER_TEMPLATE,
)

SUPPORTED_FAMILY = "os_interaction"
KNOWN_FAMILIES = ("db_bench", "knowledge_graph", SUPPORTED_FAMILY)
OS_BASE_IMAGE = "lifelong-agent-bench/os-base:ubuntu24.04"
_COMMAND_NAMES = {"bash", "python", "c", "cpp"}


class LifelongAgentBenchImportError(ValueError):
    """Raised when source rows cannot be materialized faithfully."""


class LifelongAgentBenchEnvironmentError(RuntimeError):
    """Raised when the prepared family runtime is unavailable."""


def ensure_os_base_image() -> None:
    """Require the explicitly prepared shared OS image."""
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", OS_BASE_IMAGE],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise LifelongAgentBenchEnvironmentError(
            f"cannot inspect required image {OS_BASE_IMAGE!r}: {exc}"
        ) from exc
    if completed.returncode != 0:
        raise LifelongAgentBenchEnvironmentError(
            f"required image {OS_BASE_IMAGE!r} is missing; run "
            "`medcoevo build-base-image` first"
        )


def load_lifelong_agent_bench_rows(source: Path) -> list[dict[str, Any]]:
    """Load the pinned executable-family JSONL source."""
    if not source.is_file():
        raise LifelongAgentBenchImportError(f"source file does not exist: {source}")
    if source.suffix != ".jsonl":
        raise LifelongAgentBenchImportError("executable families require JSONL")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LifelongAgentBenchImportError(
                f"invalid JSONL at {source}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise LifelongAgentBenchImportError(
                f"expected an object at {source}:{line_number}"
            )
        rows.append(value)
    return rows


def materialize_lifelong_agent_bench(
    *,
    family: str,
    rows: Iterable[Mapping[str, Any]],
    tasks_root: Path,
) -> list[Path]:
    """Materialize one family without exposing its oracle fields to the agent."""
    if family not in KNOWN_FAMILIES:
        raise LifelongAgentBenchImportError(
            f"unknown family {family!r}; choose one of {', '.join(KNOWN_FAMILIES)}"
        )
    if family != SUPPORTED_FAMILY:
        raise LifelongAgentBenchImportError(
            f"family {family!r} is not executable yet: preserving its upstream "
            "environment and verifier requires additional released resources"
        )
    selected_rows = list(rows)
    if not selected_rows:
        raise LifelongAgentBenchImportError("source contains no rows")

    prepared: list[tuple[Path, dict[Path, str]]] = []
    seen_task_ids: set[str] = set()
    for row in selected_rows:
        sample_index = _sample_index(row)
        task_id = f"lab-os-{sample_index}"
        if task_id in seen_task_ids:
            raise LifelongAgentBenchImportError(
                f"duplicate sample_index produces task ID {task_id!r}"
            )
        seen_task_ids.add(task_id)
        destination = tasks_root / family / task_id
        instruction, initialization, evaluation = _parse_os_task(row, sample_index)
        files = _os_task_files(
            task_id=task_id,
            sample_index=sample_index,
            instruction=instruction,
            initialization=initialization,
            evaluation=evaluation,
        )
        if destination.exists() and not _task_files_match(destination, files):
            raise LifelongAgentBenchImportError(
                f"existing task differs from source: {destination}"
            )
        prepared.append((destination, files))

    for destination, files in prepared:
        if not destination.exists():
            _write_task_files(destination, files)
    return [item[0] for item in prepared]


def _parse_os_task(
    row: Mapping[str, Any],
    sample_index: str,
) -> tuple[str, dict[str, str], dict[str, str]]:
    instruction = row.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise LifelongAgentBenchImportError(
            f"os_interaction sample {sample_index} has no instruction"
        )
    initialization = _command_item(
        row.get("initialization_command_item"),
        "initialization_command_item",
        allow_empty=True,
    )
    evaluation_info = _mapping(row.get("evaluation_info"), "evaluation_info")
    evaluation = _command_item(
        evaluation_info.get("evaluation_command_item"),
        "evaluation_info.evaluation_command_item",
    )
    return instruction, initialization, evaluation


def _os_task_files(
    *,
    task_id: str,
    sample_index: str,
    instruction: str,
    initialization: Mapping[str, str],
    evaluation: Mapping[str, str],
) -> dict[Path, str]:
    init_name = f"initialize.{_command_suffix(initialization['command_name'])}"
    evaluation_name = f"evaluate.{_command_suffix(evaluation['command_name'])}"
    return {
        Path("instruction.md"): instruction.rstrip() + "\n",
        Path("task.toml"): _task_toml(
            task_id=task_id,
            sample_index=sample_index,
        ),
        Path("environment/Dockerfile"): _dockerfile(
            init_name,
            initialization["command_name"],
        ),
        Path("environment") / init_name: initialization["script"].rstrip() + "\n",
        Path("tests") / evaluation_name: evaluation["script"].rstrip() + "\n",
        Path("tests/test.sh"): _verifier_script(
            evaluation_name,
            evaluation["command_name"],
        ),
    }


def _task_files_match(destination: Path, expected: Mapping[Path, str]) -> bool:
    actual_paths = {
        path.relative_to(destination)
        for path in destination.rglob("*")
        if path.is_file()
    }
    return actual_paths == set(expected) and all(
        (destination / path).read_text() == content
        for path, content in expected.items()
    )


def _write_task_files(destination: Path, files: Mapping[Path, str]) -> None:
    for relative_path, content in files.items():
        path = destination / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _sample_index(row: Mapping[str, Any]) -> str:
    raw = str(row.get("sample_index", "")).strip()
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", raw).strip("-")
    if not normalized:
        raise LifelongAgentBenchImportError("row has no safe sample_index")
    return normalized


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise LifelongAgentBenchImportError(
                    f"{field} is not a mapping"
                ) from exc
    if not isinstance(value, Mapping):
        raise LifelongAgentBenchImportError(f"{field} is not a mapping")
    return dict(value)


def _command_item(
    value: Any,
    field: str,
    *,
    allow_empty: bool = False,
) -> dict[str, str]:
    command = _mapping(value, field)
    command_name = str(command.get("command_name", "")).lower().rsplit(".", 1)[-1]
    script = command.get("script")
    if command_name not in _COMMAND_NAMES:
        raise LifelongAgentBenchImportError(
            f"{field}.command_name {command_name!r} is unsupported"
        )
    if not isinstance(script, str) or (not allow_empty and not script.strip()):
        raise LifelongAgentBenchImportError(f"{field}.script is empty")
    return {"command_name": command_name, "script": script}


def _command_suffix(command_name: str) -> str:
    return {
        "bash": "sh",
        "python": "py",
        "c": "c",
        "cpp": "cpp",
    }[command_name]


def _command_invocation(path: str, command_name: str) -> str:
    return {
        "bash": f"bash {path}",
        "python": f"python3 {path}",
        "c": f"gcc -o /tmp/lab-command {path} && /tmp/lab-command",
        "cpp": f"g++ -o /tmp/lab-command {path} && /tmp/lab-command",
    }[command_name]


def _dockerfile(init_name: str, command_name: str) -> str:
    invocation = _command_invocation(f"/opt/lab/{init_name}", command_name)
    return DOCKERFILE_TEMPLATE.substitute(
        base_image=OS_BASE_IMAGE,
        init_name=init_name,
        invocation=invocation,
    )


def _verifier_script(evaluation_name: str, command_name: str) -> str:
    invocation = _command_invocation(
        f'"${{SCRIPT_DIR}}/{evaluation_name}"',
        command_name,
    )
    return VERIFIER_TEMPLATE.substitute(invocation=invocation)


def _task_toml(*, task_id: str, sample_index: str) -> str:
    return TASK_TOML_TEMPLATE.substitute(
        task_id=task_id,
        sample_index=sample_index,
    )
