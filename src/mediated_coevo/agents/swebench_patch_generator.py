"""SWE-bench patch generation backends."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from mediated_coevo.benchmarks import swebench as swebench_helpers

if TYPE_CHECKING:
    from mediated_coevo.llm.client import LLMClient

SANDBOX_WORKSPACE_HINT = "/workspace"


@dataclass(frozen=True, slots=True)
class SWEbenchPatchGeneration:
    """Normalized SWE-bench patch generation result."""

    raw_response: str
    model_patch: str
    normalization_notes: str
    input_tokens: int = 0
    output_tokens: int = 0
    artifacts: dict[str, str] = field(default_factory=dict)


class SWEbenchPatchGenerator(Protocol):
    """Backend interface for generating a SWE-bench model patch."""

    async def generate_patch(
        self,
        *,
        task: swebench_helpers.SWEbenchTask,
        workspace: Path,
        planner_instruction: str,
        executor_skill_text: str | None,
        injected_skill_name: str,
    ) -> SWEbenchPatchGeneration:
        """Return a normalized patch for one prepared SWE-bench workspace."""
        ...


@dataclass(frozen=True, slots=True)
class SWEbenchLocalSandbox:
    """Disposable local copy of a prepared SWE-bench checkout."""

    source_workspace: Path
    sandbox: Path
    artifact_dir: Path
    artifacts: dict[str, str]


class LLMSWEbenchPatchGenerator:
    """Generate a SWE-bench patch directly from the configured executor model."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def generate_patch(
        self,
        *,
        task: swebench_helpers.SWEbenchTask,
        workspace: Path,
        planner_instruction: str,
        executor_skill_text: str | None,
        injected_skill_name: str,
    ) -> SWEbenchPatchGeneration:
        response = await self._llm_client.complete(
            messages=_generation_messages(
                task=task,
                workspace=workspace,
                planner_instruction=planner_instruction,
                executor_skill_text=executor_skill_text,
                injected_skill_name=injected_skill_name,
            ),
            temperature=0.0,
            max_tokens=4096,
        )
        raw_response = response["content"]
        model_patch, normalization_notes = normalize_swebench_model_patch(raw_response)
        return SWEbenchPatchGeneration(
            raw_response=raw_response,
            model_patch=model_patch,
            normalization_notes=normalization_notes,
            input_tokens=response["input_tokens"],
            output_tokens=response["output_tokens"],
        )


def normalize_swebench_model_patch(raw_response: str) -> tuple[str, str]:
    """Extract a unified diff from an LLM response.

    Non-diff output intentionally normalizes to an empty patch so the official
    SWE-bench harness can score the prediction as unresolved.
    """
    response = raw_response.strip()
    if not response:
        return "", "LLM response was empty; using an empty model_patch."

    fenced_diff = _extract_fenced_diff(response)
    if fenced_diff is not None:
        return _normalized_patch_result(fenced_diff, "Extracted fenced diff block.")

    inline_diff = _extract_inline_diff(response)
    if inline_diff is not None:
        return _normalized_patch_result(inline_diff, "Extracted inline unified diff.")

    return "", "No unified diff found in LLM response; using an empty model_patch."


def _generation_messages(
    *,
    task: swebench_helpers.SWEbenchTask,
    workspace: Path,
    planner_instruction: str,
    executor_skill_text: str | None,
    injected_skill_name: str,
) -> list[dict[str, str]]:
    user_content = swebench_helpers.build_swebench_executor_envelope(
        task=task,
        workspace=workspace,
        planner_instruction=planner_instruction,
        executor_policy=executor_skill_text,
    )
    return [
        {
            "role": "system",
            "content": (
                "You are the Executor for a SWE-bench task. Generate the minimal "
                "repository patch needed to resolve the issue. Follow the active "
                f"{injected_skill_name} policy when it is supplied. Output only a "
                "valid git-style unified diff with `diff --git` file headers. "
                "Do not include commit hashes, metadata, or any text outside "
                "the patch."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def _extract_fenced_diff(response: str) -> str | None:
    lines = response.splitlines()
    in_fence = False
    block_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                candidate = "\n".join(block_lines).strip()
                if _looks_like_diff(candidate):
                    return candidate
                in_fence = False
                block_lines = []
            else:
                in_fence = True
                block_lines = []
            continue
        if in_fence:
            block_lines.append(line)
    return None


def _extract_inline_diff(response: str) -> str | None:
    lines = response.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(("diff --git ", "--- ")):
            candidate = "\n".join(lines[index:]).strip()
            if _looks_like_diff(candidate):
                return candidate
    return None


def _looks_like_diff(text: str) -> bool:
    return text.startswith("diff --git ") or (
        "\n--- " in f"\n{text}" and "\n+++ " in f"\n{text}" and "\n@@ " in f"\n{text}"
    )


def _normalized_patch_result(diff_text: str, base_note: str) -> tuple[str, str]:
    normalized_diff, hunk_note = _normalize_unified_diff(diff_text)
    normalized_diff, header_note = _normalize_git_style_file_headers(normalized_diff)
    return (
        _ensure_trailing_newline(normalized_diff),
        _normalization_notes(base_note, hunk_note, header_note),
    )


def _normalize_git_style_file_headers(diff_text: str) -> tuple[str, str | None]:
    lines = diff_text.splitlines()
    normalized_lines: list[str] = []
    inserted_git_headers = 0
    rewritten_file_headers = 0
    current_section_has_git_header = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git "):
            current_section_has_git_header = True
            normalized_lines.append(line)
            index += 1
            continue

        if (
            line.startswith("--- ")
            and index + 1 < len(lines)
            and lines[index + 1].startswith("+++ ")
        ):
            old_path = line[4:].strip()
            new_path = lines[index + 1][4:].strip()
            normalized_old_path = _canonical_patch_header_path(old_path, "a")
            normalized_new_path = _canonical_patch_header_path(new_path, "b")
            if normalized_old_path != old_path or normalized_new_path != new_path:
                rewritten_file_headers += 1
            if not current_section_has_git_header:
                normalized_lines.append(
                    "diff --git "
                    f"{_diff_git_path(normalized_old_path, normalized_new_path, 'a')} "
                    f"{_diff_git_path(normalized_old_path, normalized_new_path, 'b')}"
                )
                inserted_git_headers += 1
            normalized_lines.append(f"--- {normalized_old_path}")
            normalized_lines.append(f"+++ {normalized_new_path}")
            current_section_has_git_header = False
            index += 2
            continue

        normalized_lines.append(line)
        index += 1

    notes = []
    if inserted_git_headers:
        notes.append(f"Inserted git diff headers for {inserted_git_headers} file(s).")
    if rewritten_file_headers:
        notes.append(
            f"Normalized file headers to a/ and b/ prefixes for "
            f"{rewritten_file_headers} file(s)."
        )
    return "\n".join(normalized_lines), " ".join(notes) or None


def _canonical_patch_header_path(path: str, side_prefix: str) -> str:
    if path == "/dev/null":
        return path
    normalized_path = path.removeprefix("./")
    if normalized_path.startswith(("a/", "b/")):
        normalized_path = normalized_path[2:]
    return f"{side_prefix}/{normalized_path}"


def _diff_git_path(old_path: str, new_path: str, side_prefix: str) -> str:
    path = new_path if old_path == "/dev/null" else old_path
    if new_path == "/dev/null":
        path = old_path
    if path.startswith(("a/", "b/")):
        return f"{side_prefix}/{path[2:]}"
    return f"{side_prefix}/{path}"


_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<section>.*)$"
)


def _normalize_unified_diff(diff_text: str) -> tuple[str, str | None]:
    lines = diff_text.strip().splitlines()
    normalized_lines: list[str] = []
    recomputed_hunks = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        match = _HUNK_HEADER_RE.match(line)
        if match is None:
            normalized_lines.append(line)
            index += 1
            continue

        body_lines: list[str] = []
        index += 1
        while index < len(lines) and not _starts_next_diff_section(lines, index):
            body_lines.append(lines[index])
            index += 1

        old_count, new_count = _count_hunk_body_lines(body_lines)
        original_old_count = int(match.group("old_count") or "1")
        original_new_count = int(match.group("new_count") or "1")
        if old_count != original_old_count or new_count != original_new_count:
            recomputed_hunks += 1

        normalized_lines.append(
            "@@ -{old_start},{old_count} +{new_start},{new_count} @@{section}".format(
                old_start=match.group("old_start"),
                old_count=old_count,
                new_start=match.group("new_start"),
                new_count=new_count,
                section=match.group("section"),
            )
        )
        normalized_lines.extend(body_lines)

    note = None
    if recomputed_hunks:
        note = f"Recomputed line counts for {recomputed_hunks} hunk(s)."
    return "\n".join(normalized_lines), note


def _starts_next_diff_section(lines: list[str], index: int) -> bool:
    line = lines[index]
    if _HUNK_HEADER_RE.match(line) or line.startswith("diff --git "):
        return True
    return (
        line.startswith("--- ")
        and index + 1 < len(lines)
        and lines[index + 1].startswith("+++ ")
    )


def _count_hunk_body_lines(lines: list[str]) -> tuple[int, int]:
    old_count = 0
    new_count = 0
    for line in lines:
        if line.startswith("\\ No newline at end of file"):
            continue
        if line.startswith("+"):
            new_count += 1
        elif line.startswith("-"):
            old_count += 1
        else:
            old_count += 1
            new_count += 1
    return old_count, new_count


def _normalization_notes(base_note: str, *extra_notes: str | None) -> str:
    notes = [base_note]
    notes.extend(note for note in extra_notes if note)
    return " ".join(notes)


def _ensure_trailing_newline(text: str) -> str:
    if text.endswith("\n"):
        return text
    return f"{text}\n"


def _capture_workspace_diff(workspace: Path) -> str:
    _run_checked(
        ["git", "add", "--intent-to-add", "."],
        timeout_sec=60.0,
        cwd=workspace,
    )
    return _run_checked(
        [
            "git",
            "diff",
            "--binary",
            "HEAD",
            "--",
            ".",
            ":!instruction.md",
            ":!environment/skills/**",
        ],
        timeout_sec=60.0,
        cwd=workspace,
    ).stdout


def prepare_local_swebench_sandbox(workspace: Path) -> SWEbenchLocalSandbox:
    """Copy a prepared SWE-bench checkout into a disposable local sandbox."""
    run_id = uuid.uuid4().hex[:8]
    sandbox = workspace.parent / f"{workspace.name}-sandbox-{run_id}"
    artifact_dir = workspace.parent / f"{workspace.name}-sandbox-artifacts-{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(workspace, sandbox, symlinks=True)

    artifacts = _sandbox_artifacts(artifact_dir, sandbox)
    metadata = _sandbox_metadata(
        source_workspace=workspace,
        sandbox=sandbox,
        artifacts=artifacts,
    )
    validation_error = validate_swebench_sandbox(sandbox)
    if validation_error is not None:
        _write_text(Path(artifacts["sandbox_stderr"]), validation_error)
    _write_text(
        Path(artifacts["sandbox_metadata"]),
        json.dumps(metadata, indent=2, sort_keys=True),
    )
    return SWEbenchLocalSandbox(
        source_workspace=workspace,
        sandbox=sandbox,
        artifact_dir=artifact_dir,
        artifacts=artifacts,
    )


def capture_swebench_sandbox_diff(sandbox: Path) -> str:
    """Capture the model patch from a local SWE-bench sandbox."""
    validation_error = validate_swebench_sandbox(sandbox)
    if validation_error is not None:
        raise RuntimeError(validation_error)
    return _capture_workspace_diff(sandbox)


def validate_swebench_sandbox(sandbox: Path) -> str | None:
    if not sandbox.exists():
        return f"SWE-bench sandbox does not exist: {sandbox}"
    if not (sandbox / ".git").exists():
        return f"SWE-bench sandbox is missing .git: {sandbox}"
    if not (sandbox / "instruction.md").exists():
        return f"SWE-bench sandbox is missing instruction.md: {sandbox}"
    return None


def _sandbox_artifacts(artifact_dir: Path, sandbox: Path) -> dict[str, str]:
    return {
        "sandbox_workspace": str(sandbox),
        "sandbox_stdout": str(artifact_dir / "stdout.txt"),
        "sandbox_stderr": str(artifact_dir / "stderr.txt"),
        "sandbox_metadata": str(artifact_dir / "metadata.json"),
        "sandbox_model_patch": str(artifact_dir / "model.patch"),
    }


def _sandbox_metadata(
    *,
    source_workspace: Path,
    sandbox: Path,
    artifacts: dict[str, str],
) -> dict[str, object]:
    return {
        "source_checkout": str(source_workspace),
        "sandbox_checkout": str(sandbox),
        "workspace_hint": SANDBOX_WORKSPACE_HINT,
        "git_present": (sandbox / ".git").exists(),
        "instruction_present": (sandbox / "instruction.md").exists(),
        "stdout": artifacts["sandbox_stdout"],
        "stderr": artifacts["sandbox_stderr"],
        "model_patch": artifacts["sandbox_model_patch"],
    }


def _run_checked(
    command: list[str],
    *,
    timeout_sec: float,
    cwd: Path | None,
) -> subprocess.CompletedProcess[str]:
    completed = _run(command, timeout_sec=timeout_sec, cwd=cwd)
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed with exit code "
            f"{completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    return completed


def _run(
    command: list[str],
    *,
    timeout_sec: float,
    cwd: Path | None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"command not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"command timed out after {timeout_sec}s: {' '.join(command)}"
        ) from exc


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
