"""Independent offline agent for learning and publishing harness overlays."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from difflib import unified_diff
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypedDict

from mediated_coevo.cli.experiment import PROJECT_ROOT
from mediated_coevo.cli.harness_registry import _publish_promoted_harness

Decision = Literal["HOLD", "ROLLBACK", "TARGETED_UPDATE"]
CheckName = Literal["pytest", "ruff", "mypy"]


class HLAgentResult(TypedDict):
    """Structured result returned to the infrastructure-owned episode loop."""

    response: str
    decision: dict[str, Any]
    published_update: str | None


_UPDATE_RE = re.compile(r"update_(\d{4,})")
_ANCHORS = (
    Path("src/mediated_coevo/diffusion/task_graph_agent.py"),
    Path("src/mediated_coevo/diffusion/policy_agent.py"),
)
_HARNESS_FILES = {
    *_ANCHORS,
    Path("src/mediated_coevo/diffusion/langchain_runtime.py"),
    Path("src/mediated_coevo/diffusion/renderer.py"),
}
_TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".py", ".toml", ".txt"}


class HLWorkspace:
    """Constrained repository tools for one offline HL-agent invocation."""

    def __init__(
        self,
        *,
        campaign: str,
        families: tuple[str, ...],
        project_root: Path = PROJECT_ROOT,
        source_sequence: Path | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", campaign):
            raise ValueError("campaign must be one portable path component")
        if len(families) != 4:
            raise ValueError("HL agent requires exactly four target families")

        self.project_root = project_root.resolve()
        self.campaign = campaign
        self.families = families
        self.campaign_root = self.project_root / "data" / "experiments" / campaign
        self.source_sequence = (
            self._sequence_dir(source_sequence.as_posix())
            if source_sequence is not None
            else None
        )
        self._decision: dict[str, Any] | None = None
        self._staging_dir: Path | None = None
        self._published_update: Path | None = None

    def tools(self) -> list[Any]:
        """Return the complete, deliberately narrow HL tool surface."""
        return [
            self.inspect_campaign,
            self.list_project_files,
            self.read_project_file,
            self.inspect_sequence,
            self.record_decision,
            self.prepare_update,
            self.read_staged_file,
            self.write_staged_file,
            self.inspect_staged_diff,
            self.run_focused_checks,
            self.publish_update,
        ]

    def inspect_campaign(self) -> dict[str, Any]:
        """Summarize campaign updates, promotion state, and the direct source sequence."""
        updates = [
            path.name
            for path in sorted(self.campaign_root.glob("update_*"))
            if path.is_dir() and _UPDATE_RE.fullmatch(path.name)
        ]
        channel_path = (
            self.campaign_root / "channels" / "promoted_harness.json"
        )
        promoted = _read_json(channel_path) if channel_path.is_file() else None
        decisions = [
            path.relative_to(self.project_root).as_posix()
            for path in sorted((self.campaign_root / "decisions").glob("*.json"))
        ]
        return {
            "campaign": self.campaign,
            "families": self.families,
            "source_sequence": self._relative(self.source_sequence),
            "updates": updates,
            "promoted_harness": promoted,
            "decision_records": decisions,
        }

    def list_project_files(
        self,
        relative_directory: str,
        pattern: str = "**/*",
        limit: int = 200,
    ) -> dict[str, Any]:
        """List readable evidence or harness files below an allowed project directory."""
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        glob_path = PurePosixPath(pattern)
        if glob_path.is_absolute() or ".." in glob_path.parts:
            raise ValueError("pattern must stay below relative_directory")
        directory = self._readable_path(relative_directory)
        if not directory.is_dir():
            raise ValueError(f"not a directory: {relative_directory}")

        files: list[dict[str, Any]] = []
        for path in sorted(directory.glob(pattern)):
            if not path.is_file() or self._is_cache_path(path):
                continue
            self._assert_readable(path)
            files.append(
                {
                    "path": path.relative_to(self.project_root).as_posix(),
                    "bytes": path.stat().st_size,
                }
            )
            if len(files) == limit:
                break
        return {"files": files, "limit": limit, "truncated": len(files) == limit}

    def read_project_file(
        self,
        relative_path: str,
        offset: int = 0,
        max_chars: int = 20_000,
    ) -> dict[str, Any]:
        """Read a bounded slice of an allowed text evidence or harness file."""
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if not 1 <= max_chars <= 50_000:
            raise ValueError("max_chars must be between 1 and 50000")
        path = self._readable_path(relative_path)
        if not path.is_file():
            raise ValueError(f"not a file: {relative_path}")
        if path.suffix not in _TEXT_SUFFIXES or self._is_cache_path(path):
            raise ValueError(f"not an allowed text file: {relative_path}")
        text = path.read_text(errors="replace")
        end = min(len(text), offset + max_chars)
        return {
            "path": path.relative_to(self.project_root).as_posix(),
            "content": text[offset:end],
            "offset": offset,
            "next_offset": end if end < len(text) else None,
            "total_chars": len(text),
        }

    def inspect_sequence(self, sequence_path: str) -> dict[str, Any]:
        """Return compact K-iteration rewards and harness provenance for one sequence."""
        sequence_dir = self._sequence_dir(sequence_path)
        active_path = sequence_dir / "harnesses" / "active_harness.json"
        iterations: list[dict[str, Any]] = []
        for iteration_dir in sorted(sequence_dir.glob("iter-*")):
            if not iteration_dir.is_dir():
                continue
            spec_path = iteration_dir / "sequence_spec.json"
            spec = _read_json(spec_path) if spec_path.is_file() else {}
            samples: list[dict[str, Any]] = []
            for result_path in sorted(
                (iteration_dir / "samples").glob("*/sample_result.json")
            ):
                result = _read_json(result_path)
                rewards = result.get("rewards")
                samples.append(
                    {
                        "sample": result_path.parent.name,
                        "rewards": rewards,
                        "result_path": self._relative(result_path),
                    }
                )
            iterations.append(
                {
                    "iteration": iteration_dir.name,
                    "seed": spec.get("policy_seed"),
                    "task_set_id": spec.get("task_set_id"),
                    "task_ids": [
                        task.get("task_id")
                        for task in spec.get("tasks", [])
                        if isinstance(task, dict)
                    ],
                    "samples": samples,
                }
            )
        return {
            "sequence": self._relative(sequence_dir),
            "active_harness": (
                _read_json(active_path) if active_path.is_file() else None
            ),
            "iterations": iterations,
        }

    def record_decision(
        self,
        decision: Decision,
        source_sequence: str,
        parent_update: str,
        evidence: list[str],
        protected_behavior: list[str],
    ) -> dict[str, Any]:
        """Record the required HOLD, ROLLBACK, or TARGETED_UPDATE decision."""
        if self._decision is not None:
            raise ValueError("this invocation already recorded a decision")
        if decision not in {"HOLD", "ROLLBACK", "TARGETED_UPDATE"}:
            raise ValueError(f"invalid HL decision: {decision}")
        source = self._sequence_dir(source_sequence)
        parent = self._parent_overlay(parent_update)
        now = datetime.now()
        record = {
            "schema_version": 1,
            "campaign": self.campaign,
            "decision": decision,
            "source_sequence": self._relative(source),
            "parent_update": parent_update,
            "parent_overlay": self._relative(parent),
            "evidence": evidence,
            "protected_behavior": protected_behavior,
            "created_at": now.isoformat(timespec="seconds"),
        }
        decision_dir = self.campaign_root / "decisions"
        decision_dir.mkdir(parents=True, exist_ok=True)
        record_path = decision_dir / (
            f"{now.strftime('%Y%m%d-%H%M%S-%f')}-{decision.lower()}.json"
        )
        _write_json(record_path, record)
        record["record_path"] = self._relative(record_path)
        self._decision = record
        return record

    def prepare_update(self) -> dict[str, Any]:
        """Create an invocation-owned cumulative staging overlay from the chosen parent."""
        if self._decision is None:
            raise ValueError("record a decision before preparing an update")
        if self._decision["decision"] == "HOLD":
            raise ValueError("HOLD creates no harness update")
        if self._staging_dir is not None:
            raise ValueError("this invocation already has a staging directory")

        staging_dir = (
            self.campaign_root / ".hl-agent-staging" / uuid.uuid4().hex
        )
        overlay = staging_dir / "overlay"
        parent_value = str(self._decision["parent_update"])
        parent = self._parent_overlay(parent_value)
        if parent is not None:
            self._copy_overlay(parent, overlay)

        if not any((overlay / anchor).is_file() for anchor in _ANCHORS):
            anchor = _ANCHORS[1]
            target = overlay / anchor
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.project_root / anchor, target)

        self._staging_dir = staging_dir
        return {
            "staging_directory": self._relative(staging_dir),
            "overlay_files": self._staged_files(),
        }

    def read_staged_file(
        self,
        relative_path: str,
        offset: int = 0,
        max_chars: int = 20_000,
    ) -> dict[str, Any]:
        """Read a bounded slice of a file in this invocation's staging overlay."""
        if offset < 0 or not 1 <= max_chars <= 50_000:
            raise ValueError("invalid offset or max_chars")
        path = self._staged_path(relative_path)
        if not path.is_file():
            raise ValueError(f"staged file not found: {relative_path}")
        text = path.read_text(errors="replace")
        end = min(len(text), offset + max_chars)
        return {
            "path": relative_path,
            "content": text[offset:end],
            "offset": offset,
            "next_offset": end if end < len(text) else None,
            "total_chars": len(text),
        }

    def write_staged_file(self, relative_path: str, content: str) -> dict[str, Any]:
        """Create or replace one harness-owned file in the staging overlay."""
        if self._decision is None or self._decision["decision"] != "TARGETED_UPDATE":
            raise ValueError("only TARGETED_UPDATE may edit staged files")
        if "\x00" in content:
            raise ValueError("file content cannot contain NUL bytes")
        path = self._staged_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return {"path": relative_path, "bytes": path.stat().st_size}

    def inspect_staged_diff(self, relative_path: str) -> str:
        """Return the staged file's unified diff from the repository baseline."""
        staged = self._staged_path(relative_path)
        if not staged.is_file():
            raise ValueError(f"staged file not found: {relative_path}")
        baseline = self.project_root / _relative_path(relative_path)
        before = baseline.read_text(errors="replace").splitlines(keepends=True)
        after = staged.read_text(errors="replace").splitlines(keepends=True)
        return "".join(
            unified_diff(
                before,
                after,
                fromfile=f"baseline/{relative_path}",
                tofile=f"staged/{relative_path}",
            )
        )

    def run_focused_checks(
        self,
        check: CheckName,
        paths: list[str],
    ) -> dict[str, Any]:
        """Run pytest, Ruff, or MyPy against allowed staged or repository paths."""
        if check not in {"pytest", "ruff", "mypy"}:
            raise ValueError(f"unsupported check: {check}")
        if not paths or len(paths) > 10:
            raise ValueError("provide between 1 and 10 focused paths")
        resolved_paths = [str(self._check_path(path)) for path in paths]
        action = {
            "pytest": ["-m", "pytest", "-q"],
            "ruff": ["-m", "ruff", "check"],
            "mypy": ["-m", "mypy"],
        }[check]
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        python_paths = [
            str(self._overlay_root() / "src"),
            str(self.project_root / "src"),
        ]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        completed = subprocess.run(
            [sys.executable, *action, *resolved_paths],
            cwd=self.project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        output = (completed.stdout + completed.stderr)[-20_000:]
        return {
            "check": check,
            "paths": paths,
            "exit_code": completed.returncode,
            "output": output,
        }

    def publish_update(self) -> dict[str, Any]:
        """Freeze the staged cumulative overlay as the next immutable campaign update."""
        if self._decision is None:
            raise ValueError("record a decision before publishing")
        if self._decision["decision"] == "HOLD":
            raise ValueError("HOLD creates no harness update")
        if self._staging_dir is None:
            raise ValueError("prepare an update before publishing")
        if self._published_update is not None:
            raise ValueError("this invocation already published an update")

        self._prune_baseline_duplicates()
        files = self._staged_files()
        if not any(Path(path) in _ANCHORS for path in files):
            raise ValueError("published overlay requires a direct-agent anchor")
        decision_manifest = {
            **self._decision,
            "overlay_files": files,
        }
        _write_json(self._staging_dir / "hl_decision.json", decision_manifest)

        update_dir = self._next_update_dir()
        update_dir.parent.mkdir(parents=True, exist_ok=True)
        self._staging_dir.replace(update_dir)
        self._staging_dir = None
        channel_path = _publish_promoted_harness(
            campaign=self.campaign,
            harness_dir=update_dir,
            validation_run=None,
            state_dir=None,
            source_sequence=self._sequence_dir(
                str(self._decision["source_sequence"])
            ),
        )
        self._published_update = update_dir
        return {
            "update": update_dir.name,
            "update_directory": self._relative(update_dir),
            "channel": self._relative(channel_path),
            "overlay_files": files,
        }

    def _readable_path(self, relative_path: str) -> Path:
        path = self._project_relative_path(relative_path)
        self._assert_readable(path)
        return path

    def _assert_readable(self, path: Path) -> None:
        allowed_roots = (
            self.project_root / "docs",
            self.project_root / "data" / "sequences",
            self.campaign_root,
            self.project_root / "src" / "mediated_coevo" / "diffusion",
            self.project_root / "tests",
        )
        resolved = path.resolve()
        if not any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots):
            raise ValueError(f"path is outside the HL read boundary: {path}")

    def _project_relative_path(self, value: str) -> Path:
        relative = _relative_path(value)
        path = (self.project_root / relative).resolve()
        if not path.is_relative_to(self.project_root):
            raise ValueError(f"path escapes the project root: {value}")
        return path

    def _sequence_dir(self, value: str) -> Path:
        raw = Path(value)
        if raw.is_absolute():
            path = raw.resolve()
        elif raw.parts[:2] == ("data", "sequences"):
            path = (self.project_root / raw).resolve()
        else:
            path = (self.project_root / "data" / "sequences" / raw).resolve()
        sequence_root = (self.project_root / "data" / "sequences").resolve()
        if not path.is_relative_to(sequence_root) or not path.is_dir():
            raise ValueError(f"sequence directory not found: {value}")
        return path

    def _parent_overlay(self, update: str) -> Path | None:
        if update == "baseline":
            return None
        match = _UPDATE_RE.fullmatch(update)
        if match is None:
            raise ValueError("parent_update must be baseline or update_XXXX")
        overlay = (self.campaign_root / update / "overlay").resolve()
        if (
            not overlay.is_relative_to(self.campaign_root.resolve())
            or not overlay.is_dir()
        ):
            raise ValueError(f"campaign update not found: {update}")
        return overlay

    def _copy_overlay(self, source: Path, target: Path) -> None:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or self._is_cache_path(path):
                continue
            relative = path.relative_to(source)
            self._assert_harness_path(relative)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    def _staged_path(self, value: str) -> Path:
        relative = _relative_path(value)
        self._assert_harness_path(relative)
        path = (self._overlay_root() / relative).resolve()
        if not path.is_relative_to(self._overlay_root().resolve()):
            raise ValueError(f"path escapes the staging overlay: {value}")
        return path

    def _overlay_root(self) -> Path:
        if self._staging_dir is None:
            raise ValueError("prepare an update before using staged files")
        return self._staging_dir / "overlay"

    def _assert_harness_path(self, path: Path) -> None:
        if self._is_cache_path(path):
            raise ValueError(f"cache files are not harness content: {path}")
        if path in _HARNESS_FILES:
            return
        parts = path.parts
        if parts[:4] == (
            "src",
            "mediated_coevo",
            "diffusion",
            "harness",
        ):
            return
        name = path.name
        if parts[:1] == ("tests",) and (
            name.startswith("test_harness_")
            or name.startswith("test_langchain_")
            or name == "test_diffusion_renderer.py"
        ):
            return
        raise ValueError(f"path is outside the learned harness boundary: {path}")

    def _check_path(self, value: str) -> Path:
        relative = _relative_path(value)
        self._assert_harness_path(relative)
        staged = self._overlay_root() / relative
        path = staged if staged.is_file() else self.project_root / relative
        if not path.is_file():
            raise ValueError(f"check path not found: {value}")
        return path

    def _staged_files(self) -> list[str]:
        overlay = self._overlay_root()
        return [
            path.relative_to(overlay).as_posix()
            for path in sorted(overlay.rglob("*"))
            if path.is_file() and not self._is_cache_path(path)
        ]

    def _prune_baseline_duplicates(self) -> None:
        overlay = self._overlay_root()
        files = [path for path in overlay.rglob("*") if path.is_file()]
        identical = {
            path.relative_to(overlay)
            for path in files
            if (self.project_root / path.relative_to(overlay)).is_file()
            and path.read_bytes()
            == (self.project_root / path.relative_to(overlay)).read_bytes()
        }
        changed_anchors = [
            anchor
            for anchor in _ANCHORS
            if (overlay / anchor).is_file() and anchor not in identical
        ]
        existing_anchors = [anchor for anchor in _ANCHORS if (overlay / anchor).is_file()]
        keep_anchor = (
            changed_anchors[0]
            if changed_anchors
            else existing_anchors[0] if existing_anchors else None
        )
        for relative in identical:
            if relative != keep_anchor:
                (overlay / relative).unlink()

    def _next_update_dir(self) -> Path:
        numbers = [
            int(match.group(1))
            for path in self.campaign_root.glob("update_*")
            if path.is_dir() and (match := _UPDATE_RE.fullmatch(path.name))
        ]
        return self.campaign_root / f"update_{max(numbers, default=0) + 1:04d}"

    def _relative(self, path: Path | None) -> str | None:
        if path is None:
            return None
        return path.resolve().relative_to(self.project_root).as_posix()

    @staticmethod
    def _is_cache_path(path: Path) -> bool:
        return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}


def run_hl_agent(
    *,
    model: str,
    campaign: str,
    families: tuple[str, ...],
    episode_number: int,
    episode_families: tuple[str, ...],
    project_root: Path = PROJECT_ROOT,
    source_sequence: Path,
) -> HLAgentResult:
    """Analyze one completed episode; infrastructure owns all episode execution."""
    from langchain.agents import create_agent
    from langchain.agents.middleware import ToolCallLimitMiddleware, wrap_tool_call

    if episode_number < 1:
        raise ValueError("episode_number must be at least 1")
    if not episode_families:
        raise ValueError("episode_families must contain at least one iteration")
    if set(episode_families) - set(families):
        raise ValueError("episode_families must come from the target family pool")
    prompt = (
        (project_root / "docs" / "hl_agent_prompt.md")
        .read_text()
        .replace("{CAMPAIGN}", campaign)
    )
    workspace = HLWorkspace(
        campaign=campaign,
        families=families,
        project_root=project_root,
        source_sequence=source_sequence,
    )
    middleware: list[Any] = [
        wrap_tool_call(_recover_expected_tool_error),
        ToolCallLimitMiddleware(run_limit=80, exit_behavior="error"),
    ]
    agent = create_agent(
        model=model,
        tools=workspace.tools(),
        system_prompt=prompt,
        middleware=middleware,
    )
    direct_prompt = "\n".join(
        [
            f"Campaign: {campaign}",
            f"Completed episode: {episode_number}",
            f"Episode families by iteration: {', '.join(episode_families)}",
            f"Campaign target families: {', '.join(families)}",
            "Completed source sequence: "
            + source_sequence.resolve().relative_to(project_root.resolve()).as_posix(),
            "Infrastructure owns episode count, family sampling, seeds, K, and "
            "sequence execution. Analyze this completed episode only.",
        ]
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": direct_prompt}]}
    )
    if workspace._decision is None:
        raise RuntimeError("HL agent finished without recording a decision")
    decision = workspace._decision["decision"]
    if decision != "HOLD" and workspace._published_update is None:
        raise RuntimeError(f"HL agent finished {decision} without publishing an update")
    return {
        "response": _last_message_text(result),
        "decision": workspace._decision,
        "published_update": (
            workspace._published_update.name
            if workspace._published_update is not None
            else None
        ),
    }


def _recover_expected_tool_error(request: Any, handler: Any) -> Any:
    """Return intentional workspace validation failures to the agent."""
    from langchain.messages import ToolMessage

    try:
        return handler(request)
    except ValueError as exc:
        return ToolMessage(
            content=f"Tool error: {exc}",
            tool_call_id=request.tool_call["id"],
            status="error",
        )


def _relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if (
        not value
        or value != value.strip()
        or path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"expected a relative POSIX path: {value!r}")
    return Path(*path.parts)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _last_message_text(result: Any) -> str:
    message = result["messages"][-1]
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", block)) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)
