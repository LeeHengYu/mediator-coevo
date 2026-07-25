"""Infrastructure-owned loop for the independent offline HL agent."""

from __future__ import annotations

import json
import re
import secrets
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from mediated_coevo.cli.config import _load_config_or_bad_parameter
from mediated_coevo.cli.experiment import (
    PROJECT_ROOT,
    _normalize_families,
    prepare_llm_credentials_or_exit,
)
from mediated_coevo.cli.output import console
from mediated_coevo.hl.agent import run_hl_agent

_EPISODE_RE = re.compile(r"episode_(\d{4,})\.json")


def hl_agent(
    campaign: Annotated[
        str,
        typer.Option(help="Independent HL campaign name, for example HL6."),
    ],
    family: Annotated[
        list[str],
        typer.Option(
            "--family",
            help=(
                "Exactly four campaign families, spread as evenly as possible "
                "across -K iterations."
            ),
        ),
    ],
    episodes: Annotated[
        int,
        typer.Option(
            help="Number of new deployment episodes for infrastructure to run."
        ),
    ] = 1,
    start_episode: Annotated[
        int | None,
        typer.Option(
            help="Absolute number of the first new episode; inferred for managed campaigns."
        ),
    ] = None,
    source_sequence: Annotated[
        Path | None,
        typer.Option(
            "--source-sequence",
            help=(
                "Optional completed episode to analyze before the new episodes; "
                "it does not count toward --episodes."
            ),
        ),
    ] = None,
    k: Annotated[
        int | None,
        typer.Option(
            "-K",
            help="Sequence iterations per episode; defaults to source K or 3.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(help="OpenRouter model; defaults to models.planner."),
    ] = None,
    config_dir: Annotated[
        Path,
        typer.Option(help="Config directory used by both HL and sequence commands."),
    ] = PROJECT_ROOT / "config",
) -> None:
    """Run the independent offline agent across infrastructure-owned episodes."""
    families = tuple(_normalize_families(family))
    if len(families) != 4:
        raise typer.BadParameter("hl-agent requires exactly four --family values")
    if episodes < 1:
        raise typer.BadParameter("--episodes must be at least 1")
    resolved_source = _resolve_source_sequence(source_sequence, PROJECT_ROOT)
    source_k = _source_k(resolved_source)
    resolved_k = k if k is not None else source_k
    if resolved_k < 1:
        raise typer.BadParameter("-K must be at least 1")
    if (
        resolved_source is not None
        and _completed_iterations(resolved_source) != source_k
    ):
        raise typer.BadParameter(
            "--source-sequence must contain one completed sample_result.json "
            "for every recorded source iteration"
        )

    campaign_root = _campaign_root(PROJECT_ROOT, campaign)
    resolved_start = _resolve_start_episode(
        campaign_root=campaign_root,
        requested=start_episode,
        has_source=resolved_source is not None,
    )
    if resolved_source is not None and resolved_start == 1:
        raise typer.BadParameter(
            "--source-sequence precedes the first new episode, so "
            "--start-episode must be at least 2"
        )
    if resolved_source is not None:
        try:
            source_families = _source_families(resolved_source)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        unexpected = set(source_families) - set(families)
        if unexpected:
            raise typer.BadParameter(
                "source sequence contains families outside the campaign pool: "
                + ", ".join(sorted(unexpected))
            )
    for episode_number in range(resolved_start, resolved_start + episodes):
        record_path = _episode_record_path(campaign_root, episode_number)
        if record_path.exists() and _episode_status(record_path) != "failed":
            raise typer.BadParameter(
                f"managed episode already exists: episode_{episode_number:04d}"
            )

    config = prepare_llm_credentials_or_exit(
        _load_config_or_bad_parameter(config_dir)
    )
    selected_model = model or config.models.planner
    langchain_model = (
        selected_model
        if selected_model.startswith("openrouter:")
        else f"openrouter:{selected_model.removeprefix('openrouter/')}"
    )
    run_hl_campaign(
        model=langchain_model,
        campaign=campaign,
        families=families,
        episodes=episodes,
        start_episode=resolved_start,
        k=resolved_k,
        config_dir=config_dir.resolve(),
        project_root=PROJECT_ROOT,
        source_sequence=resolved_source,
    )


def run_hl_campaign(
    *,
    model: str,
    campaign: str,
    families: tuple[str, ...],
    episodes: int,
    start_episode: int,
    k: int,
    config_dir: Path,
    project_root: Path,
    source_sequence: Path | None,
) -> list[dict[str, Any]]:
    """Run exactly ``episodes`` new episodes and analyze the final one too."""
    campaign_root = _campaign_root(project_root, campaign)
    if source_sequence is not None:
        source_episode = start_episode - 1
        source_families = _source_families(source_sequence)
        unexpected = set(source_families) - set(families)
        if unexpected:
            raise ValueError(
                "source sequence contains families outside the campaign pool: "
                + ", ".join(sorted(unexpected))
            )
        console.print(
            f"[bold]HL source analysis:[/] episode {source_episode} · "
            f"{source_families} · {source_sequence}"
        )
        source_result = run_hl_agent(
            model=model,
            campaign=campaign,
            families=families,
            episode_number=source_episode,
            episode_families=source_families,
            project_root=project_root,
            source_sequence=source_sequence,
        )
        console.print(source_result["response"])

    completed_records: list[dict[str, Any]] = []
    for episode_number in range(start_episode, start_episode + episodes):
        seed = secrets.randbits(63)
        harness_ref = _current_harness_ref(campaign_root, campaign)
        record_path = _episode_record_path(campaign_root, episode_number)
        record: dict[str, Any] = {
            "schema_version": 1,
            "campaign": campaign,
            "episode": episode_number,
            "family_pool": list(families),
            "seed": seed,
            "k": k,
            "harness_ref": harness_ref,
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_json(record_path, record)
        try:
            console.print(
                f"[bold]HL episode {episode_number}:[/] "
                f"family pool={families} · K={k} · seed={seed}"
            )
            sequence_dir, episode_families = _run_sequence_episode(
                families=families,
                seed=seed,
                k=k,
                harness_ref=harness_ref,
                config_dir=config_dir,
                project_root=project_root,
            )
            record.update(
                {
                    "status": "analyzing",
                    "sequence": _relative(project_root, sequence_dir),
                    "families": list(episode_families),
                }
            )
            _write_json(record_path, record)
            agent_result = run_hl_agent(
                model=model,
                campaign=campaign,
                families=families,
                episode_number=episode_number,
                episode_families=episode_families,
                project_root=project_root,
                source_sequence=sequence_dir,
            )
            record.update(
                {
                    "status": "complete",
                    "decision": agent_result["decision"],
                    "published_update": agent_result["published_update"],
                    "agent_response": agent_result["response"],
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            _write_json(record_path, record)
            completed_records.append(record)
            console.print(agent_result["response"])
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "failed_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            _write_json(record_path, record)
            raise
    return completed_records


def _run_sequence_episode(
    *,
    families: tuple[str, ...],
    seed: int,
    k: int,
    harness_ref: str | None,
    config_dir: Path,
    project_root: Path,
) -> tuple[Path, tuple[str, ...]]:
    sequence_root = project_root / "data" / "sequences"
    before = set(sequence_root.glob("sequence-*"))
    command = [
        sys.executable,
        "-m",
        "mediated_coevo.main",
        "sequence",
    ]
    for family in families:
        command.extend(["--family", family])
    command.extend(
        [
            "--seed",
            str(seed),
            "-K",
            str(k),
            "--graph-agent",
            "--diffusion-agent",
            "--config-dir",
            str(config_dir),
        ]
    )
    if harness_ref is not None:
        command.extend(["--harness-ref", harness_ref])
    completed = subprocess.run(command, cwd=project_root, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"sequence command failed for family pool {families!r}: "
            f"exit code {completed.returncode}"
        )
    created = sorted(set(sequence_root.glob("sequence-*")) - before)
    if len(created) != 1:
        raise RuntimeError(
            f"sequence command created {len(created)} run directories; expected 1"
        )
    result_count = len(list(created[0].glob("iter-*/samples/*/sample_result.json")))
    if result_count != k:
        raise RuntimeError(
            f"sequence episode completed {result_count}/{k} iterations: {created[0]}"
        )
    selected_families = _source_families(created[0])
    if len(selected_families) != k:
        raise RuntimeError(
            f"sequence episode recorded {len(selected_families)}/{k} iterations: "
            f"{created[0]}"
        )
    return created[0], selected_families


def _resolve_source_sequence(
    source_sequence: Path | None,
    project_root: Path,
) -> Path | None:
    if source_sequence is None:
        return None
    resolved = source_sequence.expanduser().resolve()
    sequence_root = (project_root / "data" / "sequences").resolve()
    if not resolved.is_relative_to(sequence_root) or not resolved.is_dir():
        raise typer.BadParameter(
            "--source-sequence must be a completed data/sequences/ directory"
        )
    return resolved


def _resolve_start_episode(
    *,
    campaign_root: Path,
    requested: int | None,
    has_source: bool,
) -> int:
    if requested is not None:
        if requested < 1:
            raise typer.BadParameter("--start-episode must be at least 1")
        return requested
    managed = [
        int(match.group(1))
        for path in (campaign_root / "episodes").glob("episode_*.json")
        if (match := _EPISODE_RE.fullmatch(path.name))
    ]
    if managed:
        latest = max(managed)
        latest_record = _episode_record_path(campaign_root, latest)
        return latest if _episode_status(latest_record) == "failed" else latest + 1
    if has_source:
        return 2
    has_legacy_state = (
        any(campaign_root.glob("update_*"))
        or (campaign_root / "decisions").is_dir()
        or (campaign_root / "channels" / "promoted_harness.json").is_file()
    )
    if has_legacy_state:
        raise typer.BadParameter(
            "--start-episode is required when continuing a pre-managed campaign"
        )
    return 1


def _current_harness_ref(campaign_root: Path, campaign: str) -> str | None:
    channel = campaign_root / "channels" / "promoted_harness.json"
    return f"promoted:{campaign}" if channel.is_file() else None


def _campaign_root(project_root: Path, campaign: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", campaign):
        raise typer.BadParameter("campaign must be one portable path component")
    return project_root / "data" / "experiments" / campaign


def _episode_record_path(campaign_root: Path, episode_number: int) -> Path:
    return campaign_root / "episodes" / f"episode_{episode_number:04d}.json"


def _episode_status(record_path: Path) -> str | None:
    try:
        payload = json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload.get("status") if isinstance(payload, dict) else None


def _source_k(source_sequence: Path | None) -> int:
    if source_sequence is None:
        return 3
    return len(
        [
            path
            for path in source_sequence.glob("iter-*")
            if path.is_dir() and (path / "sequence_spec.json").is_file()
        ]
    ) or 3


def _completed_iterations(sequence: Path) -> int:
    return len(list(sequence.glob("iter-*/samples/*/sample_result.json")))


def _source_families(sequence: Path) -> tuple[str, ...]:
    selected: list[str] = []
    spec_paths = sorted(
        sequence.glob("iter-*/sequence_spec.json"),
        key=lambda path: int(path.parent.name.removeprefix("iter-")),
    )
    for spec_path in spec_paths:
        payload = json.loads(spec_path.read_text())
        families: set[str] = set()
        for task in payload.get("tasks", []):
            task_id = task.get("task_id") if isinstance(task, dict) else None
            if isinstance(task_id, str) and "/" in task_id:
                families.add(task_id.split("/", maxsplit=1)[0])
        if len(families) != 1:
            raise ValueError(
                "each source iteration must contain exactly one family: "
                f"{spec_path}"
            )
        selected.append(next(iter(families)))
    if not selected:
        raise ValueError(
            f"completed source sequence contains no recorded iterations: {sequence}"
        )
    return tuple(selected)


def _relative(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def register_hl_command(app: typer.Typer) -> None:
    app.command("hl-agent")(hl_agent)
