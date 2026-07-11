"""Harness overlay and runtime-state registry helpers."""

from __future__ import annotations

import json
import shutil
import tomllib
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import typer

from mediated_coevo.cli.experiment import PROJECT_ROOT
from mediated_coevo.cli.output import console

_CHANNELS_DIR = "channels"
_BUNDLES_DIR = "bundles"
_GRAPH_STATE_CHANNEL = "graph_state.json"
_PROMOTED_HARNESS_CHANNEL = "promoted_harness.json"
_LEGACY_PROMOTED_HARNESS_FILE = "latest_promoted_harness.txt"


@dataclass(frozen=True)
class RuntimeStateSource:
    """Explicit runtime state selected for a run."""

    state_root: Path
    source: Path
    ref: str | None = None
    channel_path: Path | None = None
    manifest: dict[str, object] | None = None


def _apply_harness_overlay(harness_dir: Path, project_root: Path) -> list[str]:
    overlay_root = _harness_overlay_root(harness_dir)
    applied: list[str] = []
    for source in sorted(path for path in overlay_root.rglob("*") if path.is_file()):
        rel = source.relative_to(overlay_root)
        if len(rel.parts) == 1 and rel.name.startswith("manifest."):
            continue
        target = project_root / rel
        if source.resolve() == target.resolve():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        applied.append(rel.as_posix())
    if not applied:
        raise typer.BadParameter(
            f"harness overlay contains no source files: {harness_dir}"
        )
    return applied


def _harness_overlay_root(harness_dir: Path) -> Path:
    if not harness_dir.is_dir():
        raise typer.BadParameter(f"harness directory not found: {harness_dir}")
    overlay = harness_dir / "overlay"
    root = overlay if overlay.is_dir() else harness_dir
    if not any((root / name).exists() for name in ("src", "config", "tests")):
        raise typer.BadParameter(
            "harness directory must be a repo-root overlay containing "
            "src/, config/, tests/, or overlay/"
        )
    return root


def _prepare_harness_workspace(
    experiment_dir: Path,
    harness_dir: Path | None,
) -> None:
    harnesses_dir = experiment_dir / "harnesses"
    harnesses_dir.mkdir(parents=True, exist_ok=True)
    (harnesses_dir / "README.md").write_text(
        "# Harnesses\n\n"
        "Use this folder for learned repo-root harness overlays. Put new "
        "validated harness snapshots in subdirectories; manifest files are "
        "metadata, and every other path is treated as repo-root overlay content.\n"
    )
    if harness_dir is None:
        return

    resolved = harness_dir.expanduser().resolve()
    overlay_root = _harness_overlay_root(resolved)
    seed_dir = harnesses_dir / "seed"
    shutil.copytree(resolved, seed_dir)
    metadata = {
        "source": str(resolved),
        "overlay_root": str(overlay_root),
        "applied_files": _overlay_file_paths(overlay_root),
    }
    state_root = resolved / "state"
    if state_root.is_dir():
        metadata["bundled_state_files"] = _state_file_paths(state_root)
        metadata["state_activation"] = (
            "not_loaded_by_harness_dir; use --state-dir or --state-ref"
        )
    (harnesses_dir / "active_harness.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def _overlay_file_paths(overlay_root: Path) -> list[str]:
    paths: list[str] = []
    for path in sorted(item for item in overlay_root.rglob("*") if item.is_file()):
        rel = path.relative_to(overlay_root)
        if len(rel.parts) == 1 and rel.name.startswith("manifest."):
            continue
        paths.append(rel.as_posix())
    return paths


def _copy_explicit_state(
    experiment_dir: Path,
    state_source: RuntimeStateSource | None,
) -> dict[str, object] | None:
    """Copy explicitly selected runtime state into the new experiment."""
    if state_source is None:
        return None

    metadata: dict[str, object] = {
        "source": str(state_source.source),
        "state_root": str(state_source.state_root),
    }
    if state_source.ref is not None:
        metadata["ref"] = state_source.ref
    if state_source.channel_path is not None:
        metadata["channel_path"] = str(state_source.channel_path)
    if state_source.manifest is not None:
        metadata["manifest"] = state_source.manifest

    state_root = state_source.state_root
    diffusion_root = state_root / "diffusion"
    if diffusion_root.is_dir():
        metadata.update(_copy_diffusion_state(experiment_dir, diffusion_root))
    active_state_dir = experiment_dir / "state"
    active_state_dir.mkdir(parents=True, exist_ok=True)
    (active_state_dir / "active_state.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata if len(metadata) > 1 else None


def _copy_diffusion_state(
    experiment_dir: Path, diffusion_root: Path
) -> dict[str, object]:
    target_root = experiment_dir / "diffusion"
    metadata: dict[str, object] = {"diffusion_source": str(diffusion_root)}

    artifacts_dir = diffusion_root / "artifacts"
    if artifacts_dir.is_dir():
        skipped_artifacts = sorted(path.name for path in artifacts_dir.glob("*.json"))
        if skipped_artifacts:
            metadata["skipped_artifacts"] = skipped_artifacts
            metadata["artifact_store_reset"] = True

    graph_dir = diffusion_root / "graph_snapshots"
    copied_graph_snapshots = _copy_tree_files(
        graph_dir, target_root / "graph_snapshots"
    )
    if copied_graph_snapshots:
        metadata["graph_snapshots"] = copied_graph_snapshots

    copied_files: list[str] = []
    for source in sorted(item for item in diffusion_root.iterdir() if item.is_file()):
        target = target_root / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied_files.append(source.name)
    if copied_files:
        metadata["files"] = copied_files
    return metadata


def _copy_tree_files(source_root: Path, target_root: Path) -> list[str]:
    if not source_root.is_dir():
        return []
    copied: list[str] = []
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        rel = source.relative_to(source_root)
        target = target_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(rel.as_posix())
    return copied


def _state_file_paths(state_root: Path) -> list[str]:
    return [
        path.relative_to(state_root).as_posix()
        for path in sorted(item for item in state_root.rglob("*") if item.is_file())
    ]


def _resolve_harness_options(
    harness_dir: Path | None,
    harness_ref: str | None,
) -> Path | None:
    if harness_dir is not None and harness_ref is not None:
        raise typer.BadParameter("use either --harness-dir or --harness-ref, not both")
    if harness_ref is not None:
        return _resolve_harness_ref(harness_ref)
    return harness_dir


def _resolve_state_options(
    state_dir: Path | None,
    state_ref: str | None,
) -> RuntimeStateSource | None:
    if state_dir is not None and state_ref is not None:
        raise typer.BadParameter("use either --state-dir or --state-ref, not both")
    if state_dir is not None:
        return _runtime_state_source_from_path(state_dir, ref=None)
    if state_ref is not None:
        return _resolve_state_ref(state_ref)
    return None


def _resolve_harness_ref(ref: str) -> Path:
    channel, campaign = _parse_ref(ref)
    if channel != "promoted":
        raise typer.BadParameter(
            "--harness-ref supports promoted:<campaign> references"
        )
    campaign_root = _campaign_root(campaign)
    channel_path = campaign_root / _CHANNELS_DIR / _PROMOTED_HARNESS_CHANNEL
    if channel_path.is_file():
        payload = _read_json_mapping(channel_path)
        harness_value = payload.get("harness_dir")
        if not isinstance(harness_value, str) or not harness_value:
            raise typer.BadParameter(
                f"promoted harness channel missing harness_dir: {channel_path}"
            )
        return Path(harness_value).expanduser().resolve()

    legacy_path = campaign_root / _LEGACY_PROMOTED_HARNESS_FILE
    if legacy_path.is_file():
        return Path(legacy_path.read_text().strip()).expanduser().resolve()
    raise typer.BadParameter(f"promoted harness channel not found: {channel_path}")


def _resolve_state_ref(ref: str) -> RuntimeStateSource:
    channel, campaign = _parse_ref(ref)
    campaign_root = _campaign_root(campaign)
    if channel in {"latest-graph", "graph"}:
        channel_path = campaign_root / _CHANNELS_DIR / _GRAPH_STATE_CHANNEL
        payload = _read_json_mapping(channel_path)
        state_value = payload.get("state_dir")
        if not isinstance(state_value, str) or not state_value:
            raise typer.BadParameter(
                f"graph state channel missing state_dir: {channel_path}"
            )
        return _runtime_state_source_from_path(
            Path(state_value),
            ref=ref,
            channel_path=channel_path,
            manifest=payload,
        )
    if channel == "promoted":
        channel_path = campaign_root / _CHANNELS_DIR / _PROMOTED_HARNESS_CHANNEL
        if channel_path.is_file():
            payload = _read_json_mapping(channel_path)
            state_value = payload.get("state_dir") or payload.get("harness_dir")
            if not isinstance(state_value, str) or not state_value:
                raise typer.BadParameter(
                    f"promoted harness channel missing state_dir/harness_dir: "
                    f"{channel_path}"
                )
            return _runtime_state_source_from_path(
                Path(state_value),
                ref=ref,
                channel_path=channel_path,
                manifest=payload,
            )
        legacy_harness = _resolve_harness_ref(ref)
        return _runtime_state_source_from_path(legacy_harness, ref=ref)
    raise typer.BadParameter(
        "--state-ref supports latest-graph:<campaign>, graph:<campaign>, "
        "or promoted:<campaign>"
    )


def _runtime_state_source_from_path(
    source: Path,
    *,
    ref: str | None,
    channel_path: Path | None = None,
    manifest: dict[str, object] | None = None,
) -> RuntimeStateSource:
    resolved = source.expanduser().resolve()
    state_root = _state_root_from_path(resolved)
    return RuntimeStateSource(
        state_root=state_root,
        source=resolved,
        ref=ref,
        channel_path=channel_path,
        manifest=manifest,
    )


def _state_root_from_path(path: Path) -> Path:
    if not path.exists():
        raise typer.BadParameter(f"state source not found: {path}")
    if (path / "diffusion").is_dir():
        return path
    if (path / "state" / "diffusion").is_dir():
        return path / "state"
    raise typer.BadParameter(
        "state source must contain diffusion/ or state/diffusion/: " f"{path}"
    )


def _parse_ref(ref: str) -> tuple[str, str]:
    if ":" not in ref:
        raise typer.BadParameter(
            "references must use <channel>:<campaign>, for example latest-graph:HL3"
        )
    channel, campaign = ref.split(":", maxsplit=1)
    channel = channel.strip()
    campaign = campaign.strip()
    if not channel or not campaign:
        raise typer.BadParameter(
            "references must use <channel>:<campaign>, for example latest-graph:HL3"
        )
    return channel, campaign


def _campaign_root(campaign: str) -> Path:
    parts = Path(campaign).parts
    if len(parts) != 1 or any(part in {"", ".", ".."} for part in parts):
        raise typer.BadParameter(f"invalid campaign reference: {campaign}")
    return PROJECT_ROOT / "data" / "experiments" / campaign


def _read_json_mapping(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise typer.BadParameter(f"state channel not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"invalid JSON channel file: {path}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"channel file must contain a JSON object: {path}")
    return payload


def _publish_graph_state_ref(
    ref: str,
    *,
    experiment_dir: Path,
    split: str | None,
) -> Path:
    channel, campaign = _parse_ref(ref)
    if channel not in {"latest-graph", "graph"}:
        raise typer.BadParameter(
            "--publish-state-ref supports latest-graph:<campaign> or graph:<campaign>"
        )
    if split != "train":
        raise typer.BadParameter(
            "--publish-state-ref may only publish graph state from --split train runs"
        )
    campaign_root = _campaign_root(campaign)
    diffusion_root = experiment_dir / "diffusion"
    if not diffusion_root.is_dir():
        raise typer.BadParameter(
            f"experiment has no diffusion state to publish: {diffusion_root}"
        )
    bundle_dir, manifest = _create_graph_state_bundle(
        campaign_root=campaign_root,
        experiment_dir=experiment_dir,
        diffusion_root=diffusion_root,
    )
    channel_path = campaign_root / _CHANNELS_DIR / _GRAPH_STATE_CHANNEL
    channel_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "campaign": campaign,
        "channel": "graph_state",
        "bundle_id": manifest["bundle_id"],
        "state_dir": str(bundle_dir / "state"),
        "bundle_dir": str(bundle_dir),
        "source_run": str(experiment_dir),
        "created_from_split": split,
        "graph_digest": manifest["graph_digest"],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_write_json(channel_path, payload)
    console.print(f"[bold]Published graph state:[/] {channel_path}")
    return channel_path


def _create_graph_state_bundle(
    *,
    campaign_root: Path,
    experiment_dir: Path,
    diffusion_root: Path,
) -> tuple[Path, dict[str, object]]:
    graph_dir = diffusion_root / "graph_snapshots"
    graph_files = _relative_files(graph_dir)
    root_files = [
        path
        for path in _relative_files(diffusion_root)
        if len(path.parts) == 1 and path.name == "diffused_records.jsonl"
    ]
    if not graph_files and not root_files:
        raise typer.BadParameter(
            f"diffusion state has no graph snapshots or diffused records: "
            f"{diffusion_root}"
        )
    digest = _state_digest(diffusion_root, graph_files, root_files)
    bundle_id = f"sha256:{digest}"
    bundle_dir = campaign_root / _BUNDLES_DIR / digest
    state_diffusion = bundle_dir / "state" / "diffusion"
    if not bundle_dir.exists():
        for rel in graph_files:
            source = graph_dir / rel
            target = state_diffusion / "graph_snapshots" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for rel in root_files:
            target = state_diffusion / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(diffusion_root / rel, target)
    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_json_mapping(manifest_path)
    else:
        manifest = {
            "schema_version": 1,
            "bundle_id": bundle_id,
            "source_run": str(experiment_dir),
            "created_from_split": "train",
            "state_files": _state_file_paths(bundle_dir / "state"),
            "graph_digest": digest,
            "artifact_store_policy": "fresh_per_run; diffusion/artifacts not bundled",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        _atomic_write_json(manifest_path, manifest)
    return bundle_dir, manifest


def _publish_promoted_harness(
    *,
    campaign: str,
    harness_dir: Path,
    validation_run: str | None,
    state_dir: Path | None,
) -> Path:
    campaign_root = _campaign_root(campaign)
    resolved_harness = harness_dir.expanduser().resolve()
    overlay_root = _harness_overlay_root(resolved_harness)
    channel_path = campaign_root / _CHANNELS_DIR / _PROMOTED_HARNESS_CHANNEL
    channel_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_files = _overlay_file_paths(overlay_root)
    payload: dict[str, object] = {
        "schema_version": 1,
        "campaign": campaign,
        "channel": "promoted_harness",
        "harness_dir": str(resolved_harness),
        "overlay_root": str(overlay_root),
        "applied_files": overlay_files,
        "overlay_digest": _files_digest(
            overlay_root, [Path(path) for path in overlay_files]
        ),
        "validation_run": validation_run,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if state_dir is not None:
        state_source = _runtime_state_source_from_path(state_dir, ref=None)
        payload["state_dir"] = str(state_source.state_root)
    elif (resolved_harness / "state" / "diffusion").is_dir():
        payload["bundled_state_dir"] = str(resolved_harness / "state")
        payload["state_activation"] = (
            "not implicit; prefer latest-graph:<campaign> for forward graph state"
        )
    record_dir = campaign_root / "promotions"
    record_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f-promoted-harness")
    record_path = record_dir / f"{record_id}.json"
    record_payload = {
        **payload,
        "decision": "promoted",
        "promotion_record": str(record_path),
    }
    _atomic_write_json(record_path, record_payload)
    payload["promotion_record"] = str(record_path)
    _atomic_write_json(channel_path, payload)
    return channel_path


def _relative_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [
        path.relative_to(root)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _state_digest(
    diffusion_root: Path,
    graph_files: list[Path],
    root_files: list[Path],
) -> str:
    digest = sha256()
    for rel in [*(Path("graph_snapshots") / path for path in graph_files), *root_files]:
        source = diffusion_root / rel
        digest.update(rel.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _files_digest(root: Path, files: list[Path]) -> str:
    digest = sha256()
    for rel in files:
        source = root / rel
        digest.update(rel.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _experiment_split(experiment_dir: Path) -> str | None:
    config_path = experiment_dir / "config.toml"
    try:
        payload = tomllib.loads(config_path.read_text())
    except OSError as exc:
        raise typer.BadParameter(f"experiment config not found: {config_path}") from exc
    experiment = payload.get("experiment")
    if not isinstance(experiment, dict):
        return None
    selection = experiment.get("benchmark_selection")
    if not isinstance(selection, dict):
        return None
    split = selection.get("split")
    return split if isinstance(split, str) else None
