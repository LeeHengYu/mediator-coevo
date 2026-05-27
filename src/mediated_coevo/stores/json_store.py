"""Shared JSON and JSONL persistence helpers for file-backed stores."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

_TModel = TypeVar("_TModel", bound=BaseModel)


def write_model(
    path: Path,
    model: BaseModel,
    *,
    overwrite: bool = False,
    exists_error_prefix: str = "Artifact",
) -> Path:
    """Write one Pydantic model to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{exists_error_prefix} already exists: {path}")
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    return path


def append_jsonl(path: Path, model: BaseModel) -> Path:
    """Append one Pydantic model to a JSONL ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        payload = json.dumps(model.model_dump(mode="json"), sort_keys=True)
        output.write(payload)
        output.write("\n")
    return path


def load_model(path: Path, model_cls: type[_TModel]) -> _TModel | None:
    """Load one JSON file into a Pydantic model, if it exists."""
    if not path.exists():
        return None
    return model_cls.model_validate_json(path.read_text(encoding="utf-8"))


def load_directory_models(
    directory: Path,
    model_cls: type[_TModel],
    *,
    logger: logging.Logger,
) -> list[_TModel]:
    """Load every JSON file in a directory into validated models."""
    models: list[_TModel] = []
    for path in directory.glob("*.json"):
        try:
            loaded = load_model(path, model_cls)
        except Exception as exc:
            logger.warning("Failed to load %s %s: %s", model_cls.__name__, path, exc)
            continue
        if loaded is not None:
            models.append(loaded)
    return models


def load_jsonl_models(
    path: Path,
    model_cls: type[_TModel],
    *,
    logger: logging.Logger,
) -> list[_TModel]:
    """Load a JSONL ledger into validated models, skipping malformed lines."""
    if not path.exists():
        return []

    models: list[_TModel] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            models.append(model_cls.model_validate_json(line))
        except Exception as exc:
            logger.warning("Failed to load %s %s: %s", model_cls.__name__, path, exc)
    return models
