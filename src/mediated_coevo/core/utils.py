"""Shared utilities used across layers."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from typing import Any, cast

logger = logging.getLogger(__name__)


def as_nonempty_string(value: object) -> str | None:
    """Return stripped non-empty strings only."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def as_mapping(value: object) -> Mapping[str, Any]:
    """Return mapping values, or an empty mapping for invalid inputs."""
    if isinstance(value, Mapping):
        return value
    return {}


def as_optional_float(value: object) -> float | None:
    """Return a float conversion, or None for invalid and NaN values."""
    if not isinstance(value, str | bytes | bytearray | int | float):
        return None
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from LLM output.

    The response may be wrapped in markdown fences or prose, and JSON string
    values may themselves contain markdown fences. Avoid regex-based fence
    slicing because it can stop at an inner fence inside a quoted string.
    """
    text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        return result if isinstance(result, dict) else {}

    result = _parse_first_json_object(text)
    if result is not None:
        return result

    logger.debug("parse_json_object failed: no JSON object found | input=%r", text[:200])
    return {}


def _parse_first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            result, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result
    return None
