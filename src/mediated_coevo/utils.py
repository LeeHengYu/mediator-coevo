"""Shared utilities used across layers."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


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
