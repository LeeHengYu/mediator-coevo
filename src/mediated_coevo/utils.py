"""Shared utilities used across layers."""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from LLM output, tolerating markdown fences."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.debug("parse_json_object failed: %s | input=%r", exc, text[:200])
        return {}
    return result if isinstance(result, dict) else {}
