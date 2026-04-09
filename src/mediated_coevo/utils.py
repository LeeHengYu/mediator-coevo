"""Shared utilities used across layers."""

from __future__ import annotations

import json


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from LLM output, tolerating markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return result if isinstance(result, dict) else {}
