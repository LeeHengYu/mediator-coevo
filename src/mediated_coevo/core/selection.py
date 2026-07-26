"""Reusable deterministic selection helpers."""

from __future__ import annotations

import hashlib


def deterministic_seed(*parts: object) -> int:
    """Derive a stable integer seed from a sequence of identifying parts."""
    raw = ":".join(str(part) for part in parts).encode()
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "big")
