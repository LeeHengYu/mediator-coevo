"""Reusable deterministic selection helpers."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence
from math import ceil
from typing import Any, TypeVar

_TItem = TypeVar("_TItem")


def deterministic_seed(*parts: object) -> int:
    """Derive a stable integer seed from a sequence of identifying parts."""
    raw = ":".join(str(part) for part in parts).encode()
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "big")


def choose_seeded_top_fraction(
    items: Sequence[_TItem],
    *,
    rank_key: Callable[[_TItem], Any],
    seed: int | None,
    fraction: float,
    minimum: int = 1,
) -> _TItem | None:
    """Choose one item uniformly from the top-ranked fraction."""
    if not items:
        return None

    ranked = sorted(items, key=rank_key, reverse=True)
    top_count = max(minimum, ceil(len(ranked) * fraction))
    return random.Random(seed).choice(ranked[:top_count])
