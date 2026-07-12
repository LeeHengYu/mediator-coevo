from __future__ import annotations

from collections.abc import Iterable, Mapping


def message_text(messages: Iterable[Mapping[str, object]]) -> str:
    return "\n\n".join(str(message["content"]) for message in messages)


def assert_contains_all(text: str, expected: Iterable[str]) -> None:
    missing = [item for item in expected if item not in text]
    assert not missing, _prompt_assertion_message("missing", missing)


def assert_omits_all(text: str, unexpected: Iterable[str]) -> None:
    present = [item for item in unexpected if item in text]
    assert not present, _prompt_assertion_message("unexpected", present)


def _prompt_assertion_message(kind: str, items: list[str]) -> str:
    return f"{kind} prompt text: {', '.join(repr(item) for item in items)}"
