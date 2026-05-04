import math
from collections import UserDict

from mediated_coevo.utils import (
    as_mapping,
    as_nonempty_string,
    as_optional_float,
    parse_json_object,
)


def test_as_nonempty_string_strips_whitespace():
    assert as_nonempty_string("  value  ") == "value"


def test_as_nonempty_string_returns_none_for_empty_or_non_string_values():
    assert as_nonempty_string("  ") is None
    assert as_nonempty_string(123) is None
    assert as_nonempty_string(None) is None


def test_as_mapping_passes_through_mapping_values():
    value = {"key": "value"}
    assert as_mapping(value) is value

    user_dict = UserDict({"key": "value"})
    assert as_mapping(user_dict) is user_dict


def test_as_mapping_returns_empty_mapping_for_non_mapping_values():
    assert as_mapping(None) == {}
    assert as_mapping([("key", "value")]) == {}


def test_as_optional_float_converts_numeric_values():
    assert as_optional_float("1.25") == 1.25
    assert as_optional_float(2) == 2.0


def test_as_optional_float_returns_none_for_invalid_and_nan_values():
    assert as_optional_float("not-a-number") is None
    assert as_optional_float(None) is None
    assert as_optional_float(math.nan) is None


def test_parse_json_object_plain_object():
    assert parse_json_object('{"instruction": "run", "reasoning": "ok"}') == {
        "instruction": "run",
        "reasoning": "ok",
    }


def test_parse_json_object_fenced_object():
    text = """```json
{"instruction": "run", "reasoning": "ok"}
```"""

    assert parse_json_object(text)["instruction"] == "run"


def test_parse_json_object_with_inner_markdown_fence_in_string():
    text = '''```json
{
  "instruction": "Create a diff example:\\n```\\n-old\\n+new\\n```\\nthen apply it",
  "reasoning": "ok"
}
```'''

    parsed = parse_json_object(text)

    assert parsed["reasoning"] == "ok"
    assert "```" in parsed["instruction"]
    assert "-old" in parsed["instruction"]


def test_parse_json_object_wrapped_in_prose():
    text = 'Here is the result:\n{"no_update": true}\nDone.'

    assert parse_json_object(text) == {"no_update": True}


def test_parse_json_object_non_object_json_returns_empty_dict():
    assert parse_json_object('["not", "an", "object"]') == {}


def test_parse_json_object_invalid_json_returns_empty_dict():
    assert parse_json_object('{"instruction": "unterminated') == {}
