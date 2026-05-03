from mediated_coevo.utils import parse_json_object


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
