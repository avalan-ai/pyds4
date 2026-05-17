from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pyds4.dsml import (
    DsmlMessage,
    DsmlMessageRole,
    DsmlParseResult,
    DsmlParseStatus,
    DsmlPrompt,
    DsmlToolCall,
    DsmlToolSchema,
    normalize_tool_schemas,
    tool_schema_text,
)


def _math_schema() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "math.calculator",
            "description": "Evaluate an expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
            },
        },
    }


def test_dsml_dataclasses_are_importable_and_normalize_inputs() -> None:
    call = DsmlToolCall(
        id="call_1",
        name="math.calculator",
        arguments={"expression": "2 + 2"},
    )
    message = DsmlMessage(
        role="assistant",
        content="",
        reasoning="Use the calculator.",
        tool_calls=[call],
    )
    parsed = DsmlParseResult(
        content="",
        calls=[call],
        reasoning="Use the calculator.",
        raw_dsml="<tool_calls></tool_calls>",
    )

    assert call.arguments == {"expression": "2 + 2"}
    assert message.role is DsmlMessageRole.ASSISTANT
    assert message.tool_calls == (call,)
    assert parsed.calls == (call,)
    assert parsed.status is DsmlParseStatus.COMPLETE


def test_avalan_style_tool_schema_dicts_are_accepted_without_avalan_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "avalan", raising=False)

    schema = DsmlToolSchema(_math_schema())
    prompt = DsmlPrompt(
        system_content="System",
        messages=[
            DsmlMessage(role=DsmlMessageRole.USER, content="calculate"),
        ],
        tool_schemas=[schema],
    )

    assert schema.function_schema["name"] == "math.calculator"
    assert prompt.tool_schema_text == (
        '{"name":"math.calculator","description":"Evaluate an '
        'expression.","parameters":{"type":"object","properties":'
        '{"expression":{"type":"string"}}}}'
    )
    assert prompt.has_tool_prompt is True
    assert "avalan" not in sys.modules


def test_rendered_tool_schema_json_strings_are_accepted() -> None:
    rendered = (
        '{"name":"math.calculator","parameters":{"type":"object"}}\n'
        '{"name":"math.sqrt","parameters":{"type":"object"}}'
    )
    schema = DsmlToolSchema(rendered)

    assert schema.is_rendered is True
    assert schema.function_schema["name"] == "math.calculator"
    assert schema.rendered_json == rendered
    assert tool_schema_text(rendered) == rendered


def test_empty_tool_schemas_produce_no_tool_prompt() -> None:
    assert normalize_tool_schemas(None) == ()
    assert normalize_tool_schemas([]) == ()
    assert tool_schema_text(None) is None
    assert tool_schema_text([]) is None

    empty_prompt = DsmlPrompt(tool_schemas=None)

    assert empty_prompt.tool_schemas == ()
    assert empty_prompt.tool_schema_text is None
    assert empty_prompt.has_tool_prompt is False


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"role": "invalid", "content": ""}, "Unsupported DSML"),
        ({"role": "user", "content": object()}, "content"),
        ({"role": "user", "content": "", "reasoning": object()}, "reasoning"),
        (
            {"role": "assistant", "content": "", "tool_calls": [object()]},
            "tool_calls",
        ),
    ],
)
def test_dsml_messages_reject_invalid_inputs(
    kwargs: dict[str, object],
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        DsmlMessage(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("schema", "error_match"),
    [
        ({}, "schema"),
        ({"name": ""}, "function name"),
        ({"type": "function"}, "function payload"),
        (
            {"type": "function", "function": {"parameters": {}}},
            "function name",
        ),
        ({"name": "math.calculator", "parameters": []}, "parameters"),
        ({"name": "math.calculator", "bad": object()}, "JSON"),
        ('{"name": "math.calculator"', "valid JSON"),
        ('["math.calculator"]', "JSON object"),
        ('{"parameters": {"type": "object"}}', "function name"),
    ],
)
def test_tool_schemas_reject_malformed_inputs(
    schema: object,
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        DsmlToolSchema(schema)  # type: ignore[arg-type]


def test_dsml_parse_result_rejects_invalid_inputs() -> None:
    call = DsmlToolCall(name="math.calculator")

    with pytest.raises(TypeError, match="content"):
        DsmlParseResult(content=object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="parse status"):
        DsmlParseResult(content="", status="unknown")

    with pytest.raises(TypeError, match="calls"):
        DsmlParseResult(content="", calls=[object()])  # type: ignore[list-item]

    result = DsmlParseResult(
        content="partial",
        calls=[call],
        status="incomplete",
        error="missing closing tool_calls tag",
    )

    assert result.status is DsmlParseStatus.INCOMPLETE
    assert result.error == "missing closing tool_calls tag"


def test_dsml_import_does_not_import_native_extension_objects() -> None:
    env = {
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import sys; "
                "import pyds4.dsml as dsml; "
                "print(hasattr(dsml, 'DsmlPrompt')); "
                "print('pyds4._native' in sys.modules)"
            ),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == ["True", "False"]
