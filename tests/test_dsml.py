from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import pyds4.dsml as dsml
from pyds4.dsml import (
    PARAMETER_END_MARKERS,
    DsmlMessage,
    DsmlMessageRole,
    DsmlParseResult,
    DsmlParseStatus,
    DsmlPrompt,
    DsmlToolCall,
    DsmlToolCallBufferStatus,
    DsmlToolSchema,
    normalize_tool_schemas,
    parse_generated_message,
    parse_tool_calls,
    render_prompt,
    render_tool_calls,
    render_tool_result,
    split_reasoning,
    stream_argument_deltas,
    tool_call_buffer_status,
    tool_call_start_span,
    tool_call_start_suffix_length,
    tool_schema_text,
    tools_prompt,
)

_DOCUMENTED_DSML_PUBLIC_NAMES = frozenset(
    (
        "DsmlMessage",
        "DsmlMessageRole",
        "DsmlParseResult",
        "DsmlParseStatus",
        "DsmlPrompt",
        "DsmlToolCall",
        "DsmlToolCallBufferStatus",
        "DsmlToolSchema",
        "JsonObject",
        "JsonValue",
        "PARAMETER_END_MARKERS",
        "ReplayLookup",
        "TOOL_CALLS_END",
        "TOOL_CALLS_START",
        "TOOL_CALL_END_MARKERS",
        "TOOL_CALL_START_PREFIXES",
        "ToolSchemaInput",
        "ToolSchemasInput",
        "normalize_tool_schemas",
        "parse_generated_message",
        "parse_tool_calls",
        "render_prompt",
        "render_tool_calls",
        "render_tool_result",
        "split_reasoning",
        "stream_argument_deltas",
        "tool_call_buffer_status",
        "tool_call_start_span",
        "tool_call_start_suffix_length",
        "tool_schema_text",
        "tools_prompt",
    )
)


def test_documented_dsml_public_names_are_exported() -> None:
    exported_names = set(dsml.__all__)

    assert sorted(_DOCUMENTED_DSML_PUBLIC_NAMES - exported_names) == []
    assert sorted(exported_names - _DOCUMENTED_DSML_PUBLIC_NAMES) == []
    assert [
        name
        for name in sorted(_DOCUMENTED_DSML_PUBLIC_NAMES)
        if not hasattr(dsml, name)
    ] == []


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
    assert (
        prompt.tool_schema_text
        == '{"name":"math.calculator","description":"Evaluate an '
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


def test_tools_prompt_renders_ds4_instruction_text() -> None:
    rendered = tools_prompt([_math_schema()])

    assert rendered is not None
    assert rendered.startswith("## Tools\n\n")
    assert "<｜DSML｜tool_calls>" in rendered
    assert '<｜DSML｜invoke name="$TOOL_NAME">' in rendered
    assert '"name":"math.calculator"' in rendered
    assert "Preserve characters such as `>`, `&`, and `&&` exactly" in rendered
    assert tools_prompt(None) is None


def test_render_prompt_matches_ds4_chat_and_tool_shape() -> None:
    prompt = DsmlPrompt(
        system_content="System",
        messages=[
            DsmlMessage(role="developer", content="Developer"),
            DsmlMessage(role="user", content="hello"),
            DsmlMessage(
                role="assistant",
                content="",
                tool_calls=[
                    DsmlToolCall(
                        id="call_1",
                        name="math.calculator",
                        arguments={"expression": "2 + 2", "precision": 2},
                    )
                ],
            ),
            DsmlMessage(role="tool", content="4 > 3 & < 5"),
        ],
        tool_schemas=[_math_schema()],
    )

    rendered = render_prompt(prompt)

    assert rendered.startswith(
        "<｜begin▁of▁sentence｜>System\n\nDeveloper\n\n## Tools"
    )
    assert "<｜User｜>hello<｜Assistant｜></think>" in rendered
    assert '<｜DSML｜invoke name="math.calculator">' in rendered
    assert (
        '<｜DSML｜parameter name="expression" string="true">2 + 2'
        "</｜DSML｜parameter>"
        in rendered
    )
    assert (
        '<｜DSML｜parameter name="precision" string="false">2'
        "</｜DSML｜parameter>"
        in rendered
    )
    assert "<tool_result>4 &gt; 3 &amp; &lt; 5</tool_result>" in rendered


def test_render_prompt_emits_reasoning_for_thinking_mode() -> None:
    rendered = render_prompt(
        DsmlPrompt(
            messages=[
                DsmlMessage(role="user", content="hello"),
                DsmlMessage(
                    role="assistant",
                    content="answer",
                    reasoning="think first",
                ),
            ],
        ),
        think_mode="high",
    )

    assert "<｜Assistant｜><think>think first</think>answer" in rendered


def test_render_tool_calls_uses_replay_when_available() -> None:
    calls = (
        DsmlToolCall(name="math.calculator", arguments={"expression": "2"}),
    )

    assert render_tool_calls(calls, lambda value: "<raw/>") == "<raw/>"


def test_render_tool_calls_escapes_only_required_dsml_text() -> None:
    close_markers = " | ".join(PARAMETER_END_MARKERS)
    literal_entities = "literal &lt;tag&gt; &amp; value &lt;/parameter>"
    nested_entities = (
        "literal &lt;/parameter> and &amp;lt;/parameter> and "
        "&amp;amp;lt;/parameter>"
    )
    rendered = render_tool_calls(
        [
            DsmlToolCall(
                name='pkg.tool"&',
                arguments={
                    "command": "echo a > b && echo &",
                    "unsafe": close_markers,
                    "literal": literal_entities,
                    "payload": {
                        "value": "</｜DSML｜parameter>",
                        "variants": list(PARAMETER_END_MARKERS),
                    },
                    "nested_entities": nested_entities,
                },
            )
        ]
    )

    assert '<｜DSML｜invoke name="pkg.tool&quot;&amp;">' in rendered
    assert "echo a > b && echo &" in rendered
    assert rendered.count("</｜DSML｜parameter>") == 5
    assert "</DSML｜parameter>" not in rendered
    assert "</parameter>" not in rendered
    for marker in PARAMETER_END_MARKERS:
        assert f"&lt;{marker[1:]}" in rendered
        assert f"\\u003c{marker[1:]}" in rendered
    assert "&amp;lt;/parameter>" in rendered
    assert "&amp;amp;lt;/parameter>" in rendered
    assert "&amp;amp;amp;lt;/parameter>" in rendered

    parsed = parse_generated_message(rendered)

    assert parsed.status is DsmlParseStatus.COMPLETE
    assert parsed.calls[0].arguments == {
        "command": "echo a > b && echo &",
        "unsafe": close_markers,
        "literal": literal_entities,
        "payload": {
            "value": "</｜DSML｜parameter>",
            "variants": list(PARAMETER_END_MARKERS),
        },
        "nested_entities": nested_entities,
    }


def test_render_tool_result_escapes_xml_text() -> None:
    assert (
        render_tool_result("1 < 2 && 3 > 2")
        == "<tool_result>1 &lt; 2 &amp;&amp; 3 &gt; 2</tool_result>"
    )


def test_parse_generated_dsml_extracts_content_calls_and_raw() -> None:
    text = (
        "<think>Need math.</think>I will calculate.\n\n"
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="math.calculator">\n'
        '<｜DSML｜parameter name="expression" string="true">'
        "2 > 1 && echo &"
        "</｜DSML｜parameter>\n"
        '<｜DSML｜parameter name="precision" string="false">'
        "2"
        "</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls> ignored"
    )

    parsed = parse_generated_message(text)

    assert parsed.status is DsmlParseStatus.COMPLETE
    assert parsed.content == "I will calculate."
    assert parsed.reasoning == "Need math."
    assert (
        parsed.raw_dsml
        == "\n\n<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="math.calculator">\n'
        '<｜DSML｜parameter name="expression" string="true">'
        "2 > 1 && echo &"
        "</｜DSML｜parameter>\n"
        '<｜DSML｜parameter name="precision" string="false">'
        "2"
        "</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>"
    )
    assert len(parsed.calls) == 1
    call = parsed.calls[0]
    assert call.id is not None
    assert call.name == "math.calculator"
    assert call.arguments == {
        "expression": "2 > 1 && echo &",
        "precision": 2,
    }
    assert parse_tool_calls(text) == parsed.calls


@pytest.mark.parametrize(
    ("text", "arguments"),
    [
        (
            (
                "<DSML｜tool_calls>\n"
                '<DSML｜invoke name="math.calculator">\n'
                '<DSML｜parameter name="x" string="false">1'
                "</DSML｜parameter>\n"
                "</DSML｜invoke>\n"
                "</DSML｜tool_calls>"
            ),
            {"x": 1},
        ),
        (
            (
                "<tool_calls>\n"
                '<invoke name="math.calculator">\n'
                '<parameter name="x" string="true">1</parameter>\n'
                "</invoke>\n"
                "</tool_calls>"
            ),
            {"x": "1"},
        ),
    ],
)
def test_parse_generated_dsml_accepts_ds4_marker_variants(
    text: str,
    arguments: dict[str, object],
) -> None:
    parsed = parse_generated_message(text)

    assert parsed.status is DsmlParseStatus.COMPLETE
    assert parsed.calls[0].name == "math.calculator"
    assert parsed.calls[0].arguments == arguments


def test_parse_generated_dsml_accepts_xml_attribute_quote_variants() -> None:
    text = (
        "<tool_calls>"
        "<invoke name = 'math.calculator' id = 'call_1'>\n  "
        "<parameter name = 'expression' string = 'true'>2 + 2</parameter>"
        "\n\n"
        '<parameter name="precision" string = \'false\'>2</parameter>'
        "\n"
        "</invoke>"
        "</tool_calls>"
    )

    parsed = parse_generated_message(text)

    assert parsed.status is DsmlParseStatus.COMPLETE
    assert parsed.calls[0].id == "call_1"
    assert parsed.calls[0].name == "math.calculator"
    assert parsed.calls[0].arguments == {
        "expression": "2 + 2",
        "precision": 2,
    }


def test_parse_generated_dsml_omitted_string_attribute_is_text() -> None:
    text = (
        "<tool_calls>"
        '<invoke name="math.calculator">'
        '<parameter name="expression">{"x":1}</parameter>'
        "</invoke>"
        "</tool_calls>"
    )

    parsed = parse_generated_message(text)

    assert parsed.status is DsmlParseStatus.COMPLETE
    assert parsed.calls[0].arguments == {"expression": '{"x":1}'}


def test_parse_generated_dsml_extracts_multiple_invokes_in_order() -> None:
    text = (
        "<tool_calls>"
        "\n  "
        '<invoke name="math.calculator">'
        '<parameter name="expression" string="true">'
        "first call has a long enough body to catch cursor overshoot"
        "</parameter>"
        "</invoke>"
        "\n\n"
        '<invoke name="math.sqrt">'
        '<parameter name="value" string="false">16</parameter>'
        "</invoke>"
        "\n"
        "</tool_calls>"
    )

    parsed = parse_generated_message(text)

    assert parsed.status is DsmlParseStatus.COMPLETE
    assert [call.name for call in parsed.calls] == [
        "math.calculator",
        "math.sqrt",
    ]
    assert parsed.calls[0].arguments == {
        "expression": (
            "first call has a long enough body to catch cursor overshoot"
        )
    }
    assert parsed.calls[1].arguments == {"value": 16}


def test_parse_generated_dsml_without_tool_calls_returns_content() -> None:
    parsed = parse_generated_message("<think>hidden</think>visible")

    assert parsed.content == "visible"
    assert parsed.reasoning == "hidden"
    assert parsed.calls == ()
    assert parsed.raw_dsml is None
    assert parse_tool_calls("plain text") is None
    assert split_reasoning("plain text") == ("plain text", None)


def test_tool_call_start_span_returns_exact_span() -> None:
    assert tool_call_start_span("hello\n\n<tool_calls>") == (5, 19)
    assert tool_call_start_span("plain") is None


def test_stream_argument_deltas_emits_complete_parameter_values() -> None:
    raw = (
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="math.calculator">\n'
        '<｜DSML｜parameter name="expression" string="true">2 + 2'
        "</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>"
    )

    deltas, offset = stream_argument_deltas(raw, 0)

    assert deltas == ("2 + 2",)
    assert offset == raw.index("</｜DSML｜parameter>")


def test_stream_argument_deltas_withholds_incomplete_close_tags() -> None:
    prefix = (
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="math.calculator">\n'
        '<｜DSML｜parameter name="expression" string="true">'
    )
    raw = f"{prefix}{'x' * 30}</｜DSML｜para"
    keep = max(len(marker) for marker in PARAMETER_END_MARKERS) - 1
    safe_count = max(0, len(raw) - len(prefix) - keep)

    deltas, offset = stream_argument_deltas(raw, 0)

    assert deltas == ("x" * safe_count,)
    assert offset == len(prefix) + safe_count

    completed = f"{raw}meter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    final_deltas, final_offset = stream_argument_deltas(completed, offset)

    assert final_deltas == ("x" * (30 - safe_count),)
    assert final_offset == completed.index("</｜DSML｜parameter>")


def test_stream_argument_deltas_emits_multiple_parameters_in_order() -> None:
    raw = (
        "<tool_calls>\n"
        '<invoke name="math.calculator">\n'
        '<parameter name="expression" string="true">2 + 2</parameter>\n'
        '<parameter name="precision" string="false">2</parameter>\n'
        "</invoke>\n"
        "</tool_calls>"
    )

    deltas, offset = stream_argument_deltas(raw, 0)

    assert deltas == ("2 + 2", "2")
    assert offset == raw.index("</parameter>", raw.index(">2</parameter>"))


def test_stream_argument_deltas_rejects_invalid_offsets() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        stream_argument_deltas("", -1)

    with pytest.raises(TypeError, match="integer character offset"):
        stream_argument_deltas("", False)


def test_stream_argument_deltas_does_not_emit_tag_text() -> None:
    raw = (
        "<DSML｜tool_calls>\n"
        '<DSML｜invoke name="math.calculator">\n'
        '<DSML｜parameter name="expression" string="true">2 + '
    )
    deltas, offset = stream_argument_deltas(raw, 0)

    assert deltas == ()
    assert offset == 0

    raw += "2</DSML｜parameter>\n"
    raw += "</DSML｜invoke>\n</DSML｜tool_calls>"
    deltas, next_offset = stream_argument_deltas(raw, offset)

    assert deltas == ("2 + 2",)
    assert next_offset == raw.index("</DSML｜parameter>")

    deltas, final_offset = stream_argument_deltas(raw, next_offset)

    assert deltas == ()
    assert final_offset == next_offset


def test_parse_generated_dsml_reports_incomplete_blocks() -> None:
    parsed = parse_generated_message(
        "answer\n\n<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="math.calculator">'
    )

    assert parsed.status is DsmlParseStatus.INCOMPLETE
    assert parsed.content == "answer"
    assert parsed.calls == ()
    assert (
        parsed.raw_dsml
        == '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="math.calculator">'
    )
    assert parsed.error == "missing closing tool_calls tag"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("visible\n\n<｜DSML｜tool", len("\n\n<｜DSML｜tool")),
        ("visible\n<DSML｜tool_calls", len("\n<DSML｜tool_calls")),
        ("visible<tool", len("<tool")),
        ("visible<tool_calls>", len("<tool_calls>")),
        ("visible only", 0),
    ],
)
def test_tool_call_start_suffix_length_tracks_possible_markers(
    text: str, expected: int
) -> None:
    assert tool_call_start_suffix_length(text) == expected


def test_tool_call_start_suffix_length_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        tool_call_start_suffix_length(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain text", DsmlToolCallBufferStatus.NONE),
        ("<｜DSM", DsmlToolCallBufferStatus.PREFIX),
        ("visible\n\n<｜DSML｜tool_calls", DsmlToolCallBufferStatus.PREFIX),
        ("<｜DSML｜tool_calls>", DsmlToolCallBufferStatus.OPEN),
        (
            "<｜DSML｜tool_calls></｜DSML｜tool_calls>",
            DsmlToolCallBufferStatus.CLOSED,
        ),
        ("<DSML｜tool_calls>", DsmlToolCallBufferStatus.OPEN),
        (
            "<DSML｜tool_calls></DSML｜tool_calls>",
            DsmlToolCallBufferStatus.CLOSED,
        ),
        ("<tool_calls>", DsmlToolCallBufferStatus.OPEN),
        ("<tool_calls></tool_calls>", DsmlToolCallBufferStatus.CLOSED),
    ],
)
def test_tool_call_buffer_status_tracks_dsml_marker_variants(
    text: str,
    expected: DsmlToolCallBufferStatus,
) -> None:
    assert tool_call_buffer_status(text) is expected


def test_tool_call_buffer_status_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        tool_call_buffer_status(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "error_match"),
    [
        (
            (
                "<｜DSML｜tool_calls>\n"
                '<｜DSML｜invoke name="math.calculator">\n'
                "</｜DSML｜tool_calls>"
            ),
            "missing a close tag",
        ),
        (
            (
                "<｜DSML｜tool_calls>\n"
                '<｜DSML｜invoke name="math.calculator">\n'
                '<｜DSML｜parameter name="precision" string="false">{bad}'
                "</｜DSML｜parameter>\n"
                "</｜DSML｜invoke>\n"
                "</｜DSML｜tool_calls>"
            ),
            "malformed JSON",
        ),
        (
            (
                "<｜DSML｜tool_calls>\n"
                "<｜DSML｜invoke>\n"
                "</｜DSML｜invoke>\n"
                "</｜DSML｜tool_calls>"
            ),
            "missing a name",
        ),
        (
            (
                "<tool_calls>"
                "<invoke name=math.calculator>"
                "</invoke>"
                "</tool_calls>"
            ),
            "invoke tag contains malformed attribute syntax",
        ),
        (
            (
                "<tool_calls>"
                '<invoke name="math.calculator" mode=fast>'
                "</invoke>"
                "</tool_calls>"
            ),
            "invoke tag contains malformed attribute syntax",
        ),
        (
            (
                "<tool_calls>"
                '<invoke name="math.calculator"id="call_1">'
                "</invoke>"
                "</tool_calls>"
            ),
            "invoke tag contains malformed attribute syntax",
        ),
        (
            (
                "<｜DSML｜tool_calls>\n"
                '<｜DSML｜invoke name="math.calculator">\n'
                "<｜DSML｜parameter>2</｜DSML｜parameter>\n"
                "</｜DSML｜invoke>\n"
                "</｜DSML｜tool_calls>"
            ),
            "parameter tag is missing a name",
        ),
        (
            (
                "<tool_calls>"
                '<invoke name="math.calculator">'
                '<parameter name="value" string=false>{"x":1}</parameter>'
                "</invoke>"
                "</tool_calls>"
            ),
            "parameter tag contains malformed attribute syntax",
        ),
        (
            (
                "<tool_calls>"
                '<invoke name="math.calculator">'
                '<parameter name="value" string="">{"x":1}</parameter>'
                "</invoke>"
                "</tool_calls>"
            ),
            'parameter tag string attribute must be "true" or "false"',
        ),
        (
            (
                "<tool_calls>"
                '<invoke name="math.calculator">'
                '<parameter name="value" string="maybe">{"x":1}</parameter>'
                "</invoke>"
                "</tool_calls>"
            ),
            'parameter tag string attribute must be "true" or "false"',
        ),
        (
            (
                "<tool_calls>"
                '<invoke name="math.calculator">'
                '<parameter name="value" string="False">{"x":1}</parameter>'
                "</invoke>"
                "</tool_calls>"
            ),
            'parameter tag string attribute must be "true" or "false"',
        ),
        (
            (
                "<tool_calls>"
                '<invoke name="math.calculator" name="math.sqrt">'
                "</invoke>"
                "</tool_calls>"
            ),
            "duplicate attribute",
        ),
        (
            (
                "<tool_calls>"
                '<invoke name="math.calculator">'
                '<parameter name="value" string="true" string="false">'
                "2 + 2"
                "</parameter>"
                "</invoke>"
                "</tool_calls>"
            ),
            "duplicate attribute",
        ),
        (
            (
                "<tool_calls>"
                '<invoke name="math.calculator">'
                '<parameter name="value" string="true">first</parameter>'
                '<parameter name="value" string="true">second</parameter>'
                "</invoke>"
                "</tool_calls>"
            ),
            "duplicate name",
        ),
        (
            (
                "<tool_calls>"
                '<invoke name="math.calculator">'
                '<parameter name="expression" string="true">'
                "first call has a long enough body to catch cursor overshoot"
                "</parameter>"
                "</invoke>"
                '<invoke name="broken">'
                '<parameter name="value">missing close'
                "</invoke>"
                "</tool_calls>"
            ),
            "parameter tag is missing a close tag",
        ),
        (
            (
                "<tool_calls>"
                "not an invoke"
                '<invoke name="math.calculator"></invoke>'
                "</tool_calls>"
            ),
            "text outside invoke tags",
        ),
        (
            (
                "<tool_calls>"
                '<invoke name="math.calculator"></invoke>'
                "not an invoke"
                '<invoke name="math.sqrt"></invoke>'
                "</tool_calls>"
            ),
            "text outside invoke tags",
        ),
        (
            (
                "<tool_calls>"
                '<invoke name="math.calculator"></invoke>'
                "not an invoke"
                "</tool_calls>"
            ),
            "text outside invoke tags",
        ),
        (
            (
                "<tool_calls>"
                '<invoke name="math.calculator">'
                "not a parameter"
                '<parameter name="expression" string="true">2 + 2'
                "</parameter>"
                "</invoke>"
                "</tool_calls>"
            ),
            "text outside parameter tags",
        ),
        (
            (
                "<tool_calls>"
                '<invoke name="math.calculator">'
                '<parameter name="expression" string="true">2 + 2'
                "</parameter>"
                "not a parameter"
                '<parameter name="precision" string="false">2</parameter>'
                "</invoke>"
                "</tool_calls>"
            ),
            "text outside parameter tags",
        ),
        (
            (
                "<tool_calls>"
                '<invoke name="math.calculator">'
                '<parameter name="expression" string="true">2 + 2'
                "</parameter>"
                "not a parameter"
                "</invoke>"
                "</tool_calls>"
            ),
            "text outside parameter tags",
        ),
    ],
)
def test_parse_generated_dsml_reports_malformed_blocks(
    text: str,
    error_match: str,
) -> None:
    parsed = parse_generated_message(text)

    assert parsed.status is DsmlParseStatus.MALFORMED
    assert parsed.calls == ()
    assert parsed.error is not None
    assert error_match in parsed.error


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
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
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
