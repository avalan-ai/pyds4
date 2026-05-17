from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias, cast

JsonValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]
ToolSchemaInput: TypeAlias = Mapping[str, object] | str
ToolSchemasInput: TypeAlias = (
    ToolSchemaInput | Iterable[ToolSchemaInput] | None
)
ReplayLookup: TypeAlias = Callable[[tuple["DsmlToolCall", ...]], str | None]

TOOL_CALLS_START = "<｜DSML｜tool_calls>"
TOOL_CALLS_END = "</｜DSML｜tool_calls>"
TOOL_CALL_START_PREFIXES = (
    "<｜DSML｜tool_calls",
    "<DSML｜tool_calls",
    "<tool_calls",
)
TOOL_CALL_START_MARKERS = tuple(
    f"{line_prefix}{marker_prefix}>"
    for marker_prefix in TOOL_CALL_START_PREFIXES
    for line_prefix in ("\n\n", "\n", "")
)
TOOL_CALL_END_MARKERS = (
    "</｜DSML｜tool_calls>",
    "</DSML｜tool_calls>",
    "</tool_calls>",
)
PARAMETER_END_MARKERS = (
    "</｜DSML｜parameter>",
    "</DSML｜parameter>",
    "</parameter>",
)
_ATTR_RE = re.compile(
    r"""([A-Za-z_][\w:-]*)\s*=\s*(?:"([^"]*)"|'([^']*)')"""
)
_INVOKE_START_RE = re.compile(
    r"<(?:｜DSML｜|DSML｜)?invoke\b([^>]*)>",
    re.DOTALL,
)
_INVOKE_END_RE = re.compile(
    r"</(?:｜DSML｜|DSML｜)?invoke>",
    re.DOTALL,
)
_PARAM_START_RE = re.compile(
    r"<(?:｜DSML｜|DSML｜)?parameter\b[^>]*>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<(?:｜DSML｜|DSML｜)?parameter\b([^>]*)>"
    r"(.*?)"
    r"</(?:｜DSML｜|DSML｜)?parameter>",
    re.DOTALL,
)
_TOOL_CALLS_START_RE = re.compile(
    r"\n?\n?<(?:(?:｜DSML｜|DSML｜)tool_calls|tool_calls)>",
)
_TOOL_CALLS_END_RE = re.compile(
    r"</(?:(?:｜DSML｜|DSML｜)tool_calls|tool_calls)>",
)


class DsmlMessageRole(StrEnum):
    """Name a DSML prompt message role."""

    ASSISTANT = "assistant"
    DEVELOPER = "developer"
    SYSTEM = "system"
    TOOL = "tool"
    USER = "user"


class DsmlParseStatus(StrEnum):
    """Name the status of a generated DSML parse."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    MALFORMED = "malformed"


class DsmlToolCallBufferStatus(StrEnum):
    """Name the status of a growing generated DSML tool-call buffer."""

    NONE = "none"
    PREFIX = "prefix"
    OPEN = "open"
    CLOSED = "closed"


def _validate_str(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")


def _validate_optional_str(name: str, value: str | None) -> None:
    if value is None:
        return
    _validate_str(name, value)


def _validate_nonempty_str(name: str, value: str) -> None:
    _validate_str(name, value)
    if not value:
        raise ValueError(f"{name} must not be empty.")


def _validate_non_negative_offset(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer character offset.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _normalize_json_value(name: str, value: object) -> JsonValue:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return value
    if isinstance(value, Mapping):
        return _normalize_json_object(name, value)
    if isinstance(value, list | tuple):
        return [_normalize_json_value(f"{name} item", item) for item in value]
    raise TypeError(f"{name} must be JSON-serializable.")


def _normalize_json_object(
    name: str,
    value: Mapping[str, object],
) -> JsonObject:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a dictionary.")
    result: JsonObject = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings.")
        result[key] = _normalize_json_value(f"{name}.{key}", item)
    return result


def _function_schema_from_schema(schema: JsonObject) -> JsonObject:
    if schema.get("type") != "function":
        return schema

    function_schema = schema.get("function")
    if not isinstance(function_schema, dict):
        raise ValueError("tool schema function payload must be a JSON object.")
    return function_schema


def _validate_tool_schema(name: str, schema: JsonObject) -> JsonObject:
    if not schema:
        raise ValueError(f"{name} must not be empty.")

    function_schema = _function_schema_from_schema(schema)
    function_name = function_schema.get("name")
    if not isinstance(function_name, str) or not function_name:
        raise ValueError(f"{name} must include a non-empty function name.")

    parameters = function_schema.get("parameters")
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError(f"{name} parameters must be a JSON object.")
    return schema


def _render_json_object(value: JsonObject) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )


def _validate_rendered_tool_schema(name: str, value: str) -> str:
    _validate_nonempty_str(name, value)
    rendered = value.strip()
    if not rendered:
        raise ValueError(f"{name} must not be empty.")

    for line_number, line in enumerate(rendered.splitlines(), start=1):
        if not line:
            raise ValueError(f"{name} line {line_number} must not be empty.")
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{name} line {line_number} must be valid JSON."
            ) from error
        if not isinstance(parsed, dict):
            raise ValueError(
                f"{name} line {line_number} must be a JSON object."
            )
        _validate_tool_schema(
            f"{name} line {line_number}",
            _normalize_json_object(f"{name} line {line_number}", parsed),
        )
    return rendered


@dataclass(frozen=True, slots=True)
class DsmlToolSchema:
    """Store a DSML tool schema as a dictionary or rendered JSON text."""

    schema: ToolSchemaInput

    def __post_init__(self) -> None:
        if isinstance(self.schema, str):
            object.__setattr__(
                self,
                "schema",
                _validate_rendered_tool_schema("schema", self.schema),
            )
            return
        if not isinstance(self.schema, Mapping):
            raise TypeError("schema must be a dictionary or JSON string.")

        normalized = _validate_tool_schema(
            "schema",
            _normalize_json_object("schema", self.schema),
        )
        object.__setattr__(self, "schema", normalized)

    @property
    def is_rendered(self) -> bool:
        """Return whether this schema was supplied as rendered JSON text."""
        return isinstance(self.schema, str)

    @property
    def function_schema(self) -> JsonObject:
        """Return the function payload for OpenAI-style wrapped schemas."""
        if isinstance(self.schema, str):
            first_line = self.schema.splitlines()[0]
            parsed = json.loads(first_line)
            if not isinstance(parsed, dict):
                raise ValueError("rendered schema line must be a JSON object.")
            return _function_schema_from_schema(
                _normalize_json_object("schema", parsed)
            )
        return _function_schema_from_schema(cast(JsonObject, self.schema))

    @property
    def rendered_json(self) -> str:
        """Return newline-delimited JSON schema text."""
        if isinstance(self.schema, str):
            return self.schema
        return _render_json_object(self.function_schema)


@dataclass(frozen=True, slots=True)
class DsmlToolCall:
    """Describe a framework-neutral DSML assistant tool call."""

    name: str
    arguments: Mapping[str, object] | JsonValue | None = None
    id: str | None = None

    def __post_init__(self) -> None:
        _validate_nonempty_str("name", self.name)
        _validate_optional_str("id", self.id)

        if self.arguments is None:
            normalized: JsonObject = {}
        elif isinstance(self.arguments, Mapping):
            normalized = _normalize_json_object("arguments", self.arguments)
        else:
            normalized = {
                "arguments": _normalize_json_value(
                    "arguments",
                    self.arguments,
                )
            }
        object.__setattr__(self, "arguments", normalized)


def _normalize_tool_calls(
    calls: Iterable[DsmlToolCall],
) -> tuple[DsmlToolCall, ...]:
    if isinstance(calls, str | bytes):
        raise TypeError("tool_calls must be an iterable of DsmlToolCall.")
    result = tuple(calls)
    for call in result:
        if not isinstance(call, DsmlToolCall):
            raise TypeError("tool_calls items must be DsmlToolCall objects.")
    return result


@dataclass(frozen=True, slots=True)
class DsmlMessage:
    """Represent a framework-neutral message in DSML prompt rendering."""

    role: DsmlMessageRole | str
    content: str
    reasoning: str | None = None
    tool_calls: Iterable[DsmlToolCall] = ()

    def __post_init__(self) -> None:
        try:
            role = DsmlMessageRole(self.role)
        except ValueError as error:
            raise ValueError(
                f"Unsupported DSML message role {self.role!r}."
            ) from error
        object.__setattr__(self, "role", role)

        _validate_str("content", self.content)
        _validate_optional_str("reasoning", self.reasoning)
        object.__setattr__(
            self,
            "tool_calls",
            _normalize_tool_calls(self.tool_calls),
        )


def normalize_tool_schemas(
    schemas: ToolSchemasInput,
) -> tuple[DsmlToolSchema, ...]:
    """Return validated DSML tool schemas."""
    if schemas is None:
        return ()
    if isinstance(schemas, str | Mapping):
        return (DsmlToolSchema(schemas),)
    if isinstance(schemas, bytes) or not isinstance(schemas, Iterable):
        raise TypeError(
            "tool_schemas must be a dictionary, JSON string, iterable, or "
            "None."
        )

    result = tuple(
        (
            schema
            if isinstance(schema, DsmlToolSchema)
            else DsmlToolSchema(schema)
        )
        for schema in schemas
    )
    return result


def tool_schema_text(schemas: ToolSchemasInput) -> str | None:
    """Return newline-delimited tool schema JSON or ``None`` when empty."""
    normalized = normalize_tool_schemas(schemas)
    if not normalized:
        return None
    return "\n".join(schema.rendered_json for schema in normalized)


def _normalize_messages(
    messages: Iterable[DsmlMessage],
) -> tuple[DsmlMessage, ...]:
    if isinstance(messages, str | bytes):
        raise TypeError("messages must be an iterable of DsmlMessage.")
    result = tuple(messages)
    for message in result:
        if not isinstance(message, DsmlMessage):
            raise TypeError("messages items must be DsmlMessage objects.")
    return result


@dataclass(frozen=True, slots=True)
class DsmlPrompt:
    """Describe framework-neutral DSML prompt rendering inputs."""

    system_content: str | None = None
    messages: Iterable[DsmlMessage] = ()
    tool_schemas: ToolSchemasInput = ()

    def __post_init__(self) -> None:
        _validate_optional_str("system_content", self.system_content)
        object.__setattr__(
            self,
            "messages",
            _normalize_messages(self.messages),
        )
        object.__setattr__(
            self,
            "tool_schemas",
            normalize_tool_schemas(self.tool_schemas),
        )

    @property
    def tool_schema_text(self) -> str | None:
        """Return newline-delimited tool schema JSON or ``None``."""
        return tool_schema_text(self.tool_schemas)

    @property
    def has_tool_prompt(self) -> bool:
        """Return whether prompt rendering should include tool context."""
        return bool(self.tool_schemas)


@dataclass(frozen=True, slots=True)
class DsmlParseResult:
    """Represent parsed DSML content, tool calls, and replay metadata."""

    content: str
    calls: Iterable[DsmlToolCall] = ()
    reasoning: str | None = None
    raw_dsml: str | None = None
    status: DsmlParseStatus | str = DsmlParseStatus.COMPLETE
    error: str | None = None

    def __post_init__(self) -> None:
        _validate_str("content", self.content)
        _validate_optional_str("reasoning", self.reasoning)
        _validate_optional_str("raw_dsml", self.raw_dsml)
        try:
            status = DsmlParseStatus(self.status)
        except ValueError as error:
            raise ValueError(
                f"Unsupported DSML parse status {self.status!r}."
            ) from error
        object.__setattr__(self, "status", status)
        _validate_optional_str("error", self.error)
        object.__setattr__(
            self,
            "calls",
            _normalize_tool_calls(self.calls),
        )


def render_prompt(
    prompt: DsmlPrompt,
    think_mode: object = None,
    *,
    replay_lookup: ReplayLookup | None = None,
) -> str:
    """Return a rendered DSML chat prompt including optional tool context."""
    if not isinstance(prompt, DsmlPrompt):
        raise TypeError("prompt must be a DsmlPrompt instance.")

    messages = cast(tuple[DsmlMessage, ...], prompt.messages)
    chat_messages = tuple(
        message
        for message in messages
        if message.role
        not in {
            DsmlMessageRole.DEVELOPER,
            DsmlMessageRole.SYSTEM,
        }
    )
    tool_schemas = prompt.tool_schema_text
    system_parts = [prompt.system_content] if prompt.system_content else []
    system_parts.extend(
        message.content
        for message in messages
        if message.role in {DsmlMessageRole.DEVELOPER, DsmlMessageRole.SYSTEM}
    )
    if tool_schemas:
        system_parts.append(_tools_prompt_text(tool_schemas))

    rendered = [
        "<｜begin▁of▁sentence｜>",
        "\n\n".join(system_parts),
    ]
    pending_assistant = False
    pending_tool_result = False
    think = _thinking_enabled(think_mode)
    tool_context = bool(tool_schemas) or any(
        message.tool_calls or message.role is DsmlMessageRole.TOOL
        for message in chat_messages
    )
    last_user_index = max(
        (
            index
            for index, message in enumerate(chat_messages)
            if message.role
            in {
                DsmlMessageRole.USER,
                DsmlMessageRole.TOOL,
            }
        ),
        default=-1,
    )

    for index, message in enumerate(chat_messages):
        if message.role is DsmlMessageRole.USER:
            rendered.extend(("<｜User｜>", message.content))
            pending_assistant = True
            pending_tool_result = False
        elif message.role is DsmlMessageRole.TOOL:
            if not pending_tool_result:
                rendered.append("<｜User｜>")
            rendered.append(render_tool_result(message.content))
            pending_assistant = True
            pending_tool_result = True
        elif message.role is DsmlMessageRole.ASSISTANT:
            if pending_assistant:
                rendered.append("<｜Assistant｜>")
                if think:
                    if tool_context or index > last_user_index:
                        rendered.extend(
                            (
                                "<think>",
                                message.reasoning or "",
                                "</think>",
                            )
                        )
                    else:
                        rendered.append("</think>")
                else:
                    rendered.append("</think>")
            rendered.append(message.content)
            rendered.append(
                render_tool_calls(message.tool_calls, replay_lookup)
            )
            rendered.append("<｜end▁of▁sentence｜>")
            pending_assistant = False
            pending_tool_result = False

    if pending_assistant:
        rendered.append("<｜Assistant｜>")
        rendered.append("<think>" if think else "</think>")
    return "".join(rendered)


def tools_prompt(tool_schemas: ToolSchemasInput) -> str | None:
    """Return DSML tool-use instructions or ``None`` when no schemas exist."""
    rendered_schemas = tool_schema_text(tool_schemas)
    if rendered_schemas is None:
        return None
    return _tools_prompt_text(rendered_schemas)


def render_tool_calls(
    calls: Iterable[DsmlToolCall],
    replay_lookup: ReplayLookup | None = None,
) -> str:
    """Return canonical DSML text for assistant tool calls."""
    normalized_calls = _normalize_tool_calls(calls)
    if not normalized_calls:
        return ""
    if replay_lookup is not None:
        replay = replay_lookup(normalized_calls)
        if replay is not None:
            return replay

    parts = ["\n\n", TOOL_CALLS_START, "\n"]
    for call in normalized_calls:
        parts.extend(
            (
                '<｜DSML｜invoke name="',
                _escape_attr(call.name),
                '">\n',
            )
        )
        arguments = cast(JsonObject, call.arguments)
        for name, value in arguments.items():
            parts.append(_render_parameter(str(name), value))
        parts.append("</｜DSML｜invoke>\n")
    parts.append(TOOL_CALLS_END)
    return "".join(parts)


def render_tool_result(content: str) -> str:
    """Return canonical DSML text for a tool result."""
    _validate_str("content", content)
    return f"<tool_result>{_escape_text(content)}</tool_result>"


def parse_tool_calls(text: str) -> tuple[DsmlToolCall, ...] | None:
    """Return DSML tool calls parsed from ``text``."""
    parsed = parse_generated_message(text)
    if not parsed.calls:
        return None
    return cast(tuple[DsmlToolCall, ...], parsed.calls)


def parse_generated_message(text: str) -> DsmlParseResult:
    """Parse generated DSML text into content, calls, and replay metadata."""
    _validate_str("text", text)
    start_match = _TOOL_CALLS_START_RE.search(text)
    if not start_match:
        content, reasoning = split_reasoning(text)
        return DsmlParseResult(content=content, reasoning=reasoning)

    content, reasoning = split_reasoning(text[: start_match.start()].rstrip())
    end_match = _TOOL_CALLS_END_RE.search(text[start_match.end() :])
    if not end_match:
        return DsmlParseResult(
            content=content,
            reasoning=reasoning,
            raw_dsml=text[start_match.start() :],
            status=DsmlParseStatus.INCOMPLETE,
            error="missing closing tool_calls tag",
        )

    block_start = start_match.end()
    block_end = block_start + end_match.start()
    raw_end = start_match.end() + end_match.end()
    block = text[block_start:block_end]
    calls, error = _parse_calls(block)
    if error is not None:
        return DsmlParseResult(
            content=content,
            reasoning=reasoning,
            raw_dsml=text[start_match.start() : raw_end],
            status=DsmlParseStatus.MALFORMED,
            error=error,
        )
    return DsmlParseResult(
        content=content,
        calls=calls,
        reasoning=reasoning,
        raw_dsml=text[start_match.start() : raw_end],
    )


def tool_call_start_span(text: str) -> tuple[int, int] | None:
    """Return the first generated DSML tool-call block start span."""
    _validate_str("text", text)
    match = _TOOL_CALLS_START_RE.search(text)
    return (match.start(), match.end()) if match else None


def tool_call_start_suffix_length(text: str) -> int:
    """Return trailing text length that may become a DSML start marker."""
    _validate_str("text", text)
    max_length = min(
        len(text), max(len(marker) for marker in TOOL_CALL_START_MARKERS)
    )
    for length in range(max_length, 0, -1):
        suffix = text[-length:]
        if any(
            marker.startswith(suffix) for marker in TOOL_CALL_START_MARKERS
        ):
            return length
    return 0


def tool_call_buffer_status(text: str) -> DsmlToolCallBufferStatus:
    """Classify a growing generated text buffer relative to DSML tool calls."""
    _validate_str("text", text)
    start_match: re.Match[str] | None = None
    for match in _TOOL_CALLS_START_RE.finditer(text):
        start_match = match

    if start_match is not None:
        if _TOOL_CALLS_END_RE.search(text[start_match.end() :]):
            return DsmlToolCallBufferStatus.CLOSED
        return DsmlToolCallBufferStatus.OPEN

    if tool_call_start_suffix_length(text):
        return DsmlToolCallBufferStatus.PREFIX
    return DsmlToolCallBufferStatus.NONE


def stream_argument_deltas(
    raw_dsml: str,
    emitted_until: int,
) -> tuple[tuple[str, ...], int]:
    """Return new DSML parameter-value deltas from a growing DSML block.

    ``emitted_until`` is an absolute Python string character offset into
    ``raw_dsml``. Pass the returned offset into the next call for the same
    growing block. Incomplete parameter close tags are retained so DSML tag
    text is not emitted as argument data.
    """
    _validate_str("raw_dsml", raw_dsml)
    _validate_non_negative_offset("emitted_until", emitted_until)

    deltas: list[str] = []
    cursor = 0
    new_emitted_until = emitted_until
    keep = max(len(marker) for marker in PARAMETER_END_MARKERS) - 1

    while True:
        start_match = _PARAM_START_RE.search(raw_dsml, cursor)
        if not start_match:
            break

        value_start = start_match.end()
        end_index = _first_parameter_end_index(raw_dsml, value_start)
        if end_index is None:
            value_end = max(value_start, len(raw_dsml) - keep)
            next_cursor = len(raw_dsml)
        else:
            value_end = end_index
            next_cursor = _parameter_end_after(raw_dsml, end_index)

        segment_start = max(value_start, new_emitted_until)
        if segment_start < value_end:
            deltas.append(raw_dsml[segment_start:value_end])
            new_emitted_until = value_end

        if end_index is None:
            break
        cursor = next_cursor

    return tuple(delta for delta in deltas if delta), new_emitted_until


def split_reasoning(text: str) -> tuple[str, str | None]:
    """Return visible content and optional DSML thinking text."""
    _validate_str("text", text)
    if text.startswith("<think>") and "</think>" in text:
        reasoning, content = text.removeprefix("<think>").split("</think>", 1)
        return content, reasoning
    return text, None


def _tools_prompt_text(tool_schemas: str) -> str:
    return (
        "## Tools\n\n"
        "You have access to a set of tools to help answer the user "
        "question. You can invoke tools by writing a "
        '"<｜DSML｜tool_calls>" block like the following:\n\n'
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="$TOOL_NAME">\n'
        '<｜DSML｜parameter name="$PARAMETER_NAME" '
        'string="true|false">$PARAMETER_VALUE'
        "</｜DSML｜parameter>\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        '<｜DSML｜invoke name="$TOOL_NAME2">\n'
        "...\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>\n\n"
        "String parameters should be specified as raw text and set "
        '`string="true"`. Preserve characters such as `>`, `&`, and '
        "`&&` exactly; never replace normal string characters with XML "
        "or HTML entity escapes. Only if a string value itself contains "
        "the exact closing parameter tag `</｜DSML｜parameter>`, write "
        "that tag as `&lt;/｜DSML｜parameter>` inside the value. For all "
        "other types (numbers, booleans, arrays, objects), pass the "
        'value in JSON format and set `string="false"`.\n\n'
        "If thinking_mode is enabled (triggered by <think>), you MUST "
        "output your complete reasoning inside <think>...</think> "
        "BEFORE any tool calls or final response.\n\n"
        "Otherwise, output directly after </think> with tool calls or "
        "final response.\n\n"
        "### Available Tool Schemas\n\n"
        f"{tool_schemas}\n\n"
        "You MUST strictly follow the above defined tool name and "
        "parameter schemas to invoke tool calls. Use the exact parameter "
        "names from the schemas."
    )


def _thinking_enabled(think_mode: object) -> bool:
    value = getattr(think_mode, "value", think_mode)
    return value in {"high", "max"}


def _parse_calls(block: str) -> tuple[tuple[DsmlToolCall, ...], str | None]:
    calls: list[DsmlToolCall] = []
    position = 0
    while True:
        match = _INVOKE_START_RE.search(block, position)
        if not match:
            if block[position:].strip():
                return (), "tool_calls block contains text outside invoke tags"
            return tuple(calls), None
        if block[position : match.start()].strip():
            return (), "tool_calls block contains text outside invoke tags"

        attrs, attr_error = _parse_attrs("invoke", match.group(1))
        if attr_error is not None:
            return (), attr_error
        name = attrs.get("name")
        if not name:
            return (), "invoke tag is missing a name attribute"

        invoke_end = _INVOKE_END_RE.search(block, match.end())
        if not invoke_end:
            return (), f"invoke tag for {name!r} is missing a close tag"

        body_end = invoke_end.start()
        body = block[match.end() : body_end]
        arguments, error = _parse_parameters(body)
        if error is not None:
            return (), error

        try:
            calls.append(
                DsmlToolCall(
                    id=attrs.get("id") or f"dsml_tool_{len(calls) + 1}",
                    name=html.unescape(name),
                    arguments=arguments,
                )
            )
        except (TypeError, ValueError) as error:
            return (), str(error)
        position = invoke_end.end()


def _parse_parameters(block: str) -> tuple[JsonObject, str | None]:
    arguments: JsonObject = {}
    cursor = 0
    while True:
        start_match = _PARAM_START_RE.search(block, cursor)
        if not start_match:
            if block[cursor:].strip():
                return (
                    arguments,
                    "invoke body contains text outside parameter tags",
                )
            return arguments, None
        if block[cursor : start_match.start()].strip():
            return (
                arguments,
                "invoke body contains text outside parameter tags",
            )

        param_match = _PARAM_RE.match(block, start_match.start())
        if not param_match:
            return arguments, "parameter tag is missing a close tag"

        attrs, attr_error = _parse_attrs("parameter", param_match.group(1))
        if attr_error is not None:
            return arguments, attr_error
        name = attrs.get("name")
        if not name:
            return arguments, "parameter tag is missing a name attribute"
        parameter_name = html.unescape(name)
        string_mode = attrs.get("string", "true")
        if string_mode not in {"true", "false"}:
            return (
                arguments,
                'parameter tag string attribute must be "true" or "false"',
            )
        raw_value = param_match.group(2)
        value: JsonValue
        if string_mode == "false":
            try:
                value = _normalize_json_value(
                    f"parameter {parameter_name}",
                    json.loads(raw_value),
                )
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                return (
                    arguments,
                    (
                        f"parameter {parameter_name!r} contains malformed"
                        f" JSON: {error}"
                    ),
                )
        else:
            value = html.unescape(raw_value)
        if parameter_name in arguments:
            return (
                arguments,
                f"parameter tag contains duplicate name {parameter_name!r}",
            )
        arguments[parameter_name] = value
        cursor = param_match.end()


def _first_parameter_end_index(text: str, start: int) -> int | None:
    indexes = [
        index
        for marker in PARAMETER_END_MARKERS
        for index in (text.find(marker, start),)
        if index >= 0
    ]
    return min(indexes) if indexes else None


def _parameter_end_after(text: str, index: int) -> int:
    for marker in PARAMETER_END_MARKERS:
        if text.startswith(marker, index):
            return index + len(marker)
    return index


def _parse_attrs(
    tag_name: str,
    text: str,
) -> tuple[dict[str, str], str | None]:
    attrs: dict[str, str] = {}
    position = 0
    while position < len(text):
        space_match = re.match(r"\s*", text[position:])
        whitespace_length = 0
        if space_match is not None:
            whitespace_length = space_match.end()
            position += whitespace_length
        if position >= len(text):
            return attrs, None
        if whitespace_length == 0:
            return (
                {},
                f"{tag_name} tag contains malformed attribute syntax",
            )

        attr_match = _ATTR_RE.match(text, position)
        if attr_match is None:
            return (
                {},
                f"{tag_name} tag contains malformed attribute syntax",
            )
        name, double_quoted, single_quoted = attr_match.groups()
        if name in attrs:
            return (
                {},
                f"{tag_name} tag contains duplicate attribute {name!r}",
            )
        value = double_quoted if double_quoted is not None else single_quoted
        attrs[name] = html.unescape(value or "")
        position = attr_match.end()
    return attrs, None


def _render_parameter(name: str, value: JsonValue) -> str:
    is_string = isinstance(value, str)
    rendered_value = (
        _escape_parameter_text(cast(str, value))
        if is_string
        else _escape_json_literal(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=False,
            )
        )
    )
    return (
        f'<｜DSML｜parameter name="{_escape_attr(name)}" '
        f'string="{"true" if is_string else "false"}">'
        f"{rendered_value}</｜DSML｜parameter>\n"
    )


def _escape_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _escape_text(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _escape_parameter_text(value: str) -> str:
    return value.replace("</｜DSML｜parameter>", "&lt;/｜DSML｜parameter>")


def _escape_json_literal(value: str) -> str:
    return value.replace("</｜DSML｜parameter>", "\\u003c/｜DSML｜parameter>")


__all__ = [
    "DsmlMessage",
    "DsmlMessageRole",
    "DsmlParseResult",
    "DsmlParseStatus",
    "DsmlPrompt",
    "DsmlToolCallBufferStatus",
    "DsmlToolCall",
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
]
