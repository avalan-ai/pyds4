from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
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
        return [
            _normalize_json_value(f"{name} item", item) for item in value
        ]
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
        raise ValueError(
            "tool schema function payload must be a JSON object."
        )
    return function_schema


def _validate_tool_schema(name: str, schema: JsonObject) -> JsonObject:
    if not schema:
        raise ValueError(f"{name} must not be empty.")

    function_schema = _function_schema_from_schema(schema)
    function_name = function_schema.get("name")
    if not isinstance(function_name, str) or not function_name:
        raise ValueError(
            f"{name} must include a non-empty function name."
        )

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
        schema
        if isinstance(schema, DsmlToolSchema)
        else DsmlToolSchema(schema)
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


__all__ = [
    "DsmlMessage",
    "DsmlMessageRole",
    "DsmlParseResult",
    "DsmlParseStatus",
    "DsmlPrompt",
    "DsmlToolCall",
    "DsmlToolSchema",
    "JsonObject",
    "JsonValue",
    "ToolSchemaInput",
    "ToolSchemasInput",
    "normalize_tool_schemas",
    "tool_schema_text",
]
