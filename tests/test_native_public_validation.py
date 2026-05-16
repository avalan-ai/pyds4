from __future__ import annotations

import pytest

import pyds4


class _EngineState:
    closed = False

    def token_text(self, _: int) -> object:
        return "not bytes"

    def tokenize_text(self, _: str) -> list[int | bool]:
        return [1, True]

    def tokenize_rendered_chat(self, _: str) -> tuple[int, int]:
        return (1, 2)

    def chat_begin(self) -> list[int]:
        return [1, -1]

    def chat_append_message(
        self,
        tokens: list[int],
        role: str,
        content: str,
    ) -> None:
        tokens.extend([2, True])

    def chat_append_assistant_prefix(
        self,
        tokens: list[int],
        think_mode: str,
    ) -> None:
        tokens.extend([3, -1])

    def encode_chat_prompt(
        self,
        system: str | None,
        prompt: str,
        think_mode: str,
    ) -> object:
        return object()


class _SessionState:
    closed = False
    pos: object = 0
    ctx: object = 8
    tokens: object = [1, 2]

    def close(self) -> None:
        self.closed = True

    def argmax(self) -> object:
        return True

    def argmax_excluding(self, token_id: int) -> object:
        return -1

    def sample(
        self,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        seed: int | None,
    ) -> object:
        return object()

    def token_logprob(self, token_id: int) -> object:
        return "bad"

    def top_logprobs(self, k: int) -> object:
        return [(True, -0.25)]

    def save_snapshot(self) -> object:
        return "not bytes"

    def save_payload(self) -> object:
        return "not bytes"


class _FailingEngineState:
    closed = False

    def token_text(self, token_id: int) -> bytes:
        raise RuntimeError("native detail")

    def tokenize_text(self, text: str) -> list[int]:
        raise RuntimeError("native detail")

    def tokenize_rendered_chat(self, text: str) -> list[int]:
        raise RuntimeError("native detail")

    def chat_begin(self) -> list[int]:
        raise RuntimeError("native detail")

    def chat_append_message(
        self,
        tokens: list[int],
        role: str,
        content: str,
    ) -> None:
        raise RuntimeError("native detail")

    def chat_append_assistant_prefix(
        self,
        tokens: list[int],
        think_mode: str,
    ) -> None:
        raise RuntimeError("native detail")

    def encode_chat_prompt(
        self,
        system: str | None,
        prompt: str,
        think_mode: str,
    ) -> list[int]:
        raise RuntimeError("native detail")


class _PrefixedFailingEngineState:
    closed = False

    def token_text(self, token_id: int) -> bytes:
        raise RuntimeError("token_text failed: native detail")


class _SessionCreateFailingEngineState:
    closed = False

    def create_session(self, ctx_size: int) -> object:
        raise RuntimeError("native detail")


class _FailingSessionState:
    closed = False

    def sync(self, prompt_tokens: list[int] | tuple[int, ...]) -> None:
        raise RuntimeError("native detail")

    def eval(self, token_id: int) -> None:
        raise RuntimeError("native detail")

    def argmax(self) -> int:
        raise RuntimeError("native detail")

    def argmax_excluding(self, token_id: int) -> int:
        raise RuntimeError("native detail")

    def sample(
        self,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        seed: int | None,
    ) -> int:
        raise RuntimeError("native detail")

    def token_logprob(self, token_id: int) -> float:
        raise RuntimeError("native detail")

    def top_logprobs(self, k: int) -> list[tuple[int, float]]:
        raise RuntimeError("native detail")

    def rewind(self, pos: int) -> None:
        raise RuntimeError("native detail")

    def save_snapshot(self) -> bytes:
        raise RuntimeError("native detail")

    def load_snapshot(self, snapshot: bytes) -> None:
        raise RuntimeError("native detail")

    def save_payload(self) -> bytes:
        raise RuntimeError("native detail")

    def load_payload(self, payload: bytes) -> None:
        raise RuntimeError("native detail")

    def invalidate(self) -> None:
        raise RuntimeError("native detail")


def _engine_with_state(state: object) -> pyds4.Engine:
    engine = pyds4.Engine.__new__(pyds4.Engine)
    engine._state = state
    engine.options = pyds4.EngineOptions(
        model_path="model.gguf", backend="cpu"
    )
    return engine


def _session_with_state(state: object) -> pyds4.Session:
    session = pyds4.Session.__new__(pyds4.Session)
    session._state = state
    return session


def test_engine_rejects_malformed_native_token_list_results() -> None:
    engine = _engine_with_state(_EngineState())

    with pytest.raises(pyds4.Ds4GenerationError, match="token_text"):
        engine.token_text(1)

    with pytest.raises(pyds4.Ds4GenerationError, match="tokenize_text"):
        engine.tokenize_text("hello")

    assert engine.tokenize_rendered_chat("hello") == [1, 2]

    with pytest.raises(pyds4.Ds4GenerationError, match="chat_begin"):
        engine.chat_begin()

    with pytest.raises(pyds4.Ds4GenerationError, match="encode_chat_prompt"):
        engine.encode_chat_prompt(None, "hello", pyds4.ThinkMode.NONE)


def test_engine_rejects_malformed_chat_append_mutations() -> None:
    engine = _engine_with_state(_EngineState())

    message_tokens = [1]
    with pytest.raises(
        pyds4.Ds4GenerationError,
        match="chat_append_message",
    ):
        engine.chat_append_message(message_tokens, "user", "hello")
    assert message_tokens == [1, 2, True]

    prefix_tokens = [1]
    with pytest.raises(
        pyds4.Ds4GenerationError,
        match="chat_append_assistant_prefix",
    ):
        engine.chat_append_assistant_prefix(
            prefix_tokens,
            pyds4.ThinkMode.NONE,
        )
    assert prefix_tokens == [1, 3, -1]


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("token_text", (1,)),
        ("tokenize_text", ("hello",)),
        ("tokenize_rendered_chat", ("hello",)),
        ("chat_begin", ()),
        ("chat_append_message", ([1], "user", "hello")),
        ("chat_append_assistant_prefix", ([1], pyds4.ThinkMode.NONE)),
        ("encode_chat_prompt", (None, "hello", pyds4.ThinkMode.NONE)),
    ],
)
def test_engine_generation_failures_map_to_operation_errors(
    method_name: str,
    args: tuple[object, ...],
) -> None:
    engine = _engine_with_state(_FailingEngineState())

    with pytest.raises(
        pyds4.Ds4GenerationError,
        match=f"{method_name} failed: native detail",
    ):
        getattr(engine, method_name)(*args)


def test_generation_error_mapping_does_not_duplicate_operation_prefix() -> (
    None
):
    engine = _engine_with_state(_PrefixedFailingEngineState())

    with pytest.raises(pyds4.Ds4GenerationError) as caught:
        engine.token_text(1)

    assert str(caught.value) == "token_text failed: native detail"


def test_session_create_context_error_includes_requested_ctx_size() -> None:
    engine = _engine_with_state(_SessionCreateFailingEngineState())

    with pytest.raises(pyds4.Ds4ContextError) as caught:
        engine.create_session(123)

    message = str(caught.value)
    assert "ctx_size 123" in message
    assert "native detail" in message


@pytest.mark.parametrize(
    ("field", "value", "error_match"),
    [
        ("pos", True, "pos"),
        ("pos", -1, "pos"),
        ("ctx", False, "ctx"),
        ("ctx", 0, "ctx"),
        ("tokens", [1, True], "session tokens"),
        ("tokens", [1, -1], "session tokens"),
    ],
)
def test_session_rejects_malformed_native_inspection_values(
    field: str,
    value: object,
    error_match: str,
) -> None:
    state = _SessionState()
    setattr(state, field, value)
    session = _session_with_state(state)

    with pytest.raises(pyds4.Ds4GenerationError, match=error_match):
        getattr(session, field)


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("argmax", ()),
        ("argmax_excluding", (1,)),
        ("sample", (pyds4.SamplingOptions(),)),
        ("token_logprob", (1,)),
        ("top_logprobs", (1,)),
        ("save_snapshot", ()),
        ("save_payload", ()),
    ],
)
def test_session_rejects_malformed_native_token_results(
    method_name: str,
    args: tuple[object, ...],
) -> None:
    session = _session_with_state(_SessionState())

    with pytest.raises(pyds4.Ds4GenerationError, match=method_name):
        getattr(session, method_name)(*args)


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("sync", ([1],)),
        ("eval", (1,)),
        ("argmax", ()),
        ("argmax_excluding", (1,)),
        ("sample", (pyds4.SamplingOptions(),)),
        ("token_logprob", (1,)),
        ("top_logprobs", (1,)),
        ("rewind", (0,)),
        ("save_snapshot", ()),
        ("load_snapshot", (b"snapshot",)),
        ("save_payload", ()),
        ("load_payload", (b"payload",)),
        ("invalidate", ()),
    ],
)
def test_session_generation_failures_map_to_operation_errors(
    method_name: str,
    args: tuple[object, ...],
) -> None:
    session = _session_with_state(_FailingSessionState())

    with pytest.raises(
        pyds4.Ds4GenerationError,
        match=f"{method_name} failed: native detail",
    ):
        getattr(session, method_name)(*args)
