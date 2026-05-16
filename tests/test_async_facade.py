from __future__ import annotations

import asyncio
import gc
import importlib
import os
import threading
from collections.abc import Callable

import pytest

import pyds4

pyds4_asyncio = importlib.import_module("pyds4.asyncio")


class RecordingSession:
    close_started: threading.Event | None = None
    close_release: threading.Event | None = None

    def __init__(self, record: Callable[[str], None], ctx_size: int) -> None:
        self._record = record
        self._ctx_size = ctx_size
        self._tokens: list[int] = []
        self.closed = False

    @property
    def pos(self) -> int:
        self._record("session.pos")
        return len(self._tokens)

    @property
    def ctx(self) -> int:
        self._record("session.ctx")
        return self._ctx_size

    @property
    def tokens(self) -> list[int]:
        self._record("session.tokens")
        return list(self._tokens)

    def close(self) -> None:
        if self.closed:
            return
        self._record("session.close")
        if self.close_started is not None:
            self.close_started.set()
            assert self.close_release is not None
            self.close_release.wait(timeout=5)
        self.closed = True

    def sync(self, prompt_tokens: list[int] | tuple[int, ...]) -> None:
        self._record("session.sync")
        self._tokens = list(prompt_tokens)

    def eval(self, token_id: int) -> None:
        self._record("session.eval")
        if RecordingEngine.block_first_eval is not None:
            RecordingEngine.block_first_eval.set()
            assert RecordingEngine.release_first_eval is not None
            RecordingEngine.release_first_eval.wait(timeout=5)
        self._tokens.append(token_id)

    def argmax(self) -> int:
        self._record("session.argmax")
        return RecordingEngine.next_argmax_token

    def argmax_excluding(self, token_id: int) -> int:
        self._record(f"session.argmax_excluding:{token_id}")
        return 102

    def sample(self, options: pyds4.SamplingOptions) -> int:
        self._record(f"session.sample:{options.seed}")
        return 103

    def rewind(self, pos: int) -> None:
        self._record(f"session.rewind:{pos}")
        del self._tokens[pos:]

    def save_snapshot(self) -> bytes:
        self._record("session.save_snapshot")
        return b"snapshot"

    def load_snapshot(self, snapshot: bytes) -> None:
        self._record(f"session.load_snapshot:{snapshot!r}")
        self._tokens = [7]

    def save_payload(self) -> bytes:
        self._record("session.save_payload")
        return b"payload"

    def load_payload(self, payload: bytes) -> None:
        self._record(f"session.load_payload:{payload!r}")
        self._tokens = [8]

    def invalidate(self) -> None:
        self._record("session.invalidate")
        self._tokens.clear()


class RecordingProgressSession(RecordingSession):
    def __init__(self, record: Callable[[str], None], ctx_size: int) -> None:
        super().__init__(record, ctx_size)
        self._progress_events: list[pyds4.ProgressEvent] = []
        self._progress_wakeup_fd = -1

    def sync(self, prompt_tokens: list[int] | tuple[int, ...]) -> None:
        super().sync(prompt_tokens)
        self._progress_events.append(
            pyds4.ProgressEvent(
                event="prefill_chunk",
                current=len(prompt_tokens),
                total=len(prompt_tokens),
            )
        )
        if self._progress_wakeup_fd >= 0:
            os.write(self._progress_wakeup_fd, b"x")

    def _set_progress_wakeup_fd(self, fd: int) -> None:
        self._progress_wakeup_fd = fd

    def _drain_progress_events(self) -> list[pyds4.ProgressEvent]:
        events = list(self._progress_events)
        self._progress_events.clear()
        return events


class RecordingEngine:
    events: list[str] = []
    thread_ids: list[int] = []
    block_first_chat_append: threading.Event | None = None
    release_first_chat_append: threading.Event | None = None
    close_started: threading.Event | None = None
    close_release: threading.Event | None = None
    block_first_tokenize: threading.Event | None = None
    release_first_tokenize: threading.Event | None = None
    block_first_eval: threading.Event | None = None
    release_first_eval: threading.Event | None = None
    next_argmax_token = 101
    session_type: type[RecordingSession] = RecordingSession

    def __init__(self, options: pyds4.EngineOptions) -> None:
        self.options = options
        self.closed = False
        self._record("engine.open")

    @classmethod
    def reset(cls) -> None:
        cls.events = []
        cls.thread_ids = []
        cls.block_first_chat_append = None
        cls.release_first_chat_append = None
        cls.close_started = None
        cls.close_release = None
        cls.block_first_tokenize = None
        cls.release_first_tokenize = None
        cls.block_first_eval = None
        cls.release_first_eval = None
        RecordingSession.close_started = None
        RecordingSession.close_release = None
        cls.next_argmax_token = 101
        cls.session_type = RecordingSession

    @classmethod
    def _record(cls, event: str) -> None:
        cls.events.append(event)
        cls.thread_ids.append(threading.get_ident())

    @property
    def routed_quant_bits(self) -> int:
        self._record("engine.routed_quant_bits")
        return 2

    @property
    def has_mtp(self) -> bool:
        self._record("engine.has_mtp")
        return True

    @property
    def mtp_draft_tokens(self) -> int:
        self._record("engine.mtp_draft_tokens")
        return 1

    @property
    def eos_token_id(self) -> int:
        self._record("engine.eos_token_id")
        return 6

    def close(self) -> None:
        self._record("engine.close")
        if self.close_started is not None:
            self.close_started.set()
            assert self.close_release is not None
            self.close_release.wait(timeout=5)
        self.closed = True

    def create_session(self, ctx_size: int) -> RecordingSession:
        self._record(f"engine.create_session:{ctx_size}")
        return self.session_type(self._record, ctx_size)

    def token_text(self, token_id: int) -> bytes:
        self._record(f"engine.token_text:{token_id}")
        return bytes([token_id])

    def tokenize_text(self, text: str) -> list[int]:
        self._record(f"engine.tokenize_text:{text}")
        if text == "block" and self.block_first_tokenize is not None:
            self.block_first_tokenize.set()
            assert self.release_first_tokenize is not None
            self.release_first_tokenize.wait(timeout=5)
        return [ord(char) for char in text]

    def tokenize_rendered_chat(self, text: str) -> list[int]:
        self._record(f"engine.tokenize_rendered_chat:{text}")
        return [2000 + ord(char) for char in text]

    def chat_begin(self) -> list[int]:
        self._record("engine.chat_begin")
        return [1]

    def chat_append_message(
        self,
        tokens: list[int],
        role: str,
        content: str,
    ) -> None:
        self._record(f"engine.chat_append_message:{role}:{content}")
        tokens.append(2)
        if content == "block" and self.block_first_chat_append is not None:
            self.block_first_chat_append.set()
            assert self.release_first_chat_append is not None
            self.release_first_chat_append.wait(timeout=5)
        if role == "fail":
            raise RuntimeError("chat append failed")

    def chat_append_assistant_prefix(
        self,
        tokens: list[int],
        think_mode: pyds4.ThinkMode | str,
    ) -> None:
        self._record(f"engine.chat_append_assistant_prefix:{think_mode}")
        tokens.append(3)

    def encode_chat_prompt(
        self,
        system: str | None,
        prompt: str,
        think_mode: pyds4.ThinkMode | str,
    ) -> list[int]:
        self._record(
            f"engine.encode_chat_prompt:{system}:{prompt}:{think_mode}"
        )
        return [4, 5]


class _ValidatingSessionState:
    def __init__(self) -> None:
        self.closed = False
        self.pos = 0
        self.ctx = 64
        self.tokens: list[int] = []

    def close(self) -> None:
        self.closed = True

    def sync(self, prompt_tokens: list[int] | tuple[int, ...]) -> None:
        self.tokens = list(prompt_tokens)
        self.pos = len(self.tokens)

    def eval(self, token_id: int) -> None:
        self.tokens.append(token_id)
        self.pos = len(self.tokens)

    def argmax(self) -> int:
        return 101

    def argmax_excluding(self, token_id: int) -> int:
        return token_id + 1

    def sample(
        self,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        seed: int | None,
    ) -> int:
        return 103

    def rewind(self, pos: int) -> None:
        del self.tokens[pos:]
        self.pos = len(self.tokens)

    def save_snapshot(self) -> bytes:
        return bytes(self.tokens)

    def load_snapshot(self, snapshot: bytes) -> None:
        if not isinstance(snapshot, bytes):
            raise TypeError("snapshot must be bytes.")
        self.tokens = list(snapshot)
        self.pos = len(self.tokens)

    def save_payload(self) -> bytes:
        return bytes(self.tokens)

    def load_payload(self, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes.")
        self.tokens = list(payload)
        self.pos = len(self.tokens)

    def invalidate(self) -> None:
        self.tokens.clear()
        self.pos = 0


class _ValidatingEngineState:
    routed_quant_bits = 2
    has_mtp = False
    mtp_draft_tokens = 0
    eos_token_id = 6

    def __init__(self, session_state: object | None = None) -> None:
        self.closed = False
        self.session_state = session_state or _ValidatingSessionState()

    def close(self) -> None:
        self.closed = True

    def create_session(self, ctx_size: int) -> object:
        return self.session_state

    def token_text(self, token_id: int) -> bytes:
        return bytes([token_id])

    def tokenize_text(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def tokenize_rendered_chat(self, text: str) -> list[int]:
        return [2000 + ord(char) for char in text]

    def chat_begin(self) -> list[int]:
        return [1]

    def chat_append_message(
        self,
        tokens: list[int],
        role: str,
        content: str,
    ) -> None:
        tokens.extend(ord(char) for char in content)

    def chat_append_assistant_prefix(
        self,
        tokens: list[int],
        think_mode: str,
    ) -> None:
        tokens.append({"none": 3, "high": 4, "max": 5}[think_mode])

    def encode_chat_prompt(
        self,
        system: str | None,
        prompt: str,
        think_mode: str,
    ) -> list[int]:
        return [1, *[ord(char) for char in prompt]]


class _FailingEngineState(_ValidatingEngineState):
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


class _SessionCreateFailingEngineState(_ValidatingEngineState):
    def create_session(self, ctx_size: int) -> object:
        raise RuntimeError("native detail")


class _FailingSessionState(_ValidatingSessionState):
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


class _NextTokenEvalFailingSessionState(_ValidatingSessionState):
    def eval(self, token_id: int) -> None:
        raise RuntimeError("native detail")


def _engine_with_state(state: object) -> pyds4.Engine:
    engine = pyds4.Engine.__new__(pyds4.Engine)
    engine._state = state
    engine.options = pyds4.EngineOptions(
        model_path="model.gguf", backend="cpu"
    )
    return engine


@pytest.fixture
def recording_engine(monkeypatch: pytest.MonkeyPatch) -> type[RecordingEngine]:
    RecordingEngine.reset()
    monkeypatch.setattr(pyds4_asyncio, "_SyncEngine", RecordingEngine)
    return RecordingEngine


@pytest.fixture
def validating_sync_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> _ValidatingEngineState:
    state = _ValidatingEngineState()
    engine = _engine_with_state(state)
    monkeypatch.setattr(pyds4_asyncio, "_SyncEngine", lambda _: engine)
    return state


def test_async_exports_are_available() -> None:
    assert pyds4.AsyncEngine is pyds4_asyncio.AsyncEngine
    assert pyds4.AsyncSession is pyds4_asyncio.AsyncSession
    assert pyds4.GenerationStep is pyds4_asyncio.GenerationStep
    assert pyds4.ProgressEvent is pyds4_asyncio.ProgressEvent


def test_async_engine_methods_mirror_sync_validation_errors(
    validating_sync_engine: _ValidatingEngineState,
) -> None:
    async def scenario() -> None:
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as engine:
            with pytest.raises(TypeError, match="ctx_size"):
                await engine.create_session(False)  # type: ignore[arg-type]

            with pytest.raises(ValueError, match="ctx_size"):
                await engine.create_session(0)

            with pytest.raises(TypeError, match="token_id"):
                await engine.token_text(True)  # type: ignore[arg-type]

            with pytest.raises(TypeError, match="text"):
                await engine.tokenize_text(b"bytes")  # type: ignore[arg-type]

            with pytest.raises(TypeError, match="text"):
                await engine.tokenize_rendered_chat(123)  # type: ignore[arg-type]

            tokens = [1]
            with pytest.raises(TypeError, match="tokens"):
                await engine.chat_append_message(  # type: ignore[arg-type]
                    (1,),
                    "user",
                    "hello",
                )

            with pytest.raises(ValueError, match="Unsupported DS4 chat role"):
                await engine.chat_append_message(tokens, "tool", "hello")
            assert tokens == [1]

            with pytest.raises(TypeError, match="token"):
                await engine.chat_append_assistant_prefix(
                    [True],  # type: ignore[list-item]
                    pyds4.ThinkMode.NONE,
                )

            with pytest.raises(ValueError, match="Unsupported DS4 think mode"):
                await engine.chat_append_assistant_prefix(
                    tokens,
                    "turbo",
                )
            assert tokens == [1]

            with pytest.raises(TypeError, match="prompt"):
                await engine.encode_chat_prompt(  # type: ignore[arg-type]
                    None,
                    b"hello",
                    pyds4.ThinkMode.NONE,
                )

            assert await engine.chat_begin() == [1]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("metadata_name", "bad_value", "error_match"),
    [
        ("routed_quant_bits", True, "routed_quant_bits"),
        ("has_mtp", 1, "has_mtp"),
        ("mtp_draft_tokens", object(), "mtp_draft_tokens"),
        ("eos_token_id", None, "eos_token_id"),
    ],
)
def test_async_engine_metadata_mirrors_sync_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
    metadata_name: str,
    bad_value: object,
    error_match: str,
) -> None:
    state = _ValidatingEngineState()
    setattr(state, metadata_name, bad_value)
    engine = _engine_with_state(state)
    monkeypatch.setattr(pyds4_asyncio, "_SyncEngine", lambda _: engine)

    async def scenario() -> None:
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as async_engine:
            with pytest.raises(pyds4.Ds4LoadError, match=error_match):
                await getattr(async_engine, metadata_name)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("method_name", "call"),
    [
        ("token_text", lambda engine: engine.token_text(1)),
        ("tokenize_text", lambda engine: engine.tokenize_text("hello")),
        (
            "tokenize_rendered_chat",
            lambda engine: engine.tokenize_rendered_chat("hello"),
        ),
        ("chat_begin", lambda engine: engine.chat_begin()),
        (
            "chat_append_message",
            lambda engine: engine.chat_append_message([1], "user", "hello"),
        ),
        (
            "chat_append_assistant_prefix",
            lambda engine: engine.chat_append_assistant_prefix(
                [1],
                pyds4.ThinkMode.NONE,
            ),
        ),
        (
            "encode_chat_prompt",
            lambda engine: engine.encode_chat_prompt(
                None,
                "hello",
                pyds4.ThinkMode.NONE,
            ),
        ),
    ],
)
def test_async_engine_generation_failures_mirror_sync_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    call: Callable[[pyds4_asyncio.AsyncEngine], object],
) -> None:
    engine = _engine_with_state(_FailingEngineState())
    monkeypatch.setattr(pyds4_asyncio, "_SyncEngine", lambda _: engine)

    async def scenario() -> None:
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as async_engine:
            with pytest.raises(
                pyds4.Ds4GenerationError,
                match=f"{method_name} failed: native detail",
            ):
                await call(async_engine)

    asyncio.run(scenario())


def test_async_create_session_mirrors_sync_context_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine_with_state(_SessionCreateFailingEngineState())
    monkeypatch.setattr(pyds4_asyncio, "_SyncEngine", lambda _: engine)

    async def scenario() -> None:
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as async_engine:
            with pytest.raises(pyds4.Ds4ContextError) as caught:
                await async_engine.create_session(123)

        message = str(caught.value)
        assert "ctx_size 123" in message
        assert "native detail" in message

    asyncio.run(scenario())


def test_async_engine_uses_one_owner_thread_and_serializes_calls(
    recording_engine: type[RecordingEngine],
) -> None:
    async def scenario() -> None:
        main_thread_id = threading.get_ident()
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as engine:
            tokenized = await asyncio.gather(
                *(engine.tokenize_text(str(index)) for index in range(5))
            )
            assert tokenized == [[48], [49], [50], [51], [52]]
            assert await engine.eos_token_id == 6

            async with await engine.create_session(64) as session:
                await session.sync([1, 2])
                assert await session.pos == 2
                assert await session.ctx == 64
                assert await session.tokens == [1, 2]
                assert await session.argmax() == 101
                assert await session.argmax_excluding(101) == 102
                assert (
                    await session.sample(pyds4.SamplingOptions(seed=7)) == 103
                )
                await session.eval(104)
                assert await session.tokens == [1, 2, 104]
                assert await session.save_snapshot() == b"snapshot"
                await session.load_snapshot(b"snapshot")
                assert await session.tokens == [7]
                assert await session.save_payload() == b"payload"
                await session.load_payload(b"payload")
                assert await session.tokens == [8]

        assert engine.closed is True
        assert len(set(recording_engine.thread_ids)) == 1
        assert recording_engine.thread_ids[0] != main_thread_id

    asyncio.run(scenario())

    assert recording_engine.events == [
        "engine.open",
        "engine.tokenize_text:0",
        "engine.tokenize_text:1",
        "engine.tokenize_text:2",
        "engine.tokenize_text:3",
        "engine.tokenize_text:4",
        "engine.eos_token_id",
        "engine.create_session:64",
        "session.sync",
        "session.pos",
        "session.ctx",
        "session.tokens",
        "session.argmax",
        "session.argmax_excluding:101",
        "session.sample:7",
        "session.eval",
        "session.tokens",
        "session.save_snapshot",
        "session.load_snapshot:b'snapshot'",
        "session.tokens",
        "session.save_payload",
        "session.load_payload:b'payload'",
        "session.tokens",
        "session.close",
        "engine.close",
    ]


def test_async_session_progress_events_are_queued(
    recording_engine: type[RecordingEngine],
) -> None:
    recording_engine.session_type = RecordingProgressSession

    async def scenario() -> None:
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as engine:
            async with await engine.create_session(64) as session:
                await session.sync([1, 2, 3])

                event = await asyncio.wait_for(
                    session.progress.get(),
                    timeout=1,
                )
                assert event == pyds4.ProgressEvent(
                    event="prefill_chunk",
                    current=3,
                    total=3,
                )

    asyncio.run(scenario())


def test_async_session_methods_mirror_sync_validation_errors(
    validating_sync_engine: _ValidatingEngineState,
) -> None:
    async def scenario() -> None:
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as engine:
            async with await engine.create_session(64) as session:
                with pytest.raises(TypeError, match="prompt_tokens"):
                    await session.sync([True])  # type: ignore[list-item]

                with pytest.raises(TypeError, match="token_id"):
                    await session.eval(True)  # type: ignore[arg-type]

                with pytest.raises(ValueError, match="token_id"):
                    await session.argmax_excluding(-1)

                with pytest.raises(TypeError, match="SamplingOptions"):
                    await session.sample(object())  # type: ignore[arg-type]

                with pytest.raises(TypeError, match="pos"):
                    await session.rewind(False)  # type: ignore[arg-type]

                with pytest.raises(TypeError, match="snapshot"):
                    await session.load_snapshot("bad")  # type: ignore[arg-type]

                with pytest.raises(TypeError, match="payload"):
                    await session.load_payload(bytearray(b"bad"))  # type: ignore[arg-type]

                with pytest.raises(TypeError, match="advance"):
                    await session.next_token(advance=1)  # type: ignore[arg-type]

                with pytest.raises(TypeError, match="decode"):
                    await session.next_token(decode=1)  # type: ignore[arg-type]

                with pytest.raises(TypeError, match="stop_on_eos"):
                    await session.next_token(  # type: ignore[arg-type]
                        stop_on_eos=1,
                    )

                with pytest.raises(ValueError, match="exclude_token_id"):
                    await session.next_token(
                        pyds4.SamplingOptions(seed=1),
                        exclude_token_id=7,
                    )

                with pytest.raises(TypeError, match="token_id"):
                    await session.next_token(
                        exclude_token_id=True,  # type: ignore[arg-type]
                    )

                await session.sync([1, 2])
                assert await session.tokens == [1, 2]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("field", "bad_value", "error_match"),
    [
        ("pos", True, "pos"),
        ("ctx", 0, "ctx"),
        ("tokens", [1, True], "session tokens"),
    ],
)
def test_async_session_properties_mirror_sync_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad_value: object,
    error_match: str,
) -> None:
    session_state = _ValidatingSessionState()
    setattr(session_state, field, bad_value)
    state = _ValidatingEngineState(session_state)
    engine = _engine_with_state(state)
    monkeypatch.setattr(pyds4_asyncio, "_SyncEngine", lambda _: engine)

    async def scenario() -> None:
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as async_engine:
            async with await async_engine.create_session(64) as session:
                with pytest.raises(
                    pyds4.Ds4GenerationError, match=error_match
                ):
                    await getattr(session, field)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("method_name", "call"),
    [
        ("sync", lambda session: session.sync([1])),
        ("eval", lambda session: session.eval(1)),
        ("argmax", lambda session: session.argmax()),
        ("argmax_excluding", lambda session: session.argmax_excluding(1)),
        ("sample", lambda session: session.sample(pyds4.SamplingOptions())),
        ("rewind", lambda session: session.rewind(0)),
        ("save_snapshot", lambda session: session.save_snapshot()),
        ("load_snapshot", lambda session: session.load_snapshot(b"snapshot")),
        ("save_payload", lambda session: session.save_payload()),
        ("load_payload", lambda session: session.load_payload(b"payload")),
        ("invalidate", lambda session: session.invalidate()),
        ("argmax", lambda session: session.next_token()),
    ],
)
def test_async_session_generation_failures_mirror_sync_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    call: Callable[[pyds4_asyncio.AsyncSession], object],
) -> None:
    state = _ValidatingEngineState(_FailingSessionState())
    engine = _engine_with_state(state)
    monkeypatch.setattr(pyds4_asyncio, "_SyncEngine", lambda _: engine)

    async def scenario() -> None:
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as async_engine:
            async with await async_engine.create_session(64) as session:
                with pytest.raises(
                    pyds4.Ds4GenerationError,
                    match=f"{method_name} failed: native detail",
                ):
                    await call(session)

    asyncio.run(scenario())


def test_async_next_token_eval_failure_mirrors_sync_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _ValidatingEngineState(_NextTokenEvalFailingSessionState())
    engine = _engine_with_state(state)
    monkeypatch.setattr(pyds4_asyncio, "_SyncEngine", lambda _: engine)

    async def scenario() -> None:
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as async_engine:
            async with await async_engine.create_session(64) as session:
                with pytest.raises(
                    pyds4.Ds4GenerationError,
                    match="eval failed: native detail",
                ):
                    await session.next_token()

    asyncio.run(scenario())


def test_async_chat_append_mutates_original_only_after_success(
    recording_engine: type[RecordingEngine],
) -> None:
    async def scenario() -> None:
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as engine:
            tokens = [1]
            await engine.chat_append_message(tokens, "user", "hello")
            assert tokens == [1, 2]

            await engine.chat_append_assistant_prefix(
                tokens,
                pyds4.ThinkMode.NONE,
            )
            assert tokens == [1, 2, 3]

            failing_tokens = [9]
            with pytest.raises(RuntimeError, match="chat append failed"):
                await engine.chat_append_message(
                    failing_tokens,
                    "fail",
                    "boom",
                )
            assert failing_tokens == [9]

    asyncio.run(scenario())


def test_async_cancelled_chat_append_does_not_mutate_original(
    recording_engine: type[RecordingEngine],
) -> None:
    async def scenario() -> None:
        started = threading.Event()
        release = threading.Event()
        recording_engine.block_first_chat_append = started
        recording_engine.release_first_chat_append = release
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as engine:
            tokens = [1]
            task = asyncio.create_task(
                engine.chat_append_message(tokens, "user", "block")
            )
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set()

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert tokens == [1]

            release.set()
            assert await engine.tokenize_text("after") == [
                97,
                102,
                116,
                101,
                114,
            ]

    asyncio.run(scenario())

    assert "engine.chat_append_message:user:block" in recording_engine.events
    assert "engine.tokenize_text:after" in recording_engine.events


def test_async_next_token_selects_decodes_and_commits_in_one_job(
    recording_engine: type[RecordingEngine],
) -> None:
    async def scenario() -> None:
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as engine:
            async with await engine.create_session(64) as session:
                await session.sync([1, 2])
                first, second = await asyncio.gather(
                    session.next_token(
                        pyds4.SamplingOptions(seed=1),
                        decode=True,
                    ),
                    session.next_token(
                        pyds4.SamplingOptions(seed=2),
                        decode=True,
                    ),
                )

                assert first == pyds4.GenerationStep(
                    token_id=103,
                    is_eos=False,
                    advanced=True,
                    token_bytes=b"g",
                )
                assert second == pyds4.GenerationStep(
                    token_id=103,
                    is_eos=False,
                    advanced=True,
                    token_bytes=b"g",
                )
                assert await session.tokens == [1, 2, 103, 103]

    asyncio.run(scenario())

    assert recording_engine.events == [
        "engine.open",
        "engine.create_session:64",
        "session.sync",
        "session.sample:1",
        "engine.eos_token_id",
        "session.eval",
        "engine.token_text:103",
        "session.sample:2",
        "engine.eos_token_id",
        "session.eval",
        "engine.token_text:103",
        "session.tokens",
        "session.close",
        "engine.close",
    ]


def test_async_next_token_stops_before_eos_eval_and_decode(
    recording_engine: type[RecordingEngine],
) -> None:
    async def scenario() -> None:
        recording_engine.next_argmax_token = 6
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as engine:
            async with await engine.create_session(64) as session:
                await session.sync([1, 2])
                step = await session.next_token(decode=True)

                assert step == pyds4.GenerationStep(
                    token_id=6,
                    is_eos=True,
                    advanced=False,
                    token_bytes=None,
                )
                assert await session.tokens == [1, 2]

    asyncio.run(scenario())

    assert "session.eval" not in recording_engine.events
    assert "engine.token_text:6" not in recording_engine.events


def test_async_next_token_rejects_sampling_with_exclusion(
    recording_engine: type[RecordingEngine],
) -> None:
    async def scenario() -> None:
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as engine:
            async with await engine.create_session(64) as session:
                with pytest.raises(ValueError, match="exclude_token_id"):
                    await session.next_token(
                        pyds4.SamplingOptions(seed=1),
                        exclude_token_id=7,
                    )

    asyncio.run(scenario())


def test_async_queued_cancelled_job_is_not_run(
    recording_engine: type[RecordingEngine],
) -> None:
    async def scenario() -> None:
        started = threading.Event()
        release = threading.Event()
        recording_engine.block_first_tokenize = started
        recording_engine.release_first_tokenize = release
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as engine:
            first = asyncio.create_task(engine.tokenize_text("block"))
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set()

            second = asyncio.create_task(engine.tokenize_text("cancelled"))
            await asyncio.sleep(0)
            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second

            release.set()
            assert await first == [98, 108, 111, 99, 107]

    asyncio.run(scenario())

    assert "engine.tokenize_text:block" in recording_engine.events
    assert "engine.tokenize_text:cancelled" not in recording_engine.events


def test_async_in_flight_non_mutating_cancellation_discards_result(
    recording_engine: type[RecordingEngine],
) -> None:
    async def scenario() -> None:
        started = threading.Event()
        release = threading.Event()
        recording_engine.block_first_tokenize = started
        recording_engine.release_first_tokenize = release
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as engine:
            task = asyncio.create_task(engine.tokenize_text("block"))
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set()

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            release.set()
            assert await engine.tokenize_text("after") == [
                97,
                102,
                116,
                101,
                114,
            ]

    asyncio.run(scenario())

    assert "engine.tokenize_text:block" in recording_engine.events
    assert "engine.tokenize_text:after" in recording_engine.events


def test_async_engine_close_is_shielded_from_caller_cancellation(
    recording_engine: type[RecordingEngine],
) -> None:
    async def scenario() -> None:
        close_started = threading.Event()
        close_release = threading.Event()
        recording_engine.close_started = close_started
        recording_engine.close_release = close_release
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        engine = await pyds4_asyncio.AsyncEngine.open(options)
        close_task = asyncio.create_task(engine.aclose())
        for _ in range(100):
            if close_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert close_started.is_set()

        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_task

        close_release.set()
        await engine.aclose()
        assert engine.closed is True

    asyncio.run(scenario())

    assert recording_engine.events == ["engine.open", "engine.close"]


def test_async_session_close_is_shielded_from_caller_cancellation(
    recording_engine: type[RecordingEngine],
) -> None:
    async def scenario() -> None:
        close_started = threading.Event()
        close_release = threading.Event()
        RecordingSession.close_started = close_started
        RecordingSession.close_release = close_release
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as engine:
            session = await engine.create_session(64)
            close_task = asyncio.create_task(session.aclose())
            for _ in range(100):
                if close_started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert close_started.is_set()

            close_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await close_task

            close_release.set()
            await engine.tokenize_text("after")
            assert session.closed is True

    asyncio.run(scenario())

    assert recording_engine.events == [
        "engine.open",
        "engine.create_session:64",
        "session.close",
        "engine.tokenize_text:after",
        "engine.close",
    ]


def test_async_in_flight_mutating_cancellation_poisons_session(
    recording_engine: type[RecordingEngine],
) -> None:
    async def scenario() -> None:
        started = threading.Event()
        release = threading.Event()
        recording_engine.block_first_eval = started
        recording_engine.release_first_eval = release
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4_asyncio.AsyncEngine(options) as engine:
            session = await engine.create_session(64)
            await session.sync([1, 2])

            task = asyncio.create_task(session.eval(104))
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set()

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            with pytest.raises(pyds4.Ds4Cancelled, match="mutating operation"):
                await session.pos

            release.set()
            await session.aclose()

    asyncio.run(scenario())

    assert recording_engine.events == [
        "engine.open",
        "engine.create_session:64",
        "session.sync",
        "session.eval",
        "session.close",
        "engine.close",
    ]


def test_async_fake_native_context_managers_close_resources_once() -> None:
    native = pytest.importorskip("pyds4._native")
    if not getattr(native, "__ds4_fake_native__", False):
        pytest.skip("requires a PYDS4_USE_FAKE_DS4 build")

    async def scenario() -> None:
        native.fake_reset_counters()
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4.AsyncEngine(options) as engine:
            async with await engine.create_session(64) as session:
                await session.sync([1, 2])
                assert await session.pos == 2
                assert await session.ctx == 64
                assert await session.tokens == [1, 2]

        gc.collect()
        snapshot = dict(native.fake_counters())
        assert snapshot["engine_open_calls"] == 1
        assert snapshot["engine_close_calls"] == 1
        assert snapshot["session_create_calls"] == 1
        assert snapshot["session_free_calls"] == 1
        assert snapshot["token_live_allocations"] == 0
        assert (
            snapshot["last_session_free_sequence"]
            < snapshot["last_engine_close_sequence"]
        )

    asyncio.run(scenario())


def test_async_fake_native_progress_notifications_are_queued() -> None:
    native = pytest.importorskip("pyds4._native")
    if not getattr(native, "__ds4_fake_native__", False):
        pytest.skip("requires a PYDS4_USE_FAKE_DS4 build")

    async def scenario() -> None:
        native.fake_reset_counters()
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4.AsyncEngine(options) as engine:
            async with await engine.create_session(64) as session:
                await session.sync([1, 2, 3, 4])
                event = await asyncio.wait_for(
                    session.progress.get(),
                    timeout=1,
                )
                assert event == pyds4.ProgressEvent(
                    event="prefill_chunk",
                    current=4,
                    total=4,
                )

    asyncio.run(scenario())


def test_async_fake_native_snapshot_and_payload_round_trip() -> None:
    native = pytest.importorskip("pyds4._native")
    if not getattr(native, "__ds4_fake_native__", False):
        pytest.skip("requires a PYDS4_USE_FAKE_DS4 build")

    async def scenario() -> None:
        native.fake_reset_counters()
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4.AsyncEngine(options) as engine:
            async with await engine.create_session(64) as session:
                await session.sync([1, 2])
                await session.eval(1000 + ord("A"))
                snapshot = await session.save_snapshot()
                payload = await session.save_payload()

                await session.eval(1000 + ord("B"))
                assert await session.tokens == [
                    1,
                    2,
                    1000 + ord("A"),
                    1000 + ord("B"),
                ]

                await session.load_snapshot(snapshot)
                assert await session.tokens == [1, 2, 1000 + ord("A")]

                await session.invalidate()
                await session.load_payload(payload)
                assert await session.tokens == [1, 2, 1000 + ord("A")]

        counts = dict(native.fake_counters())
        assert counts["save_snapshot_calls"] == 1
        assert counts["load_snapshot_calls"] == 1
        assert counts["save_payload_calls"] >= 2
        assert counts["load_payload_calls"] >= 2
        assert counts["snapshot_live_allocations"] == 0

    asyncio.run(scenario())


def test_async_fake_native_mutating_cancellation_closes_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = pytest.importorskip("pyds4._native")
    if not getattr(native, "__ds4_fake_native__", False):
        pytest.skip("requires a PYDS4_USE_FAKE_DS4 build")

    async def scenario() -> None:
        native.fake_reset_counters()
        monkeypatch.setenv("PYDS4_FAKE_DELAY_EVAL_MS", "200")
        options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

        async with pyds4.AsyncEngine(options) as engine:
            session = await engine.create_session(64)
            await session.sync([1, 2])

            task = asyncio.create_task(session.eval(1000 + ord("A")))
            for _ in range(100):
                if dict(native.fake_counters())["eval_calls"] == 1:
                    break
                await asyncio.sleep(0.01)
            assert dict(native.fake_counters())["eval_calls"] == 1

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            with pytest.raises(pyds4.Ds4Cancelled, match="mutating operation"):
                await session.tokens

            await session.aclose()

        snapshot = dict(native.fake_counters())
        assert snapshot["eval_calls"] == 1
        assert snapshot["session_free_calls"] == 1
        assert snapshot["engine_close_calls"] == 1
        assert (
            snapshot["last_eval_sequence"]
            < snapshot["last_session_free_sequence"]
        )

    asyncio.run(scenario())
