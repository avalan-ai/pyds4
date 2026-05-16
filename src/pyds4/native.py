from __future__ import annotations

import ctypes
import os
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import replace
from math import isfinite
from numbers import Real
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn, TypeGuard

from ._metadata import C_INT_MAX
from .availability import backend_unavailable_reason, is_backend_available
from .errors import (
    Ds4BackendUnavailable,
    Ds4ContextError,
    Ds4Error,
    Ds4GenerationError,
    Ds4InvalidModel,
    Ds4LoadError,
)
from .types import EngineOptions, ProgressEvent, SamplingOptions, ThinkMode

_native: ModuleType | None
try:
    from . import _native as _native
except ImportError:  # Source-tree fallback before the extension is built.
    _native = None

ENGINE_CONSTRUCTOR_CALLS = 0
SESSION_CONSTRUCTOR_CALLS = 0

_INVALID_MODEL_ERROR_MARKERS = (
    "bad gguf",
    "cannot open model",
    "does not exist",
    "failed to open model",
    "file not found",
    "incompatible model",
    "invalid gguf",
    "invalid model",
    "is a directory",
    "missing tensor",
    "missing tokenizer",
    "must be a file",
    "no such file",
    "not a gguf",
    "unsupported gguf",
    "unsupported model",
    "for deepseek4 flash",
    "metadata key has a non-float type",
    "metadata key has a non-integer type",
    "required metadata key is missing",
    "required tensor is missing",
    "required tokenizer token is missing",
)
_BACKEND_UNAVAILABLE_ERROR_MARKERS = (
    "backend requested but",
    "backend unavailable",
    "gpu support is not compiled",
    "has no graph backend support",
    "linked with cuda",
    "linked with metal",
)
_DS4_GGUF_MAGIC = b"GGUF"
_DS4_MIN_GGUF_HEADER_BYTES = 32
_DS4_SUPPORTED_GGUF_VERSION = 3


class _CapturedNativeStderr:
    text = ""


def _raise_native_unavailable() -> NoReturn:
    raise Ds4LoadError(
        "pyds4 native engine binding is not implemented in this build."
    )


def _flush_c_stdio() -> None:
    try:
        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass


def _decode_native_stderr(data: bytes) -> str:
    lines = [
        line.strip()
        for line in data.decode("utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    return "\n".join(lines)


def _replay_native_stderr(text: str) -> None:
    if not text:
        return
    try:
        os.write(2, f"{text}\n".encode("utf-8", errors="replace"))
    except OSError:
        pass


@contextmanager
def _capture_native_stderr() -> Any:
    captured = _CapturedNativeStderr()
    if os.name != "posix":
        yield captured
        return

    original_fd = os.dup(2)
    try:
        with tempfile.TemporaryFile() as stderr_file:
            _flush_c_stdio()
            os.dup2(stderr_file.fileno(), 2)
            try:
                yield captured
            finally:
                _flush_c_stdio()
                stderr_file.seek(0)
                captured.text = _decode_native_stderr(stderr_file.read())
                os.dup2(original_fd, 2)
    finally:
        os.close(original_fd)


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    if value > C_INT_MAX:
        raise ValueError(f"{name} must be <= {C_INT_MAX}.")


def _validate_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    if value > C_INT_MAX:
        raise ValueError(f"{name} must be <= {C_INT_MAX}.")
    return value


def _validate_str(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")


def _is_token_id(value: object) -> TypeGuard[int]:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        and value <= C_INT_MAX
    )


def _validate_token_id(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer token id.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    if value > C_INT_MAX:
        raise ValueError(f"{name} must be <= {C_INT_MAX}.")


def _validate_token_sequence_for_native(
    name: str,
    value: object,
    *,
    allow_tuple: bool = False,
) -> None:
    tokens: Iterable[object]
    if isinstance(value, list):
        tokens = value
    elif allow_tuple and isinstance(value, tuple):
        tokens = value
    else:
        container = "list or tuple" if allow_tuple else "list"
        raise TypeError(f"{name} must be a {container} of token ids.")
    for token in tokens:
        _validate_token_id(name, token)


def _validate_token_list(tokens: object, name: str) -> list[int]:
    if isinstance(tokens, tuple):
        tokens = list(tokens)
    if isinstance(tokens, list) and all(
        _is_token_id(token) for token in tokens
    ):
        return tokens
    raise Ds4GenerationError(
        f"{name} must be a list of non-negative C int token ids."
    )


def _validate_native_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Ds4GenerationError(f"{name} must be an integer.")
    if minimum is not None and value < minimum:
        raise Ds4GenerationError(f"{name} must be >= {minimum}.")
    return value


def _validate_token_result(operation: str, value: object) -> int:
    if _is_token_id(value):
        return value
    raise Ds4GenerationError(
        f"{operation} failed: DS4 returned invalid token id."
    )


def _validate_bytes_result(operation: str, value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    raise Ds4GenerationError(
        f"{operation} failed: DS4 returned non-bytes result."
    )


def _validate_bytes_input(name: str, value: object) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes.")
    return value


def _generation_exception(
    operation: str,
    error: RuntimeError,
) -> Ds4GenerationError:
    message = str(error) or type(error).__name__
    prefix = f"{operation} failed"
    if message.startswith(prefix):
        return Ds4GenerationError(message)
    return Ds4GenerationError(f"{prefix}: {message}")


def _validate_sampling_real(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number.")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum:g}.")
    return result


def _validate_sampling_options(
    options: SamplingOptions,
) -> tuple[float, int, float, float, int | None]:
    if not isinstance(options, SamplingOptions):
        raise TypeError("options must be a SamplingOptions instance.")

    temperature = _validate_sampling_real(
        "temperature",
        options.temperature,
        minimum=0.0,
    )
    top_k = _validate_non_negative_int("top_k", options.top_k)
    top_p = _validate_sampling_real(
        "top_p",
        options.top_p,
        minimum=0.0,
        maximum=1.0,
    )
    min_p = _validate_sampling_real(
        "min_p",
        options.min_p,
        minimum=0.0,
        maximum=1.0,
    )
    seed = options.seed
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int)
    ):
        raise TypeError("seed must be an integer or None.")
    return temperature, top_k, top_p, min_p, seed


def _coerce_think_mode(value: ThinkMode | str) -> ThinkMode:
    try:
        return ThinkMode(value)
    except ValueError as error:
        raise ValueError(f"Unsupported DS4 think mode {value!r}.") from error


def _native_uses_fake_ds4() -> bool:
    return bool(getattr(_native, "__ds4_fake_native__", False))


def _validate_real_file_path(
    name: str,
    value: str,
    *,
    missing_error: type[Ds4InvalidModel] | type[Ds4LoadError],
) -> Path:
    if not value:
        raise missing_error(f"{name} path must not be empty.")
    path = Path(value).expanduser()
    if not path.exists():
        raise missing_error(f"{name} path does not exist: {path}.")
    if path.is_dir():
        raise missing_error(
            f"{name} path must be a file, got directory: {path}."
        )
    if not path.is_file():
        raise missing_error(f"{name} path must be a regular file: {path}.")
    return path


def _validate_ds4_gguf_header(name: str, path: Path) -> None:
    try:
        size = path.stat().st_size
        with path.open("rb") as file:
            header = file.read(8)
    except OSError as error:
        detail = error.strerror or str(error)
        raise Ds4InvalidModel(
            f"cannot open {name} '{path}': {detail}."
        ) from error

    if size < _DS4_MIN_GGUF_HEADER_BYTES:
        raise Ds4InvalidModel(f"{name} file is too small to be GGUF: {path}.")
    if header[:4] != _DS4_GGUF_MAGIC:
        raise Ds4InvalidModel(f"{name} is not a GGUF file: {path}.")

    version = int.from_bytes(header[4:8], byteorder="little", signed=False)
    if version != _DS4_SUPPORTED_GGUF_VERSION:
        raise Ds4InvalidModel(
            f"{name} only GGUF v{_DS4_SUPPORTED_GGUF_VERSION} is supported; "
            f"got v{version}: {path}."
        )


def _validate_real_model_path(name: str, model_path: str) -> str:
    path = _validate_real_file_path(
        name,
        model_path,
        missing_error=Ds4InvalidModel,
    )
    _validate_ds4_gguf_header(name, path)
    return str(path)


def _validate_directional_steering_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    path = _validate_real_file_path(
        "DS4 directional steering file",
        path_value,
        missing_error=Ds4LoadError,
    )
    return str(path)


def _validate_directional_steering_options(
    options: EngineOptions,
) -> str | None:
    if options.directional_steering_file is None and (
        options.directional_steering_attn != 0.0
        or options.directional_steering_ffn != 0.0
    ):
        raise Ds4LoadError(
            "DS4 directional steering needs directional_steering_file when "
            "directional steering scales are non-zero."
        )
    return _validate_directional_steering_path(
        options.directional_steering_file
    )


def _validate_real_engine_options(options: EngineOptions) -> EngineOptions:
    model_path = _validate_real_model_path("DS4 model", options.model_path)
    mtp_path = (
        _validate_real_model_path("DS4 MTP model", options.mtp_path)
        if options.mtp_path is not None
        else None
    )
    directional_steering_file = _validate_directional_steering_options(options)
    return replace(
        options,
        model_path=model_path,
        mtp_path=mtp_path,
        directional_steering_file=directional_steering_file,
    )


def _engine_open_exception(
    error: RuntimeError,
    native_stderr: str = "",
) -> Ds4BackendUnavailable | Ds4InvalidModel | Ds4LoadError:
    message = str(error) or type(error).__name__
    if native_stderr and native_stderr not in message:
        message = f"{message}\n{native_stderr}"
    lowered = message.lower()
    if any(marker in lowered for marker in _BACKEND_UNAVAILABLE_ERROR_MARKERS):
        return Ds4BackendUnavailable(message)
    if any(marker in lowered for marker in _INVALID_MODEL_ERROR_MARKERS):
        return Ds4InvalidModel(message)
    return Ds4LoadError(message)


def _context_exception(ctx_size: int, error: RuntimeError) -> Ds4ContextError:
    message = str(error) or type(error).__name__
    if f"ctx_size {ctx_size}" in message:
        return Ds4ContextError(message)
    return Ds4ContextError(f"ctx_size {ctx_size}: {message}")


class Engine:
    """Own a native DS4 engine handle."""

    __slots__ = ("_state", "options")

    def __init__(self, options: EngineOptions) -> None:
        global ENGINE_CONSTRUCTOR_CALLS

        ENGINE_CONSTRUCTOR_CALLS += 1
        if not isinstance(options, EngineOptions):
            raise TypeError("options must be an EngineOptions instance.")
        if not options.model_path:
            raise Ds4InvalidModel("DS4 model_path must not be empty.")

        backend = options.backend.value
        if not is_backend_available(backend):
            raise Ds4BackendUnavailable(backend_unavailable_reason(backend))

        state_type = getattr(_native, "EngineState", None)
        if state_type is None:
            _raise_native_unavailable()

        model_path = options.model_path
        if not _native_uses_fake_ds4():
            options = _validate_real_engine_options(options)
            model_path = options.model_path

        self.options = options
        captured_stderr = _CapturedNativeStderr()
        try:
            if _native_uses_fake_ds4():
                self._state = state_type(
                    model_path,
                    backend,
                    options.mtp_path,
                    options.n_threads,
                    options.mtp_draft_tokens,
                    options.mtp_margin,
                    options.directional_steering_file,
                    options.directional_steering_attn,
                    options.directional_steering_ffn,
                    options.warm_weights,
                    options.quality,
                )
            else:
                with _capture_native_stderr() as captured_stderr:
                    self._state = state_type(
                        model_path,
                        backend,
                        options.mtp_path,
                        options.n_threads,
                        options.mtp_draft_tokens,
                        options.mtp_margin,
                        options.directional_steering_file,
                        options.directional_steering_attn,
                        options.directional_steering_ffn,
                        options.warm_weights,
                        options.quality,
                    )
                if options.native_log:
                    _replay_native_stderr(captured_stderr.text)
        except (Ds4BackendUnavailable, Ds4InvalidModel, Ds4LoadError):
            raise
        except RuntimeError as error:
            raise _engine_open_exception(
                error, captured_stderr.text
            ) from error

    @property
    def closed(self) -> bool:
        return self._state is None or bool(self._state.closed)

    @property
    def routed_quant_bits(self) -> int:
        return self._metadata_int("routed_quant_bits")

    @property
    def has_mtp(self) -> bool:
        return self._metadata_bool("has_mtp")

    @property
    def mtp_draft_tokens(self) -> int:
        return self._metadata_int("mtp_draft_tokens")

    @property
    def eos_token_id(self) -> int:
        return self._metadata_int("eos_token_id")

    def close(self) -> None:
        state = self._state
        if state is not None:
            state.close()
            self._state = None

    def __enter__(self) -> "Engine":
        self._require_state()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        self.close()

    def create_session(self, ctx_size: int) -> "Session":
        _validate_positive_int("ctx_size", ctx_size)
        try:
            state = self._require_state()
            return Session._from_state(state.create_session(ctx_size))
        except Ds4Error:
            raise
        except RuntimeError as error:
            raise _context_exception(ctx_size, error) from error

    def token_text(self, token_id: int) -> bytes:
        _validate_token_id("token_id", token_id)
        try:
            return _validate_bytes_result(
                "token_text",
                self._require_state().token_text(token_id),
            )
        except (Ds4Error, TypeError, ValueError):
            raise
        except RuntimeError as error:
            raise _generation_exception("token_text", error) from error

    def tokenize_text(self, text: str) -> list[int]:
        _validate_str("text", text)
        try:
            return _validate_token_list(
                self._require_state().tokenize_text(text),
                "tokenize_text",
            )
        except Ds4Error:
            raise
        except RuntimeError as error:
            raise _generation_exception("tokenize_text", error) from error

    def tokenize_rendered_chat(self, text: str) -> list[int]:
        _validate_str("text", text)
        try:
            return _validate_token_list(
                self._require_state().tokenize_rendered_chat(text),
                "tokenize_rendered_chat",
            )
        except Ds4Error:
            raise
        except RuntimeError as error:
            raise _generation_exception(
                "tokenize_rendered_chat",
                error,
            ) from error

    def chat_begin(self) -> list[int]:
        try:
            return _validate_token_list(
                self._require_state().chat_begin(),
                "chat_begin",
            )
        except Ds4Error:
            raise
        except RuntimeError as error:
            raise _generation_exception("chat_begin", error) from error

    def chat_append_message(
        self,
        tokens: list[int],
        role: str,
        content: str,
    ) -> None:
        _validate_str("role", role)
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported DS4 chat role {role!r}.")
        _validate_str("content", content)
        _validate_token_sequence_for_native("tokens", tokens)
        try:
            self._require_state().chat_append_message(tokens, role, content)
            _validate_token_list(tokens, "chat_append_message")
        except Ds4Error:
            raise
        except (TypeError, ValueError):
            raise
        except RuntimeError as error:
            raise _generation_exception(
                "chat_append_message",
                error,
            ) from error

    def chat_append_assistant_prefix(
        self,
        tokens: list[int],
        think_mode: ThinkMode | str,
    ) -> None:
        mode = _coerce_think_mode(think_mode)
        _validate_token_sequence_for_native("tokens", tokens)
        try:
            self._require_state().chat_append_assistant_prefix(
                tokens,
                mode.value,
            )
            _validate_token_list(tokens, "chat_append_assistant_prefix")
        except Ds4Error:
            raise
        except (TypeError, ValueError):
            raise
        except RuntimeError as error:
            raise _generation_exception(
                "chat_append_assistant_prefix",
                error,
            ) from error

    def encode_chat_prompt(
        self,
        system: str | None,
        prompt: str,
        think_mode: ThinkMode | str,
    ) -> list[int]:
        if system is not None:
            _validate_str("system", system)
        _validate_str("prompt", prompt)
        mode = _coerce_think_mode(think_mode)
        try:
            return _validate_token_list(
                self._require_state().encode_chat_prompt(
                    system,
                    prompt,
                    mode.value,
                ),
                "encode_chat_prompt",
            )
        except Ds4Error:
            raise
        except RuntimeError as error:
            raise _generation_exception("encode_chat_prompt", error) from error

    def _require_state(self) -> Any:
        if self._state is None or self._state.closed:
            raise Ds4LoadError("DS4 engine is closed.")
        return self._state

    def _metadata_value(self, name: str) -> object:
        value = getattr(self._require_state(), name, None)
        if callable(value):
            value = value()
        if value is None:
            raise Ds4LoadError(f"DS4 engine metadata {name!r} is unavailable.")
        return value

    def _metadata_int(self, name: str) -> int:
        value = self._metadata_value(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise Ds4LoadError(
                f"DS4 engine metadata {name!r} must be an integer."
            )
        return value

    def _metadata_bool(self, name: str) -> bool:
        value = self._metadata_value(name)
        if not isinstance(value, bool):
            raise Ds4LoadError(
                f"DS4 engine metadata {name!r} must be a boolean."
            )
        return value


class Session:
    """Own a native DS4 session handle."""

    __slots__ = ("_state",)
    _state: Any | None

    def __init__(self, *_: object, **__: object) -> None:
        global SESSION_CONSTRUCTOR_CALLS

        SESSION_CONSTRUCTOR_CALLS += 1
        raise Ds4LoadError(
            "pyds4 Session objects are created by Engine.create_session()."
        )

    @classmethod
    def _from_state(cls, state: Any) -> "Session":
        global SESSION_CONSTRUCTOR_CALLS

        SESSION_CONSTRUCTOR_CALLS += 1
        session = cls.__new__(cls)
        session._state = state
        return session

    @property
    def closed(self) -> bool:
        return self._state is None or bool(self._state.closed)

    @property
    def pos(self) -> int:
        return _validate_native_int(
            "DS4 session pos",
            self._require_state().pos,
            minimum=0,
        )

    @property
    def ctx(self) -> int:
        return _validate_native_int(
            "DS4 session ctx",
            self._require_state().ctx,
            minimum=1,
        )

    @property
    def tokens(self) -> list[int]:
        return _validate_token_list(
            self._require_state().tokens,
            "session tokens",
        )

    def close(self) -> None:
        state = self._state
        if state is not None:
            state.close()
            self._state = None

    def __enter__(self) -> "Session":
        self._require_state()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        self.close()

    def sync(self, prompt_tokens: list[int] | tuple[int, ...]) -> None:
        state = self._require_state()
        _validate_token_sequence_for_native(
            "prompt_tokens",
            prompt_tokens,
            allow_tuple=True,
        )
        try:
            state.sync(prompt_tokens)
        except Ds4Error:
            raise
        except (TypeError, ValueError):
            raise
        except RuntimeError as error:
            raise _generation_exception("sync", error) from error

    def eval(self, token_id: int) -> None:
        _validate_token_id("token_id", token_id)
        try:
            self._require_state().eval(token_id)
        except Ds4Error:
            raise
        except (TypeError, ValueError):
            raise
        except RuntimeError as error:
            raise _generation_exception("eval", error) from error

    def argmax(self) -> int:
        try:
            return _validate_token_result(
                "argmax",
                self._require_state().argmax(),
            )
        except Ds4Error:
            raise
        except RuntimeError as error:
            raise _generation_exception("argmax", error) from error

    def argmax_excluding(self, token_id: int) -> int:
        _validate_token_id("token_id", token_id)
        try:
            return _validate_token_result(
                "argmax_excluding",
                self._require_state().argmax_excluding(token_id),
            )
        except Ds4Error:
            raise
        except (TypeError, ValueError):
            raise
        except RuntimeError as error:
            raise _generation_exception("argmax_excluding", error) from error

    def sample(self, options: SamplingOptions) -> int:
        temperature, top_k, top_p, min_p, seed = _validate_sampling_options(
            options
        )
        try:
            return _validate_token_result(
                "sample",
                self._require_state().sample(
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    seed,
                ),
            )
        except Ds4Error:
            raise
        except (TypeError, ValueError):
            raise
        except RuntimeError as error:
            raise _generation_exception("sample", error) from error

    def rewind(self, pos: int) -> None:
        _validate_non_negative_int("pos", pos)
        try:
            self._require_state().rewind(pos)
        except Ds4Error:
            raise
        except (TypeError, ValueError):
            raise
        except RuntimeError as error:
            raise _generation_exception("rewind", error) from error

    def save_snapshot(self) -> bytes:
        try:
            return _validate_bytes_result(
                "save_snapshot",
                self._require_state().save_snapshot(),
            )
        except Ds4Error:
            raise
        except RuntimeError as error:
            raise _generation_exception("save_snapshot", error) from error

    def load_snapshot(self, snapshot: bytes) -> None:
        snapshot = _validate_bytes_input("snapshot", snapshot)
        try:
            self._require_state().load_snapshot(snapshot)
        except Ds4Error:
            raise
        except TypeError:
            raise
        except RuntimeError as error:
            raise _generation_exception("load_snapshot", error) from error

    def save_payload(self) -> bytes:
        try:
            return _validate_bytes_result(
                "save_payload",
                self._require_state().save_payload(),
            )
        except Ds4Error:
            raise
        except RuntimeError as error:
            raise _generation_exception("save_payload", error) from error

    def load_payload(self, payload: bytes) -> None:
        payload = _validate_bytes_input("payload", payload)
        try:
            self._require_state().load_payload(payload)
        except Ds4Error:
            raise
        except TypeError:
            raise
        except RuntimeError as error:
            raise _generation_exception("load_payload", error) from error

    def invalidate(self) -> None:
        try:
            self._require_state().invalidate()
        except Ds4Error:
            raise
        except RuntimeError as error:
            raise _generation_exception("invalidate", error) from error

    def _set_progress_wakeup_fd(self, fd: int) -> None:
        state = self._require_state()
        setter = getattr(state, "set_progress_wakeup_fd", None)
        if setter is not None:
            setter(fd)

    def _drain_progress_events(self) -> list[ProgressEvent]:
        state = self._state
        if state is None or state.closed:
            return []
        drain = getattr(state, "drain_progress_events", None)
        if drain is None:
            return []
        events: list[ProgressEvent] = []
        for item in drain():
            try:
                event, current, total = item
            except (TypeError, ValueError):
                raise Ds4GenerationError(
                    "progress failed: DS4 returned malformed progress event."
                ) from None
            if not isinstance(event, str):
                raise Ds4GenerationError(
                    "progress failed: DS4 returned non-string event name."
                )
            events.append(
                ProgressEvent(
                    event=event,
                    current=_validate_native_int(
                        "progress current",
                        current,
                        minimum=0,
                    ),
                    total=_validate_native_int(
                        "progress total",
                        total,
                        minimum=0,
                    ),
                )
            )
        return events

    def _require_state(self) -> Any:
        if self._state is None or self._state.closed:
            raise Ds4ContextError("DS4 session is closed.")
        return self._state
