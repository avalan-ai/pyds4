from __future__ import annotations

from collections.abc import Collection
from types import ModuleType

from . import availability as _availability
from ._metadata import REQUIRED_C_SYMBOLS
from ._version import (
    __ds4_api_version__,
    __ds4_available_backends__,
    __ds4_commit__,
    __ds4_native_backend__,
    __ds4_symbols__,
)
from .errors import Ds4ApiVersionError
from .types import Ds4Capabilities

_native: ModuleType | None
try:
    from . import _native as _native
except ImportError:  # Source-tree fallback before the extension is built.
    _native = None


def _metadata_symbols() -> frozenset[str]:
    if not isinstance(__ds4_symbols__, Collection) or isinstance(
        __ds4_symbols__, (bytes, str)
    ):
        raise Ds4ApiVersionError(
            "DS4 binding metadata __ds4_symbols__ must be a collection of "
            "public C symbol names."
        )

    symbols: set[str] = set()
    for symbol in __ds4_symbols__:
        if not isinstance(symbol, str):
            raise Ds4ApiVersionError(
                "DS4 binding metadata __ds4_symbols__ must contain strings."
            )
        symbols.add(symbol)
    return frozenset(symbols)


def _validate_required_symbols() -> None:
    symbols = _metadata_symbols()
    missing = tuple(
        symbol for symbol in REQUIRED_C_SYMBOLS if symbol not in symbols
    )
    if missing:
        missing_symbols = ", ".join(missing)
        raise Ds4ApiVersionError(
            "DS4 binding API mismatch: missing required public C symbols "
            f"{missing_symbols}."
        )


def _native_type(name: str) -> object:
    if _native is None:
        return None
    return getattr(_native, name, None)


def _has_methods(target: object, *names: str) -> bool:
    return target is not None and all(
        callable(getattr(target, name, None)) for name in names
    )


def _has_attributes(target: object, *names: str) -> bool:
    return target is not None and all(hasattr(target, name) for name in names)


def _native_runtime_available() -> bool:
    return _native is not None and _availability.is_backend_available(
        __ds4_native_backend__
    )


def capabilities() -> Ds4Capabilities:
    """Return import-safe DS4 capabilities for this pyds4 build."""
    _validate_required_symbols()

    runtime_available = _native_runtime_available()
    engine_type = _native_type("EngineState")
    session_type = _native_type("SessionState")

    return Ds4Capabilities(
        backend=__ds4_native_backend__,
        ds4_commit=__ds4_commit__,
        ds4_api_version=__ds4_api_version__,
        required_symbols=tuple(REQUIRED_C_SYMBOLS),
        available_backends=tuple(__ds4_available_backends__),
        snapshots=runtime_available
        and _has_methods(session_type, "save_snapshot", "load_snapshot"),
        payloads=runtime_available
        and _has_methods(session_type, "save_payload", "load_payload"),
        logprobs=runtime_available
        and _has_methods(session_type, "token_logprob"),
        top_logprobs=runtime_available
        and _has_methods(session_type, "top_logprobs"),
        progress=runtime_available
        and _has_methods(
            session_type,
            "set_progress_wakeup_fd",
            "drain_progress_events",
        ),
        mtp=runtime_available
        and _has_attributes(engine_type, "has_mtp", "mtp_draft_tokens"),
        speculative_eval=runtime_available
        and _has_methods(session_type, "eval_speculative_argmax"),
    )


__all__ = ["capabilities"]
