from __future__ import annotations

from types import ModuleType

from ._metadata import (
    CPU_DIAGNOSTIC_NOTE,
    PRODUCTION_TARGETS,
    SUPPORTED_NATIVE_BACKENDS,
)
from ._version import __ds4_native_backend__

_build_config: ModuleType | None
try:
    from . import _build_config as _build_config
except ImportError:  # Source-tree fallback before the extension is built.
    _build_config = None

_native: ModuleType | None
try:
    from . import _native as _native

    _native_import_error: ImportError | None = None
except ImportError as error:
    # Source-tree fallback before the extension is built.
    _native = None
    _native_import_error = error


def _normalize_backend(backend: str) -> str:
    if not isinstance(backend, str):
        return ""
    return backend.strip().lower()


def _built_extension_failed_to_import() -> bool:
    return _build_config is not None and _native_import_error is not None


def _native_import_failure_reason() -> str:
    if not _built_extension_failed_to_import():
        return ""
    return (
        "The pyds4 native extension could not be imported: "
        f"{_native_import_error}."
    )


def _native_extension_missing_reason() -> str:
    if _native is not None or _build_config is not None:
        return ""
    return "The pyds4 native extension has not been built or installed."


def _with_target_diagnostics(reason: str) -> str:
    if not reason:
        return ""

    parts = [reason.rstrip()]
    if "Supported production targets:" not in reason:
        parts.append(f"Supported production targets: {PRODUCTION_TARGETS}.")
    if CPU_DIAGNOSTIC_NOTE not in reason:
        parts.append(CPU_DIAGNOSTIC_NOTE)
    return " ".join(parts)


def is_backend_available(backend: str) -> bool:
    normalized = _normalize_backend(backend)
    if _native is not None:
        available = getattr(_native, "is_backend_available", None)
        if callable(available):
            return bool(available(normalized))
    return False


def backend_unavailable_reason(backend: str) -> str:
    normalized = _normalize_backend(backend)
    if _native is not None:
        reason = getattr(_native, "backend_unavailable_reason", None)
        if callable(reason):
            value = reason(normalized)
            if isinstance(value, str):
                return _with_target_diagnostics(value)

    if normalized not in SUPPORTED_NATIVE_BACKENDS:
        supported = ", ".join(SUPPORTED_NATIVE_BACKENDS)
        return (
            f"Unsupported DS4 native backend {backend!r}. "
            f"Supported native backends: {supported}. "
            f"Supported production targets: {PRODUCTION_TARGETS}. "
            f"{CPU_DIAGNOSTIC_NOTE}"
        )
    native_failure = _native_import_failure_reason()
    if native_failure:
        return (
            f"This pyds4 build was configured for {__ds4_native_backend__!r}, "
            f"but it is not usable. {native_failure} "
            f"Supported production targets: {PRODUCTION_TARGETS}. "
            f"{CPU_DIAGNOSTIC_NOTE}"
        )
    missing_extension = _native_extension_missing_reason()
    if missing_extension:
        return (
            "This pyds4 source tree has no usable native backend. "
            f"{missing_extension} "
            f"Supported production targets: {PRODUCTION_TARGETS}. "
            f"{CPU_DIAGNOSTIC_NOTE}"
        )
    if normalized == __ds4_native_backend__:
        return ""
    return (
        f"This pyds4 build was configured for {__ds4_native_backend__!r}. "
        f"Supported production targets: {PRODUCTION_TARGETS}. "
        f"{CPU_DIAGNOSTIC_NOTE}"
    )
