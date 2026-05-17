from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType

from ._metadata import DS4_API_VERSION, DS4_COMMIT, REQUIRED_C_SYMBOLS

_build_config: ModuleType | None
try:
    from . import _build_config as _build_config
except ImportError:  # Source-tree fallback before the extension is built.
    _build_config = None

_native: ModuleType | None
try:
    from . import _native as _native
except ImportError:  # Source-tree fallback before the extension is built.
    _native = None


def _string_from_native(name: str, fallback: str) -> str:
    if _native is None:
        return fallback
    value = getattr(_native, name, fallback)
    return value if isinstance(value, str) else fallback


def _backend_from_build_config() -> str:
    if _build_config is None:
        return "cpu"
    value = getattr(_build_config, "NATIVE_BACKEND", "cpu")
    return value if isinstance(value, str) and value else "cpu"


def _available_backends_from_native() -> tuple[str, ...]:
    if _native is None:
        return ()
    available = getattr(_native, "available_backends", None)
    if callable(available):
        value = available()
        if isinstance(value, (list, tuple, set)):
            return tuple(str(backend) for backend in value)
    if getattr(_native, "__ds4_source_present__", True) and getattr(
        _native, "__ds4_backend_host_supported__", True
    ):
        return (__ds4_native_backend__,)
    return ()


def _source_tree_version() -> str:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return "0+unknown"
    match = re.search(r'(?m)^version = "([^"]+)"$', text)
    return match.group(1) if match is not None else "0+unknown"


__version__ = (
    getattr(_build_config, "VERSION", _source_tree_version())
    if _build_config is not None
    else _source_tree_version()
)
__ds4_import_safe__ = True
__ds4_commit__ = _string_from_native("__ds4_commit__", DS4_COMMIT)
__ds4_api_version__ = DS4_API_VERSION
__ds4_native_backend__ = _string_from_native(
    "__ds4_native_backend__", _backend_from_build_config()
)
__ds4_symbols__ = REQUIRED_C_SYMBOLS
__ds4_available_backends__ = _available_backends_from_native()
