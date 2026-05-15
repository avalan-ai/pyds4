from __future__ import annotations

from types import ModuleType

from ._metadata import C_INT_MAX, THINK_MAX_MIN_CONTEXT
from .errors import Ds4GenerationError
from .types import ThinkMode

_native: ModuleType | None
try:
    from . import _native as _native
except ImportError:  # Source-tree fallback before the extension is built.
    _native = None


def think_mode_for_context(mode: ThinkMode, ctx_size: int) -> ThinkMode:
    """Return the effective thinking mode for a context size.

    DS4 downgrades max thinking below its pinned minimum context size. When
    the extension was built against DS4 source, delegate to the native helper;
    otherwise preserve the same pinned behavior in Python.
    """
    if isinstance(ctx_size, bool) or not isinstance(ctx_size, int):
        raise TypeError("DS4 context size must be a positive integer.")
    if ctx_size <= 0:
        raise ValueError("DS4 context size must be a positive integer.")
    if ctx_size > C_INT_MAX:
        raise ValueError(f"DS4 context size must be <= {C_INT_MAX}.")
    try:
        normalized_mode = ThinkMode(mode)
    except ValueError as error:
        raise ValueError(f"Unsupported DS4 think mode {mode!r}.") from error

    if _native is not None:
        helper = getattr(_native, "think_mode_for_context", None)
        if callable(helper):
            try:
                value = helper(normalized_mode.value, ctx_size)
            except (RuntimeError, ValueError) as error:
                detail = str(error) or type(error).__name__
                raise Ds4GenerationError(
                    f"think_mode_for_context failed: {detail}"
                ) from error
            try:
                return ThinkMode(value)
            except ValueError as error:
                raise Ds4GenerationError(
                    f"DS4 returned unsupported thinking mode {value!r}."
                ) from error

    if normalized_mode is ThinkMode.MAX and ctx_size < THINK_MAX_MIN_CONTEXT:
        return ThinkMode.HIGH
    return normalized_mode
