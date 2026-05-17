"""Import-safe Python surface for the DS4 native binding."""

from ._capabilities import capabilities
from ._metadata import (
    DS4_API_VERSION,
    DS4_COMMIT,
    REQUIRED_C_SYMBOLS,
)
from ._metal import (
    configure_metal_source_paths as _configure_metal_source_paths,
)
from ._version import (
    __ds4_api_version__,
    __ds4_available_backends__,
    __ds4_commit__,
    __ds4_import_safe__,
    __ds4_native_backend__,
    __ds4_symbols__,
    __version__,
)
from .asyncio import AsyncEngine, AsyncSession
from .availability import backend_unavailable_reason, is_backend_available
from .errors import (
    Ds4ApiVersionError,
    Ds4BackendUnavailable,
    Ds4Cancelled,
    Ds4ContextError,
    Ds4Error,
    Ds4GenerationError,
    Ds4InvalidModel,
    Ds4LoadError,
)
from .native import Engine, Session
from .thinking import think_mode_for_context
from .types import (
    Backend,
    Ds4Capabilities,
    EngineOptions,
    GenerationOptions,
    GenerationScoreOptions,
    GenerationStep,
    ProgressEvent,
    SamplingOptions,
    ThinkMode,
    TokenScore,
    TokenScoreMode,
)

_configure_metal_source_paths()
del _configure_metal_source_paths

__all__ = [
    "Backend",
    "AsyncEngine",
    "AsyncSession",
    "Ds4Capabilities",
    "DS4_API_VERSION",
    "DS4_COMMIT",
    "Ds4ApiVersionError",
    "Ds4BackendUnavailable",
    "Ds4Cancelled",
    "Ds4ContextError",
    "Ds4Error",
    "Ds4GenerationError",
    "Ds4InvalidModel",
    "Ds4LoadError",
    "Engine",
    "EngineOptions",
    "GenerationOptions",
    "GenerationScoreOptions",
    "GenerationStep",
    "ProgressEvent",
    "REQUIRED_C_SYMBOLS",
    "SamplingOptions",
    "Session",
    "ThinkMode",
    "TokenScore",
    "TokenScoreMode",
    "__ds4_api_version__",
    "__ds4_available_backends__",
    "__ds4_commit__",
    "__ds4_import_safe__",
    "__ds4_native_backend__",
    "__ds4_symbols__",
    "__version__",
    "backend_unavailable_reason",
    "capabilities",
    "is_backend_available",
    "think_mode_for_context",
]
