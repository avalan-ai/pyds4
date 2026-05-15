from __future__ import annotations

import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

import pyds4

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPLICIT_ENV = ("PYDS4_MODEL", "PYDS4_BACKEND", "PYDS4_CTX")
_LOCAL_DS4_MODEL = Path(".local/ds4/ds4flash.gguf")
_PROFILE_NAMES = ("macos-metal", "linux-cuda", "cpu-diagnostic")


class _RealDs4ConfigError(ValueError):
    pass


class _RealDs4Skip(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _ResolvedRealDs4Settings:
    model_path: Path
    backend_name: str
    ctx_size: int
    profile_name: str | None = None


def _env_value(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    return value if value else None


def _parse_ctx_size(value: str, name: str) -> int:
    try:
        ctx_size = int(value)
    except ValueError as error:
        raise _RealDs4ConfigError(
            f"{name} must be an integer: {value!r}"
        ) from error
    if ctx_size <= 0:
        raise _RealDs4ConfigError(f"{name} must be positive: {ctx_size}")
    return ctx_size


def _model_path(
    environ: Mapping[str, str],
    env_name: str,
    repo_root: Path,
    default: Path | None,
) -> Path:
    value = _env_value(environ, env_name)
    if value is None:
        value = _env_value(environ, "PYDS4_MODEL")
    if value is not None:
        return Path(value).expanduser()
    if default is None:
        raise _RealDs4Skip(
            f"{env_name} or PYDS4_MODEL must point to a DS4 GGUF."
        )
    return repo_root / default


def _ctx_size(
    environ: Mapping[str, str],
    env_name: str,
    default: int,
) -> int:
    value = _env_value(environ, env_name)
    value_name = env_name
    if value is None:
        value = _env_value(environ, "PYDS4_CTX")
        value_name = "PYDS4_CTX"
    return _parse_ctx_size(value, value_name) if value is not None else default


def _explicit_settings(
    environ: Mapping[str, str],
) -> _ResolvedRealDs4Settings | None:
    present = [
        name for name in _EXPLICIT_ENV if _env_value(environ, name) is not None
    ]
    if len(present) == len(_EXPLICIT_ENV):
        model_path = Path(environ["PYDS4_MODEL"]).expanduser()
        return _ResolvedRealDs4Settings(
            model_path=model_path,
            backend_name=environ["PYDS4_BACKEND"].strip().lower(),
            ctx_size=_parse_ctx_size(environ["PYDS4_CTX"], "PYDS4_CTX"),
        )

    if not present or _env_value(environ, "PYDS4_REAL_PROFILE") is not None:
        return None

    missing = [
        name for name in _EXPLICIT_ENV if _env_value(environ, name) is None
    ]
    if missing:
        raise _RealDs4Skip(
            "real DS4 integration requires "
            + ", ".join(missing)
            + " to be set"
        )


def resolve_real_ds4_settings(
    environ: Mapping[str, str],
    *,
    repo_root: Path = _REPO_ROOT,
    platform_name: str = sys.platform,
    machine: str | None = None,
) -> _ResolvedRealDs4Settings:
    explicit = _explicit_settings(environ)
    if explicit is not None:
        return explicit

    profile_name = _env_value(environ, "PYDS4_REAL_PROFILE")
    if profile_name is None:
        profiles = ", ".join(_PROFILE_NAMES)
        raise _RealDs4Skip(
            "real DS4 integration requires PYDS4_MODEL, PYDS4_BACKEND, "
            "and PYDS4_CTX, or PYDS4_REAL_PROFILE set to one of: "
            f"{profiles}"
        )

    normalized = profile_name.strip().lower()
    machine_name = machine if machine is not None else platform.machine()
    if normalized == "macos-metal":
        if platform_name != "darwin" or machine_name != "arm64":
            raise _RealDs4Skip(
                "PYDS4_REAL_PROFILE=macos-metal requires macOS arm64."
            )
        return _ResolvedRealDs4Settings(
            model_path=_model_path(
                environ,
                "PYDS4_METAL_MODEL",
                repo_root,
                _LOCAL_DS4_MODEL,
            ),
            backend_name="metal",
            ctx_size=_ctx_size(environ, "PYDS4_METAL_CTX", 4096),
            profile_name=normalized,
        )

    if normalized == "linux-cuda":
        if not platform_name.startswith("linux"):
            raise _RealDs4Skip("PYDS4_REAL_PROFILE=linux-cuda requires Linux.")
        return _ResolvedRealDs4Settings(
            model_path=_model_path(
                environ,
                "PYDS4_CUDA_MODEL",
                repo_root,
                None,
            ),
            backend_name="cuda",
            ctx_size=_ctx_size(environ, "PYDS4_CUDA_CTX", 4096),
            profile_name=normalized,
        )

    if normalized == "cpu-diagnostic":
        if _env_value(environ, "PYDS4_ALLOW_CPU_DIAGNOSTIC") != "1":
            raise _RealDs4Skip(
                "PYDS4_REAL_PROFILE=cpu-diagnostic requires "
                "PYDS4_ALLOW_CPU_DIAGNOSTIC=1."
            )
        return _ResolvedRealDs4Settings(
            model_path=_model_path(
                environ,
                "PYDS4_CPU_MODEL",
                repo_root,
                _LOCAL_DS4_MODEL,
            ),
            backend_name="cpu",
            ctx_size=_ctx_size(environ, "PYDS4_CPU_CTX", 4096),
            profile_name=normalized,
        )

    profiles = ", ".join(_PROFILE_NAMES)
    raise _RealDs4ConfigError(
        f"Unknown PYDS4_REAL_PROFILE {profile_name!r}. "
        f"Supported profiles: {profiles}."
    )


def real_ds4_settings() -> tuple[Path, pyds4.Backend, int]:
    import os

    try:
        settings = resolve_real_ds4_settings(os.environ)
    except _RealDs4Skip as error:
        pytest.skip(str(error))
    except _RealDs4ConfigError as error:
        pytest.fail(str(error))

    if not settings.model_path.is_file():
        message = (
            f"DS4 model path does not point to a file: {settings.model_path}"
        )
        if settings.profile_name is not None:
            pytest.skip(message)
        pytest.fail(message)

    try:
        backend = pyds4.Backend(settings.backend_name)
    except ValueError:
        pytest.fail(f"Unsupported PYDS4_BACKEND: {settings.backend_name!r}")

    if not pyds4.is_backend_available(backend.value):
        pytest.fail(pyds4.backend_unavailable_reason(backend.value))

    return settings.model_path, backend, settings.ctx_size
