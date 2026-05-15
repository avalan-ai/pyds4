from __future__ import annotations

from pathlib import Path

import pytest
from _real_ds4_profiles import (
    _RealDs4ConfigError,
    _RealDs4Skip,
    resolve_real_ds4_settings,
)


def test_explicit_real_ds4_environment_takes_precedence() -> None:
    settings = resolve_real_ds4_settings(
        {
            "PYDS4_MODEL": "/tmp/model.gguf",
            "PYDS4_BACKEND": " CUDA ",
            "PYDS4_CTX": "8192",
            "PYDS4_REAL_PROFILE": "macos-metal",
        },
        repo_root=Path("/repo"),
        platform_name="darwin",
        machine="arm64",
    )

    assert settings.model_path == Path("/tmp/model.gguf")
    assert settings.backend_name == "cuda"
    assert settings.ctx_size == 8192
    assert settings.profile_name is None


def test_macos_metal_profile_uses_local_model_default() -> None:
    settings = resolve_real_ds4_settings(
        {"PYDS4_REAL_PROFILE": "macos-metal"},
        repo_root=Path("/repo"),
        platform_name="darwin",
        machine="arm64",
    )

    assert settings.model_path == Path("/repo/.local/ds4/ds4flash.gguf")
    assert settings.backend_name == "metal"
    assert settings.ctx_size == 4096
    assert settings.profile_name == "macos-metal"


def test_linux_cuda_profile_requires_linux_and_explicit_model() -> None:
    with pytest.raises(_RealDs4Skip, match="requires Linux"):
        resolve_real_ds4_settings(
            {"PYDS4_REAL_PROFILE": "linux-cuda"},
            platform_name="darwin",
            machine="arm64",
        )

    with pytest.raises(_RealDs4Skip, match="PYDS4_CUDA_MODEL"):
        resolve_real_ds4_settings(
            {"PYDS4_REAL_PROFILE": "linux-cuda"},
            platform_name="linux",
            machine="x86_64",
        )

    settings = resolve_real_ds4_settings(
        {
            "PYDS4_REAL_PROFILE": "linux-cuda",
            "PYDS4_MODEL": "/models/ds4.gguf",
            "PYDS4_CTX": "16384",
        },
        platform_name="linux",
        machine="x86_64",
    )

    assert settings.model_path == Path("/models/ds4.gguf")
    assert settings.backend_name == "cuda"
    assert settings.ctx_size == 16384
    assert settings.profile_name == "linux-cuda"


def test_cpu_diagnostic_profile_requires_explicit_opt_in() -> None:
    with pytest.raises(_RealDs4Skip, match="PYDS4_ALLOW_CPU_DIAGNOSTIC=1"):
        resolve_real_ds4_settings(
            {"PYDS4_REAL_PROFILE": "cpu-diagnostic"},
            repo_root=Path("/repo"),
        )

    settings = resolve_real_ds4_settings(
        {
            "PYDS4_REAL_PROFILE": "cpu-diagnostic",
            "PYDS4_ALLOW_CPU_DIAGNOSTIC": "1",
            "PYDS4_CPU_MODEL": "/models/cpu.gguf",
            "PYDS4_CPU_CTX": "512",
        },
        repo_root=Path("/repo"),
    )

    assert settings.model_path == Path("/models/cpu.gguf")
    assert settings.backend_name == "cpu"
    assert settings.ctx_size == 512
    assert settings.profile_name == "cpu-diagnostic"


def test_unknown_profile_and_invalid_context_fail_clearly() -> None:
    with pytest.raises(
        _RealDs4ConfigError, match="Unknown PYDS4_REAL_PROFILE"
    ):
        resolve_real_ds4_settings({"PYDS4_REAL_PROFILE": "bogus"})

    with pytest.raises(
        _RealDs4ConfigError, match="PYDS4_CTX must be positive"
    ):
        resolve_real_ds4_settings(
            {
                "PYDS4_MODEL": "/tmp/model.gguf",
                "PYDS4_BACKEND": "metal",
                "PYDS4_CTX": "0",
            }
        )
