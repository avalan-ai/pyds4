from __future__ import annotations

from pathlib import Path

import pytest

import pyds4
from pyds4._metadata import DS4_COMMIT
from pyds4._metal import METAL_SOURCE_ENV


def test_build_config_matches_public_native_metadata() -> None:
    build_config = pytest.importorskip("pyds4._build_config")
    native = pytest.importorskip("pyds4._native")

    assert build_config.VERSION == pyds4.__version__
    assert build_config.DS4_COMMIT == DS4_COMMIT
    assert build_config.NATIVE_BACKEND == pyds4.__ds4_native_backend__
    assert native.__ds4_commit__ == pyds4.__ds4_commit__
    assert native.__ds4_native_backend__ == pyds4.__ds4_native_backend__
    assert native.__ds4_source_present__ is build_config.DS4_SOURCE_PRESENT
    assert native.__ds4_fake_native__ is build_config.FAKE_NATIVE
    assert (
        native.__ds4_backend_host_supported__
        is build_config.HOST_SUPPORTS_NATIVE_BACKEND
    )


def test_cmake_metal_source_inventory_matches_runtime_configuration() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cmake = (repo_root / "CMakeLists.txt").read_text(encoding="utf-8")

    for filename in METAL_SOURCE_ENV.values():
        assert f"metal/{filename}" in cmake
