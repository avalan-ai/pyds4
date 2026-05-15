from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Collection
from pathlib import Path
from textwrap import dedent

import pytest

import pyds4
from pyds4 import _version, availability


def test_import_safe_metadata() -> None:
    assert pyds4.__version__ == "0.1.0"
    assert pyds4.__ds4_import_safe__ is True
    assert pyds4.__ds4_commit__ == "8809b90a1e3247389d7652b565ab6772e036f1ea"
    assert pyds4.__ds4_api_version__ is None
    assert pyds4.__ds4_native_backend__ in {"metal", "cuda", "cpu"}


def test_required_symbols_are_exposed_as_collection() -> None:
    assert isinstance(pyds4.__ds4_symbols__, Collection)
    assert not isinstance(pyds4.__ds4_symbols__, (bytes, str))
    assert set(pyds4.REQUIRED_C_SYMBOLS) <= set(pyds4.__ds4_symbols__)


def test_import_does_not_instantiate_engine_or_expose_native_open() -> None:
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_path
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import importlib.util, pyds4, pyds4.native as native; "
                "print(native.ENGINE_CONSTRUCTOR_CALLS); "
                "print(native.SESSION_CONSTRUCTOR_CALLS); "
                "print(hasattr(pyds4, 'ds4_engine_open')); "
                "print(hasattr(pyds4, '_ds4_engine_open')); "
                "print(importlib.util.find_spec('pyds4._native') is None)"
            ),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "0",
        "0",
        "False",
        "False",
        "True",
    ]


def test_backend_availability_is_limited_to_selected_backend() -> None:
    backend = pyds4.__ds4_native_backend__
    host_supported = backend in pyds4.__ds4_available_backends__

    assert pyds4.is_backend_available(backend) is host_supported
    assert (
        pyds4.is_backend_available(f"  {backend.upper()}  ") is host_supported
    )
    if host_supported:
        assert pyds4.backend_unavailable_reason(backend) == ""
        assert pyds4.backend_unavailable_reason(f"  {backend.upper()}  ") == ""
    else:
        reason = pyds4.backend_unavailable_reason(backend)
        assert "macOS arm64 + Metal" in reason
        assert "Linux + CUDA" in reason

    unavailable = next(
        candidate
        for candidate in ("metal", "cuda", "cpu")
        if candidate != backend
    )
    assert pyds4.is_backend_available(unavailable) is False
    reason = pyds4.backend_unavailable_reason(unavailable)
    assert "macOS arm64 + Metal" in reason
    assert "Linux + CUDA" in reason
    assert "CPU mode is diagnostic/reference only" in reason


def test_available_backends_require_native_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeWithoutSource:
        __ds4_source_present__ = False
        __ds4_backend_host_supported__ = True

    monkeypatch.setattr(_version, "_native", NativeWithoutSource())

    assert _version._available_backends_from_native() == ()


def test_native_backend_availability_respects_host_support() -> None:
    native = pytest.importorskip("pyds4._native")
    host_supported = bool(
        getattr(native, "__ds4_backend_host_supported__", True)
    )
    source_present = bool(getattr(native, "__ds4_source_present__", True))
    backend_available = host_supported and source_present

    assert (
        pyds4.is_backend_available(pyds4.__ds4_native_backend__)
        is backend_available
    )
    assert (
        pyds4.__ds4_native_backend__ in pyds4.__ds4_available_backends__
    ) is backend_available
    if not source_present:
        reason = pyds4.backend_unavailable_reason(pyds4.__ds4_native_backend__)
        assert "does not include DS4 native source" in reason
        assert "DS4_SOURCE_DIR" in reason
    elif not host_supported:
        reason = pyds4.backend_unavailable_reason(pyds4.__ds4_native_backend__)
        assert "does not match that backend target" in reason
        assert "macOS arm64 + Metal" in reason
        assert "Linux + CUDA" in reason


def test_unknown_backend_is_unavailable_with_reason() -> None:
    assert pyds4.is_backend_available("bogus") is False
    reason = pyds4.backend_unavailable_reason("bogus")
    assert "Unsupported DS4 native backend" in reason
    assert "metal, cuda, cpu" in reason
    assert "macOS arm64 + Metal" in reason
    assert "Linux + CUDA" in reason
    assert "CPU mode is diagnostic/reference only" in reason


def test_built_backend_is_unavailable_when_native_extension_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(availability, "_native", None)
    monkeypatch.setattr(
        availability,
        "_native_import_error",
        ImportError("missing native dependency"),
    )
    monkeypatch.setattr(availability, "_build_config", object())
    monkeypatch.setattr(availability, "__ds4_native_backend__", "metal")

    assert availability.is_backend_available("metal") is False
    reason = availability.backend_unavailable_reason("metal")
    assert "could not be imported" in reason
    assert "missing native dependency" in reason
    assert "macOS arm64 + Metal" in reason
    assert "Linux + CUDA" in reason


def test_import_remains_safe_when_built_native_extension_cannot_load() -> None:
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_path
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            dedent("""
                import importlib.abc
                import importlib.util
                import sys
                import types

                class NativeLoader(importlib.abc.Loader):
                    def create_module(self, spec):
                        return None

                    def exec_module(self, module):
                        raise ImportError(
                            "dlopen(pyds4/_native.so): missing Metal symbols"
                        )

                class BuildConfigLoader(importlib.abc.Loader):
                    def create_module(self, spec):
                        return types.ModuleType(spec.name)

                    def exec_module(self, module):
                        module.VERSION = "0.1.0"
                        module.DS4_COMMIT = (
                            "8809b90a1e3247389d7652b565ab6772e036f1ea"
                        )
                        module.DS4_API_VERSION = None
                        module.NATIVE_BACKEND = "metal"
                        module.DS4_SOURCE_PRESENT = True
                        module.FAKE_NATIVE = False
                        module.HOST_SUPPORTS_NATIVE_BACKEND = False

                class Finder(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path, target=None):
                        if fullname == "pyds4._native":
                            return importlib.util.spec_from_loader(
                                fullname,
                                NativeLoader(),
                            )
                        if fullname == "pyds4._build_config":
                            return importlib.util.spec_from_loader(
                                fullname,
                                BuildConfigLoader(),
                            )
                        return None

                sys.meta_path.insert(0, Finder())

                import pyds4

                print(pyds4.__ds4_native_backend__)
                print(pyds4.__ds4_available_backends__)
                print(pyds4.is_backend_available("metal"))
                reason = pyds4.backend_unavailable_reason("metal")
                print("dlopen" in reason)
                print("missing Metal symbols" in reason)
                print("macOS arm64 + Metal" in reason)
                print("Linux + CUDA" in reason)
                try:
                    pyds4.Engine(
                        pyds4.EngineOptions(
                            model_path="model.gguf",
                            backend="metal",
                        )
                    )
                except Exception as error:
                    print(type(error).__name__)
                    print("dlopen" in str(error))
                """),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == [
        "metal",
        "()",
        "False",
        "True",
        "True",
        "True",
        "True",
        "Ds4BackendUnavailable",
        "True",
    ]


def test_native_backend_reason_is_decorated_with_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OldNative:
        @staticmethod
        def backend_unavailable_reason(_: str) -> str:
            return "old native reason"

    monkeypatch.setattr(availability, "_native", OldNative())

    reason = availability.backend_unavailable_reason("cpu")

    assert "old native reason" in reason
    assert "macOS arm64 + Metal" in reason
    assert "Linux + CUDA" in reason
    assert "CPU mode is diagnostic/reference only" in reason


def test_empty_native_backend_reason_remains_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AvailableNative:
        @staticmethod
        def backend_unavailable_reason(_: str) -> str:
            return ""

    monkeypatch.setattr(availability, "_native", AvailableNative())

    assert availability.backend_unavailable_reason("cpu") == ""


def test_source_tree_without_native_extension_reports_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(availability, "_native", None)
    monkeypatch.setattr(availability, "_native_import_error", None)
    monkeypatch.setattr(availability, "_build_config", None)
    monkeypatch.setattr(availability, "__ds4_native_backend__", "cpu")

    assert availability.is_backend_available("cpu") is False
    reason = availability.backend_unavailable_reason("cpu")
    assert "native extension has not been built or installed" in reason
    assert "macOS arm64 + Metal" in reason
    assert "Linux + CUDA" in reason
