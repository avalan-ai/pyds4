from __future__ import annotations

import os
from pathlib import Path

import pytest

import pyds4
import pyds4.native as native


def _write_gguf_header(path: Path, *, version: int = 3) -> Path:
    path.write_bytes(
        b"GGUF"
        + version.to_bytes(4, byteorder="little", signed=False)
        + b"\0" * 24
    )
    return path


def _rejecting_real_native() -> type:
    class Native:
        __ds4_fake_native__ = False

        class EngineState:
            def __init__(self, *_: object) -> None:
                raise AssertionError("EngineState must not be constructed")

    return Native


@pytest.fixture(autouse=True)
def backend_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native, "is_backend_available", lambda _: True)


def test_missing_model_file_is_rejected_before_real_native_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(native, "_native", _rejecting_real_native())

    with pytest.raises(pyds4.Ds4InvalidModel, match="does not exist"):
        pyds4.Engine(
            pyds4.EngineOptions(
                model_path=str(tmp_path / "missing.gguf"),
                backend="cpu",
            )
        )


def test_directory_model_path_is_rejected_before_real_native_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(native, "_native", _rejecting_real_native())

    with pytest.raises(pyds4.Ds4InvalidModel, match="must be a file"):
        pyds4.Engine(
            pyds4.EngineOptions(
                model_path=str(tmp_path),
                backend="cpu",
            )
        )


@pytest.mark.parametrize(
    ("payload", "error_match"),
    [
        (b"", "too small"),
        (b"NOPE" + b"\0" * 28, "not a GGUF"),
        (b"GGUF" + (2).to_bytes(4, "little") + b"\0" * 24, "GGUF v3"),
    ],
)
def test_invalid_gguf_model_headers_are_rejected_before_native_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: bytes,
    error_match: str,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(payload)
    monkeypatch.setattr(native, "_native", _rejecting_real_native())

    with pytest.raises(pyds4.Ds4InvalidModel, match=error_match):
        pyds4.Engine(
            pyds4.EngineOptions(
                model_path=str(model_path),
                backend="cpu",
            )
        )


def test_mtp_model_path_is_rejected_before_real_native_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = _write_gguf_header(tmp_path / "model.gguf")
    monkeypatch.setattr(native, "_native", _rejecting_real_native())

    with pytest.raises(pyds4.Ds4InvalidModel, match="DS4 MTP model"):
        pyds4.Engine(
            pyds4.EngineOptions(
                model_path=str(model_path),
                backend="cpu",
                mtp_path=str(tmp_path / "missing-mtp.gguf"),
            )
        )


def test_directional_steering_requires_file_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = _write_gguf_header(tmp_path / "model.gguf")
    monkeypatch.setattr(native, "_native", _rejecting_real_native())

    with pytest.raises(pyds4.Ds4LoadError, match="directional_steering_file"):
        pyds4.Engine(
            pyds4.EngineOptions(
                model_path=str(model_path),
                backend="cpu",
                directional_steering_attn=0.25,
            )
        )


def test_real_native_open_receives_preflighted_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = _write_gguf_header(tmp_path / "model.gguf")
    mtp_path = _write_gguf_header(tmp_path / "mtp.gguf")
    steering_path = tmp_path / "steering.bin"
    steering_path.write_bytes(b"steering")
    captured: dict[str, object] = {}

    class Native:
        __ds4_fake_native__ = False

        class EngineState:
            closed = False

            def __init__(
                self,
                model_path: str,
                backend: str,
                mtp_path: str | None,
                n_threads: int,
                mtp_draft_tokens: int,
                mtp_margin: float,
                directional_steering_file: str | None,
                directional_steering_attn: float,
                directional_steering_ffn: float,
                warm_weights: bool,
                quality: bool,
            ) -> None:
                captured["model_path"] = model_path
                captured["backend"] = backend
                captured["mtp_path"] = mtp_path
                captured["directional_steering_file"] = (
                    directional_steering_file
                )

            def close(self) -> None:
                self.closed = True

    monkeypatch.setattr(native, "_native", Native)

    engine = pyds4.Engine(
        pyds4.EngineOptions(
            model_path=str(model_path),
            backend="cpu",
            mtp_path=str(mtp_path),
            directional_steering_file=str(steering_path),
        )
    )

    assert captured == {
        "model_path": str(model_path),
        "backend": "cpu",
        "mtp_path": str(mtp_path),
        "directional_steering_file": str(steering_path),
    }
    assert engine.options.model_path == str(model_path)
    assert engine.options.mtp_path == str(mtp_path)
    assert engine.options.directional_steering_file == str(steering_path)
    engine.close()


def test_native_invalid_model_message_maps_to_invalid_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Native:
        __ds4_fake_native__ = True

        class EngineState:
            def __init__(self, *_: object) -> None:
                raise RuntimeError("unsupported GGUF: missing tensor blk.0")

    monkeypatch.setattr(native, "_native", Native)

    with pytest.raises(
        pyds4.Ds4InvalidModel,
        match="unsupported GGUF: missing tensor",
    ):
        pyds4.Engine(
            pyds4.EngineOptions(model_path="model.gguf", backend="cpu")
        )


def test_public_invalid_model_from_native_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Native:
        __ds4_fake_native__ = True

        class EngineState:
            def __init__(self, *_: object) -> None:
                raise pyds4.Ds4InvalidModel("native says invalid model")

    monkeypatch.setattr(native, "_native", Native)

    with pytest.raises(pyds4.Ds4InvalidModel, match="native says invalid"):
        pyds4.Engine(
            pyds4.EngineOptions(model_path="model.gguf", backend="cpu")
        )


def test_public_backend_unavailable_from_native_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Native:
        __ds4_fake_native__ = True

        class EngineState:
            def __init__(self, *_: object) -> None:
                raise pyds4.Ds4BackendUnavailable("metal unavailable")

    monkeypatch.setattr(native, "_native", Native)

    with pytest.raises(pyds4.Ds4BackendUnavailable, match="metal unavailable"):
        pyds4.Engine(
            pyds4.EngineOptions(model_path="model.gguf", backend="cpu")
        )


def test_runtime_backend_unavailable_message_maps_to_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Native:
        __ds4_fake_native__ = True

        class EngineState:
            def __init__(self, *_: object) -> None:
                raise RuntimeError("ds4: metal backend unavailable")

    monkeypatch.setattr(native, "_native", Native)

    with pytest.raises(
        pyds4.Ds4BackendUnavailable,
        match="metal backend unavailable",
    ):
        pyds4.Engine(
            pyds4.EngineOptions(model_path="model.gguf", backend="cpu")
        )


def test_native_engine_open_stderr_participates_in_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = _write_gguf_header(tmp_path / "model.gguf")

    class Native:
        __ds4_fake_native__ = False

        class EngineState:
            def __init__(self, *_: object) -> None:
                os.write(
                    2,
                    b"ds4: Metal device not available\n"
                    b"ds4: metal backend unavailable; aborting startup\n",
                )
                raise RuntimeError("ds4_engine_open failed")

    monkeypatch.setattr(native, "_native", Native)

    with pytest.raises(pyds4.Ds4BackendUnavailable) as caught:
        pyds4.Engine(
            pyds4.EngineOptions(model_path=str(model_path), backend="cpu")
        )

    message = str(caught.value)
    assert "ds4_engine_open failed" in message
    assert "metal backend unavailable" in message


def test_native_engine_open_replays_stderr_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    model_path = _write_gguf_header(tmp_path / "model.gguf")

    class Native:
        __ds4_fake_native__ = False

        class EngineState:
            def __init__(self, *_: object) -> None:
                os.write(2, b"ds4: native startup detail\n")

            def close(self) -> None:
                pass

    monkeypatch.setattr(native, "_native", Native)

    engine = pyds4.Engine(
        pyds4.EngineOptions(model_path=str(model_path), backend="cpu")
    )

    engine.close()
    assert "ds4: native startup detail" in capfd.readouterr().err


def test_native_engine_open_can_suppress_success_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    model_path = _write_gguf_header(tmp_path / "model.gguf")

    class Native:
        __ds4_fake_native__ = False

        class EngineState:
            def __init__(self, *_: object) -> None:
                os.write(2, b"ds4: native startup detail\n")

            def close(self) -> None:
                pass

    monkeypatch.setattr(native, "_native", Native)

    engine = pyds4.Engine(
        pyds4.EngineOptions(
            model_path=str(model_path),
            backend="cpu",
            native_log=False,
        )
    )

    engine.close()
    assert "ds4: native startup detail" not in capfd.readouterr().err


def test_native_ds4_model_compatibility_message_maps_to_invalid_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Native:
        __ds4_fake_native__ = True

        class EngineState:
            def __init__(self, *_: object) -> None:
                raise RuntimeError(
                    "ds4: required metadata key is missing: tokenizer.ggml"
                )

    monkeypatch.setattr(native, "_native", Native)

    with pytest.raises(
        pyds4.Ds4InvalidModel,
        match="required metadata key is missing",
    ):
        pyds4.Engine(
            pyds4.EngineOptions(model_path="model.gguf", backend="cpu")
        )


def test_unavailable_backend_maps_to_backend_unavailable_before_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native, "_native", _rejecting_real_native())
    monkeypatch.setattr(native, "is_backend_available", lambda _: False)
    monkeypatch.setattr(
        native,
        "backend_unavailable_reason",
        lambda _: (
            "backend unavailable. Supported production targets: "
            "macOS arm64 + Metal and Linux + CUDA. "
            "CPU mode is diagnostic/reference only."
        ),
    )

    with pytest.raises(
        pyds4.Ds4BackendUnavailable,
        match="macOS arm64 \\+ Metal.*Linux \\+ CUDA",
    ) as caught:
        pyds4.Engine(
            pyds4.EngineOptions(model_path="model.gguf", backend="cpu")
        )

    assert "CPU mode is diagnostic/reference only" in str(caught.value)


def test_generic_native_open_failure_maps_to_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Native:
        __ds4_fake_native__ = True

        class EngineState:
            def __init__(self, *_: object) -> None:
                raise RuntimeError("native open failed")

    monkeypatch.setattr(native, "_native", Native)

    with pytest.raises(pyds4.Ds4LoadError, match="native open failed"):
        pyds4.Engine(
            pyds4.EngineOptions(model_path="model.gguf", backend="cpu")
        )
