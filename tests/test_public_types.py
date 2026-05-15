from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

import pyds4
import pyds4.errors as errors


def test_public_type_defaults_and_enum_coercion() -> None:
    options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")

    assert options.model_path == "model.gguf"
    assert options.backend is pyds4.Backend.CPU
    assert options.mtp_path is None
    assert options.n_threads == 0
    assert options.warm_weights is False
    assert options.quality is False

    sampling = pyds4.SamplingOptions()
    assert sampling.temperature == 0.0
    assert sampling.top_k == 0
    assert sampling.top_p == 1.0
    assert sampling.min_p == 0.0
    assert sampling.seed is None

    custom_sampling = pyds4.SamplingOptions(
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        min_p=0.05,
        seed=123,
    )
    assert custom_sampling.temperature == 0.7
    assert custom_sampling.top_k == 40
    assert custom_sampling.top_p == 0.9
    assert custom_sampling.min_p == 0.05
    assert custom_sampling.seed == 123

    step = pyds4.GenerationStep(token_id=5, is_eos=False, advanced=True)
    assert step.token_id == 5
    assert step.is_eos is False
    assert step.advanced is True
    assert step.token_bytes is None

    progress = pyds4.ProgressEvent(
        event="prefill_chunk",
        current=1,
        total=2,
    )
    assert progress.event == "prefill_chunk"
    assert progress.current == 1
    assert progress.total == 2


def test_public_types_are_frozen_and_slotted() -> None:
    options = pyds4.EngineOptions(model_path="model.gguf")
    step = pyds4.GenerationStep(
        token_id=7,
        is_eos=False,
        advanced=True,
        token_bytes=b"x",
    )
    progress = pyds4.ProgressEvent(
        event="prefill_chunk",
        current=1,
        total=2,
    )

    with pytest.raises(FrozenInstanceError):
        options.model_path = "other.gguf"  # type: ignore[misc]

    with pytest.raises((AttributeError, TypeError)):
        options.extra = True  # type: ignore[attr-defined]

    with pytest.raises(FrozenInstanceError):
        step.token_id = 8  # type: ignore[misc]

    with pytest.raises((AttributeError, TypeError)):
        step.extra = True  # type: ignore[attr-defined]

    with pytest.raises(FrozenInstanceError):
        progress.current = 2  # type: ignore[misc]

    with pytest.raises((AttributeError, TypeError)):
        progress.extra = True  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"temperature": -0.1}, "temperature"),
        ({"top_k": -1}, "top_k"),
        ({"top_p": -0.1}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
        ({"min_p": -0.1}, "min_p"),
        ({"min_p": 1.1}, "min_p"),
        ({"temperature": math.inf}, "temperature"),
        ({"temperature": math.nan}, "temperature"),
        ({"top_p": math.nan}, "top_p"),
        ({"min_p": math.inf}, "min_p"),
        ({"seed": 1.2}, "seed"),
        ({"seed": True}, "seed"),
        ({"top_k": False}, "top_k"),
        ({"top_k": 2**31}, "top_k"),
    ],
)
def test_sampling_options_reject_invalid_values(
    kwargs: dict[str, object],
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        pyds4.SamplingOptions(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_threads": True},
        {"mtp_draft_tokens": False},
        {"n_threads": 1.5},
        {"mtp_draft_tokens": "2"},
    ],
)
def test_engine_options_reject_non_integer_counts(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        pyds4.EngineOptions(  # type: ignore[arg-type]
            model_path="model.gguf",
            **kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_threads": -1},
        {"mtp_draft_tokens": -1},
    ],
)
def test_engine_options_reject_negative_counts(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=">= 0"):
        pyds4.EngineOptions(  # type: ignore[arg-type]
            model_path="model.gguf",
            **kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_threads": 2**31},
        {"mtp_draft_tokens": 2**31},
    ],
)
def test_engine_options_reject_counts_outside_c_int_range(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="2147483647"):
        pyds4.EngineOptions(  # type: ignore[arg-type]
            model_path="model.gguf",
            **kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_path": 123},
        {"mtp_path": 123},
        {"directional_steering_file": 123},
    ],
)
def test_engine_options_reject_non_string_paths(
    kwargs: dict[str, object],
) -> None:
    values: dict[str, object] = {"model_path": "model.gguf"}
    values.update(kwargs)
    with pytest.raises(TypeError, match="must be a string"):
        pyds4.EngineOptions(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mtp_margin": True},
        {"directional_steering_attn": False},
        {"directional_steering_ffn": "0.1"},
    ],
)
def test_engine_options_reject_non_numeric_float_options(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(TypeError, match="must be a number"):
        pyds4.EngineOptions(  # type: ignore[arg-type]
            model_path="model.gguf",
            **kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mtp_margin": math.inf},
        {"directional_steering_attn": math.nan},
        {"directional_steering_ffn": -math.inf},
    ],
)
def test_engine_options_reject_non_finite_float_options(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="finite"):
        pyds4.EngineOptions(
            model_path="model.gguf",
            **kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warm_weights": 1},
        {"quality": "yes"},
    ],
)
def test_engine_options_reject_non_boolean_flags(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        pyds4.EngineOptions(  # type: ignore[arg-type]
            model_path="model.gguf",
            **kwargs,
        )


def test_public_exceptions_are_reexported() -> None:
    assert issubclass(pyds4.Ds4ApiVersionError, pyds4.Ds4Error)
    assert issubclass(pyds4.Ds4BackendUnavailable, pyds4.Ds4Error)
    assert issubclass(pyds4.Ds4LoadError, pyds4.Ds4Error)
    assert issubclass(pyds4.Ds4InvalidModel, pyds4.Ds4Error)
    assert issubclass(pyds4.Ds4ContextError, pyds4.Ds4Error)
    assert issubclass(pyds4.Ds4GenerationError, pyds4.Ds4Error)
    assert issubclass(pyds4.Ds4Cancelled, pyds4.Ds4Error)
    assert errors.Ds4Error is pyds4.Ds4Error
    assert errors.Ds4ApiVersionError is pyds4.Ds4ApiVersionError
    assert errors.Ds4BackendUnavailable is pyds4.Ds4BackendUnavailable
    assert errors.Ds4LoadError is pyds4.Ds4LoadError
    assert errors.Ds4InvalidModel is pyds4.Ds4InvalidModel
    assert errors.Ds4ContextError is pyds4.Ds4ContextError
    assert errors.Ds4GenerationError is pyds4.Ds4GenerationError
    assert errors.Ds4Cancelled is pyds4.Ds4Cancelled


@pytest.mark.parametrize(
    "error_type",
    [
        pyds4.Ds4Error,
        pyds4.Ds4ApiVersionError,
        pyds4.Ds4BackendUnavailable,
        pyds4.Ds4LoadError,
        pyds4.Ds4InvalidModel,
        pyds4.Ds4ContextError,
        pyds4.Ds4GenerationError,
        pyds4.Ds4Cancelled,
    ],
)
def test_public_exceptions_are_raiseable(
    error_type: type[pyds4.Ds4Error],
) -> None:
    with pytest.raises(error_type, match="probe"):
        raise error_type("probe")


def test_think_mode_for_context_validates_context_size() -> None:
    assert (
        pyds4.think_mode_for_context(pyds4.ThinkMode.MAX, 4096)
        is pyds4.ThinkMode.HIGH
    )
    assert pyds4.think_mode_for_context("max", 393216) is pyds4.ThinkMode.MAX
    assert (
        pyds4.think_mode_for_context(pyds4.ThinkMode.NONE, 4096)
        is pyds4.ThinkMode.NONE
    )
    assert (
        pyds4.think_mode_for_context(pyds4.ThinkMode.HIGH, 4096)
        is pyds4.ThinkMode.HIGH
    )

    with pytest.raises(ValueError, match="positive integer"):
        pyds4.think_mode_for_context(pyds4.ThinkMode.NONE, 0)

    with pytest.raises(ValueError, match="2147483647"):
        pyds4.think_mode_for_context(pyds4.ThinkMode.NONE, 2**31)

    with pytest.raises(TypeError, match="positive integer"):
        pyds4.think_mode_for_context(  # type: ignore[arg-type]
            pyds4.ThinkMode.NONE,
            True,
        )

    with pytest.raises(ValueError, match="Unsupported DS4 think mode"):
        pyds4.think_mode_for_context(  # type: ignore[arg-type]
            "turbo",
            4096,
        )


def test_think_mode_for_context_maps_invalid_native_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidThinkModeNative:
        @staticmethod
        def think_mode_for_context(mode: str, ctx_size: int) -> str:
            assert mode == "max"
            assert ctx_size == 4096
            return "turbo"

    import pyds4.thinking as thinking

    monkeypatch.setattr(thinking, "_native", InvalidThinkModeNative())

    with pytest.raises(pyds4.Ds4GenerationError, match="unsupported"):
        pyds4.think_mode_for_context(pyds4.ThinkMode.MAX, 4096)


def test_think_mode_for_context_maps_native_helper_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingThinkModeNative:
        @staticmethod
        def think_mode_for_context(mode: str, ctx_size: int) -> str:
            assert mode == "max"
            assert ctx_size == 4096
            raise ValueError("DS4 returned an unsupported think mode.")

    import pyds4.thinking as thinking

    monkeypatch.setattr(thinking, "_native", FailingThinkModeNative())

    with pytest.raises(
        pyds4.Ds4GenerationError,
        match="think_mode_for_context failed",
    ):
        pyds4.think_mode_for_context(pyds4.ThinkMode.MAX, 4096)
