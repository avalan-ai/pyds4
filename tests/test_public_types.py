from __future__ import annotations

import math
import sys
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
    assert options.native_log is True

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

    capabilities = pyds4.Ds4Capabilities(
        backend="cpu",
        ds4_commit="commit",
        ds4_api_version=None,
        required_symbols=["ds4_engine_open", "ds4_session_eval"],
        available_backends=["cpu"],
        snapshots=False,
        payloads=False,
        logprobs=False,
        top_logprobs=False,
        progress=True,
        mtp=True,
        speculative_eval=False,
    )
    assert capabilities.backend == "cpu"
    assert capabilities.required_symbols == (
        "ds4_engine_open",
        "ds4_session_eval",
    )
    assert capabilities.available_backends == ("cpu",)
    assert capabilities.progress is True
    assert capabilities.mtp is True

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

    score = pyds4.TokenScore(token_id=5, logprob=-0.25)
    assert score.token_id == 5
    assert score.logprob == -0.25

    score_options = pyds4.GenerationScoreOptions()
    assert score_options.mode is pyds4.TokenScoreMode.NONE
    assert score_options.top_k == 0

    generation = pyds4.GenerationOptions()
    assert generation.max_new_tokens == 1
    assert generation.sampling is None
    assert generation.stop_strings == ()
    assert generation.stop_on_eos is True
    assert generation.decode is False
    assert generation.scores == pyds4.GenerationScoreOptions()
    assert generation.advance is True

    stop_buffer = pyds4.StopStringBuffer(["stop"])
    assert stop_buffer.stop_strings == ("stop",)
    assert stop_buffer.stopped is False
    assert stop_buffer.pending_text == ""


def test_public_types_are_frozen_and_slotted() -> None:
    options = pyds4.EngineOptions(model_path="model.gguf")
    capabilities = pyds4.Ds4Capabilities(
        backend="cpu",
        ds4_commit="commit",
        ds4_api_version=None,
        required_symbols=(),
        available_backends=(),
        snapshots=False,
        payloads=False,
        logprobs=False,
        top_logprobs=False,
        progress=False,
        mtp=False,
        speculative_eval=False,
    )
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
    score = pyds4.TokenScore(token_id=5, logprob=-0.25)
    score_options = pyds4.GenerationScoreOptions()
    generation = pyds4.GenerationOptions(stop_strings=["stop"])

    with pytest.raises(FrozenInstanceError):
        options.model_path = "other.gguf"  # type: ignore[misc]

    with pytest.raises((AttributeError, TypeError)):
        options.extra = True  # type: ignore[attr-defined]

    with pytest.raises(FrozenInstanceError):
        capabilities.backend = "metal"  # type: ignore[misc]

    with pytest.raises((AttributeError, TypeError)):
        capabilities.extra = True  # type: ignore[attr-defined]

    with pytest.raises(FrozenInstanceError):
        step.token_id = 8  # type: ignore[misc]

    with pytest.raises((AttributeError, TypeError)):
        step.extra = True  # type: ignore[attr-defined]

    with pytest.raises(FrozenInstanceError):
        progress.current = 2  # type: ignore[misc]

    with pytest.raises((AttributeError, TypeError)):
        progress.extra = True  # type: ignore[attr-defined]

    with pytest.raises(FrozenInstanceError):
        score.logprob = -1.0  # type: ignore[misc]

    with pytest.raises((AttributeError, TypeError)):
        score.extra = True  # type: ignore[attr-defined]

    with pytest.raises(FrozenInstanceError):
        score_options.mode = pyds4.TokenScoreMode.TOKEN_LOGPROB  # type: ignore[misc]

    with pytest.raises((AttributeError, TypeError)):
        score_options.extra = True  # type: ignore[attr-defined]

    with pytest.raises(FrozenInstanceError):
        generation.max_new_tokens = 2  # type: ignore[misc]

    with pytest.raises((AttributeError, TypeError)):
        generation.extra = True  # type: ignore[attr-defined]

    stop_buffer = pyds4.StopStringBuffer("stop")
    with pytest.raises((AttributeError, TypeError)):
        stop_buffer.extra = True  # type: ignore[attr-defined]


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
        ({"seed": -1}, "seed"),
        ({"seed": 1.2}, "seed"),
        ({"seed": True}, "seed"),
        ({"seed": 2**64}, "seed"),
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
    ("kwargs", "error_match"),
    [
        ({"token_id": True, "logprob": -0.25}, "token_id"),
        ({"token_id": -1, "logprob": -0.25}, "token_id"),
        ({"token_id": 2**31, "logprob": -0.25}, "token_id"),
        ({"token_id": 1, "logprob": True}, "logprob"),
        ({"token_id": 1, "logprob": "bad"}, "logprob"),
        ({"token_id": 1, "logprob": math.inf}, "logprob"),
        ({"token_id": 1, "logprob": math.nan}, "logprob"),
    ],
)
def test_token_score_rejects_invalid_values(
    kwargs: dict[str, object],
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        pyds4.TokenScore(**kwargs)  # type: ignore[arg-type]


def test_generation_score_options_support_public_request_shapes() -> None:
    none = pyds4.GenerationScoreOptions()
    chosen = pyds4.GenerationScoreOptions(
        mode=pyds4.TokenScoreMode.TOKEN_LOGPROB,
    )
    top = pyds4.GenerationScoreOptions(
        mode="top_logprobs",
        top_k=3,
    )
    combined = pyds4.GenerationScoreOptions(
        mode=pyds4.TokenScoreMode.TOKEN_LOGPROB_AND_TOP_LOGPROBS,
        top_k=5,
    )

    assert none.mode is pyds4.TokenScoreMode.NONE
    assert none.top_k == 0
    assert chosen.mode is pyds4.TokenScoreMode.TOKEN_LOGPROB
    assert chosen.top_k == 0
    assert top.mode is pyds4.TokenScoreMode.TOP_LOGPROBS
    assert top.top_k == 3
    assert combined.mode is (
        pyds4.TokenScoreMode.TOKEN_LOGPROB_AND_TOP_LOGPROBS
    )
    assert combined.top_k == 5


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"mode": "unsupported"}, "Unsupported token score mode"),
        ({"mode": pyds4.TokenScoreMode.TOP_LOGPROBS}, "top_k"),
        (
            {
                "mode": pyds4.TokenScoreMode.TOKEN_LOGPROB_AND_TOP_LOGPROBS,
            },
            "top_k",
        ),
        ({"mode": pyds4.TokenScoreMode.NONE, "top_k": 1}, "top_k"),
        ({"mode": pyds4.TokenScoreMode.TOKEN_LOGPROB, "top_k": 1}, "top_k"),
        ({"top_k": -1}, "top_k"),
        ({"top_k": True}, "top_k"),
    ],
)
def test_generation_score_options_reject_invalid_requests(
    kwargs: dict[str, object],
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        pyds4.GenerationScoreOptions(**kwargs)  # type: ignore[arg-type]


def test_generation_options_accept_stop_string_shapes() -> None:
    assert pyds4.GenerationOptions(
        stop_strings="</s>",
    ).stop_strings == ("</s>",)
    assert pyds4.GenerationOptions(
        stop_strings=("stop", "done"),
    ).stop_strings == ("stop", "done")

    source = ["stop"]
    options = pyds4.GenerationOptions(stop_strings=source)
    source.append("mutated")

    assert options.stop_strings == ("stop",)


def test_generation_options_accept_sampling_and_score_options() -> None:
    sampling = pyds4.SamplingOptions(temperature=0.7, seed=7)
    scores = pyds4.GenerationScoreOptions(
        mode=pyds4.TokenScoreMode.TOP_LOGPROBS,
        top_k=2,
    )

    options = pyds4.GenerationOptions(
        max_new_tokens=12,
        sampling=sampling,
        stop_strings=["stop"],
        stop_on_eos=False,
        decode=True,
        scores=scores,
        advance=False,
    )

    assert options.max_new_tokens == 12
    assert options.sampling == sampling
    assert options.stop_strings == ("stop",)
    assert options.stop_on_eos is False
    assert options.decode is True
    assert options.scores == scores
    assert options.advance is False


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"max_new_tokens": True}, "max_new_tokens"),
        ({"max_new_tokens": 2**31}, "max_new_tokens"),
        ({"sampling": object()}, "sampling"),
        ({"stop_strings": ""}, "stop_strings"),
        ({"stop_strings": [""]}, "stop_strings"),
        ({"stop_strings": [1]}, "stop_strings"),
        ({"stop_strings": {">"}}, "stop_strings"),
        ({"stop_on_eos": 1}, "stop_on_eos"),
        ({"decode": 1}, "decode"),
        ({"scores": object()}, "scores"),
        ({"advance": 1}, "advance"),
    ],
)
def test_generation_options_reject_invalid_values(
    kwargs: dict[str, object],
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        pyds4.GenerationOptions(**kwargs)  # type: ignore[arg-type]


def test_stop_string_buffer_emits_plain_chunks_without_stops() -> None:
    buffer = pyds4.StopStringBuffer()

    assert buffer.stop_strings == ()
    assert buffer.push("alpha") == ("alpha",)
    assert buffer.push("") == ()
    assert buffer.flush() == ()
    assert buffer.stopped is False


def test_stop_string_buffer_suppresses_stop_within_chunk() -> None:
    buffer = pyds4.StopStringBuffer(" STOP")

    assert buffer.push("alpha STOP omega") == ("alpha",)
    assert buffer.stopped is True
    assert buffer.pending_text == ""
    assert buffer.push("ignored") == ()
    assert buffer.flush() == ()


def test_stop_string_buffer_suppresses_stop_across_chunks() -> None:
    buffer = pyds4.StopStringBuffer(["STOP"])

    assert buffer.push("alpha ST") == ("alpha",)
    assert buffer.pending_text == " ST"
    assert buffer.push("OP omega") == (" ",)
    assert buffer.stopped is True
    assert buffer.flush() == ()


def test_stop_string_buffer_flushes_without_stop() -> None:
    buffer = pyds4.StopStringBuffer(("STOP",))

    assert buffer.push("abcd") == ("a",)
    assert buffer.push("e") == ("b",)
    assert buffer.flush() == ("cde",)


@pytest.mark.parametrize(
    ("stop_strings", "error_match"),
    [
        ("", "stop_strings"),
        ([""], "stop_strings"),
        ([1], "stop_strings"),
        ({">"}, "stop_strings"),
    ],
)
def test_stop_string_buffer_rejects_invalid_stop_strings(
    stop_strings: object,
    error_match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error_match):
        pyds4.StopStringBuffer(stop_strings)  # type: ignore[arg-type]


def test_stop_string_buffer_rejects_non_string_text() -> None:
    buffer = pyds4.StopStringBuffer("STOP")

    with pytest.raises(TypeError, match="text"):
        buffer.push(b"bad")  # type: ignore[arg-type]


def test_generation_step_carries_decoded_text_and_scores() -> None:
    score = pyds4.TokenScore(token_id=7, logprob=-0.5)
    step = pyds4.GenerationStep(
        token_id=7,
        is_eos=False,
        advanced=True,
        token_bytes=b"x",
        decoded_text="x",
        token_logprob=-0.5,
        top_logprobs=[score],  # type: ignore[arg-type]
    )

    assert step.token_id == 7
    assert step.token_bytes == b"x"
    assert step.decoded_text == "x"
    assert step.token_logprob == -0.5
    assert step.top_logprobs == (score,)


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"token_id": True}, "token_id"),
        ({"token_id": -1}, "token_id"),
        ({"is_eos": 1}, "is_eos"),
        ({"advanced": 1}, "advanced"),
        ({"token_bytes": "x"}, "token_bytes"),
        ({"decoded_text": b"x"}, "decoded_text"),
        ({"token_logprob": math.inf}, "token_logprob"),
        ({"top_logprobs": [object()]}, "top_logprobs"),
    ],
)
def test_generation_step_rejects_invalid_values(
    kwargs: dict[str, object],
    error_match: str,
) -> None:
    values: dict[str, object] = {
        "token_id": 1,
        "is_eos": False,
        "advanced": True,
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=error_match):
        pyds4.GenerationStep(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"event": b"prefill_chunk"}, "event"),
        ({"current": True}, "current"),
        ({"current": -1}, "current"),
        ({"current": 2**31}, "current"),
        ({"total": False}, "total"),
        ({"total": -1}, "total"),
        ({"total": 2**31}, "total"),
    ],
)
def test_progress_event_rejects_invalid_values(
    kwargs: dict[str, object],
    error_match: str,
) -> None:
    values: dict[str, object] = {
        "event": "prefill_chunk",
        "current": 1,
        "total": 2,
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=error_match):
        pyds4.ProgressEvent(**values)  # type: ignore[arg-type]


def test_generation_option_construction_does_not_import_avalan() -> None:
    sys.modules.pop("avalan", None)

    pyds4.GenerationOptions(
        stop_strings="stop",
        scores=pyds4.GenerationScoreOptions(
            mode=pyds4.TokenScoreMode.TOKEN_LOGPROB,
        ),
    )

    assert "avalan" not in sys.modules


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
        {"native_log": "yes"},
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
