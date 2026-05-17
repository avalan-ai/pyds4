from __future__ import annotations

import gc
import time
from collections.abc import Iterator
from threading import Thread

import pytest

import pyds4

_native = pytest.importorskip("pyds4._native")
if not getattr(_native, "__ds4_fake_native__", False):
    pytest.skip("requires a PYDS4_USE_FAKE_DS4 build", allow_module_level=True)


@pytest.fixture(autouse=True)
def reset_fake_counters() -> Iterator[None]:
    _native.fake_reset_counters()
    yield
    gc.collect()
    assert counters()["token_live_allocations"] == 0
    assert counters()["snapshot_live_allocations"] == 0


def counters() -> dict[str, int]:
    return dict(_native.fake_counters())


def make_engine(**kwargs: object) -> pyds4.Engine:
    option_values: dict[str, object] = {
        "model_path": "model.gguf",
        "backend": "cpu",
    }
    option_values.update(kwargs)
    options = pyds4.EngineOptions(**option_values)  # type: ignore[arg-type]
    return pyds4.Engine(options)


def text_tokens(text: str) -> list[int]:
    return [1000 + ord(char) for char in text]


def test_fake_native_capabilities_match_bound_runtime_surface() -> None:
    caps = pyds4.capabilities()

    assert caps.backend == "cpu"
    assert caps.available_backends == ("cpu",)
    assert caps.required_symbols == tuple(pyds4.REQUIRED_C_SYMBOLS)
    assert caps.progress is True
    assert caps.mtp is True
    assert caps.snapshots is True
    assert caps.payloads is True
    assert caps.logprobs is True
    assert caps.top_logprobs is True
    assert caps.speculative_eval is True
    assert counters()["engine_open_calls"] == 0
    assert counters()["session_create_calls"] == 0


def test_engine_and_session_close_are_idempotent() -> None:
    engine = make_engine()
    session = engine.create_session(64)

    session.close()
    session.close()
    engine.close()
    engine.close()

    snapshot = counters()
    assert snapshot["engine_open_calls"] == 1
    assert snapshot["engine_close_calls"] == 1
    assert snapshot["session_create_calls"] == 1
    assert snapshot["session_free_calls"] == 1


def test_engine_and_session_context_managers_close_handles() -> None:
    with make_engine() as engine:
        with engine.create_session(64) as session:
            session.sync([1])

        assert session.closed is True
        assert counters()["session_free_calls"] == 1
        assert engine.closed is False

    snapshot = counters()
    assert engine.closed is True
    assert snapshot["engine_close_calls"] == 1
    assert snapshot["session_free_calls"] == 1


def test_closed_engine_and_session_errors_keep_public_error_classes() -> None:
    engine = make_engine()
    session = engine.create_session(64)

    session.close()

    with pytest.raises(pyds4.Ds4ContextError, match="session is closed"):
        session.argmax()
    with pytest.raises(pyds4.Ds4ContextError, match="session is closed"):
        session.eval(1)

    engine.close()

    with pytest.raises(pyds4.Ds4LoadError, match="engine is closed"):
        engine.tokenize_text("x")
    with pytest.raises(pyds4.Ds4LoadError, match="engine is closed"):
        engine.create_session(64)


def test_engine_close_closes_live_sessions_before_engine() -> None:
    engine = make_engine()
    session = engine.create_session(64)

    engine.close()

    assert engine.closed is True
    assert session.closed is True
    snapshot = counters()
    assert snapshot["session_free_calls"] == 1
    assert snapshot["engine_close_calls"] == 1
    assert (
        snapshot["last_session_free_sequence"]
        < snapshot["last_engine_close_sequence"]
    )

    session.close()
    assert counters()["session_free_calls"] == 1


def test_engine_close_waits_for_in_flight_session_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYDS4_FAKE_DELAY_EVAL_MS", "200")
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1])
    errors: list[BaseException] = []

    def eval_token() -> None:
        try:
            session.eval(1000 + ord("A"))
        except BaseException as error:
            errors.append(error)

    thread = Thread(target=eval_token)
    thread.start()
    for _ in range(100):
        if counters()["eval_calls"] == 1:
            break
        time.sleep(0.01)
    else:
        pytest.fail("fake eval did not start")

    started = time.monotonic()
    engine.close()
    elapsed = time.monotonic() - started
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert errors == []
    assert elapsed >= 0.1
    snapshot = counters()
    assert snapshot["eval_calls"] == 1
    assert snapshot["session_free_calls"] == 1
    assert snapshot["engine_close_calls"] == 1
    assert (
        snapshot["last_eval_sequence"]
        < snapshot["last_session_free_sequence"]
        < snapshot["last_engine_close_sequence"]
    )


def test_session_keeps_engine_alive_until_session_close() -> None:
    engine = make_engine()
    session = engine.create_session(64)

    del engine
    gc.collect()

    snapshot = counters()
    assert snapshot["engine_close_calls"] == 0
    assert snapshot["session_free_calls"] == 0

    session.close()
    gc.collect()

    snapshot = counters()
    assert snapshot["session_free_calls"] == 1
    assert snapshot["engine_close_calls"] == 1
    assert (
        snapshot["last_session_free_sequence"]
        < snapshot["last_engine_close_sequence"]
    )


def test_sync_uses_temporary_token_buffer_and_round_trips_tokens() -> None:
    engine = make_engine()
    session = engine.create_session(64)

    session.sync([1, 2, 1003])

    assert session.pos == 3
    assert session.ctx == 64
    assert session.tokens == [1, 2, 1003]
    snapshot = counters()
    assert snapshot["sync_calls"] == 1
    assert snapshot["tokens_free_calls"] == 1

    session.close()
    engine.close()
    assert counters()["tokens_free_calls"] == 2


def test_sync_failure_preserves_operation_and_native_message() -> None:
    engine = make_engine()
    session = engine.create_session(3)

    with pytest.raises(
        pyds4.Ds4GenerationError,
        match="sync failed: prompt exceeds context",
    ):
        session.sync([1, 2, 3])

    snapshot = counters()
    assert snapshot["sync_calls"] == 1
    assert snapshot["tokens_free_calls"] == 1

    session.close()
    engine.close()


def test_candidate_selection_does_not_advance_session_state() -> None:
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1, 2])

    greedy = session.argmax()
    excluded_greedy = session.argmax_excluding(greedy)
    sampled = session.sample(
        pyds4.SamplingOptions(
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            min_p=0.05,
            seed=123,
        )
    )

    assert greedy == 1000 + ord("A")
    assert excluded_greedy == 1000 + ord("B")
    assert isinstance(sampled, int)
    assert sampled >= 0
    assert session.pos == 2
    assert session.tokens == [1, 2]

    snapshot = counters()
    assert snapshot["argmax_calls"] == 1
    assert snapshot["argmax_excluding_calls"] == 1
    assert snapshot["sample_calls"] == 1

    session.close()
    engine.close()


def test_candidate_selection_requires_initialized_logits() -> None:
    engine = make_engine()
    session = engine.create_session(64)

    with pytest.raises(
        pyds4.Ds4GenerationError,
        match="no synchronized prompt or evaluated token",
    ):
        session.argmax()
    with pytest.raises(
        pyds4.Ds4GenerationError,
        match="no synchronized prompt or evaluated token",
    ):
        session.argmax_excluding(1)
    with pytest.raises(
        pyds4.Ds4GenerationError,
        match="no synchronized prompt or evaluated token",
    ):
        session.sample(pyds4.SamplingOptions(seed=123))

    snapshot = counters()
    assert snapshot["argmax_calls"] == 0
    assert snapshot["argmax_excluding_calls"] == 0
    assert snapshot["sample_calls"] == 0

    session.eval(1000 + ord("A"))
    assert session.argmax() == 1000 + ord("A")

    session.close()
    engine.close()


def test_rewind_requires_resync_before_candidate_selection() -> None:
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1, 2, 3])

    assert session.argmax() == 1000 + ord("A")

    session.rewind(2)

    with pytest.raises(
        pyds4.Ds4GenerationError,
        match="no synchronized prompt or evaluated token",
    ):
        session.argmax()

    assert counters()["argmax_calls"] == 1

    session.sync([1, 2])
    assert session.argmax() == 1000 + ord("A")

    session.close()
    engine.close()


def test_sample_preserves_rng_state_until_sync_resets_stream() -> None:
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1])
    sampling = pyds4.SamplingOptions(
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        min_p=0.05,
        seed=123,
    )

    first = session.sample(sampling)
    second = session.sample(sampling)
    third = session.sample(sampling)

    assert [first, second, third] == [
        1000 + ord("s"),
        1000 + ord("t"),
        1000 + ord("o"),
    ]
    assert session.pos == 1
    assert session.tokens == [1]

    session.sync([1])

    assert session.sample(sampling) == first
    assert counters()["sample_calls"] == 4

    session.close()
    engine.close()


def test_top_logprobs_return_stable_sorted_token_scores() -> None:
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1, 2])

    scores = session.top_logprobs(3)

    assert scores == [
        pyds4.TokenScore(token_id=1000 + ord("A"), logprob=-0.25),
        pyds4.TokenScore(token_id=1000 + ord("B"), logprob=-1.25),
        pyds4.TokenScore(token_id=1000 + ord("C"), logprob=-2.25),
    ]
    assert counters()["top_logprobs_calls"] == 1

    session.close()
    engine.close()


def test_token_logprob_can_be_requested_before_eval() -> None:
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1, 2])

    token_id = session.argmax()
    assert token_id == 1000 + ord("A")
    assert session.token_logprob(token_id) == -0.25
    assert session.pos == 2

    session.eval(token_id)
    assert session.pos == 3
    assert counters()["token_logprob_calls"] == 1

    session.close()
    engine.close()


def test_speculative_eval_argmax_advances_expected_tokens() -> None:
    engine = make_engine(mtp_path="mtp.gguf", mtp_draft_tokens=4)
    session = engine.create_session(64)
    session.sync([1, 2])

    accepted = session.eval_speculative_argmax(
        1000 + ord("A"),
        3,
        engine.eos_token_id,
    )

    assert accepted == [
        1000 + ord("A"),
        1000 + ord("B"),
        1000 + ord("C"),
    ]
    assert session.tokens == [1, 2, *accepted]
    counts = counters()
    assert counts["speculative_eval_calls"] == 1
    assert counts["eval_calls"] == 0

    session.close()
    engine.close()


def test_speculative_eval_argmax_requires_mtp_draft_tokens() -> None:
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1])

    with pytest.raises(
        pyds4.Ds4GenerationError,
        match="requires an engine opened with MTP draft tokens > 1",
    ):
        session.eval_speculative_argmax(
            1000 + ord("A"),
            2,
            engine.eos_token_id,
        )

    assert counters()["speculative_eval_calls"] == 0

    session.close()
    engine.close()


@pytest.mark.parametrize(
    ("first_token", "max_tokens", "eos_token_id"),
    [
        (True, 2, 6),
        (-1, 2, 6),
        (1000 + ord("A"), False, 6),
        (1000 + ord("A"), 0, 6),
        (1000 + ord("A"), 2, True),
        (1000 + ord("A"), 2, -1),
    ],
)
def test_speculative_eval_argmax_rejects_invalid_inputs_before_native_call(
    first_token: object,
    max_tokens: object,
    eos_token_id: object,
) -> None:
    engine = make_engine(mtp_path="mtp.gguf", mtp_draft_tokens=4)
    session = engine.create_session(64)
    session.sync([1])

    with pytest.raises((TypeError, ValueError)):
        session.eval_speculative_argmax(  # type: ignore[arg-type]
            first_token,
            max_tokens,
            eos_token_id,
        )

    assert counters()["speculative_eval_calls"] == 0

    session.close()
    engine.close()


def test_speculative_eval_argmax_native_failure_maps_to_generation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = make_engine(mtp_path="mtp.gguf", mtp_draft_tokens=4)
    session = engine.create_session(64)
    session.sync([1])
    monkeypatch.setenv("PYDS4_FAKE_FAIL_EVAL_SPECULATIVE_ARGMAX", "1")

    with pytest.raises(
        pyds4.Ds4GenerationError,
        match="eval_speculative_argmax failed: fake speculative eval failure",
    ):
        session.eval_speculative_argmax(
            1000 + ord("A"),
            2,
            engine.eos_token_id,
        )

    assert counters()["speculative_eval_calls"] == 1

    session.close()
    engine.close()


@pytest.mark.parametrize("k", [False, -1, 1.5])
def test_top_logprobs_rejects_invalid_k_before_native_call(
    k: object,
) -> None:
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1])

    with pytest.raises((TypeError, ValueError)):
        session.top_logprobs(k)  # type: ignore[arg-type]

    assert counters()["top_logprobs_calls"] == 0

    session.close()
    engine.close()


@pytest.mark.parametrize("token_id", [True, -1, 1.5])
def test_token_logprob_rejects_invalid_token_ids_before_native_call(
    token_id: object,
) -> None:
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1])

    with pytest.raises((TypeError, ValueError)):
        session.token_logprob(token_id)  # type: ignore[arg-type]

    assert counters()["token_logprob_calls"] == 0

    session.close()
    engine.close()


@pytest.mark.parametrize("method_name", ["top_logprobs", "token_logprob"])
def test_logprob_methods_require_ready_logits(method_name: str) -> None:
    engine = make_engine()
    session = engine.create_session(64)

    with pytest.raises(
        pyds4.Ds4GenerationError,
        match="no synchronized prompt or evaluated token",
    ):
        if method_name == "top_logprobs":
            session.top_logprobs(1)
        else:
            session.token_logprob(1000 + ord("A"))

    counts = counters()
    assert counts["top_logprobs_calls"] == 0
    assert counts["token_logprob_calls"] == 0

    session.close()
    engine.close()


def test_malformed_top_logprobs_raise_generation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1])
    monkeypatch.setenv("PYDS4_FAKE_MALFORMED_TOP_LOGPROBS", "1")

    with pytest.raises(pyds4.Ds4GenerationError, match="malformed"):
        session.top_logprobs(1)

    assert counters()["top_logprobs_calls"] == 1

    session.close()
    engine.close()


def test_malformed_token_logprob_raises_generation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1])
    monkeypatch.setenv("PYDS4_FAKE_MALFORMED_TOKEN_LOGPROB", "1")

    with pytest.raises(pyds4.Ds4GenerationError, match="malformed"):
        session.token_logprob(1000 + ord("A"))

    assert counters()["token_logprob_calls"] == 1

    session.close()
    engine.close()


def test_greedy_sample_does_not_start_seeded_rng_stream() -> None:
    engine = make_engine()
    expected_session = engine.create_session(64)
    expected_session.sync([1])
    sampling = pyds4.SamplingOptions(
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        min_p=0.05,
        seed=456,
    )
    expected = expected_session.sample(sampling)

    session = engine.create_session(64)
    session.sync([1])

    greedy = session.sample(pyds4.SamplingOptions(temperature=0.0, seed=123))

    assert greedy == 1000 + ord("A")
    assert session.sample(sampling) == expected

    session.close()
    expected_session.close()
    engine.close()


def test_rewind_resets_sampling_stream_when_timeline_changes() -> None:
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1])
    sampling = pyds4.SamplingOptions(
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        min_p=0.05,
        seed=123,
    )

    first = session.sample(sampling)
    session.eval(first)
    assert session.sample(sampling) != first

    session.rewind(1)
    session.eval(first)

    assert session.sample(sampling) == first

    session.close()
    engine.close()


def test_failed_sample_does_not_start_seeded_rng_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = make_engine()
    expected_sampling = pyds4.SamplingOptions(temperature=0.7, seed=456)
    expected_session = engine.create_session(64)
    expected_session.sync([1])
    expected = expected_session.sample(expected_sampling)

    session = engine.create_session(64)
    session.sync([1])

    monkeypatch.setenv("PYDS4_FAKE_FAIL_SAMPLE", "1")
    with pytest.raises(pyds4.Ds4GenerationError, match="sample failed"):
        session.sample(pyds4.SamplingOptions(temperature=0.7, seed=123))

    monkeypatch.delenv("PYDS4_FAKE_FAIL_SAMPLE")
    assert session.sample(expected_sampling) == expected

    session.close()
    expected_session.close()
    engine.close()


def test_eval_advances_session_by_one_token() -> None:
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1, 2])
    token_id = session.argmax()

    session.eval(token_id)

    assert session.pos == 3
    assert session.tokens == [1, 2, 1000 + ord("A")]
    assert counters()["eval_calls"] == 1

    session.close()
    engine.close()


def test_eval_failure_preserves_operation_and_native_message() -> None:
    engine = make_engine()
    session = engine.create_session(4)
    session.sync([1, 2, 3])

    with pytest.raises(
        pyds4.Ds4GenerationError,
        match="eval failed: prompt exceeds context",
    ):
        session.eval(1000 + ord("A"))

    assert session.pos == 3
    assert session.tokens == [1, 2, 3]
    assert counters()["eval_calls"] == 0

    session.close()
    engine.close()


def test_rewind_and_invalidate_update_native_session_state() -> None:
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1, 2, 3])
    session.eval(1000 + ord("A"))

    session.rewind(2)

    assert session.pos == 2
    assert session.tokens == [1, 2]
    assert counters()["rewind_calls"] == 1

    session.invalidate()

    assert session.pos == 0
    assert session.tokens == []
    snapshot = counters()
    assert snapshot["invalidate_calls"] == 1

    with pytest.raises(pyds4.Ds4GenerationError, match="argmax failed"):
        session.argmax()

    session.close()
    engine.close()


def test_snapshot_save_load_restores_position_and_tokens() -> None:
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1, 2])
    session.eval(1000 + ord("A"))

    snapshot = session.save_snapshot()
    assert isinstance(snapshot, bytes)
    assert snapshot

    session.eval(1000 + ord("B"))
    assert session.tokens == [1, 2, 1000 + ord("A"), 1000 + ord("B")]

    session.load_snapshot(snapshot)

    assert session.pos == 3
    assert session.tokens == [1, 2, 1000 + ord("A")]
    assert session.argmax() == 1000 + ord("A")

    counts = counters()
    assert counts["save_snapshot_calls"] == 1
    assert counts["load_snapshot_calls"] == 1
    assert counts["snapshot_allocation_calls"] == 1
    assert counts["snapshot_free_calls"] == 1
    assert counts["snapshot_live_allocations"] == 0

    session.close()
    engine.close()


def test_payload_save_load_restores_prompt_synchronized_session() -> None:
    engine = make_engine()
    source = engine.create_session(64)
    target = engine.create_session(64)
    source.sync([1, 2, 3])

    payload = source.save_payload()
    assert isinstance(payload, bytes)
    assert payload

    target.sync([9])
    target.load_payload(payload)

    assert target.pos == 3
    assert target.tokens == [1, 2, 3]
    assert target.argmax() == 1000 + ord("A")

    counts = counters()
    assert counts["payload_bytes_calls"] == 1
    assert counts["save_payload_calls"] == 1
    assert counts["load_payload_calls"] == 1

    source.close()
    target.close()
    engine.close()


@pytest.mark.parametrize(
    ("method_name", "bad_value", "error_match"),
    [
        ("load_snapshot", bytearray(b"snapshot"), "snapshot must be bytes"),
        ("load_payload", "payload", "payload must be bytes"),
    ],
)
def test_snapshot_and_payload_load_reject_non_bytes_before_native_call(
    method_name: str,
    bad_value: object,
    error_match: str,
) -> None:
    engine = make_engine()
    session = engine.create_session(64)

    with pytest.raises(TypeError, match=error_match):
        getattr(session, method_name)(bad_value)

    counts = counters()
    assert counts["load_snapshot_calls"] == 0
    assert counts["load_payload_calls"] == 0

    session.close()
    engine.close()


@pytest.mark.parametrize(
    ("method_name", "bad_value", "error_match"),
    [
        ("load_snapshot", b"not a snapshot", "load_snapshot failed"),
        ("load_payload", b"not a payload", "load_payload failed"),
    ],
)
def test_corrupt_snapshot_and_payload_raise_generation_errors(
    method_name: str,
    bad_value: bytes,
    error_match: str,
) -> None:
    engine = make_engine()
    session = engine.create_session(64)

    with pytest.raises(pyds4.Ds4GenerationError, match=error_match):
        getattr(session, method_name)(bad_value)

    session.close()
    engine.close()


@pytest.mark.parametrize("method_name", ["load_snapshot", "load_payload"])
def test_snapshot_and_payload_from_wrong_context_are_rejected(
    method_name: str,
) -> None:
    engine = make_engine()
    source = engine.create_session(64)
    target = engine.create_session(32)
    source.sync([1, 2])
    data = (
        source.save_snapshot()
        if method_name == "load_snapshot"
        else source.save_payload()
    )

    with pytest.raises(pyds4.Ds4GenerationError, match="context"):
        getattr(target, method_name)(data)

    source.close()
    target.close()
    engine.close()


@pytest.mark.parametrize("method_name", ["load_snapshot", "load_payload"])
def test_snapshot_and_payload_from_wrong_model_are_rejected(
    method_name: str,
) -> None:
    source_engine = make_engine(model_path="source.gguf")
    target_engine = make_engine(model_path="target.gguf")
    source = source_engine.create_session(64)
    target = target_engine.create_session(64)
    source.sync([1, 2])
    data = (
        source.save_snapshot()
        if method_name == "load_snapshot"
        else source.save_payload()
    )

    with pytest.raises(pyds4.Ds4GenerationError, match="model"):
        getattr(target, method_name)(data)

    source.close()
    target.close()
    source_engine.close()
    target_engine.close()


def test_snapshot_and_payload_calls_after_close_raise_context_error() -> None:
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1])
    snapshot = session.save_snapshot()
    payload = session.save_payload()

    session.close()

    with pytest.raises(pyds4.Ds4ContextError, match="session is closed"):
        session.save_snapshot()
    with pytest.raises(pyds4.Ds4ContextError, match="session is closed"):
        session.load_snapshot(snapshot)
    with pytest.raises(pyds4.Ds4ContextError, match="session is closed"):
        session.save_payload()
    with pytest.raises(pyds4.Ds4ContextError, match="session is closed"):
        session.load_payload(payload)

    engine.close()


def test_invalidate_can_be_resynchronized_or_reinitialized() -> None:
    engine = make_engine()
    session = engine.create_session(64)
    session.sync([1])

    session.invalidate()
    session.sync([1, 2])

    assert session.pos == 2
    assert session.tokens == [1, 2]
    assert session.argmax() == 1000 + ord("A")

    session.invalidate()
    session.eval(1000 + ord("A"))

    assert session.pos == 1
    assert session.tokens == [1000 + ord("A")]
    assert session.argmax() == 1000 + ord("A")

    session.close()
    engine.close()


@pytest.mark.parametrize("token_id", [True, -1, 1.5])
def test_eval_rejects_invalid_token_ids_before_native_call(
    token_id: object,
) -> None:
    engine = make_engine()
    session = engine.create_session(64)

    with pytest.raises((TypeError, ValueError)):
        session.eval(token_id)  # type: ignore[arg-type]

    assert counters()["eval_calls"] == 0

    session.close()
    engine.close()


@pytest.mark.parametrize("token_id", [False, -1, 1.5])
def test_argmax_excluding_rejects_invalid_token_ids_before_native_call(
    token_id: object,
) -> None:
    engine = make_engine()
    session = engine.create_session(64)

    with pytest.raises((TypeError, ValueError)):
        session.argmax_excluding(token_id)  # type: ignore[arg-type]

    assert counters()["argmax_excluding_calls"] == 0

    session.close()
    engine.close()


@pytest.mark.parametrize("pos", [False, -1, 1.5])
def test_rewind_rejects_invalid_positions_before_native_call(
    pos: object,
) -> None:
    engine = make_engine()
    session = engine.create_session(64)

    with pytest.raises((TypeError, ValueError)):
        session.rewind(pos)  # type: ignore[arg-type]

    assert counters()["rewind_calls"] == 0

    session.close()
    engine.close()


def test_sample_rejects_invalid_options_before_native_call() -> None:
    engine = make_engine()
    session = engine.create_session(64)

    with pytest.raises(TypeError, match="SamplingOptions"):
        session.sample(object())  # type: ignore[arg-type]

    assert counters()["sample_calls"] == 0

    session.close()
    engine.close()


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_native_session_sample_rejects_invalid_seed_range(seed: int) -> None:
    engine_state = _native.EngineState(
        "model.gguf",
        "cpu",
        None,
        0,
        0,
        0.0,
        None,
        0.0,
        0.0,
        False,
        False,
    )
    session_state = engine_state.create_session(64)

    with pytest.raises(ValueError, match="seed"):
        session_state.sample(0.7, 0, 1.0, 0.0, seed)

    assert counters()["sample_calls"] == 0

    session_state.close()
    engine_state.close()


def test_session_create_failure_maps_to_context_error_with_ctx_size() -> None:
    engine = make_engine(model_path="ctx-fail-model.gguf")

    with pytest.raises(pyds4.Ds4ContextError, match="ctx_size 64"):
        engine.create_session(64)

    snapshot = counters()
    assert snapshot["session_create_calls"] == 1
    assert snapshot["session_free_calls"] == 0

    engine.close()


@pytest.mark.parametrize(
    "prompt_tokens",
    [
        [True],
        [1, False],
        [-1],
        [1.5],
        "not a token list",
    ],
)
def test_invalid_token_ids_are_rejected_before_native_sync(
    prompt_tokens: object,
) -> None:
    engine = make_engine()
    session = engine.create_session(64)

    with pytest.raises((TypeError, ValueError)):
        session.sync(prompt_tokens)  # type: ignore[arg-type]

    snapshot = counters()
    assert snapshot["sync_calls"] == 0
    assert snapshot["tokens_free_calls"] == 0

    session.close()
    engine.close()


def test_token_text_returns_raw_bytes() -> None:
    engine = make_engine()

    assert engine.token_text(1000) == b"\x00"
    assert engine.token_text(1000 + ord("A")) == b"A"
    assert isinstance(engine.token_text(6), bytes)
    with pytest.raises(ValueError, match="non-negative"):
        engine.token_text(-1)
    with pytest.raises(TypeError, match="integer token id"):
        engine.token_text(True)  # type: ignore[arg-type]

    engine.close()


def test_engine_metadata_properties_use_native_values() -> None:
    engine = make_engine(
        mtp_path="mtp.gguf",
        mtp_draft_tokens=7,
        quality=True,
    )

    assert engine.routed_quant_bits == 4
    assert engine.has_mtp is True
    assert engine.mtp_draft_tokens == 7
    assert engine.eos_token_id == 6

    engine.close()


def test_engine_tokenization_and_chat_begin_return_fresh_token_lists() -> None:
    engine = make_engine()

    text = engine.tokenize_text("Az")
    rendered = engine.tokenize_rendered_chat("Hi")
    chat = engine.chat_begin()

    assert text == text_tokens("Az")
    assert rendered == text_tokens("Hi")
    assert chat == [1]
    assert text is not engine.tokenize_text("Az")

    engine.close()


def test_engine_chat_append_message_mutates_input_tokens() -> None:
    engine = make_engine()
    tokens = engine.chat_begin()

    result = engine.chat_append_message(tokens, "system", "S")
    assert result is None
    assert tokens == [1, *text_tokens("S")]

    engine.chat_append_message(tokens, "user", "U")
    assert tokens == [1, *text_tokens("S"), 2, *text_tokens("U")]

    engine.chat_append_message(tokens, "assistant", "A")
    assert tokens == [
        1,
        *text_tokens("S"),
        2,
        *text_tokens("U"),
        3,
        *text_tokens("A"),
    ]

    engine.close()


def test_engine_chat_prefix_and_prompt_rendering_use_think_modes() -> None:
    engine = make_engine()

    none_tokens = [42]
    engine.chat_append_assistant_prefix(none_tokens, pyds4.ThinkMode.NONE)
    assert none_tokens == [42, 3, 5]

    high_tokens = [42]
    engine.chat_append_assistant_prefix(high_tokens, "high")
    assert high_tokens == [42, 3, 4]

    assert engine.encode_chat_prompt("S", "P", pyds4.ThinkMode.HIGH) == [
        1,
        *text_tokens("S"),
        2,
        *text_tokens("P"),
        3,
        4,
    ]
    assert engine.encode_chat_prompt(None, "P", pyds4.ThinkMode.NONE) == [
        1,
        2,
        *text_tokens("P"),
        3,
        5,
    ]

    engine.close()


def test_engine_chat_helpers_reject_invalid_inputs_before_mutation() -> None:
    engine = make_engine()
    tokens = [1]

    with pytest.raises(TypeError, match="text"):
        engine.tokenize_text(b"bytes")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="tokens"):
        engine.chat_append_message((1,), "user", "U")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported DS4 chat role"):
        engine.chat_append_message(tokens, "tool", "T")
    assert tokens == [1]

    tokens_with_bool = [1, True]
    with pytest.raises(TypeError, match="token"):
        engine.chat_append_message(tokens_with_bool, "user", "U")
    assert tokens_with_bool == [1, True]

    with pytest.raises(TypeError, match="token"):
        engine.chat_append_assistant_prefix(
            tokens_with_bool,
            pyds4.ThinkMode.NONE,
        )
    assert tokens_with_bool == [1, True]

    with pytest.raises(ValueError, match="Unsupported DS4 think mode"):
        engine.chat_append_assistant_prefix(  # type: ignore[arg-type]
            tokens,
            "turbo",
        )
    assert tokens == [1]

    with pytest.raises(TypeError, match="prompt"):
        engine.encode_chat_prompt(  # type: ignore[arg-type]
            None,
            b"P",
            pyds4.ThinkMode.NONE,
        )

    engine.close()


def test_engine_chat_helpers_free_temporary_native_token_buffers() -> None:
    engine = make_engine()

    tokens = engine.chat_begin()
    engine.tokenize_text("a")
    engine.tokenize_rendered_chat("b")
    engine.chat_append_message(tokens, "user", "c")
    engine.chat_append_assistant_prefix(tokens, pyds4.ThinkMode.HIGH)
    engine.encode_chat_prompt(None, "d", pyds4.ThinkMode.NONE)

    snapshot = counters()
    assert snapshot["tokens_free_calls"] == 6
    assert snapshot["token_allocation_calls"] >= 6
    assert snapshot["token_live_allocations"] == 0

    engine.close()


def test_public_c_int_boundaries_are_rejected_before_native_calls() -> None:
    engine = make_engine()

    with pytest.raises(ValueError, match="2147483647"):
        engine.create_session(2**31)
    assert counters()["session_create_calls"] == 0

    session = engine.create_session(64)

    with pytest.raises(ValueError, match="2147483647"):
        engine.token_text(2**31)
    with pytest.raises(ValueError, match="2147483647"):
        session.sync([2**31])
    with pytest.raises(ValueError, match="2147483647"):
        session.eval(2**31)
    with pytest.raises(ValueError, match="2147483647"):
        session.argmax_excluding(2**31)
    with pytest.raises(ValueError, match="2147483647"):
        session.rewind(2**31)

    snapshot = counters()
    assert snapshot["sync_calls"] == 0
    assert snapshot["eval_calls"] == 0
    assert snapshot["argmax_excluding_calls"] == 0
    assert snapshot["rewind_calls"] == 0

    session.close()
    engine.close()


def test_fake_native_error_injection_can_use_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYDS4_FAKE_FAIL", "engine_open")

    with pytest.raises(pyds4.Ds4LoadError, match="ds4_engine_open failed"):
        make_engine()

    assert counters()["engine_open_calls"] == 1


def test_fake_native_error_injection_maps_generation_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = make_engine()
    session = engine.create_session(64)

    monkeypatch.setenv("PYDS4_FAKE_FAIL_SYNC", "1")
    with pytest.raises(
        pyds4.Ds4GenerationError,
        match="sync failed: fake env sync failure",
    ):
        session.sync([1])

    monkeypatch.delenv("PYDS4_FAKE_FAIL_SYNC")
    session.sync([1])

    monkeypatch.setenv("PYDS4_FAKE_FAIL", "argmax_excluding, sample")
    with pytest.raises(pyds4.Ds4GenerationError, match="argmax_excluding"):
        session.argmax_excluding(1)
    with pytest.raises(pyds4.Ds4GenerationError, match="sample"):
        session.sample(pyds4.SamplingOptions(seed=123))

    session.close()
    engine.close()


def test_fake_native_token_id_error_injection_maps_public_errors() -> None:
    engine = make_engine()
    session = engine.create_session(64)

    with pytest.raises(pyds4.Ds4GenerationError, match="token_text failed"):
        engine.token_text(13)
    with pytest.raises(
        pyds4.Ds4GenerationError,
        match="argmax_excluding failed",
    ):
        session.argmax_excluding(13)
    with pytest.raises(
        pyds4.Ds4GenerationError,
        match="eval failed: fake token eval failure",
    ):
        session.eval(13)

    session.close()
    engine.close()
