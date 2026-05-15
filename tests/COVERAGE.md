# pyds4 Coverage Manifest

This manifest maps the implemented public surface from
`specs/DS4-BINDING-IMPL.md` to the tests that currently cover it.

## Phase 0: Repository And Build Baseline

- Package metadata and import-safe constants:
  `tests/test_metadata.py::test_import_safe_metadata`
- Required C symbol metadata:
  `tests/test_metadata.py::test_required_symbols_are_exposed_as_collection`
- Import does not instantiate `Engine` or expose raw native open symbols:
  `tests/test_metadata.py::test_import_does_not_instantiate_engine_or_expose_native_open`
- Generated build metadata agrees with the native extension:
  `tests/test_build_configuration.py::test_build_config_matches_public_native_metadata`
- Avalan import compatibility:
  covered by Avalan's `tests/model/ds4_binding_packaging_test.py` and
  `tests/model/ds4_api_verification_test.py`

## Phase 1: Public Python Types And Errors

- `Backend`, `ThinkMode`, `EngineOptions`, `SamplingOptions`,
  `GenerationStep`, and `ProgressEvent` construction, defaults where
  applicable, enum coercion where applicable, immutability, and slots:
  `tests/test_public_types.py::test_public_type_defaults_and_enum_coercion`
  and `tests/test_public_types.py::test_public_types_are_frozen_and_slotted`
- Sampling boundary validation and bool rejection for integer fields:
  `tests/test_public_types.py::test_sampling_options_reject_invalid_values`
- Engine integer-count validation:
  `tests/test_public_types.py::test_engine_options_reject_non_integer_counts`
  `tests/test_public_types.py::test_engine_options_reject_negative_counts`,
  and `tests/test_public_types.py::test_engine_options_reject_counts_outside_c_int_range`
- Engine path and floating-point option validation:
  `tests/test_public_types.py::test_engine_options_reject_non_string_paths`,
  `tests/test_public_types.py::test_engine_options_reject_non_numeric_float_options`,
  `tests/test_public_types.py::test_engine_options_reject_non_finite_float_options`,
  and `tests/test_public_types.py::test_engine_options_reject_non_boolean_flags`
- Exception class exports from `pyds4` and `pyds4.errors`:
  `tests/test_public_types.py::test_public_exceptions_are_reexported`
- Context-aware thinking mode behavior:
  `tests/test_public_types.py::test_think_mode_for_context_validates_context_size`
- Unexpected native thinking-mode values and native helper errors map to
  `Ds4GenerationError`:
  `tests/test_public_types.py::test_think_mode_for_context_maps_invalid_native_output`
  and `tests/test_public_types.py::test_think_mode_for_context_maps_native_helper_errors`

## Phase 2: DS4 Source Integration

- Fake DS4 shim declares and implements all required and wrapper-used C
  symbols:
  `tests/test_fake_ds4_source.py::test_fake_ds4_source_declares_and_implements_required_symbols`
- Direct native C++ tests cover shared helper logic, fake DS4 C API lifecycle,
  native token-buffer RAII, tokenization/chat helpers, and sanitizer-compatible
  teardown:
  `tests/native/test_native_cxx.cpp`
- Backend availability helpers, selected backend, unavailable backends,
  unknown backend input, and actionable reason strings:
  `tests/test_metadata.py::test_backend_availability_is_limited_to_selected_backend`,
  `tests/test_metadata.py::test_native_backend_availability_respects_host_support`,
  `tests/test_metadata.py::test_unknown_backend_is_unavailable_with_reason`,
  `tests/test_metadata.py::test_built_backend_is_unavailable_when_native_extension_fails`,
  `tests/test_metadata.py::test_import_remains_safe_when_built_native_extension_cannot_load`,
  and `tests/test_metadata.py::test_source_tree_without_native_extension_reports_unavailable`
- CMake Metal source inventory matches runtime environment configuration:
  `tests/test_build_configuration.py::test_cmake_metal_source_inventory_matches_runtime_configuration`
- Metal kernel source path configuration:
  `tests/test_metal_sources.py`

## Phase 3: Native RAII Layer

- Engine/session native open and close counters, idempotent public close
  methods, and exactly-once native frees:
  `tests/test_native_raii.py::test_engine_and_session_close_are_idempotent`
- Session creation failures map to `Ds4ContextError` and include the requested
  context size:
  `tests/test_native_raii.py::test_session_create_failure_maps_to_context_error_with_ctx_size`
  and `tests/test_native_public_validation.py::test_session_create_context_error_includes_requested_ctx_size`
- Explicit engine close deterministically closes live sessions before the
  engine handle:
  `tests/test_native_raii.py::test_engine_close_closes_live_sessions_before_engine`
- Engine close waits for an in-flight native session call before freeing the
  session and engine handles:
  `tests/test_native_raii.py::test_engine_close_waits_for_in_flight_session_call`
- Sessions keep their engine state alive until session close:
  `tests/test_native_raii.py::test_session_keeps_engine_alive_until_session_close`
- Temporary native token buffers are freed after conversion, and native
  session token buffers are converted back to `list[int]`:
  `tests/test_native_raii.py::test_sync_uses_temporary_token_buffer_and_round_trips_tokens`
- Public session inspection rejects malformed native `pos`, `ctx`, and token
  history values:
  `tests/test_native_public_validation.py::test_session_rejects_malformed_native_inspection_values`
- Invalid token sequences, bool token IDs, and negative token IDs are rejected
  before native sync calls:
  `tests/test_native_raii.py::test_invalid_token_ids_are_rejected_before_native_sync`
- Token text is returned as raw `bytes`:
  `tests/test_native_raii.py::test_token_text_returns_raw_bytes`

## Phase 4: Engine Binding

- Engine metadata properties call through the native layer and return typed
  public values:
  `tests/test_native_raii.py::test_engine_metadata_properties_use_native_values`
- Tokenization and `chat_begin()` return fresh Python token lists:
  `tests/test_native_raii.py::test_engine_tokenization_and_chat_begin_return_fresh_token_lists`
- Tokenization, `chat_begin()`, and `encode_chat_prompt()` reject malformed
  native token list outputs:
  `tests/test_native_public_validation.py::test_engine_rejects_malformed_native_token_list_results`
- Chat append helpers mutate input lists in place, support system/user/
  assistant roles, and apply assistant prefix thinking modes:
  `tests/test_native_raii.py::test_engine_chat_append_message_mutates_input_tokens`
  and `tests/test_native_raii.py::test_engine_chat_prefix_and_prompt_rendering_use_think_modes`
- Chat append helpers reject malformed token mutations from the native layer:
  `tests/test_native_public_validation.py::test_engine_rejects_malformed_chat_append_mutations`
- Chat rendering rejects invalid roles, invalid token lists, invalid think
  modes, and non-string text before native calls:
  `tests/test_native_raii.py::test_engine_chat_helpers_reject_invalid_inputs_before_mutation`
- Engine token text preserves raw bytes and rejects invalid token ids:
  `tests/test_native_raii.py::test_token_text_returns_raw_bytes` and
  `tests/test_native_raii.py::test_public_c_int_boundaries_are_rejected_before_native_calls`
- Temporary native token buffers used by engine tokenization/chat helpers are
  freed:
  `tests/test_native_raii.py::test_engine_chat_helpers_free_temporary_native_token_buffers`
- Model path, GGUF header, MTP path, directional steering preflight, and
  native open failure mapping:
  `tests/test_engine_open_errors.py`

## Phase 5: Thinking Mode And Context Helpers

- Public `think_mode_for_context()` validates context sizes, covers all
  public `ThinkMode` values, and proves max-mode downgrade behavior:
  `tests/test_public_types.py::test_think_mode_for_context_validates_context_size`
- Unsupported native thinking-mode values and native helper errors surface as
  `Ds4GenerationError`:
  `tests/test_public_types.py::test_think_mode_for_context_maps_invalid_native_output`
  and `tests/test_public_types.py::test_think_mode_for_context_maps_native_helper_errors`
- Required thinking C symbols are included in public metadata:
  `tests/test_metadata.py::test_required_symbols_are_exposed_as_collection`

## Phase 6: Session Binding And Generation Semantics

- Session `sync()` sets native session token history and position:
  `tests/test_native_raii.py::test_sync_uses_temporary_token_buffer_and_round_trips_tokens`
- Session `eval()` advances position and token history by exactly one accepted
  token:
  `tests/test_native_raii.py::test_eval_advances_session_by_one_token`
- Native `eval()` failures preserve the operation name and message:
  `tests/test_native_raii.py::test_eval_failure_preserves_operation_and_native_message`
- `argmax()`, `argmax_excluding()`, and `sample()` return candidate token IDs
  without advancing session state:
  `tests/test_native_raii.py::test_candidate_selection_does_not_advance_session_state`
- Candidate selection is rejected before logits are initialized and after
  `rewind()` changes the session position until a later `sync()` or `eval()`:
  `tests/test_native_raii.py::test_candidate_selection_requires_initialized_logits`
  and `tests/test_native_raii.py::test_rewind_requires_resync_before_candidate_selection`
- Sampled generation keeps mutable native RNG state across repeated
  `sample()` calls and starts a new deterministic stream after `sync()`:
  `tests/test_native_raii.py::test_sample_preserves_rng_state_until_sync_resets_stream`
  and `tests/test_native_raii.py::test_failed_sample_does_not_start_seeded_rng_stream`
- `rewind()` and `invalidate()` update the native session according to the
  fake DS4 contract:
  `tests/test_native_raii.py::test_rewind_and_invalidate_update_native_session_state`
- Invalid eval/exclusion token IDs, rewind positions, and sampling options
  are rejected before native calls:
  `tests/test_native_raii.py::test_eval_rejects_invalid_token_ids_before_native_call`,
  `tests/test_native_raii.py::test_argmax_excluding_rejects_invalid_token_ids_before_native_call`,
  `tests/test_native_raii.py::test_rewind_rejects_invalid_positions_before_native_call`,
  `tests/test_native_raii.py::test_sample_rejects_invalid_options_before_native_call`,
  and `tests/test_native_raii.py::test_public_c_int_boundaries_are_rejected_before_native_calls`
- Public session candidate methods reject malformed native token results:
  `tests/test_native_public_validation.py::test_session_rejects_malformed_native_token_results`

## Phase 7: Error Mapping And Diagnostics

- Public exception classes remain stable, importable, and raiseable:
  `tests/test_public_types.py::test_public_exceptions_are_reexported`
  and `tests/test_public_types.py::test_public_exceptions_are_raiseable`
- Model path preflight, invalid GGUF headers, native invalid-model messages,
  preserved native public exceptions, and generic native open failures:
  `tests/test_engine_open_errors.py`
- Runtime native backend-unavailable messages map to
  `Ds4BackendUnavailable`, and native DS4 model compatibility diagnostics
  that mention missing metadata/tensors map to `Ds4InvalidModel`:
  `tests/test_engine_open_errors.py::test_runtime_backend_unavailable_message_maps_to_backend_error`
  and `tests/test_engine_open_errors.py::test_native_ds4_model_compatibility_message_maps_to_invalid_model`
- Unavailable backend construction maps to `Ds4BackendUnavailable` before
  native open and includes production-target plus CPU diagnostic guidance:
  `tests/test_engine_open_errors.py::test_unavailable_backend_maps_to_backend_unavailable_before_open`
- Unsupported backend reasons include supported native backend names,
  production targets, and CPU diagnostic guidance:
  `tests/test_metadata.py::test_unknown_backend_is_unavailable_with_reason`
- Session allocation/context failures map to `Ds4ContextError` with requested
  context size:
  `tests/test_native_raii.py::test_session_create_failure_maps_to_context_error_with_ctx_size`
- Closed engine and session operations preserve their already-mapped public
  `Ds4LoadError` and `Ds4ContextError` classes instead of being rewrapped as
  generation failures:
  `tests/test_native_raii.py::test_closed_engine_and_session_errors_keep_public_error_classes`
- Engine tokenization, chat rendering, token text, and prompt-rendering
  failures map to operation-specific `Ds4GenerationError` messages while
  preserving native detail:
  `tests/test_native_public_validation.py::test_engine_generation_failures_map_to_operation_errors`
  and `tests/test_native_public_validation.py::test_generation_error_mapping_does_not_duplicate_operation_prefix`
- Session `sync`, `eval`, `argmax`, `argmax_excluding`, `sample`, `rewind`,
  and `invalidate` failures map to operation-specific
  `Ds4GenerationError` messages:
  `tests/test_native_public_validation.py::test_session_generation_failures_map_to_operation_errors`
- Fake-native generation failure coverage includes failed `sync`, failed
  `eval`, failed candidate generation after invalidation, and invalid
  sampling options rejected before native calls:
  `tests/test_native_raii.py::test_sync_failure_preserves_operation_and_native_message`,
  `tests/test_native_raii.py::test_eval_failure_preserves_operation_and_native_message`,
  `tests/test_native_raii.py::test_rewind_and_invalidate_update_native_session_state`,
  `tests/test_native_raii.py::test_sample_rejects_invalid_options_before_native_call`,
  `tests/test_native_raii.py::test_fake_native_error_injection_maps_generation_failures`,
  and `tests/test_native_raii.py::test_fake_native_token_id_error_injection_maps_public_errors`

## Phase 8: Fake DS4 C Shim

- Fake DS4 declares and implements every required C symbol exposed through
  public metadata:
  `tests/test_fake_ds4_source.py::test_fake_ds4_source_declares_and_implements_required_symbols`
- Fake-native tests execute against an explicit `PYDS4_USE_FAKE_DS4=1` build,
  and candidate generation is verified not to advance state:
  `tests/test_native_raii.py::test_candidate_selection_does_not_advance_session_state`
- CI builds and tests the explicit fake-native configuration on Python 3.11
  and 3.12:
  `.github/workflows/fake-native.yml`
- The fake shim exposes live token-buffer allocation counters and every
  fake-native test asserts teardown returns to zero live token buffers:
  `tests/test_native_raii.py::reset_fake_counters`
- Engine temporary token buffers are explicitly counted and checked after
  tokenization and chat helper calls:
  `tests/test_native_raii.py::test_engine_chat_helpers_free_temporary_native_token_buffers`
- Environment-driven fake-native error injection is covered by:
  `tests/test_native_raii.py::test_fake_native_error_injection_can_use_environment`
  and `tests/test_native_raii.py::test_fake_native_error_injection_maps_generation_failures`

## Phase 9: Real DS4 Integration Profiles

- Named real-model profile resolution for explicit environment variables,
  local macOS Metal, Linux CUDA, and opt-in CPU diagnostic/reference runs:
  `tests/test_real_ds4_profiles.py`
- Synchronous real-model smoke coverage for every configured profile:
  `tests/test_real_ds4_integration.py::test_real_ds4_generation_smoke`
- Async real-model smoke coverage for every configured profile:
  `tests/test_async_real_ds4_integration.py::test_async_real_ds4_generation_smoke`

## Phase 12: Async Facade And Generation Optimizations

- Async facade exports from both `pyds4` and `pyds4.asyncio`:
  `tests/test_async_facade.py::test_async_exports_are_available`
- Dedicated owner-thread execution and FIFO serialization for engine and
  session operations:
  `tests/test_async_facade.py::test_async_engine_uses_one_owner_thread_and_serializes_calls`
- Async validation and public exception mapping mirror the synchronous
  `Engine` and `Session` wrappers for engine methods, metadata properties,
  session inspection, primitive session operations, `create_session()`, and
  `next_token()`:
  `tests/test_async_facade.py::test_async_engine_methods_mirror_sync_validation_errors`,
  `tests/test_async_facade.py::test_async_engine_metadata_mirrors_sync_validation_errors`,
  `tests/test_async_facade.py::test_async_engine_generation_failures_mirror_sync_error_mapping`,
  `tests/test_async_facade.py::test_async_create_session_mirrors_sync_context_error_mapping`,
  `tests/test_async_facade.py::test_async_session_methods_mirror_sync_validation_errors`,
  `tests/test_async_facade.py::test_async_session_properties_mirror_sync_validation_errors`,
  `tests/test_async_facade.py::test_async_session_generation_failures_mirror_sync_error_mapping`,
  and `tests/test_async_facade.py::test_async_next_token_eval_failure_mirrors_sync_error_mapping`
- Async chat append helpers work on copied token lists and mutate caller
  lists only after successful native work:
  `tests/test_async_facade.py::test_async_chat_append_mutates_original_only_after_success`
- Cancelled async chat append helpers do not mutate caller-owned token lists,
  even when the native work started on the copied list:
  `tests/test_async_facade.py::test_async_cancelled_chat_append_does_not_mutate_original`
- Queued cancelled async jobs are skipped before native work starts:
  `tests/test_async_facade.py::test_async_queued_cancelled_job_is_not_run`
- In-flight non-mutating async cancellation propagates
  `asyncio.CancelledError` and discards the later result:
  `tests/test_async_facade.py::test_async_in_flight_non_mutating_cancellation_discards_result`
- Async engine and session cleanup is shielded from caller cancellation:
  `tests/test_async_facade.py::test_async_engine_close_is_shielded_from_caller_cancellation`
  and `tests/test_async_facade.py::test_async_session_close_is_shielded_from_caller_cancellation`
- In-flight mutating async session cancellation poisons the async session with
  `Ds4Cancelled` and closes the native session after the in-flight call
  returns:
  `tests/test_async_facade.py::test_async_in_flight_mutating_cancellation_poisons_session`
  and `tests/test_async_facade.py::test_async_fake_native_mutating_cancellation_closes_session`
- Async `next_token()` selects a candidate, checks EOS, optionally decodes
  bytes, and commits with `eval()` as one serialized worker job:
  `tests/test_async_facade.py::test_async_next_token_selects_decodes_and_commits_in_one_job`
  and `tests/test_async_facade.py::test_async_next_token_stops_before_eos_eval_and_decode`
- Async progress notifications are queued through `AsyncSession.progress`
  without invoking Python from the native DS4 callback path:
  `tests/test_async_facade.py::test_async_session_progress_events_are_queued`
  and
  `tests/test_async_facade.py::test_async_fake_native_progress_notifications_are_queued`
- Fake-native operation delay controls for cancellation and close-while-busy
  tests:
  `tests/test_async_facade.py::test_async_fake_native_mutating_cancellation_closes_session`
- Async fake-native context managers close engine/session resources exactly
  once and in native-safe order:
  `tests/test_async_facade.py::test_async_fake_native_context_managers_close_resources_once`
- Async real-model generation mirrors the synchronous hardware-gated smoke:
  `tests/test_async_real_ds4_integration.py::test_async_real_ds4_generation_smoke`

## Remaining Coverage

- Real-model integration is covered by
  `tests/test_real_ds4_integration.py::test_real_ds4_generation_smoke`
  and
  `tests/test_async_real_ds4_integration.py::test_async_real_ds4_generation_smoke`.
  These are skipped until `PYDS4_MODEL`, `PYDS4_BACKEND`, and `PYDS4_CTX`
  are configured with a supported DS4 GGUF, or a supported
  `PYDS4_REAL_PROFILE` is selected. The local macOS arm64 + Metal fixture is
  documented in `specs/DS4-BINDING-IMPL.md`.
- Release wheel smoke coverage is provided by `scripts/wheel_smoke.py`. It
  installs a built wheel into a fresh virtual environment, runs `pip check`,
  verifies import-safe metadata, checks backend availability expectations, and
  exercises fake-native engine/session semantics when the wheel was built with
  `PYDS4_USE_FAKE_DS4=1`. With `--model`, it also runs a one-token real DS4
  smoke against a hardware-supported backend.
