DS4_COMMIT = "8809b90a1e3247389d7652b565ab6772e036f1ea"
DS4_API_VERSION: int | None = None
THINK_MAX_MIN_CONTEXT = 393216
C_INT_MAX = 2_147_483_647

SUPPORTED_NATIVE_BACKENDS = ("metal", "cuda", "cpu")

PRODUCTION_TARGETS = "macOS arm64 + Metal and Linux + CUDA"
CPU_DIAGNOSTIC_NOTE = "CPU mode is diagnostic/reference only."

REQUIRED_C_SYMBOLS = (
    "ds4_engine_open",
    "ds4_engine_close",
    "ds4_backend_name",
    "ds4_think_mode_enabled",
    "ds4_think_mode_name",
    "ds4_think_max_prefix",
    "ds4_think_max_min_context",
    "ds4_think_mode_for_context",
    "ds4_context_memory_estimate",
    "ds4_tokens_push",
    "ds4_tokens_free",
    "ds4_tokens_copy",
    "ds4_tokens_starts_with",
    "ds4_tokenize_text",
    "ds4_tokenize_rendered_chat",
    "ds4_chat_begin",
    "ds4_encode_chat_prompt",
    "ds4_chat_append_message",
    "ds4_chat_append_assistant_prefix",
    "ds4_token_text",
    "ds4_token_eos",
    "ds4_session_create",
    "ds4_session_free",
    "ds4_session_set_progress",
    "ds4_session_sync",
    "ds4_session_argmax",
    "ds4_session_argmax_excluding",
    "ds4_session_sample",
    "ds4_session_eval",
    "ds4_session_invalidate",
    "ds4_session_rewind",
    "ds4_session_pos",
    "ds4_session_ctx",
    "ds4_engine_routed_quant_bits",
    "ds4_engine_has_mtp",
    "ds4_engine_mtp_draft_tokens",
    "ds4_session_tokens",
    "ds4_session_payload_bytes",
    "ds4_session_save_payload",
    "ds4_session_load_payload",
    "ds4_session_save_snapshot",
    "ds4_session_load_snapshot",
    "ds4_session_snapshot_free",
)
