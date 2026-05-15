#include "ds4.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

enum {
    FAKE_TOKEN_BOS = 1,
    FAKE_TOKEN_USER = 2,
    FAKE_TOKEN_ASSISTANT = 3,
    FAKE_TOKEN_THINK_START = 4,
    FAKE_TOKEN_THINK_END = 5,
    FAKE_TOKEN_EOS = 6,
    FAKE_TOKEN_BYTE_BASE = 1000,
};

static const uint32_t FAKE_THINK_MAX_MIN_CONTEXT = 393216u;
static const char FAKE_THINK_MAX_PREFIX[] =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n";

struct ds4_engine {
    ds4_backend backend;
    char* model_path;
    char* mtp_path;
    int mtp_draft_tokens;
    bool quality;
};

struct ds4_session {
    ds4_engine* engine;
    ds4_tokens checkpoint;
    ds4_session_progress_fn progress;
    void* progress_ud;
    int ctx_size;
    bool valid;
};

static pyds4_fake_counters counters;

static uint64_t next_call(void) {
    counters.call_sequence++;
    return counters.call_sequence;
}

static char* fake_strdup(const char* value) {
    if (!value)
        return NULL;
    const size_t len = strlen(value);
    char* copy = malloc(len + 1);
    if (!copy)
        abort();
    memcpy(copy, value, len + 1);
    return copy;
}

static bool should_fail(const char* value, const char* needle) {
    return value && strstr(value, needle) != NULL;
}

static bool env_value_is_enabled(const char* value) {
    return value && value[0] && strcmp(value, "0") != 0 &&
           strcmp(value, "false") != 0 && strcmp(value, "False") != 0;
}

static bool env_list_contains(const char* value, const char* operation) {
    if (!env_value_is_enabled(value))
        return false;

    const char* cursor = value;
    while (*cursor) {
        while (*cursor == ',' || *cursor == ' ' || *cursor == '\t')
            cursor++;
        const char* start = cursor;
        while (*cursor && *cursor != ',')
            cursor++;
        const char* end = cursor;
        while (end > start && (end[-1] == ' ' || end[-1] == '\t'))
            end--;

        const size_t len = (size_t)(end - start);
        if ((len == 3 && strncmp(start, "all", len) == 0) ||
            (strlen(operation) == len &&
             strncmp(start, operation, len) == 0)) {
            return true;
        }
    }
    return false;
}

static bool env_should_fail(const char* operation) {
    char specific[128] = "PYDS4_FAKE_FAIL_";
    const size_t prefix_len = strlen(specific);
    size_t i = 0;
    for (; operation[i] && prefix_len + i + 1 < sizeof(specific); i++) {
        specific[prefix_len + i] = (char)toupper((unsigned char)operation[i]);
    }
    specific[prefix_len + i] = '\0';

    return env_value_is_enabled(getenv(specific)) ||
           env_list_contains(getenv("PYDS4_FAKE_FAIL"), operation);
}

static void operation_env_name(char* out, size_t out_size, const char* prefix,
                               const char* operation, const char* suffix) {
    if (out_size == 0)
        return;
    const int written = snprintf(out, out_size, "%s", prefix);
    if (written < 0 || (size_t)written >= out_size) {
        out[out_size - 1] = '\0';
        return;
    }

    size_t cursor = (size_t)written;
    for (size_t i = 0; operation[i] && cursor + 1 < out_size; i++) {
        out[cursor++] = (char)toupper((unsigned char)operation[i]);
    }
    out[cursor] = '\0';
    if (suffix) {
        snprintf(out + cursor, out_size - cursor, "%s", suffix);
    }
}

static unsigned long env_delay_ms(const char* operation) {
    char specific[128];
    operation_env_name(specific, sizeof(specific), "PYDS4_FAKE_DELAY_",
                       operation, "_MS");
    const char* value = getenv(specific);
    if (!env_value_is_enabled(value)) {
        return 0;
    }

    char* end = NULL;
    errno = 0;
    const unsigned long delay = strtoul(value, &end, 10);
    if (errno != 0 || end == value || delay == ULONG_MAX) {
        return 0;
    }
    return delay;
}

static void sleep_ms(unsigned long delay_ms) {
    if (delay_ms == 0)
        return;
#if defined(_WIN32)
    Sleep(delay_ms);
#else
    struct timespec remaining;
    remaining.tv_sec = (time_t)(delay_ms / 1000ul);
    remaining.tv_nsec = (long)((delay_ms % 1000ul) * 1000000ul);
    while (nanosleep(&remaining, &remaining) == -1 && errno == EINTR) {
    }
#endif
}

static void fake_delay_if_requested(const char* operation) {
    sleep_ms(env_delay_ms(operation));
}

static void set_error(char* err, size_t errlen, const char* message) {
    if (err && errlen != 0)
        snprintf(err, errlen, "%s", message);
}

static void ensure_token_capacity(ds4_tokens* tv, int needed) {
    if (!tv || needed <= tv->cap)
        return;
    int cap = tv->cap > 0 ? tv->cap : 8;
    while (cap < needed)
        cap *= 2;
    const bool had_buffer = tv->v != NULL;
    int* next = realloc(tv->v, (size_t)cap * sizeof(*next));
    if (!next)
        abort();
    if (!had_buffer) {
        counters.token_allocation_calls++;
        counters.token_live_allocations++;
        if (counters.token_live_allocations >
            counters.token_peak_live_allocations) {
            counters.token_peak_live_allocations =
                counters.token_live_allocations;
        }
    }
    tv->v = next;
    tv->cap = cap;
}

static void append_text_tokens(ds4_tokens* out, const char* text) {
    if (!text)
        text = "";
    const unsigned char* cursor = (const unsigned char*)text;
    while (*cursor) {
        ds4_tokens_push(out, FAKE_TOKEN_BYTE_BASE + (int)*cursor);
        cursor++;
    }
}

static void append_think_marker(ds4_tokens* tokens, ds4_think_mode mode) {
    ds4_tokens_push(tokens, ds4_think_mode_enabled(mode)
                                ? FAKE_TOKEN_THINK_START
                                : FAKE_TOKEN_THINK_END);
}

void pyds4_fake_reset_counters(void) {
    memset(&counters, 0, sizeof(counters));
}

pyds4_fake_counters pyds4_fake_get_counters(void) {
    return counters;
}

int ds4_engine_open(ds4_engine** out, const ds4_engine_options* opt) {
    counters.last_engine_open_sequence = next_call();
    counters.engine_open_calls++;
    if (out)
        *out = NULL;
    if (!out || !opt || !opt->model_path || !opt->model_path[0])
        return 1;
    if (env_should_fail("engine_open"))
        return 1;
    if (should_fail(opt->model_path, "open-fail"))
        return 1;

    ds4_engine* engine = calloc(1, sizeof(*engine));
    if (!engine)
        abort();
    engine->backend = opt->backend;
    engine->model_path = fake_strdup(opt->model_path);
    engine->mtp_path = fake_strdup(opt->mtp_path);
    engine->mtp_draft_tokens =
        opt->mtp_draft_tokens > 0 ? opt->mtp_draft_tokens : 1;
    engine->quality = opt->quality;
    *out = engine;
    return 0;
}

void ds4_engine_close(ds4_engine* e) {
    if (!e)
        return;
    counters.last_engine_close_sequence = next_call();
    counters.engine_close_calls++;
    free(e->model_path);
    free(e->mtp_path);
    free(e);
}

const char* ds4_backend_name(ds4_backend backend) {
    switch (backend) {
    case DS4_BACKEND_METAL:
        return "metal";
    case DS4_BACKEND_CUDA:
        return "cuda";
    case DS4_BACKEND_CPU:
        return "cpu";
    }
    return "unknown";
}

bool ds4_think_mode_enabled(ds4_think_mode mode) {
    return mode == DS4_THINK_HIGH || mode == DS4_THINK_MAX;
}

const char* ds4_think_mode_name(ds4_think_mode mode) {
    switch (mode) {
    case DS4_THINK_NONE:
        return "none";
    case DS4_THINK_HIGH:
        return "high";
    case DS4_THINK_MAX:
        return "max";
    }
    return "unknown";
}

const char* ds4_think_max_prefix(void) {
    return FAKE_THINK_MAX_PREFIX;
}

uint32_t ds4_think_max_min_context(void) {
    return FAKE_THINK_MAX_MIN_CONTEXT;
}

ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size) {
    if (mode == DS4_THINK_MAX &&
        (uint32_t)(ctx_size > 0 ? ctx_size : 0) < FAKE_THINK_MAX_MIN_CONTEXT) {
        return DS4_THINK_HIGH;
    }
    return mode;
}

ds4_context_memory ds4_context_memory_estimate(ds4_backend backend,
                                               int ctx_size) {
    (void)backend;
    const uint64_t ctx = ctx_size > 0 ? (uint64_t)ctx_size : 1u;
    ds4_context_memory memory = {0};
    memory.prefill_cap = ctx > 512u ? 512u : (uint32_t)ctx;
    memory.raw_cap = (uint32_t)ctx;
    memory.comp_cap = (uint32_t)(ctx / 4u + 2u);
    memory.raw_bytes = ctx * 64u;
    memory.compressed_bytes = ctx * 16u;
    memory.scratch_bytes = ctx * 8u;
    memory.total_bytes =
        memory.raw_bytes + memory.compressed_bytes + memory.scratch_bytes;
    return memory;
}

void ds4_tokens_push(ds4_tokens* tv, int token) {
    if (!tv)
        return;
    ensure_token_capacity(tv, tv->len + 1);
    tv->v[tv->len++] = token;
}

void ds4_tokens_free(ds4_tokens* tv) {
    if (!tv)
        return;
    counters.last_tokens_free_sequence = next_call();
    counters.tokens_free_calls++;
    if (tv->v && counters.token_live_allocations > 0) {
        counters.token_live_allocations--;
    }
    free(tv->v);
    tv->v = NULL;
    tv->len = 0;
    tv->cap = 0;
}

void ds4_tokens_copy(ds4_tokens* dst, const ds4_tokens* src) {
    if (!dst || !src)
        return;
    dst->len = 0;
    ensure_token_capacity(dst, src->len);
    if (src->len > 0) {
        memcpy(dst->v, src->v, (size_t)src->len * sizeof(*dst->v));
    }
    dst->len = src->len;
}

bool ds4_tokens_starts_with(const ds4_tokens* tokens,
                            const ds4_tokens* prefix) {
    if (!tokens || !prefix || prefix->len > tokens->len)
        return false;
    for (int i = 0; i < prefix->len; i++) {
        if (tokens->v[i] != prefix->v[i])
            return false;
    }
    return true;
}

void ds4_tokenize_text(ds4_engine* e, const char* text, ds4_tokens* out) {
    (void)e;
    append_text_tokens(out, text);
}

void ds4_tokenize_rendered_chat(ds4_engine* e, const char* text,
                                ds4_tokens* out) {
    ds4_tokenize_text(e, text, out);
}

void ds4_chat_begin(ds4_engine* e, ds4_tokens* tokens) {
    (void)e;
    ds4_tokens_push(tokens, FAKE_TOKEN_BOS);
}

void ds4_encode_chat_prompt(ds4_engine* e, const char* system,
                            const char* prompt, ds4_think_mode think_mode,
                            ds4_tokens* out) {
    ds4_chat_begin(e, out);
    if (think_mode == DS4_THINK_MAX)
        append_text_tokens(out, FAKE_THINK_MAX_PREFIX);
    append_text_tokens(out, system);
    ds4_tokens_push(out, FAKE_TOKEN_USER);
    append_text_tokens(out, prompt);
    ds4_tokens_push(out, FAKE_TOKEN_ASSISTANT);
    append_think_marker(out, think_mode);
}

void ds4_chat_append_message(ds4_engine* e, ds4_tokens* tokens,
                             const char* role, const char* content) {
    (void)e;
    if (!role)
        role = "user";
    if (strcmp(role, "system") == 0) {
        append_text_tokens(tokens, content);
    } else if (strcmp(role, "assistant") == 0) {
        ds4_tokens_push(tokens, FAKE_TOKEN_ASSISTANT);
        append_text_tokens(tokens, content);
    } else {
        ds4_tokens_push(tokens, FAKE_TOKEN_USER);
        append_text_tokens(tokens, content);
    }
}

void ds4_chat_append_assistant_prefix(ds4_engine* e, ds4_tokens* tokens,
                                      ds4_think_mode think_mode) {
    (void)e;
    ds4_tokens_push(tokens, FAKE_TOKEN_ASSISTANT);
    append_think_marker(tokens, think_mode);
}

static char* copy_bytes(const char* bytes, size_t len) {
    char* out = malloc(len + 1);
    if (!out)
        abort();
    memcpy(out, bytes, len);
    out[len] = '\0';
    return out;
}

char* ds4_token_text(ds4_engine* e, int token, size_t* len) {
    (void)e;
    if (env_should_fail("token_text") || token == 13) {
        if (len)
            *len = 0;
        return NULL;
    }
    if (token >= FAKE_TOKEN_BYTE_BASE && token < FAKE_TOKEN_BYTE_BASE + 256) {
        char value = (char)(token - FAKE_TOKEN_BYTE_BASE);
        if (len)
            *len = 1;
        return copy_bytes(&value, 1);
    }

    const char* text = "";
    switch (token) {
    case FAKE_TOKEN_BOS:
        text = "<BOS>";
        break;
    case FAKE_TOKEN_USER:
        text = "<User>";
        break;
    case FAKE_TOKEN_ASSISTANT:
        text = "<Assistant>";
        break;
    case FAKE_TOKEN_THINK_START:
        text = "<think>";
        break;
    case FAKE_TOKEN_THINK_END:
        text = "</think>";
        break;
    case FAKE_TOKEN_EOS:
        text = "<EOS>";
        break;
    default:
        text = "";
        break;
    }
    const size_t text_len = strlen(text);
    if (len)
        *len = text_len;
    return copy_bytes(text, text_len);
}

int ds4_token_eos(ds4_engine* e) {
    (void)e;
    return FAKE_TOKEN_EOS;
}

int ds4_session_create(ds4_session** out, ds4_engine* e, int ctx_size) {
    counters.last_session_create_sequence = next_call();
    counters.session_create_calls++;
    if (out)
        *out = NULL;
    if (!out || !e || ctx_size <= 0 ||
        should_fail(e->model_path, "ctx-fail")) {
        return 1;
    }
    if (env_should_fail("session_create"))
        return 1;
    ds4_session* session = calloc(1, sizeof(*session));
    if (!session)
        abort();
    session->engine = e;
    session->ctx_size = ctx_size;
    session->valid = true;
    *out = session;
    return 0;
}

void ds4_session_free(ds4_session* s) {
    if (!s)
        return;
    counters.last_session_free_sequence = next_call();
    counters.session_free_calls++;
    ds4_tokens_free(&s->checkpoint);
    free(s);
}

void ds4_session_set_progress(ds4_session* s, ds4_session_progress_fn fn,
                              void* ud) {
    if (!s)
        return;
    s->progress = fn;
    s->progress_ud = ud;
}

int ds4_session_sync(ds4_session* s, const ds4_tokens* prompt, char* err,
                     size_t errlen) {
    counters.last_sync_sequence = next_call();
    counters.sync_calls++;
    fake_delay_if_requested("sync");
    if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
        set_error(err, errlen, "prompt exceeds context");
        return 1;
    }
    if (env_should_fail("sync")) {
        set_error(err, errlen, "fake env sync failure");
        return 1;
    }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->valid = true;
    if (s->progress) {
        s->progress(s->progress_ud, "prefill_chunk", prompt->len, prompt->len);
    }
    return 0;
}

int ds4_session_argmax(ds4_session* s) {
    counters.last_argmax_sequence = next_call();
    counters.argmax_calls++;
    fake_delay_if_requested("argmax");
    if (env_should_fail("argmax"))
        return -1;
    if (!s || !s->valid)
        return -1;
    return FAKE_TOKEN_BYTE_BASE + 'A';
}

int ds4_session_argmax_excluding(ds4_session* s, int excluded_id) {
    counters.last_argmax_excluding_sequence = next_call();
    counters.argmax_excluding_calls++;
    fake_delay_if_requested("argmax_excluding");
    if (env_should_fail("argmax_excluding") || excluded_id == 13)
        return -1;
    if (!s || !s->valid)
        return -1;
    const int first = FAKE_TOKEN_BYTE_BASE + 'A';
    return excluded_id == first ? FAKE_TOKEN_BYTE_BASE + 'B' : first;
}

int ds4_session_sample(ds4_session* s, float temperature, int top_k,
                       float top_p, float min_p, uint64_t* rng) {
    (void)top_k;
    (void)top_p;
    (void)min_p;
    counters.last_sample_sequence = next_call();
    counters.sample_calls++;
    fake_delay_if_requested("sample");
    if (env_should_fail("sample"))
        return -1;
    if (!s || !s->valid)
        return -1;
    if (temperature <= 0.0f)
        return FAKE_TOKEN_BYTE_BASE + 'A';
    uint64_t value = rng && *rng ? *rng : 88172645463393265ull;
    value = value * 2862933555777941757ull + 3037000493ull;
    if (rng)
        *rng = value;
    return FAKE_TOKEN_BYTE_BASE + (int)('a' + (value % 26u));
}

int ds4_session_eval(ds4_session* s, int token, char* err, size_t errlen) {
    counters.last_eval_sequence = next_call();
    counters.eval_calls++;
    fake_delay_if_requested("eval");
    if (env_should_fail("eval") || token == 13) {
        set_error(err, errlen, "fake token eval failure");
        return 1;
    }
    if (!s || token < 0) {
        set_error(err, errlen, "invalid eval token");
        return 1;
    }
    if (s->checkpoint.len + 1 >= s->ctx_size) {
        set_error(err, errlen, "prompt exceeds context");
        return 1;
    }
    ds4_tokens_push(&s->checkpoint, token);
    s->valid = true;
    return 0;
}

void ds4_session_invalidate(ds4_session* s) {
    counters.last_invalidate_sequence = next_call();
    counters.invalidate_calls++;
    fake_delay_if_requested("invalidate");
    if (!s)
        return;
    s->valid = false;
    s->checkpoint.len = 0;
}

void ds4_session_rewind(ds4_session* s, int pos) {
    counters.last_rewind_sequence = next_call();
    counters.rewind_calls++;
    fake_delay_if_requested("rewind");
    if (!s || pos < 0)
        return;
    if (pos < s->checkpoint.len)
        s->checkpoint.len = pos;
}

int ds4_session_pos(ds4_session* s) {
    return s ? s->checkpoint.len : 0;
}

int ds4_session_ctx(ds4_session* s) {
    return s ? s->ctx_size : 0;
}

int ds4_engine_routed_quant_bits(ds4_engine* e) {
    return e && e->quality ? 4 : 2;
}

bool ds4_engine_has_mtp(ds4_engine* e) {
    return e && e->mtp_path && e->mtp_path[0];
}

int ds4_engine_mtp_draft_tokens(ds4_engine* e) {
    return ds4_engine_has_mtp(e) ? e->mtp_draft_tokens : 0;
}

const ds4_tokens* ds4_session_tokens(ds4_session* s) {
    return s ? &s->checkpoint : NULL;
}
