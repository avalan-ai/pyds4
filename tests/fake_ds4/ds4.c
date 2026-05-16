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
static const uint32_t FAKE_PAYLOAD_MAGIC = 0x34445350u; /* PDS4 */
static const uint32_t FAKE_PAYLOAD_VERSION = 1u;
static const uint64_t FAKE_PAYLOAD_FIXED_BYTES = 5u * sizeof(uint32_t);

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

static void write_le32(uint8_t* out, uint32_t value) {
    out[0] = (uint8_t)value;
    out[1] = (uint8_t)(value >> 8);
    out[2] = (uint8_t)(value >> 16);
    out[3] = (uint8_t)(value >> 24);
}

static uint32_t read_le32(const uint8_t* in) {
    return (uint32_t)in[0] | ((uint32_t)in[1] << 8) | ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

static int write_u32(FILE* fp, uint32_t value, char* err, size_t errlen) {
    uint8_t data[4];
    write_le32(data, value);
    if (fwrite(data, 1, sizeof(data), fp) != sizeof(data)) {
        set_error(err, errlen, "failed to write fake session payload");
        return 1;
    }
    return 0;
}

static int read_exact(FILE* fp, uint8_t* out, uint64_t bytes,
                      uint64_t* remaining, char* err, size_t errlen) {
    if (*remaining < bytes || bytes > (uint64_t)SIZE_MAX) {
        set_error(err, errlen, "truncated fake session payload");
        return 1;
    }
    if (bytes > 0 && fread(out, 1, (size_t)bytes, fp) != (size_t)bytes) {
        set_error(err, errlen, "failed to read fake session payload");
        return 1;
    }
    *remaining -= bytes;
    return 0;
}

static int read_u32(FILE* fp, uint32_t* out, uint64_t* remaining, char* err,
                    size_t errlen) {
    uint8_t data[4];
    if (read_exact(fp, data, sizeof(data), remaining, err, errlen) != 0)
        return 1;
    *out = read_le32(data);
    return 0;
}

static uint64_t fake_payload_bytes(ds4_session* s) {
    if (!s || !s->valid || !s->engine || !s->engine->model_path)
        return 0;
    const uint64_t model_len = (uint64_t)strlen(s->engine->model_path);
    return FAKE_PAYLOAD_FIXED_BYTES + model_len +
           (uint64_t)s->checkpoint.len * sizeof(uint32_t);
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

uint64_t ds4_session_payload_bytes(ds4_session* s) {
    counters.last_payload_bytes_sequence = next_call();
    counters.payload_bytes_calls++;
    return fake_payload_bytes(s);
}

int ds4_session_save_payload(ds4_session* s, FILE* fp, char* err,
                             size_t errlen) {
    counters.last_save_payload_sequence = next_call();
    counters.save_payload_calls++;
    fake_delay_if_requested("save_payload");
    if (!s || !fp || !s->valid || !s->engine || !s->engine->model_path) {
        set_error(err, errlen, "session has no valid checkpoint to save");
        return 1;
    }
    if (env_should_fail("save_payload")) {
        set_error(err, errlen, "fake env save payload failure");
        return 1;
    }

    const uint32_t model_len = (uint32_t)strlen(s->engine->model_path);
    if (write_u32(fp, FAKE_PAYLOAD_MAGIC, err, errlen) != 0 ||
        write_u32(fp, FAKE_PAYLOAD_VERSION, err, errlen) != 0 ||
        write_u32(fp, (uint32_t)s->ctx_size, err, errlen) != 0 ||
        write_u32(fp, (uint32_t)s->checkpoint.len, err, errlen) != 0 ||
        write_u32(fp, model_len, err, errlen) != 0) {
        return 1;
    }
    if (model_len > 0 &&
        fwrite(s->engine->model_path, 1, model_len, fp) != model_len) {
        set_error(err, errlen, "failed to write fake session payload");
        return 1;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        if (write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0)
            return 1;
    }
    return 0;
}

int ds4_session_load_payload(ds4_session* s, FILE* fp, uint64_t payload_bytes,
                             char* err, size_t errlen) {
    counters.last_load_payload_sequence = next_call();
    counters.load_payload_calls++;
    fake_delay_if_requested("load_payload");
    if (!s || !fp || !s->engine || !s->engine->model_path) {
        set_error(err, errlen, "invalid session payload load");
        return 1;
    }
    if (env_should_fail("load_payload")) {
        set_error(err, errlen, "fake env load payload failure");
        return 1;
    }

    uint64_t remaining = payload_bytes;
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t ctx_size = 0;
    uint32_t token_count = 0;
    uint32_t model_len = 0;
    if (read_u32(fp, &magic, &remaining, err, errlen) != 0 ||
        read_u32(fp, &version, &remaining, err, errlen) != 0 ||
        read_u32(fp, &ctx_size, &remaining, err, errlen) != 0 ||
        read_u32(fp, &token_count, &remaining, err, errlen) != 0 ||
        read_u32(fp, &model_len, &remaining, err, errlen) != 0) {
        return 1;
    }
    if (magic != FAKE_PAYLOAD_MAGIC || version != FAKE_PAYLOAD_VERSION) {
        set_error(err, errlen, "unsupported fake session payload version");
        return 1;
    }
    if (ctx_size != (uint32_t)s->ctx_size || token_count >= ctx_size) {
        set_error(
            err, errlen,
            "fake session payload context does not match current session");
        return 1;
    }
    const size_t current_model_len = strlen(s->engine->model_path);
    if ((uint64_t)model_len > remaining ||
        model_len != (uint32_t)current_model_len) {
        set_error(err, errlen,
                  "fake session payload model does not match current engine");
        return 1;
    }

    char* model = malloc((size_t)model_len + 1);
    if (!model)
        abort();
    if (read_exact(fp, (uint8_t*)model, model_len, &remaining, err, errlen) !=
        0) {
        free(model);
        return 1;
    }
    model[model_len] = '\0';
    if (strcmp(model, s->engine->model_path) != 0) {
        free(model);
        set_error(err, errlen,
                  "fake session payload model does not match current engine");
        return 1;
    }
    free(model);

    ds4_tokens next = {0};
    for (uint32_t i = 0; i < token_count; i++) {
        uint32_t token = 0;
        if (read_u32(fp, &token, &remaining, err, errlen) != 0) {
            ds4_tokens_free(&next);
            return 1;
        }
        if (token > (uint32_t)INT_MAX) {
            ds4_tokens_free(&next);
            set_error(err, errlen,
                      "fake session payload contains invalid token id");
            return 1;
        }
        ds4_tokens_push(&next, (int)token);
    }
    if (remaining != 0) {
        ds4_tokens_free(&next);
        set_error(err, errlen, "fake session payload has trailing bytes");
        return 1;
    }

    ds4_tokens_free(&s->checkpoint);
    s->checkpoint = next;
    s->valid = true;
    return 0;
}

int ds4_session_save_snapshot(ds4_session* s, ds4_session_snapshot* snap,
                              char* err, size_t errlen) {
    counters.last_save_snapshot_sequence = next_call();
    counters.save_snapshot_calls++;
    fake_delay_if_requested("save_snapshot");
    if (!s || !snap) {
        set_error(err, errlen, "invalid session snapshot save");
        return 1;
    }
    if (env_should_fail("save_snapshot")) {
        set_error(err, errlen, "fake env save snapshot failure");
        return 1;
    }
    const uint64_t bytes = fake_payload_bytes(s);
    if (bytes == 0 || bytes > (uint64_t)SIZE_MAX) {
        set_error(err, errlen, "session has no valid checkpoint to snapshot");
        return 1;
    }
    if (snap->cap < bytes) {
        uint8_t* next = realloc(snap->ptr, (size_t)bytes);
        if (!next) {
            set_error(err, errlen,
                      "out of memory while allocating fake session snapshot");
            return 1;
        }
        if (!snap->ptr) {
            counters.snapshot_allocation_calls++;
            counters.snapshot_live_allocations++;
            if (counters.snapshot_live_allocations >
                counters.snapshot_peak_live_allocations) {
                counters.snapshot_peak_live_allocations =
                    counters.snapshot_live_allocations;
            }
        }
        snap->ptr = next;
        snap->cap = bytes;
    }

    FILE* fp = tmpfile();
    if (!fp) {
        set_error(err, errlen, "failed to open fake snapshot memory file");
        return 1;
    }
    const int rc = ds4_session_save_payload(s, fp, err, errlen);
    if (rc == 0 && fflush(fp) != 0) {
        set_error(err, errlen, "failed to flush fake session snapshot");
        fclose(fp);
        return 1;
    }
    if (rc == 0 && fseek(fp, 0, SEEK_SET) != 0) {
        set_error(err, errlen, "failed to rewind fake session snapshot");
        fclose(fp);
        return 1;
    }
    if (rc == 0 && fread(snap->ptr, 1, (size_t)bytes, fp) != (size_t)bytes) {
        set_error(err, errlen, "failed to read fake session snapshot");
        fclose(fp);
        return 1;
    }
    fclose(fp);
    if (rc != 0)
        return 1;
    snap->len = bytes;
    return 0;
}

int ds4_session_load_snapshot(ds4_session* s, const ds4_session_snapshot* snap,
                              char* err, size_t errlen) {
    counters.last_load_snapshot_sequence = next_call();
    counters.load_snapshot_calls++;
    fake_delay_if_requested("load_snapshot");
    if (!s || !snap || !snap->ptr || snap->len == 0) {
        set_error(err, errlen, "invalid session snapshot load");
        return 1;
    }
    if (snap->len > (uint64_t)SIZE_MAX) {
        set_error(err, errlen,
                  "fake session snapshot is too large for this platform");
        return 1;
    }
    if (env_should_fail("load_snapshot")) {
        set_error(err, errlen, "fake env load snapshot failure");
        return 1;
    }

    FILE* fp = tmpfile();
    if (!fp) {
        set_error(err, errlen, "failed to open fake snapshot memory file");
        return 1;
    }
    if (fwrite(snap->ptr, 1, (size_t)snap->len, fp) != (size_t)snap->len ||
        fflush(fp) != 0 || fseek(fp, 0, SEEK_SET) != 0) {
        set_error(err, errlen, "failed to prepare fake session snapshot");
        fclose(fp);
        return 1;
    }
    const int rc = ds4_session_load_payload(s, fp, snap->len, err, errlen);
    fclose(fp);
    return rc;
}

void ds4_session_snapshot_free(ds4_session_snapshot* snap) {
    if (!snap)
        return;
    counters.last_snapshot_free_sequence = next_call();
    counters.snapshot_free_calls++;
    if (snap->ptr && counters.snapshot_live_allocations > 0)
        counters.snapshot_live_allocations--;
    free(snap->ptr);
    snap->ptr = NULL;
    snap->len = 0;
    snap->cap = 0;
}
