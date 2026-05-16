#ifndef PYDS4_FAKE_DS4_H
#define PYDS4_FAKE_DS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef enum {
    DS4_BACKEND_METAL,
    DS4_BACKEND_CUDA,
    DS4_BACKEND_CPU,
} ds4_backend;

typedef enum {
    DS4_THINK_NONE,
    DS4_THINK_HIGH,
    DS4_THINK_MAX,
} ds4_think_mode;

typedef enum {
    DS4_LOG_DEFAULT,
    DS4_LOG_PREFILL,
    DS4_LOG_GENERATION,
    DS4_LOG_KVCACHE,
    DS4_LOG_TOOL,
    DS4_LOG_WARNING,
    DS4_LOG_TIMING,
    DS4_LOG_OK,
    DS4_LOG_ERROR,
} ds4_log_type;

typedef struct {
    int* v;
    int len;
    int cap;
} ds4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_token_score;

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef void (*ds4_session_progress_fn)(void* ud, const char* event,
                                        int current, int total);

typedef struct {
    const char* model_path;
    const char* mtp_path;
    ds4_backend backend;
    int n_threads;
    int mtp_draft_tokens;
    float mtp_margin;
    const char* directional_steering_file;
    float directional_steering_attn;
    float directional_steering_ffn;
    bool warm_weights;
    bool quality;
} ds4_engine_options;

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ds4_context_memory;

typedef struct {
    uint8_t* ptr;
    uint64_t len;
    uint64_t cap;
} ds4_session_snapshot;

int ds4_engine_open(ds4_engine** out, const ds4_engine_options* opt);
void ds4_engine_close(ds4_engine* e);
const char* ds4_backend_name(ds4_backend backend);
bool ds4_think_mode_enabled(ds4_think_mode mode);
const char* ds4_think_mode_name(ds4_think_mode mode);
const char* ds4_think_max_prefix(void);
uint32_t ds4_think_max_min_context(void);
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend,
                                               int ctx_size);

void ds4_tokens_push(ds4_tokens* tv, int token);
void ds4_tokens_free(ds4_tokens* tv);
void ds4_tokens_copy(ds4_tokens* dst, const ds4_tokens* src);
bool ds4_tokens_starts_with(const ds4_tokens* tokens,
                            const ds4_tokens* prefix);

void ds4_tokenize_text(ds4_engine* e, const char* text, ds4_tokens* out);
void ds4_tokenize_rendered_chat(ds4_engine* e, const char* text,
                                ds4_tokens* out);
void ds4_chat_begin(ds4_engine* e, ds4_tokens* tokens);
void ds4_encode_chat_prompt(ds4_engine* e, const char* system,
                            const char* prompt, ds4_think_mode think_mode,
                            ds4_tokens* out);
void ds4_chat_append_message(ds4_engine* e, ds4_tokens* tokens,
                             const char* role, const char* content);
void ds4_chat_append_assistant_prefix(ds4_engine* e, ds4_tokens* tokens,
                                      ds4_think_mode think_mode);

char* ds4_token_text(ds4_engine* e, int token, size_t* len);
int ds4_token_eos(ds4_engine* e);

int ds4_session_create(ds4_session** out, ds4_engine* e, int ctx_size);
void ds4_session_free(ds4_session* s);
void ds4_session_set_progress(ds4_session* s, ds4_session_progress_fn fn,
                              void* ud);
int ds4_session_sync(ds4_session* s, const ds4_tokens* prompt, char* err,
                     size_t errlen);
int ds4_session_argmax(ds4_session* s);
int ds4_session_argmax_excluding(ds4_session* s, int excluded_id);
int ds4_session_sample(ds4_session* s, float temperature, int top_k,
                       float top_p, float min_p, uint64_t* rng);
int ds4_session_eval(ds4_session* s, int token, char* err, size_t errlen);
void ds4_session_invalidate(ds4_session* s);
void ds4_session_rewind(ds4_session* s, int pos);
int ds4_session_pos(ds4_session* s);
int ds4_session_ctx(ds4_session* s);
int ds4_engine_routed_quant_bits(ds4_engine* e);
bool ds4_engine_has_mtp(ds4_engine* e);
int ds4_engine_mtp_draft_tokens(ds4_engine* e);
const ds4_tokens* ds4_session_tokens(ds4_session* s);
uint64_t ds4_session_payload_bytes(ds4_session* s);
int ds4_session_save_payload(ds4_session* s, FILE* fp, char* err,
                             size_t errlen);
int ds4_session_load_payload(ds4_session* s, FILE* fp, uint64_t payload_bytes,
                             char* err, size_t errlen);
int ds4_session_save_snapshot(ds4_session* s, ds4_session_snapshot* snap,
                              char* err, size_t errlen);
int ds4_session_load_snapshot(ds4_session* s, const ds4_session_snapshot* snap,
                              char* err, size_t errlen);
void ds4_session_snapshot_free(ds4_session_snapshot* snap);

typedef struct {
    uint64_t engine_open_calls;
    uint64_t engine_close_calls;
    uint64_t session_create_calls;
    uint64_t session_free_calls;
    uint64_t tokens_free_calls;
    uint64_t sync_calls;
    uint64_t eval_calls;
    uint64_t argmax_calls;
    uint64_t argmax_excluding_calls;
    uint64_t sample_calls;
    uint64_t rewind_calls;
    uint64_t invalidate_calls;
    uint64_t payload_bytes_calls;
    uint64_t save_payload_calls;
    uint64_t load_payload_calls;
    uint64_t save_snapshot_calls;
    uint64_t load_snapshot_calls;
    uint64_t snapshot_free_calls;
    uint64_t token_allocation_calls;
    uint64_t token_live_allocations;
    uint64_t token_peak_live_allocations;
    uint64_t snapshot_allocation_calls;
    uint64_t snapshot_live_allocations;
    uint64_t snapshot_peak_live_allocations;
    uint64_t call_sequence;
    uint64_t last_engine_open_sequence;
    uint64_t last_engine_close_sequence;
    uint64_t last_session_create_sequence;
    uint64_t last_session_free_sequence;
    uint64_t last_tokens_free_sequence;
    uint64_t last_sync_sequence;
    uint64_t last_eval_sequence;
    uint64_t last_argmax_sequence;
    uint64_t last_argmax_excluding_sequence;
    uint64_t last_sample_sequence;
    uint64_t last_rewind_sequence;
    uint64_t last_invalidate_sequence;
    uint64_t last_payload_bytes_sequence;
    uint64_t last_save_payload_sequence;
    uint64_t last_load_payload_sequence;
    uint64_t last_save_snapshot_sequence;
    uint64_t last_load_snapshot_sequence;
    uint64_t last_snapshot_free_sequence;
} pyds4_fake_counters;

void pyds4_fake_reset_counters(void);
pyds4_fake_counters pyds4_fake_get_counters(void);

#endif
