#include "pyds4/_native_common.hpp"
#include "pyds4/_native_tokens.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class TestFailure : public std::runtime_error {
  public:
    TestFailure(const char* file, int line, const char* expression)
        : std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                             ": check failed: " + expression) {}
};

#define CHECK(expression)                                                     \
    do {                                                                      \
        if (!(expression)) {                                                  \
            throw TestFailure(__FILE__, __LINE__, #expression);               \
        }                                                                     \
    } while (false)

struct ProgressCapture {
    int calls = 0;
    std::string event;
    int current = 0;
    int total = 0;
};

void capture_progress(void* userdata, const char* event, int current,
                      int total) {
    auto* capture = static_cast<ProgressCapture*>(userdata);
    capture->calls += 1;
    capture->event = event == nullptr ? "" : event;
    capture->current = current;
    capture->total = total;
}

ds4_engine* open_fake_engine() {
    ds4_engine_options options = {};
    options.model_path = "model.gguf";
    options.mtp_path = "mtp.gguf";
    options.backend = DS4_BACKEND_CPU;
    options.mtp_draft_tokens = 7;
    options.quality = true;

    ds4_engine* engine = nullptr;
    CHECK(ds4_engine_open(&engine, &options) == 0);
    CHECK(engine != nullptr);
    return engine;
}

void close_engine(ds4_engine* engine) {
    ds4_engine_close(engine);
}

std::vector<int> token_values(const ds4_tokens* tokens) {
    CHECK(tokens != nullptr);
    std::vector<int> values;
    values.reserve(static_cast<std::size_t>(tokens->len));
    for (int i = 0; i < tokens->len; ++i) {
        values.push_back(tokens->v[i]);
    }
    return values;
}

void test_native_common_helpers() {
    using pyds4::native::is_known_backend;
    using pyds4::native::is_known_think_mode;
    using pyds4::native::normalized;
    using pyds4::native::selected_backend_is_available;
    using pyds4::native::unavailable_reason;

    CHECK(normalized("MeTaL") == "metal");
    CHECK(is_known_backend("metal"));
    CHECK(is_known_backend("cuda"));
    CHECK(is_known_backend("cpu"));
    CHECK(!is_known_backend("tpu"));
    CHECK(is_known_think_mode("none"));
    CHECK(is_known_think_mode("high"));
    CHECK(is_known_think_mode("max"));
    CHECK(!is_known_think_mode("medium"));
    CHECK(selected_backend_is_available(true, true));
    CHECK(!selected_backend_is_available(false, true));
    CHECK(!selected_backend_is_available(true, false));

    CHECK(unavailable_reason("cpu", "cpu", true, true).empty());
    CHECK(unavailable_reason("tpu", "cpu", true, true)
              .find("Unsupported DS4 native backend") != std::string::npos);
    CHECK(unavailable_reason("cpu", "cpu", false, true)
              .find("does not include DS4 native source") !=
          std::string::npos);
    CHECK(unavailable_reason("cuda", "cpu", true, true)
              .find("configured for 'cpu'") != std::string::npos);
}

void test_native_tokens_raii_frees_buffers() {
    pyds4_fake_reset_counters();

    {
        const pyds4::native::NativeTokens tokens({1, 2, 3});
        CHECK(token_values(tokens.get()) == std::vector<int>({1, 2, 3}));

        const pyds4_fake_counters counters = pyds4_fake_get_counters();
        CHECK(counters.token_allocation_calls == 1);
        CHECK(counters.token_live_allocations == 1);
        CHECK(counters.token_peak_live_allocations == 1);
    }

    const pyds4_fake_counters counters = pyds4_fake_get_counters();
    CHECK(counters.tokens_free_calls == 1);
    CHECK(counters.token_live_allocations == 0);
}

void test_fake_engine_session_lifecycle() {
    pyds4_fake_reset_counters();

    ds4_engine* engine = open_fake_engine();
    CHECK(ds4_engine_routed_quant_bits(engine) == 4);
    CHECK(ds4_engine_has_mtp(engine));
    CHECK(ds4_engine_mtp_draft_tokens(engine) == 7);
    CHECK(ds4_token_eos(engine) == 6);

    ds4_session* session = nullptr;
    CHECK(ds4_session_create(&session, engine, 8) == 0);
    CHECK(session != nullptr);

    ProgressCapture progress;
    ds4_session_set_progress(session, capture_progress, &progress);

    {
        const pyds4::native::NativeTokens prompt({1, 2});
        char error[128] = {};
        CHECK(ds4_session_sync(session, prompt.get(), error, sizeof(error)) ==
              0);
    }

    CHECK(progress.calls == 1);
    CHECK(progress.event == "prefill_chunk");
    CHECK(progress.current == 2);
    CHECK(progress.total == 2);
    CHECK(ds4_session_pos(session) == 2);
    CHECK(token_values(ds4_session_tokens(session)) ==
          std::vector<int>({1, 2}));

    constexpr int first_generated_token = 1000 + 'A';
    constexpr int second_generated_token = 1000 + 'B';
    CHECK(ds4_session_argmax(session) == first_generated_token);
    CHECK(ds4_session_argmax_excluding(session, first_generated_token) ==
          second_generated_token);

    uint64_t rng = 123;
    const int sampled =
        ds4_session_sample(session, 0.7F, 40, 0.95F, 0.05F, &rng);
    CHECK(sampled >= 1000 + 'a');
    CHECK(sampled <= 1000 + 'z');
    CHECK(rng != 123);

    ds4_token_score scores[3] = {};
    CHECK(ds4_session_top_logprobs(session, scores, 3) == 3);
    CHECK(scores[0].id == first_generated_token);
    CHECK(scores[0].logprob == -0.25F);
    CHECK(scores[1].id == second_generated_token);
    CHECK(scores[1].logprob == -1.25F);

    ds4_token_score sampled_score = {};
    CHECK(ds4_session_token_logprob(session, first_generated_token,
                                    &sampled_score) == 1);
    CHECK(sampled_score.id == first_generated_token);
    CHECK(sampled_score.logprob == -0.25F);

    char error[128] = {};
    CHECK(ds4_session_eval(session, sampled, error, sizeof(error)) == 0);
    CHECK(ds4_session_pos(session) == 3);

    ds4_session_rewind(session, 2);
    CHECK(ds4_session_pos(session) == 2);
    CHECK(token_values(ds4_session_tokens(session)) ==
          std::vector<int>({1, 2}));

    const uint64_t payload_bytes = ds4_session_payload_bytes(session);
    CHECK(payload_bytes > 0);

    ds4_session_snapshot snapshot = {};
    CHECK(ds4_session_save_snapshot(session, &snapshot, error,
                                    sizeof(error)) == 0);
    CHECK(snapshot.ptr != nullptr);
    CHECK(snapshot.len == payload_bytes);

    CHECK(ds4_session_eval(session, first_generated_token, error,
                           sizeof(error)) == 0);
    CHECK(ds4_session_pos(session) == 3);
    CHECK(ds4_session_load_snapshot(session, &snapshot, error,
                                    sizeof(error)) == 0);
    CHECK(ds4_session_pos(session) == 2);
    CHECK(token_values(ds4_session_tokens(session)) ==
          std::vector<int>({1, 2}));
    ds4_session_snapshot_free(&snapshot);

    ds4_session_invalidate(session);
    CHECK(ds4_session_pos(session) == 0);
    CHECK(ds4_session_argmax(session) == -1);

    ds4_session_free(session);
    close_engine(engine);

    const pyds4_fake_counters counters = pyds4_fake_get_counters();
    CHECK(counters.engine_open_calls == 1);
    CHECK(counters.engine_close_calls == 1);
    CHECK(counters.session_create_calls == 1);
    CHECK(counters.session_free_calls == 1);
    CHECK(counters.sync_calls == 1);
    CHECK(counters.eval_calls == 2);
    CHECK(counters.top_logprobs_calls == 1);
    CHECK(counters.token_logprob_calls == 1);
    CHECK(counters.rewind_calls == 1);
    CHECK(counters.invalidate_calls == 1);
    CHECK(counters.token_live_allocations == 0);
    CHECK(counters.snapshot_live_allocations == 0);
}

void test_fake_tokenization_and_chat_helpers() {
    pyds4_fake_reset_counters();
    ds4_engine* engine = open_fake_engine();

    {
        pyds4::native::NativeTokens text;
        ds4_tokenize_text(engine, "Az", text.mutable_get());
        CHECK(token_values(text.get()) ==
              std::vector<int>({1000 + 'A', 1000 + 'z'}));
    }

    {
        pyds4::native::NativeTokens chat;
        ds4_chat_begin(engine, chat.mutable_get());
        ds4_chat_append_message(engine, chat.mutable_get(), "user", "Q");
        ds4_chat_append_assistant_prefix(engine, chat.mutable_get(),
                                         DS4_THINK_HIGH);
        CHECK(token_values(chat.get()) ==
              std::vector<int>({1, 2, 1000 + 'Q', 3, 4}));
    }

    close_engine(engine);
    CHECK(pyds4_fake_get_counters().token_live_allocations == 0);
}

using TestFn = void (*)();

struct TestCase {
    std::string_view name;
    TestFn function;
};

} // namespace

int main() {
    const TestCase tests[] = {
        {"native common helpers", test_native_common_helpers},
        {"native token RAII", test_native_tokens_raii_frees_buffers},
        {"fake engine/session lifecycle", test_fake_engine_session_lifecycle},
        {"fake tokenization/chat helpers",
         test_fake_tokenization_and_chat_helpers},
    };

    int failures = 0;
    for (const TestCase& test : tests) {
        try {
            test.function();
            std::cout << "[ OK ] " << test.name << '\n';
        } catch (const std::exception& error) {
            failures += 1;
            std::cerr << "[FAIL] " << test.name << ": " << error.what()
                      << '\n';
        } catch (...) {
            failures += 1;
            std::cerr << "[FAIL] " << test.name << ": unknown exception\n";
        }
    }

    return failures == 0 ? 0 : 1;
}
