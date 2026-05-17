#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "_native_common.hpp"

#if PYDS4_DS4_SOURCE_PRESENT
#include "_native_tokens.hpp"
#endif

#ifndef PYDS4_VERSION
#define PYDS4_VERSION "1.0.1"
#endif

#ifndef PYDS4_DS4_COMMIT
#define PYDS4_DS4_COMMIT "8809b90a1e3247389d7652b565ab6772e036f1ea"
#endif

#ifndef PYDS4_NATIVE_BACKEND
#define PYDS4_NATIVE_BACKEND "cpu"
#endif

#ifndef PYDS4_THINK_MAX_MIN_CONTEXT
#define PYDS4_THINK_MAX_MIN_CONTEXT 393216
#endif

#ifndef PYDS4_DS4_SOURCE_PRESENT
#define PYDS4_DS4_SOURCE_PRESENT 0
#endif

#ifndef PYDS4_HOST_SUPPORTS_SELECTED_BACKEND
#define PYDS4_HOST_SUPPORTS_SELECTED_BACKEND 1
#endif

#ifndef PYDS4_FAKE_NATIVE
#define PYDS4_FAKE_NATIVE 0
#endif

namespace py = pybind11;

namespace {

using pyds4::native::is_known_backend;
using pyds4::native::is_known_think_mode;
using pyds4::native::normalized;

#if PYDS4_DS4_SOURCE_PRESENT
using pyds4::native::NativeTokens;

struct ProgressEvent {
    std::string event;
    int current;
    int total;
};

struct FileCloser {
    void operator()(std::FILE* file) const {
        if (file != nullptr) {
            std::fclose(file);
        }
    }
};

using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

class NativeSnapshot {
  public:
    NativeSnapshot() = default;
    NativeSnapshot(const NativeSnapshot&) = delete;
    NativeSnapshot& operator=(const NativeSnapshot&) = delete;

    ~NativeSnapshot() {
        ds4_session_snapshot_free(&snapshot_);
    }

    ds4_session_snapshot* get() {
        return &snapshot_;
    }

    const ds4_session_snapshot* get() const {
        return &snapshot_;
    }

  private:
    ds4_session_snapshot snapshot_ = {};
};

FilePtr open_temporary_file(const std::string& operation) {
    std::FILE* file = std::tmpfile();
    if (file == nullptr) {
        throw std::runtime_error(operation +
                                 " failed: could not open a temporary file.");
    }
    return FilePtr(file);
}

void write_all(std::FILE* file, const char* data, std::size_t size,
               const std::string& operation) {
    if (size == 0) {
        return;
    }
    if (std::fwrite(data, 1, size, file) != size) {
        throw std::runtime_error(operation +
                                 " failed: could not write temporary data.");
    }
}

void rewind_after_write(std::FILE* file, const std::string& operation) {
    if (std::fflush(file) != 0 || std::fseek(file, 0, SEEK_SET) != 0) {
        throw std::runtime_error(operation +
                                 " failed: could not rewind temporary data.");
    }
}

std::vector<char> read_payload_bytes(std::FILE* file, uint64_t byte_count,
                                     const std::string& operation) {
    if (byte_count >
        static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(operation +
                                 " failed: payload is too large to copy.");
    }

    std::vector<char> data(static_cast<std::size_t>(byte_count));
    if (!data.empty() &&
        std::fread(data.data(), 1, data.size(), file) != data.size()) {
        throw std::runtime_error(operation +
                                 " failed: could not read temporary data.");
    }
    return data;
}

py::bytes bytes_from_vector(const std::vector<char>& data) {
    return py::bytes(data.empty() ? "" : data.data(), data.size());
}

ds4_backend to_native_backend(std::string backend) {
    backend = normalized(std::move(backend));
    if (backend == "metal") {
        return DS4_BACKEND_METAL;
    }
    if (backend == "cuda") {
        return DS4_BACKEND_CUDA;
    }
    if (backend == "cpu") {
        return DS4_BACKEND_CPU;
    }
    throw py::value_error("Unsupported DS4 native backend '" + backend + "'.");
}
#endif

bool selected_backend_is_available() {
    return pyds4::native::selected_backend_is_available(
        static_cast<bool>(PYDS4_DS4_SOURCE_PRESENT),
        static_cast<bool>(PYDS4_HOST_SUPPORTS_SELECTED_BACKEND));
}

std::string unavailable_reason(const std::string& backend) {
    return pyds4::native::unavailable_reason(
        backend, PYDS4_NATIVE_BACKEND,
        static_cast<bool>(PYDS4_DS4_SOURCE_PRESENT),
        static_cast<bool>(PYDS4_HOST_SUPPORTS_SELECTED_BACKEND));
}

#if PYDS4_DS4_SOURCE_PRESENT
ds4_think_mode to_native_think_mode(const std::string& mode) {
    if (mode == "none") {
        return DS4_THINK_NONE;
    }
    if (mode == "high") {
        return DS4_THINK_HIGH;
    }
    if (mode == "max") {
        return DS4_THINK_MAX;
    }
    throw py::value_error("Unsupported DS4 think mode '" + mode + "'.");
}

int py_token_id(py::handle value, const std::string& name) {
    if (py::isinstance<py::bool_>(value) || !py::isinstance<py::int_>(value)) {
        throw py::type_error(name + " must be an integer token id.");
    }

    int overflow = 0;
    const long long token =
        PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw py::value_error(name +
                              " is outside the supported token id range.");
    }
    if (overflow != 0 || token > INT_MAX) {
        throw py::value_error(name +
                              " is outside the supported token id range.");
    }
    if (token < 0) {
        throw py::value_error(name + " must be non-negative.");
    }
    return static_cast<int>(token);
}

int py_non_negative_int(py::handle value, const std::string& name) {
    if (py::isinstance<py::bool_>(value) || !py::isinstance<py::int_>(value)) {
        throw py::type_error(name + " must be a non-negative integer.");
    }

    int overflow = 0;
    const long long number =
        PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw py::value_error(name +
                              " is outside the supported integer range.");
    }
    if (overflow != 0 || number > INT_MAX) {
        throw py::value_error(name +
                              " is outside the supported integer range.");
    }
    if (number < 0) {
        throw py::value_error(name + " must be non-negative.");
    }
    return static_cast<int>(number);
}

int py_positive_int(py::handle value, const std::string& name) {
    if (py::isinstance<py::bool_>(value) || !py::isinstance<py::int_>(value)) {
        throw py::type_error(name + " must be a positive integer.");
    }

    int overflow = 0;
    const long long number =
        PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw py::value_error(name +
                              " is outside the supported integer range.");
    }
    if (overflow != 0 || number > INT_MAX) {
        throw py::value_error(name +
                              " is outside the supported integer range.");
    }
    if (number <= 0) {
        throw py::value_error(name + " must be positive.");
    }
    return static_cast<int>(number);
}

uint64_t py_seed(py::handle value) {
    if (value.is_none()) {
        return 0;
    }
    if (py::isinstance<py::bool_>(value) || !py::isinstance<py::int_>(value)) {
        throw py::type_error("seed must be an integer or None.");
    }

    const unsigned long long seed = PyLong_AsUnsignedLongLong(value.ptr());
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw py::value_error("seed is outside the supported integer range.");
    }
    return static_cast<uint64_t>(seed);
}

std::vector<int> validate_token_sequence(py::handle value,
                                         const std::string& name) {
    if (!py::isinstance<py::list>(value) &&
        !py::isinstance<py::tuple>(value)) {
        throw py::type_error(name + " must be a list or tuple of token ids.");
    }

    py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
    std::vector<int> tokens;
    tokens.reserve(static_cast<std::size_t>(py::len(sequence)));
    for (py::handle item : sequence) {
        tokens.push_back(py_token_id(item, name));
    }
    return tokens;
}

py::list tokens_to_list(const ds4_tokens* tokens) {
    if (tokens == nullptr) {
        throw std::runtime_error("DS4 returned a null token buffer.");
    }

    py::list result;
    for (int i = 0; i < tokens->len; ++i) {
        result.append(tokens->v[i]);
    }
    return result;
}

void replace_py_list_from_tokens(py::list target, const ds4_tokens* tokens) {
    if (tokens == nullptr) {
        throw std::runtime_error("DS4 returned a null token buffer.");
    }
    target.attr("clear")();
    for (int i = 0; i < tokens->len; ++i) {
        target.append(tokens->v[i]);
    }
}

class SessionState;

class EngineState : public std::enable_shared_from_this<EngineState> {
  public:
    EngineState(const std::string& model_path, const std::string& backend,
                const std::optional<std::string>& mtp_path, int n_threads,
                int mtp_draft_tokens, float mtp_margin,
                const std::optional<std::string>& directional_steering_file,
                float directional_steering_attn,
                float directional_steering_ffn, bool warm_weights,
                bool quality);

    EngineState(const EngineState&) = delete;
    EngineState& operator=(const EngineState&) = delete;

    ~EngineState();

    void close();
    bool closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return engine_ == nullptr;
    }

    std::shared_ptr<SessionState> create_session(int ctx_size);
    int routed_quant_bits() const;
    bool has_mtp() const;
    int mtp_draft_tokens() const;
    int eos_token_id() const;
    py::bytes token_text(py::handle token_id);
    py::list tokenize_text(const std::string& text);
    py::list tokenize_rendered_chat(const std::string& text);
    py::list chat_begin();
    void chat_append_message(py::list tokens, const std::string& role,
                             const std::string& content);
    void chat_append_assistant_prefix(py::list tokens, std::string think_mode);
    py::list encode_chat_prompt(const std::optional<std::string>& system,
                                const std::string& prompt,
                                std::string think_mode);

    ds4_engine* get_locked() const {
        if (engine_ == nullptr) {
            throw std::runtime_error("DS4 engine is closed.");
        }
        return engine_;
    }

  private:
    ds4_engine* engine_ = nullptr;
    std::vector<std::weak_ptr<SessionState>> sessions_;
    mutable std::mutex mutex_;
};

class SessionState : public std::enable_shared_from_this<SessionState> {
  public:
    SessionState(std::shared_ptr<EngineState> engine_state,
                 ds4_session* session, bool speculative_eval_supported)
        : engine_state_(std::move(engine_state)), session_(session),
          speculative_eval_supported_(speculative_eval_supported) {
        ds4_session_set_progress(session_, &SessionState::progress_callback,
                                 this);
    }

    SessionState(const SessionState&) = delete;
    SessionState& operator=(const SessionState&) = delete;

    ~SessionState() {
        close();
    }

    void close() {
        ds4_session* session = nullptr;
        {
            py::gil_scoped_release release;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                session = std::exchange(session_, nullptr);
                logits_ready_ = false;
                rng_state_.reset();
            }
            {
                std::lock_guard<std::mutex> progress_lock(progress_mutex_);
                progress_wakeup_fd_ = -1;
                progress_events_.clear();
            }
            if (session != nullptr) {
                ds4_session_set_progress(session, nullptr, nullptr);
                ds4_session_free(session);
            }
        }
        engine_state_.reset();
    }

    bool closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return session_ == nullptr;
    }

    int pos() const {
        std::lock_guard<std::mutex> lock(mutex_);
        ds4_session* session = get_locked();
        return ds4_session_pos(session);
    }

    int ctx() const {
        std::lock_guard<std::mutex> lock(mutex_);
        ds4_session* session = get_locked();
        return ds4_session_ctx(session);
    }

    py::list tokens() const {
        std::lock_guard<std::mutex> lock(mutex_);
        ds4_session* session = get_locked();
        return tokens_to_list(ds4_session_tokens(session));
    }

    void sync(py::handle prompt_tokens) {
        const std::vector<int> tokens =
            validate_token_sequence(prompt_tokens, "prompt_tokens");
        NativeTokens native_tokens(tokens);

        std::array<char, 4096> error = {};
        int result = 0;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ds4_session* session = get_locked();
            result = ds4_session_sync(session, native_tokens.get(),
                                      error.data(), error.size());
            if (result != 0) {
                logits_ready_ = false;
            } else {
                rng_state_.reset();
                logits_ready_ = true;
            }
        }
        if (result != 0) {
            std::string message = error.data();
            if (message.empty())
                message = "ds4_session_sync failed.";
            throw std::runtime_error("sync failed: " + message);
        }
    }

    void eval(py::handle token_id) {
        const int token = py_token_id(token_id, "token_id");
        std::array<char, 4096> error = {};
        int result = 0;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ds4_session* session = get_locked();
            const int current_pos = ds4_session_pos(session);
            const int ctx_size = ds4_session_ctx(session);
            if (ctx_size <= 0 || current_pos >= ctx_size - 1) {
                throw std::runtime_error(
                    "eval failed: prompt exceeds context");
            }
            result =
                ds4_session_eval(session, token, error.data(), error.size());
            if (result != 0) {
                logits_ready_ = false;
            } else {
                logits_ready_ = true;
            }
        }
        if (result != 0) {
            std::string message = error.data();
            if (message.empty())
                message = "ds4_session_eval failed.";
            throw std::runtime_error("eval failed: " + message);
        }
    }

    int argmax() {
        int token = -1;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ds4_session* session = get_locked();
            ensure_logits_ready("argmax");
            token = ds4_session_argmax(session);
        }
        if (token < 0) {
            throw std::runtime_error(
                "argmax failed: DS4 returned invalid token id.");
        }
        return token;
    }

    int argmax_excluding(py::handle token_id) {
        const int excluded_id = py_token_id(token_id, "token_id");

        int token = -1;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ds4_session* session = get_locked();
            ensure_logits_ready("argmax_excluding");
            token = ds4_session_argmax_excluding(session, excluded_id);
        }
        if (token < 0) {
            throw std::runtime_error(
                "argmax_excluding failed: DS4 returned invalid token id.");
        }
        return token;
    }

    int sample(double temperature, int top_k, double top_p, double min_p,
               py::handle seed) {
        const uint64_t seed_value = py_seed(seed);
        uint64_t rng = 0;

        int token = -1;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ds4_session* session = get_locked();
            ensure_logits_ready("sample");
            rng = rng_state_.has_value() ? *rng_state_ : seed_value;
            const uint64_t call_initial_rng = rng;
            token = ds4_session_sample(
                session, static_cast<float>(temperature), top_k,
                static_cast<float>(top_p), static_cast<float>(min_p), &rng);
            if (token >= 0 && rng != call_initial_rng) {
                rng_state_ = rng;
            }
        }
        if (token < 0) {
            throw std::runtime_error(
                "sample failed: DS4 returned invalid token id.");
        }
        return token;
    }

    double token_logprob(py::handle token_id) {
        const int token = py_token_id(token_id, "token_id");

        ds4_token_score score = {};
        int result = 0;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ds4_session* session = get_locked();
            ensure_logits_ready("token_logprob");
            result = ds4_session_token_logprob(session, token, &score);
        }
        if (result != 1) {
            throw std::runtime_error(
                "token_logprob failed: DS4 returned no token score.");
        }
        if (score.id != token || !std::isfinite(score.logprob)) {
            throw std::runtime_error(
                "token_logprob failed: DS4 returned malformed token score.");
        }
        return static_cast<double>(score.logprob);
    }

    py::list top_logprobs(py::handle k_value) {
        const int k = py_non_negative_int(k_value, "k");
        py::list result;
        if (k == 0) {
            return result;
        }

        std::vector<ds4_token_score> scores(static_cast<std::size_t>(k));
        int count = 0;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ds4_session* session = get_locked();
            ensure_logits_ready("top_logprobs");
            count = ds4_session_top_logprobs(session, scores.data(), k);
        }
        if (count <= 0 || count > k) {
            throw std::runtime_error(
                "top_logprobs failed: DS4 returned invalid score count.");
        }

        for (int i = 0; i < count; ++i) {
            const ds4_token_score& score = scores[static_cast<std::size_t>(i)];
            if (score.id < 0 || !std::isfinite(score.logprob)) {
                throw std::runtime_error(
                    "top_logprobs failed: DS4 returned malformed token "
                    "score.");
            }
            result.append(
                py::make_tuple(score.id, static_cast<double>(score.logprob)));
        }
        return result;
    }

    py::list eval_speculative_argmax(py::handle first_token_id,
                                     py::handle max_tokens_value,
                                     py::handle eos_token_id) {
        const int first_token = py_token_id(first_token_id, "first_token");
        const int max_tokens = py_positive_int(max_tokens_value, "max_tokens");
        const int eos_token = py_token_id(eos_token_id, "eos_token_id");

        std::array<char, 4096> error = {};
        std::vector<int> accepted;
        int count = 0;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ds4_session* session = get_locked();
            if (!speculative_eval_supported_) {
                throw std::runtime_error(
                    "eval_speculative_argmax failed: DS4 speculative eval "
                    "requires an engine opened with MTP draft tokens > 1.");
            }

            const int current_pos = ds4_session_pos(session);
            const int ctx_size = ds4_session_ctx(session);
            if (ctx_size <= 0 || current_pos >= ctx_size - 1) {
                throw std::runtime_error(
                    "eval_speculative_argmax failed: prompt exceeds context");
            }

            const int remaining = ctx_size - current_pos - 1;
            const int accepted_cap = std::min(max_tokens, remaining);
            accepted.resize(static_cast<std::size_t>(accepted_cap));
            count = ds4_session_eval_speculative_argmax(
                session, first_token, accepted_cap, eos_token, accepted.data(),
                accepted_cap, error.data(), error.size());
            if (count < 0) {
                logits_ready_ = false;
            } else {
                logits_ready_ = true;
            }
        }
        if (count < 0) {
            std::string message = error.data();
            if (message.empty())
                message = "ds4_session_eval_speculative_argmax failed.";
            throw std::runtime_error("eval_speculative_argmax failed: " +
                                     message);
        }
        if (count <= 0 || count > static_cast<int>(accepted.size())) {
            throw std::runtime_error(
                "eval_speculative_argmax failed: DS4 returned invalid "
                "accepted token count.");
        }

        py::list result;
        for (int i = 0; i < count; ++i) {
            const int token = accepted[static_cast<std::size_t>(i)];
            if (token < 0) {
                throw std::runtime_error(
                    "eval_speculative_argmax failed: DS4 returned invalid "
                    "token id.");
            }
            result.append(token);
        }
        return result;
    }

    void rewind(py::handle pos_value) {
        const int rewind_pos = py_non_negative_int(pos_value, "pos");
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ds4_session* session = get_locked();
            const int previous_pos = ds4_session_pos(session);
            ds4_session_rewind(session, rewind_pos);
            const int current_pos = ds4_session_pos(session);
            if (current_pos != previous_pos) {
                logits_ready_ = false;
                rng_state_.reset();
            }
        }
    }

    py::bytes save_snapshot() {
        std::vector<char> data;
        std::array<char, 4096> error = {};
        int result = 0;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ds4_session* session = get_locked();
            NativeSnapshot snapshot;
            result = ds4_session_save_snapshot(session, snapshot.get(),
                                               error.data(), error.size());
            if (result == 0) {
                const ds4_session_snapshot* raw_snapshot = snapshot.get();
                if (raw_snapshot->ptr == nullptr || raw_snapshot->len == 0) {
                    throw std::runtime_error(
                        "save_snapshot failed: DS4 returned an empty "
                        "snapshot.");
                }
                if (raw_snapshot->len >
                    static_cast<uint64_t>(
                        std::numeric_limits<std::size_t>::max())) {
                    throw std::runtime_error(
                        "save_snapshot failed: snapshot is too large to "
                        "copy.");
                }
                const auto* begin =
                    reinterpret_cast<const char*>(raw_snapshot->ptr);
                data.assign(begin, begin + static_cast<std::size_t>(
                                               raw_snapshot->len));
            }
        }
        if (result != 0) {
            std::string message = error.data();
            if (message.empty())
                message = "ds4_session_save_snapshot failed.";
            throw std::runtime_error("save_snapshot failed: " + message);
        }
        return bytes_from_vector(data);
    }

    void load_snapshot(py::bytes snapshot_bytes) {
        const std::string data = py::cast<std::string>(snapshot_bytes);
        std::array<char, 4096> error = {};
        int result = 0;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ds4_session* session = get_locked();
            ds4_session_snapshot snapshot = {};
            snapshot.ptr =
                reinterpret_cast<uint8_t*>(const_cast<char*>(data.data()));
            snapshot.len = static_cast<uint64_t>(data.size());
            snapshot.cap = snapshot.len;
            result = ds4_session_load_snapshot(session, &snapshot,
                                               error.data(), error.size());
            if (result != 0) {
                logits_ready_ = false;
                rng_state_.reset();
            } else {
                logits_ready_ = true;
                rng_state_.reset();
            }
        }
        if (result != 0) {
            std::string message = error.data();
            if (message.empty())
                message = "ds4_session_load_snapshot failed.";
            throw std::runtime_error("load_snapshot failed: " + message);
        }
    }

    py::bytes save_payload() {
        std::vector<char> data;
        std::array<char, 4096> error = {};
        int result = 0;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ds4_session* session = get_locked();
            const uint64_t byte_count = ds4_session_payload_bytes(session);
            FilePtr file = open_temporary_file("save_payload");
            result = ds4_session_save_payload(session, file.get(),
                                              error.data(), error.size());
            if (result == 0) {
                rewind_after_write(file.get(), "save_payload");
                data =
                    read_payload_bytes(file.get(), byte_count, "save_payload");
            }
        }
        if (result != 0) {
            std::string message = error.data();
            if (message.empty())
                message = "ds4_session_save_payload failed.";
            throw std::runtime_error("save_payload failed: " + message);
        }
        return bytes_from_vector(data);
    }

    void load_payload(py::bytes payload_bytes) {
        const std::string data = py::cast<std::string>(payload_bytes);
        std::array<char, 4096> error = {};
        int result = 0;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ds4_session* session = get_locked();
            FilePtr file = open_temporary_file("load_payload");
            write_all(file.get(), data.data(), data.size(), "load_payload");
            rewind_after_write(file.get(), "load_payload");
            result = ds4_session_load_payload(
                session, file.get(), static_cast<uint64_t>(data.size()),
                error.data(), error.size());
            if (result != 0) {
                logits_ready_ = false;
                rng_state_.reset();
            } else {
                logits_ready_ = true;
                rng_state_.reset();
            }
        }
        if (result != 0) {
            std::string message = error.data();
            if (message.empty())
                message = "ds4_session_load_payload failed.";
            throw std::runtime_error("load_payload failed: " + message);
        }
    }

    void invalidate() {
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ds4_session* session = get_locked();
            ds4_session_invalidate(session);
            logits_ready_ = false;
            rng_state_.reset();
        }
    }

    void set_progress_wakeup_fd(int fd) {
        std::lock_guard<std::mutex> lock(progress_mutex_);
        progress_wakeup_fd_ = fd;
    }

    py::list drain_progress_events() {
        std::vector<ProgressEvent> events;
        {
            std::lock_guard<std::mutex> lock(progress_mutex_);
            events.swap(progress_events_);
        }

        py::list result;
        for (const ProgressEvent& event : events) {
            result.append(
                py::make_tuple(event.event, event.current, event.total));
        }
        return result;
    }

  private:
    static void progress_callback(void* ud, const char* event, int current,
                                  int total) noexcept {
        auto* state = static_cast<SessionState*>(ud);
        if (state == nullptr) {
            return;
        }

        int wakeup_fd = -1;
        try {
            std::lock_guard<std::mutex> lock(state->progress_mutex_);
            state->progress_events_.push_back(
                ProgressEvent{event == nullptr ? "" : event, current, total});
            wakeup_fd = state->progress_wakeup_fd_;
        } catch (...) {
            return;
        }

#if defined(__unix__) || defined(__APPLE__)
        if (wakeup_fd >= 0) {
            const char byte = 1;
            const ssize_t ignored = ::write(wakeup_fd, &byte, 1);
            (void)ignored;
        }
#else
        (void)wakeup_fd;
#endif
    }

    ds4_session* get_locked() const {
        if (session_ == nullptr) {
            throw std::runtime_error("DS4 session is closed.");
        }
        return session_;
    }

    void ensure_logits_ready(const std::string& operation) const {
        if (!logits_ready_) {
            throw std::runtime_error(
                operation +
                " failed: DS4 session has no synchronized prompt or "
                "evaluated token.");
        }
    }

    std::shared_ptr<EngineState> engine_state_;
    ds4_session* session_ = nullptr;
    std::optional<uint64_t> rng_state_;
    bool logits_ready_ = false;
    bool speculative_eval_supported_ = false;
    mutable std::mutex mutex_;
    std::vector<ProgressEvent> progress_events_;
    int progress_wakeup_fd_ = -1;
    mutable std::mutex progress_mutex_;
};

EngineState::EngineState(
    const std::string& model_path, const std::string& backend,
    const std::optional<std::string>& mtp_path, int n_threads,
    int mtp_draft_tokens, float mtp_margin,
    const std::optional<std::string>& directional_steering_file,
    float directional_steering_attn, float directional_steering_ffn,
    bool warm_weights, bool quality) {
    ds4_engine_options options = {};
    options.model_path = model_path.c_str();
    options.mtp_path = mtp_path ? mtp_path->c_str() : nullptr;
    options.backend = to_native_backend(backend);
    options.n_threads = n_threads;
    options.mtp_draft_tokens = mtp_draft_tokens;
    options.mtp_margin = mtp_margin;
    options.directional_steering_file =
        directional_steering_file ? directional_steering_file->c_str()
                                  : nullptr;
    options.directional_steering_attn = directional_steering_attn;
    options.directional_steering_ffn = directional_steering_ffn;
    options.warm_weights = warm_weights;
    options.quality = quality;

    ds4_engine* opened = nullptr;
    int result = 0;
    {
        py::gil_scoped_release release;
        result = ds4_engine_open(&opened, &options);
    }
    if (result != 0 || opened == nullptr) {
        if (opened != nullptr) {
            py::gil_scoped_release release;
            ds4_engine_close(opened);
        }
        throw std::runtime_error("ds4_engine_open failed for model '" +
                                 model_path + "'.");
    }
    engine_ = opened;
}

EngineState::~EngineState() {
    close();
}

void EngineState::close() {
    std::vector<std::shared_ptr<SessionState>> live_sessions;
    ds4_engine* engine = nullptr;
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        live_sessions.reserve(sessions_.size());
        for (const std::weak_ptr<SessionState>& weak_session : sessions_) {
            if (std::shared_ptr<SessionState> session = weak_session.lock()) {
                live_sessions.push_back(std::move(session));
            }
        }
        sessions_.clear();
        engine = std::exchange(engine_, nullptr);
    }
    for (const std::shared_ptr<SessionState>& session : live_sessions) {
        session->close();
    }

    if (engine != nullptr) {
        py::gil_scoped_release release;
        ds4_engine_close(engine);
    }
}

std::shared_ptr<SessionState> EngineState::create_session(int ctx_size) {
    if (ctx_size <= 0) {
        throw py::value_error("ctx_size must be a positive integer.");
    }
    ds4_session* opened = nullptr;
    std::shared_ptr<SessionState> session;
    int result = 0;
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        result = ds4_session_create(&opened, get_locked(), ctx_size);
        if (result == 0 && opened != nullptr) {
            try {
                const bool speculative_eval_supported =
                    ds4_engine_mtp_draft_tokens(get_locked()) > 1;
                session = std::make_shared<SessionState>(
                    shared_from_this(), opened, speculative_eval_supported);
                sessions_.push_back(session);
                opened = nullptr;
            } catch (...) {
                ds4_session_free(opened);
                throw;
            }
        }
    }
    if (result != 0 || session == nullptr) {
        if (opened != nullptr) {
            py::gil_scoped_release release;
            ds4_session_free(opened);
        }
        throw std::runtime_error("ds4_session_create failed for ctx_size " +
                                 std::to_string(ctx_size) + ".");
    }
    return session;
}

int EngineState::routed_quant_bits() const {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mutex_);
    return ds4_engine_routed_quant_bits(get_locked());
}

bool EngineState::has_mtp() const {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mutex_);
    return ds4_engine_has_mtp(get_locked());
}

int EngineState::mtp_draft_tokens() const {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mutex_);
    return ds4_engine_mtp_draft_tokens(get_locked());
}

int EngineState::eos_token_id() const {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mutex_);
    return ds4_token_eos(get_locked());
}

py::bytes EngineState::token_text(py::handle token_id) {
    const int token = py_token_id(token_id, "token_id");

    size_t len = 0;
    char* raw_text = nullptr;
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        raw_text = ds4_token_text(get_locked(), token, &len);
    }
    if (raw_text == nullptr) {
        throw std::runtime_error("ds4_token_text failed.");
    }

    std::unique_ptr<char, decltype(&std::free)> text(raw_text, &std::free);
    return py::bytes(text.get(), len);
}

py::list EngineState::tokenize_text(const std::string& text) {
    NativeTokens tokens;
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        ds4_tokenize_text(get_locked(), text.c_str(), tokens.mutable_get());
    }
    return tokens_to_list(tokens.get());
}

py::list EngineState::tokenize_rendered_chat(const std::string& text) {
    NativeTokens tokens;
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        ds4_tokenize_rendered_chat(get_locked(), text.c_str(),
                                   tokens.mutable_get());
    }
    return tokens_to_list(tokens.get());
}

py::list EngineState::chat_begin() {
    NativeTokens tokens;
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        ds4_chat_begin(get_locked(), tokens.mutable_get());
    }
    return tokens_to_list(tokens.get());
}

void EngineState::chat_append_message(py::list tokens, const std::string& role,
                                      const std::string& content) {
    if (role != "system" && role != "user" && role != "assistant") {
        throw py::value_error("Unsupported DS4 chat role '" + role + "'.");
    }

    NativeTokens native_tokens(validate_token_sequence(tokens, "tokens"));
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        ds4_chat_append_message(get_locked(), native_tokens.mutable_get(),
                                role.c_str(), content.c_str());
    }
    replace_py_list_from_tokens(tokens, native_tokens.get());
}

void EngineState::chat_append_assistant_prefix(py::list tokens,
                                               std::string think_mode) {
    think_mode = normalized(std::move(think_mode));
    if (!is_known_think_mode(think_mode)) {
        throw py::value_error("Unsupported DS4 think mode '" + think_mode +
                              "'.");
    }

    NativeTokens native_tokens(validate_token_sequence(tokens, "tokens"));
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        ds4_chat_append_assistant_prefix(get_locked(),
                                         native_tokens.mutable_get(),
                                         to_native_think_mode(think_mode));
    }
    replace_py_list_from_tokens(tokens, native_tokens.get());
}

py::list
EngineState::encode_chat_prompt(const std::optional<std::string>& system,
                                const std::string& prompt,
                                std::string think_mode) {
    think_mode = normalized(std::move(think_mode));
    if (!is_known_think_mode(think_mode)) {
        throw py::value_error("Unsupported DS4 think mode '" + think_mode +
                              "'.");
    }

    NativeTokens tokens;
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        ds4_encode_chat_prompt(
            get_locked(), system ? system->c_str() : nullptr, prompt.c_str(),
            to_native_think_mode(think_mode), tokens.mutable_get());
    }
    return tokens_to_list(tokens.get());
}
#endif

std::string think_mode_for_context(std::string mode, int ctx_size) {
    mode = normalized(std::move(mode));
    if (!is_known_think_mode(mode)) {
        throw py::value_error("Unsupported DS4 think mode '" + mode + "'.");
    }
#if PYDS4_DS4_SOURCE_PRESENT
    const ds4_think_mode result =
        ds4_think_mode_for_context(to_native_think_mode(mode), ctx_size);
    const char* name = ds4_think_mode_name(result);
    if (name != nullptr && is_known_think_mode(name)) {
        return name;
    }
    throw py::value_error("DS4 returned an unsupported think mode.");
#else
    if (mode == "max" && ctx_size < PYDS4_THINK_MAX_MIN_CONTEXT) {
        return "high";
    }
    return mode;
#endif
}

} // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "Import-safe pyds4 native metadata module";

    module.attr("__version__") = PYDS4_VERSION;
    module.attr("__ds4_commit__") = PYDS4_DS4_COMMIT;
    module.attr("__ds4_api_version__") = py::none();
    module.attr("__ds4_native_backend__") = PYDS4_NATIVE_BACKEND;
    module.attr("__ds4_source_present__") =
        static_cast<bool>(PYDS4_DS4_SOURCE_PRESENT);
    module.attr("__ds4_backend_host_supported__") =
        static_cast<bool>(PYDS4_HOST_SUPPORTS_SELECTED_BACKEND);
    module.attr("__ds4_fake_native__") = static_cast<bool>(PYDS4_FAKE_NATIVE);

    module.def("native_backend",
               []() { return std::string(PYDS4_NATIVE_BACKEND); });
    module.def("available_backends", []() -> py::tuple {
        if (selected_backend_is_available()) {
            return py::make_tuple(std::string(PYDS4_NATIVE_BACKEND));
        }
        return py::tuple();
    });
    module.def("is_backend_available", [](std::string backend) {
        return normalized(std::move(backend)) == PYDS4_NATIVE_BACKEND &&
               selected_backend_is_available();
    });
    module.def("backend_unavailable_reason", [](std::string backend) {
        return unavailable_reason(normalized(std::move(backend)));
    });
    module.def("think_max_min_context",
               []() { return static_cast<int>(PYDS4_THINK_MAX_MIN_CONTEXT); });
    module.def("think_mode_for_context", &think_mode_for_context);

#if PYDS4_DS4_SOURCE_PRESENT
    py::class_<EngineState, std::shared_ptr<EngineState>>(module,
                                                          "EngineState")
        .def(py::init<const std::string&, const std::string&,
                      const std::optional<std::string>&, int, int, float,
                      const std::optional<std::string>&, float, float, bool,
                      bool>(),
             py::arg("model_path"), py::arg("backend"), py::arg("mtp_path"),
             py::arg("n_threads"), py::arg("mtp_draft_tokens"),
             py::arg("mtp_margin"), py::arg("directional_steering_file"),
             py::arg("directional_steering_attn"),
             py::arg("directional_steering_ffn"), py::arg("warm_weights"),
             py::arg("quality"))
        .def("close", &EngineState::close)
        .def_property_readonly("closed", &EngineState::closed)
        .def_property_readonly("routed_quant_bits",
                               &EngineState::routed_quant_bits)
        .def_property_readonly("has_mtp", &EngineState::has_mtp)
        .def_property_readonly("mtp_draft_tokens",
                               &EngineState::mtp_draft_tokens)
        .def_property_readonly("eos_token_id", &EngineState::eos_token_id)
        .def("create_session", &EngineState::create_session,
             py::arg("ctx_size"))
        .def("token_text", &EngineState::token_text, py::arg("token_id"))
        .def("tokenize_text", &EngineState::tokenize_text, py::arg("text"))
        .def("tokenize_rendered_chat", &EngineState::tokenize_rendered_chat,
             py::arg("text"))
        .def("chat_begin", &EngineState::chat_begin)
        .def("chat_append_message", &EngineState::chat_append_message,
             py::arg("tokens"), py::arg("role"), py::arg("content"))
        .def("chat_append_assistant_prefix",
             &EngineState::chat_append_assistant_prefix, py::arg("tokens"),
             py::arg("think_mode"))
        .def("encode_chat_prompt", &EngineState::encode_chat_prompt,
             py::arg("system"), py::arg("prompt"), py::arg("think_mode"));

    py::class_<SessionState, std::shared_ptr<SessionState>>(module,
                                                            "SessionState")
        .def("close", &SessionState::close)
        .def_property_readonly("closed", &SessionState::closed)
        .def_property_readonly("pos", &SessionState::pos)
        .def_property_readonly("ctx", &SessionState::ctx)
        .def_property_readonly("tokens", &SessionState::tokens)
        .def("sync", &SessionState::sync, py::arg("prompt_tokens"))
        .def("eval", &SessionState::eval, py::arg("token_id"))
        .def("argmax", &SessionState::argmax)
        .def("argmax_excluding", &SessionState::argmax_excluding,
             py::arg("token_id"))
        .def("sample", &SessionState::sample, py::arg("temperature"),
             py::arg("top_k"), py::arg("top_p"), py::arg("min_p"),
             py::arg("seed") = py::none())
        .def("token_logprob", &SessionState::token_logprob,
             py::arg("token_id"))
        .def("top_logprobs", &SessionState::top_logprobs, py::arg("k"))
        .def("eval_speculative_argmax", &SessionState::eval_speculative_argmax,
             py::arg("first_token"), py::arg("max_tokens"),
             py::arg("eos_token_id"))
        .def("rewind", &SessionState::rewind, py::arg("pos"))
        .def("save_snapshot", &SessionState::save_snapshot)
        .def("load_snapshot", &SessionState::load_snapshot,
             py::arg("snapshot"))
        .def("save_payload", &SessionState::save_payload)
        .def("load_payload", &SessionState::load_payload, py::arg("payload"))
        .def("invalidate", &SessionState::invalidate)
        .def("set_progress_wakeup_fd", &SessionState::set_progress_wakeup_fd,
             py::arg("fd"))
        .def("drain_progress_events", &SessionState::drain_progress_events);
#endif

#if PYDS4_FAKE_NATIVE
    module.def("fake_reset_counters", &pyds4_fake_reset_counters);
    module.def("fake_counters", []() {
        const pyds4_fake_counters counters = pyds4_fake_get_counters();
        py::dict result;
        result["engine_open_calls"] = counters.engine_open_calls;
        result["engine_close_calls"] = counters.engine_close_calls;
        result["session_create_calls"] = counters.session_create_calls;
        result["session_free_calls"] = counters.session_free_calls;
        result["tokens_free_calls"] = counters.tokens_free_calls;
        result["sync_calls"] = counters.sync_calls;
        result["eval_calls"] = counters.eval_calls;
        result["argmax_calls"] = counters.argmax_calls;
        result["argmax_excluding_calls"] = counters.argmax_excluding_calls;
        result["sample_calls"] = counters.sample_calls;
        result["top_logprobs_calls"] = counters.top_logprobs_calls;
        result["token_logprob_calls"] = counters.token_logprob_calls;
        result["speculative_eval_calls"] = counters.speculative_eval_calls;
        result["rewind_calls"] = counters.rewind_calls;
        result["invalidate_calls"] = counters.invalidate_calls;
        result["payload_bytes_calls"] = counters.payload_bytes_calls;
        result["save_payload_calls"] = counters.save_payload_calls;
        result["load_payload_calls"] = counters.load_payload_calls;
        result["save_snapshot_calls"] = counters.save_snapshot_calls;
        result["load_snapshot_calls"] = counters.load_snapshot_calls;
        result["snapshot_free_calls"] = counters.snapshot_free_calls;
        result["token_allocation_calls"] = counters.token_allocation_calls;
        result["token_live_allocations"] = counters.token_live_allocations;
        result["token_peak_live_allocations"] =
            counters.token_peak_live_allocations;
        result["snapshot_allocation_calls"] =
            counters.snapshot_allocation_calls;
        result["snapshot_live_allocations"] =
            counters.snapshot_live_allocations;
        result["snapshot_peak_live_allocations"] =
            counters.snapshot_peak_live_allocations;
        result["call_sequence"] = counters.call_sequence;
        result["last_engine_open_sequence"] =
            counters.last_engine_open_sequence;
        result["last_engine_close_sequence"] =
            counters.last_engine_close_sequence;
        result["last_session_create_sequence"] =
            counters.last_session_create_sequence;
        result["last_session_free_sequence"] =
            counters.last_session_free_sequence;
        result["last_tokens_free_sequence"] =
            counters.last_tokens_free_sequence;
        result["last_sync_sequence"] = counters.last_sync_sequence;
        result["last_eval_sequence"] = counters.last_eval_sequence;
        result["last_argmax_sequence"] = counters.last_argmax_sequence;
        result["last_argmax_excluding_sequence"] =
            counters.last_argmax_excluding_sequence;
        result["last_sample_sequence"] = counters.last_sample_sequence;
        result["last_top_logprobs_sequence"] =
            counters.last_top_logprobs_sequence;
        result["last_token_logprob_sequence"] =
            counters.last_token_logprob_sequence;
        result["last_speculative_eval_sequence"] =
            counters.last_speculative_eval_sequence;
        result["last_rewind_sequence"] = counters.last_rewind_sequence;
        result["last_invalidate_sequence"] = counters.last_invalidate_sequence;
        result["last_payload_bytes_sequence"] =
            counters.last_payload_bytes_sequence;
        result["last_save_payload_sequence"] =
            counters.last_save_payload_sequence;
        result["last_load_payload_sequence"] =
            counters.last_load_payload_sequence;
        result["last_save_snapshot_sequence"] =
            counters.last_save_snapshot_sequence;
        result["last_load_snapshot_sequence"] =
            counters.last_load_snapshot_sequence;
        result["last_snapshot_free_sequence"] =
            counters.last_snapshot_free_sequence;
        return result;
    });
#endif
}
