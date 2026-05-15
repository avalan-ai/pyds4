#ifndef PYDS4_NATIVE_TOKENS_HPP
#define PYDS4_NATIVE_TOKENS_HPP

#include <vector>

extern "C" {
#include "ds4.h"
}

namespace pyds4::native {

class NativeTokens {
  public:
    NativeTokens() = default;

    explicit NativeTokens(const std::vector<int>& values) {
        for (const int token : values) {
            ds4_tokens_push(&tokens_, token);
        }
    }

    NativeTokens(const NativeTokens&) = delete;
    NativeTokens& operator=(const NativeTokens&) = delete;

    ~NativeTokens() {
        ds4_tokens_free(&tokens_);
    }

    const ds4_tokens* get() const {
        return &tokens_;
    }

    ds4_tokens* mutable_get() {
        return &tokens_;
    }

  private:
    ds4_tokens tokens_ = {nullptr, 0, 0};
};

} // namespace pyds4::native

#endif // PYDS4_NATIVE_TOKENS_HPP
