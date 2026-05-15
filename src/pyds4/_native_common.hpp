#ifndef PYDS4_NATIVE_COMMON_HPP
#define PYDS4_NATIVE_COMMON_HPP

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace pyds4::native {

inline std::string normalized(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

inline bool is_known_backend(std::string_view backend) {
    return backend == "metal" || backend == "cuda" || backend == "cpu";
}

inline bool is_known_think_mode(std::string_view mode) {
    return mode == "none" || mode == "high" || mode == "max";
}

inline bool
selected_backend_is_available(bool ds4_source_present,
                              bool host_supports_selected_backend) {
    return ds4_source_present && host_supports_selected_backend;
}

inline std::string unavailable_reason(std::string_view requested_backend,
                                      std::string_view selected_backend,
                                      bool ds4_source_present,
                                      bool host_supports_selected_backend) {
    const std::string backend(requested_backend);
    const std::string selected(selected_backend);

    if (!is_known_backend(backend)) {
        return "Unsupported DS4 native backend '" + backend +
               "'. Supported native backends: metal, cuda, cpu. "
               "Supported production targets: macOS arm64 + Metal and Linux + "
               "CUDA. CPU mode is diagnostic/reference only.";
    }
    if (backend == selected) {
        if (!ds4_source_present) {
            return "This pyds4 build was configured for '" + selected +
                   "', but it does not include DS4 native source. "
                   "Rebuild with DS4_SOURCE_DIR for real DS4 or "
                   "PYDS4_USE_FAKE_DS4=1 for fake-native tests. "
                   "Supported production targets: macOS arm64 + Metal and "
                   "Linux + CUDA. CPU mode is diagnostic/reference only.";
        }
        if (!host_supports_selected_backend) {
            return "This pyds4 build was configured for '" + selected +
                   "', but this host does not match that backend target. "
                   "Supported production targets: macOS arm64 + Metal and "
                   "Linux + CUDA. CPU mode is diagnostic/reference only.";
        }
        return "";
    }
    return "This pyds4 build was configured for '" + selected +
           "'. Supported production targets: macOS arm64 + Metal and Linux + "
           "CUDA. CPU mode is diagnostic/reference only.";
}

} // namespace pyds4::native

#endif // PYDS4_NATIVE_COMMON_HPP
