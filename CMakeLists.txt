cmake_minimum_required(VERSION 3.20)

if(DEFINED SKBUILD_PROJECT_VERSION AND NOT "${SKBUILD_PROJECT_VERSION}" STREQUAL "")
  set(PYDS4_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
  file(READ "${CMAKE_CURRENT_LIST_DIR}/pyproject.toml" PYDS4_PYPROJECT_TOML)
  string(REGEX MATCH "\nversion = \"([^\"]+)\"" _ "${PYDS4_PYPROJECT_TOML}")
  set(PYDS4_VERSION "${CMAKE_MATCH_1}")
endif()

if("${PYDS4_VERSION}" STREQUAL "")
  message(FATAL_ERROR "Could not determine pyds4 version.")
endif()

project(pyds4 LANGUAGES C CXX)

option(PYDS4_BUILD_PYTHON_EXTENSION "Build the pyds4 Python extension" ON)
option(PYDS4_BUILD_CXX_TESTS "Build direct native C++ tests" OFF)
option(PYDS4_ENABLE_NATIVE_WARNINGS "Enable warnings for pyds4 native targets" ON)
option(PYDS4_WARNINGS_AS_ERRORS "Treat pyds4 native warnings as errors" OFF)
option(PYDS4_ENABLE_SANITIZERS "Build pyds4 native targets with ASan/UBSan" OFF)

function(pyds4_enable_project_warnings target)
  if(NOT PYDS4_ENABLE_NATIVE_WARNINGS)
    return()
  endif()

  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
    if(PYDS4_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(
      ${target}
      PRIVATE
        "$<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wpedantic>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
        "$<$<COMPILE_LANGUAGE:OBJC>:-Wall;-Wextra>")
    if(PYDS4_WARNINGS_AS_ERRORS)
      target_compile_options(
        ${target}
        PRIVATE
          "$<$<COMPILE_LANGUAGE:C>:-Werror>"
          "$<$<COMPILE_LANGUAGE:CXX>:-Werror>"
          "$<$<COMPILE_LANGUAGE:OBJC>:-Werror>")
    endif()
  endif()
endfunction()

function(pyds4_enable_sanitizers target)
  if(NOT PYDS4_ENABLE_SANITIZERS)
    return()
  endif()

  if(MSVC)
    message(WARNING "PYDS4_ENABLE_SANITIZERS is not wired for MSVC builds.")
    return()
  endif()

  if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU"
     OR CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(
      ${target}
      PRIVATE
        "$<$<COMPILE_LANGUAGE:C>:-fsanitize=address,undefined;-fno-omit-frame-pointer>"
        "$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined;-fno-omit-frame-pointer>"
        "$<$<COMPILE_LANGUAGE:OBJC>:-fsanitize=address,undefined;-fno-omit-frame-pointer>")
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  else()
    message(WARNING
        "PYDS4_ENABLE_SANITIZERS requires Clang or GCC-compatible flags.")
  endif()
endfunction()

set(PYDS4_EXPECTED_DS4_SOURCE_REF
    "8809b90a1e3247389d7652b565ab6772e036f1ea")
set(PYDS4_THINK_MAX_MIN_CONTEXT "393216")
set(PYDS4_REQUIRED_METAL_SOURCE_FILES
    metal/flash_attn.metal
    metal/dense.metal
    metal/moe.metal
    metal/dsv4_hc.metal
    metal/unary.metal
    metal/dsv4_kv.metal
    metal/dsv4_rope.metal
    metal/dsv4_misc.metal
    metal/argsort.metal
    metal/cpy.metal
    metal/concat.metal
    metal/get_rows.metal
    metal/sum_rows.metal
    metal/softmax.metal
    metal/repeat.metal
    metal/glu.metal
    metal/norm.metal
    metal/bin.metal
    metal/set_rows.metal)

set(PYDS4_HOST_SUPPORTS_METAL OFF)
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  find_library(PYDS4_METAL_FRAMEWORK Metal)
  find_library(PYDS4_FOUNDATION_FRAMEWORK Foundation)
  if(PYDS4_METAL_FRAMEWORK AND PYDS4_FOUNDATION_FRAMEWORK)
    set(PYDS4_HOST_SUPPORTS_METAL ON)
  endif()
endif()

set(PYDS4_HOST_SUPPORTS_CUDA OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(PYDS4_HOST_SUPPORTS_CUDA ON)
  endif()
endif()

if(DEFINED ENV{PYDS4_BACKEND} AND NOT "$ENV{PYDS4_BACKEND}" STREQUAL "")
  set(PYDS4_BACKEND "$ENV{PYDS4_BACKEND}" CACHE STRING
      "DS4 native backend: metal, cuda, or cpu" FORCE)
elseif(NOT DEFINED PYDS4_BACKEND OR "${PYDS4_BACKEND}" STREQUAL "")
  if(PYDS4_HOST_SUPPORTS_METAL)
    set(PYDS4_BACKEND "metal" CACHE STRING
        "DS4 native backend: metal, cuda, or cpu" FORCE)
  elseif(PYDS4_HOST_SUPPORTS_CUDA)
    set(PYDS4_BACKEND "cuda" CACHE STRING
        "DS4 native backend: metal, cuda, or cpu" FORCE)
  else()
    set(PYDS4_BACKEND "cpu" CACHE STRING
        "DS4 native backend: metal, cuda, or cpu" FORCE)
  endif()
endif()

string(STRIP "${PYDS4_BACKEND}" PYDS4_BACKEND_STRIPPED)
string(TOLOWER "${PYDS4_BACKEND_STRIPPED}" PYDS4_BACKEND_NORMALIZED)
set(PYDS4_SUPPORTED_BACKENDS metal cuda cpu)
if(NOT PYDS4_BACKEND_NORMALIZED IN_LIST PYDS4_SUPPORTED_BACKENDS)
  message(FATAL_ERROR
      "Unsupported PYDS4_BACKEND='${PYDS4_BACKEND}'. "
      "Expected one of: metal, cuda, cpu.")
endif()

message(STATUS
    "pyds4 backend selection: selected='${PYDS4_BACKEND_NORMALIZED}', "
    "macOS arm64 Metal available=${PYDS4_HOST_SUPPORTS_METAL}, "
    "Linux CUDA available=${PYDS4_HOST_SUPPORTS_CUDA}")
if(PYDS4_BACKEND_NORMALIZED STREQUAL "cpu")
  message(STATUS
      "PYDS4_BACKEND=cpu selected; CPU mode is diagnostic/reference only.")
elseif(PYDS4_BACKEND_NORMALIZED STREQUAL "metal"
       AND NOT PYDS4_HOST_SUPPORTS_METAL)
  message(WARNING
      "PYDS4_BACKEND=metal was selected, but this host does not look like "
      "macOS arm64 with Metal frameworks. Native DS4 linking may fail until "
      "built on a supported production target.")
elseif(PYDS4_BACKEND_NORMALIZED STREQUAL "cuda"
       AND NOT PYDS4_HOST_SUPPORTS_CUDA)
  message(WARNING
      "PYDS4_BACKEND=cuda was selected, but CUDA Toolkit was not detected on "
      "Linux. Native DS4 linking may fail until built on a supported "
      "production target.")
endif()

set(PYDS4_HOST_SUPPORTS_SELECTED_BACKEND OFF)
if(PYDS4_BACKEND_NORMALIZED STREQUAL "cpu")
  set(PYDS4_HOST_SUPPORTS_SELECTED_BACKEND ON)
elseif(PYDS4_BACKEND_NORMALIZED STREQUAL "metal"
       AND PYDS4_HOST_SUPPORTS_METAL)
  set(PYDS4_HOST_SUPPORTS_SELECTED_BACKEND ON)
elseif(PYDS4_BACKEND_NORMALIZED STREQUAL "cuda"
       AND PYDS4_HOST_SUPPORTS_CUDA)
  set(PYDS4_HOST_SUPPORTS_SELECTED_BACKEND ON)
endif()

if(DEFINED ENV{DS4_SOURCE_REF} AND NOT "$ENV{DS4_SOURCE_REF}" STREQUAL "")
  set(DS4_SOURCE_REF "$ENV{DS4_SOURCE_REF}" CACHE STRING
      "Pinned DS4 source commit" FORCE)
elseif(NOT DEFINED DS4_SOURCE_REF OR "${DS4_SOURCE_REF}" STREQUAL "")
  set(DS4_SOURCE_REF "${PYDS4_EXPECTED_DS4_SOURCE_REF}" CACHE STRING
      "Pinned DS4 source commit" FORCE)
endif()

if(NOT DS4_SOURCE_REF STREQUAL PYDS4_EXPECTED_DS4_SOURCE_REF)
  message(FATAL_ERROR
      "DS4_SOURCE_REF must be ${PYDS4_EXPECTED_DS4_SOURCE_REF}; "
      "got '${DS4_SOURCE_REF}'.")
endif()

if(DEFINED ENV{DS4_SOURCE_DIR} AND NOT "$ENV{DS4_SOURCE_DIR}" STREQUAL "")
  set(DS4_SOURCE_DIR "$ENV{DS4_SOURCE_DIR}" CACHE PATH
      "Path to the pinned antirez/ds4 checkout" FORCE)
endif()

if(DEFINED ENV{CUDA_ARCH} AND NOT "$ENV{CUDA_ARCH}" STREQUAL "")
  set(CUDA_ARCH "$ENV{CUDA_ARCH}" CACHE STRING
      "CUDA architecture list for future DS4 CUDA builds" FORCE)
endif()

option(PYDS4_REQUIRE_DS4_SOURCE
       "Require DS4_SOURCE_DIR during this metadata-only bootstrap build" OFF)
option(PYDS4_USE_FAKE_DS4
       "Build against the deterministic test fake DS4 C shim" OFF)

if(DEFINED ENV{PYDS4_REQUIRE_DS4_SOURCE}
   AND NOT "$ENV{PYDS4_REQUIRE_DS4_SOURCE}" STREQUAL "")
  set(PYDS4_REQUIRE_DS4_SOURCE "$ENV{PYDS4_REQUIRE_DS4_SOURCE}" CACHE BOOL
      "Require DS4_SOURCE_DIR during this metadata-only bootstrap build" FORCE)
endif()

if(DEFINED ENV{PYDS4_USE_FAKE_DS4}
   AND NOT "$ENV{PYDS4_USE_FAKE_DS4}" STREQUAL "")
  set(PYDS4_USE_FAKE_DS4 "$ENV{PYDS4_USE_FAKE_DS4}" CACHE BOOL
      "Build against the deterministic test fake DS4 C shim" FORCE)
endif()

set(PYDS4_DS4_SOURCE_PRESENT OFF)
if(PYDS4_USE_FAKE_DS4)
  set(DS4_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/fake_ds4" CACHE PATH
      "Path to the fake DS4 C shim" FORCE)
  set(PYDS4_REQUIRED_DS4_FILES ds4.h ds4.c)
elseif(DEFINED DS4_SOURCE_DIR AND NOT "${DS4_SOURCE_DIR}" STREQUAL "")
  set(PYDS4_REQUIRED_DS4_FILES
      ds4.h
      ds4.c
      rax.c
      rax.h
      rax_malloc.h)
  if(PYDS4_BACKEND_NORMALIZED STREQUAL "metal")
    list(APPEND PYDS4_REQUIRED_DS4_FILES
         ds4_gpu.h
         ds4_metal.m
         ${PYDS4_REQUIRED_METAL_SOURCE_FILES})
  elseif(PYDS4_BACKEND_NORMALIZED STREQUAL "cuda")
    list(APPEND PYDS4_REQUIRED_DS4_FILES
         ds4_gpu.h
         ds4_cuda.cu
         ds4_iq2_tables_cuda.inc)
  endif()
endif()

if(DEFINED DS4_SOURCE_DIR AND NOT "${DS4_SOURCE_DIR}" STREQUAL "")
  foreach(required_file IN LISTS PYDS4_REQUIRED_DS4_FILES)
    if(NOT EXISTS "${DS4_SOURCE_DIR}/${required_file}")
      message(FATAL_ERROR
          "DS4_SOURCE_DIR='${DS4_SOURCE_DIR}' is missing required DS4 "
          "file '${required_file}'.")
    endif()
  endforeach()

  find_package(Git QUIET)
  if(Git_FOUND AND NOT PYDS4_USE_FAKE_DS4)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${DS4_SOURCE_DIR}" rev-parse HEAD
      OUTPUT_VARIABLE PYDS4_ACTUAL_DS4_SOURCE_REF
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(PYDS4_ACTUAL_DS4_SOURCE_REF
       AND NOT PYDS4_ACTUAL_DS4_SOURCE_REF STREQUAL PYDS4_EXPECTED_DS4_SOURCE_REF)
      message(FATAL_ERROR
          "DS4_SOURCE_DIR is at '${PYDS4_ACTUAL_DS4_SOURCE_REF}', but pyds4 "
          "targets '${PYDS4_EXPECTED_DS4_SOURCE_REF}'.")
    endif()
  endif()

  set(PYDS4_DS4_SOURCE_PRESENT ON)
  if(PYDS4_USE_FAKE_DS4)
    message(STATUS
        "PYDS4_USE_FAKE_DS4=ON selected; compiling the deterministic "
        "test fake DS4 C shim instead of upstream native DS4.")
  endif()
elseif(PYDS4_REQUIRE_DS4_SOURCE)
  message(FATAL_ERROR
      "DS4_SOURCE_DIR is required when PYDS4_REQUIRE_DS4_SOURCE=ON.")
else()
  message(STATUS
      "DS4_SOURCE_DIR was not provided; building import-safe metadata "
      "extension only. Native inference requires rebuilding with "
      "DS4_SOURCE_DIR or PYDS4_USE_FAKE_DS4=1.")
endif()

set(PYDS4_NATIVE_DS4_OBJECT_TARGET "")
if(PYDS4_DS4_SOURCE_PRESENT AND NOT PYDS4_USE_FAKE_DS4)
  if(PYDS4_BACKEND_NORMALIZED STREQUAL "metal")
    if(NOT PYDS4_HOST_SUPPORTS_METAL)
      message(FATAL_ERROR
          "PYDS4_BACKEND=metal with DS4_SOURCE_DIR requires macOS arm64 "
          "with Metal and Foundation frameworks.")
    endif()
    enable_language(OBJC)
  elseif(PYDS4_BACKEND_NORMALIZED STREQUAL "cuda")
    if(NOT CUDAToolkit_FOUND)
      message(FATAL_ERROR
          "PYDS4_BACKEND=cuda with DS4_SOURCE_DIR requires CUDA Toolkit "
          "on Linux.")
    endif()
    if(DEFINED CUDA_ARCH AND NOT "${CUDA_ARCH}" STREQUAL "")
      set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH}" CACHE STRING
          "CUDA architecture list for DS4 CUDA builds" FORCE)
    endif()
    enable_language(CUDA)
  endif()

  set(PYDS4_NATIVE_DS4_SOURCES "${DS4_SOURCE_DIR}/ds4.c")
  if(PYDS4_BACKEND_NORMALIZED STREQUAL "metal")
    list(APPEND PYDS4_NATIVE_DS4_SOURCES "${DS4_SOURCE_DIR}/ds4_metal.m")
  elseif(PYDS4_BACKEND_NORMALIZED STREQUAL "cuda")
    list(APPEND PYDS4_NATIVE_DS4_SOURCES "${DS4_SOURCE_DIR}/ds4_cuda.cu")
  endif()

  add_library(pyds4_ds4_objects OBJECT ${PYDS4_NATIVE_DS4_SOURCES})
  set(PYDS4_NATIVE_DS4_OBJECT_TARGET pyds4_ds4_objects)
  set_target_properties(
    pyds4_ds4_objects
    PROPERTIES
      C_STANDARD 99
      C_STANDARD_REQUIRED ON
      POSITION_INDEPENDENT_CODE ON)
  pyds4_enable_sanitizers(pyds4_ds4_objects)
  target_include_directories(pyds4_ds4_objects PRIVATE "${DS4_SOURCE_DIR}")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(pyds4_ds4_objects PRIVATE _GNU_SOURCE)
    target_compile_options(
      pyds4_ds4_objects
      PRIVATE $<$<COMPILE_LANGUAGE:C>:-fno-finite-math-only>)
  endif()
  if(PYDS4_BACKEND_NORMALIZED STREQUAL "cpu")
    target_compile_definitions(pyds4_ds4_objects PRIVATE DS4_NO_GPU)
  elseif(PYDS4_BACKEND_NORMALIZED STREQUAL "metal")
    target_compile_options(
      pyds4_ds4_objects
      PRIVATE $<$<COMPILE_LANGUAGE:OBJC>:-fobjc-arc>)
  elseif(PYDS4_BACKEND_NORMALIZED STREQUAL "cuda")
    set_target_properties(
      pyds4_ds4_objects
      PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)
  endif()
endif()

if(PYDS4_DS4_SOURCE_PRESENT)
  set(PYDS4_DS4_SOURCE_PRESENT_PY "True")
else()
  set(PYDS4_DS4_SOURCE_PRESENT_PY "False")
endif()
if(PYDS4_USE_FAKE_DS4)
  set(PYDS4_USE_FAKE_DS4_PY "True")
else()
  set(PYDS4_USE_FAKE_DS4_PY "False")
endif()
if(PYDS4_HOST_SUPPORTS_SELECTED_BACKEND)
  set(PYDS4_HOST_SUPPORTS_SELECTED_BACKEND_PY "True")
else()
  set(PYDS4_HOST_SUPPORTS_SELECTED_BACKEND_PY "False")
endif()

if(PYDS4_BUILD_PYTHON_EXTENSION)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_native src/pyds4/_native.cpp)
  target_compile_features(_native PRIVATE cxx_std_17)
  target_compile_definitions(
    _native
    PRIVATE
      PYDS4_VERSION="${PYDS4_VERSION}"
      PYDS4_DS4_COMMIT="${PYDS4_EXPECTED_DS4_SOURCE_REF}"
      PYDS4_NATIVE_BACKEND="${PYDS4_BACKEND_NORMALIZED}"
      PYDS4_THINK_MAX_MIN_CONTEXT=${PYDS4_THINK_MAX_MIN_CONTEXT}
      PYDS4_DS4_SOURCE_PRESENT=$<BOOL:${PYDS4_DS4_SOURCE_PRESENT}>
      PYDS4_HOST_SUPPORTS_SELECTED_BACKEND=$<BOOL:${PYDS4_HOST_SUPPORTS_SELECTED_BACKEND}>
      PYDS4_FAKE_NATIVE=$<BOOL:${PYDS4_USE_FAKE_DS4}>)
  pyds4_enable_project_warnings(_native)
  pyds4_enable_sanitizers(_native)

  if(PYDS4_USE_FAKE_DS4)
    target_sources(_native PRIVATE tests/fake_ds4/ds4.c)
    target_include_directories(_native PRIVATE tests/fake_ds4)
  elseif(PYDS4_NATIVE_DS4_OBJECT_TARGET)
    find_package(Threads REQUIRED)
    target_sources(_native PRIVATE $<TARGET_OBJECTS:pyds4_ds4_objects>)
    target_include_directories(_native PRIVATE "${DS4_SOURCE_DIR}")
    target_link_libraries(_native PRIVATE Threads::Threads)
    if(NOT APPLE)
      target_link_libraries(_native PRIVATE m)
    endif()
    if(PYDS4_BACKEND_NORMALIZED STREQUAL "metal")
      target_link_libraries(
        _native
        PRIVATE
          "${PYDS4_METAL_FRAMEWORK}"
          "${PYDS4_FOUNDATION_FRAMEWORK}")
    elseif(PYDS4_BACKEND_NORMALIZED STREQUAL "cuda")
      target_link_libraries(_native PRIVATE CUDA::cudart CUDA::cublas)
    endif()
  endif()

  configure_file(
    src/pyds4/_build_config.py.in
    "${CMAKE_CURRENT_BINARY_DIR}/generated/pyds4/_build_config.py"
    @ONLY)

  install(TARGETS _native DESTINATION pyds4)
  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/pyds4/_build_config.py"
    DESTINATION pyds4)
  if(PYDS4_DS4_SOURCE_PRESENT
     AND NOT PYDS4_USE_FAKE_DS4
     AND PYDS4_BACKEND_NORMALIZED STREQUAL "metal")
    install(
      DIRECTORY "${DS4_SOURCE_DIR}/metal/"
      DESTINATION pyds4/metal
      FILES_MATCHING PATTERN "*.metal")
  endif()
endif()

if(PYDS4_BUILD_CXX_TESTS)
  enable_testing()

  add_executable(
    pyds4_native_cxx_tests
    tests/native/test_native_cxx.cpp
    tests/fake_ds4/ds4.c)
  target_compile_features(pyds4_native_cxx_tests PRIVATE cxx_std_17)
  target_include_directories(
    pyds4_native_cxx_tests
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/fake_ds4")
  pyds4_enable_project_warnings(pyds4_native_cxx_tests)
  pyds4_enable_sanitizers(pyds4_native_cxx_tests)

  add_test(NAME pyds4_native_cxx_tests COMMAND pyds4_native_cxx_tests)
endif()
