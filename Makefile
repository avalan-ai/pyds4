.PHONY: all bridge build install install-dev fake-bridge ds4-bridge format format-python format-cpp lint lint-python lint-cpp lint-cpp-format lint-cpp-tidy lint-cpp-cppcheck cmake-cpp-lint test test-python test-cpp test-cpp-sanitizers tests mypy typecheck release-tools clean-dist sdist wheel dist release-sdist release-wheel audit-wheel repair-wheel release-dist dist-check publish release version pip-check wheel-smoke

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
PIP ?= $(PYTHON) -m pip
BUILD ?= $(PYTHON) -m build
TWINE ?= $(PYTHON) -m twine
AUDITWHEEL ?= $(PYTHON) -m auditwheel
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= $(PYTHON) -m ruff
BLACK ?= $(PYTHON) -m black
MYPY ?= $(PYTHON) -m mypy
CMAKE ?= cmake
CTEST ?= ctest
CLANG_FORMAT ?= $(or $(shell command -v clang-format 2>/dev/null),$(shell xcrun --find clang-format 2>/dev/null),clang-format)
CLANG_TIDY ?= $(shell command -v clang-tidy 2>/dev/null)
CPPCHECK ?= $(shell command -v cppcheck 2>/dev/null)
PYTHON_SOURCES ?= src/ tests/ scripts/ examples/
CXX_FORMAT_SOURCES ?= \
	src/pyds4/_native.cpp \
	src/pyds4/_native_common.hpp \
	src/pyds4/_native_tokens.hpp \
	tests/fake_ds4/ds4.c \
	tests/fake_ds4/ds4.h \
	tests/native/test_native_cxx.cpp
CXX_TIDY_SOURCES ?= src/pyds4/_native.cpp tests/native/test_native_cxx.cpp
CXX_LINT_BUILD_DIR ?= build/cpp-lint
CXX_TEST_BUILD_DIR ?= build/cpp-tests
CXX_SANITIZER_BUILD_DIR ?= build/cpp-sanitizers
PYBIND11_DIR ?= $(shell $(PYTHON) -m pybind11 --cmakedir 2>/dev/null)
PYDS4_BACKEND ?=
PYDS4_USE_FAKE_DS4 ?= 0
DS4_SOURCE_DIR ?=
DS4_SOURCE_REF ?= 8809b90a1e3247389d7652b565ab6772e036f1ea
CUDA_ARCH ?=
DIST_DIR ?= dist
REPAIRED_DIST_DIR ?= wheelhouse
RELEASE_BUILD_DIR ?= build/release/$(if $(PYDS4_BACKEND),$(PYDS4_BACKEND),auto)-fake$(PYDS4_USE_FAKE_DS4)/{wheel_tag}
WHEEL ?= $(DIST_DIR)/pyds4-*.whl
MANYLINUX_PLAT ?= manylinux_2_28_x86_64
SMOKE_WHEEL = $(firstword $(wildcard $(WHEEL)))
SMOKE_BACKEND ?=
SMOKE_EXPECT_AVAILABLE ?=
SMOKE_MODEL ?=
SMOKE_CTX ?= 4096
TWINE_UPLOAD_ARGS ?= --repository pypi
VERSION ?=

all: bridge

bridge:
	$(PIP) install -e .

build: bridge

install:
	$(PIP) install -e '.[test]'

install-dev:
	$(PIP) install -e '.[test,dev]'

fake-bridge:
	PYDS4_USE_FAKE_DS4=1 PYDS4_BACKEND=cpu $(PIP) install -e '.[test]'

ds4-bridge:
	@test -n "$(DS4_SOURCE_DIR)" || { \
		echo "DS4_SOURCE_DIR is required for make ds4-bridge"; \
		exit 2; \
	}
	DS4_SOURCE_DIR="$(DS4_SOURCE_DIR)" \
	DS4_SOURCE_REF="$(DS4_SOURCE_REF)" \
	PYDS4_BACKEND="$(PYDS4_BACKEND)" \
	PYDS4_USE_FAKE_DS4="$(PYDS4_USE_FAKE_DS4)" \
	CUDA_ARCH="$(CUDA_ARCH)" \
	$(PIP) install -e .

format: format-python format-cpp

format-python:
	$(RUFF) format --preview $(PYTHON_SOURCES)
	$(BLACK) --preview --enable-unstable-feature=string_processing $(PYTHON_SOURCES)

format-cpp:
	$(CLANG_FORMAT) -i $(CXX_FORMAT_SOURCES)

lint: format lint-python lint-cpp

lint-python:
	$(RUFF) check --fix $(PYTHON_SOURCES)
	$(MYPY)

lint-cpp: lint-cpp-format lint-cpp-tidy lint-cpp-cppcheck

lint-cpp-format:
	$(CLANG_FORMAT) --dry-run --Werror $(CXX_FORMAT_SOURCES)

cmake-cpp-lint:
	$(CMAKE) -S . -B "$(CXX_LINT_BUILD_DIR)" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPYDS4_BUILD_PYTHON_EXTENSION=ON -DPYDS4_BUILD_CXX_TESTS=ON -DPYDS4_USE_FAKE_DS4=ON -DPYDS4_BACKEND=cpu $(if $(PYBIND11_DIR),-Dpybind11_DIR="$(PYBIND11_DIR)")

lint-cpp-tidy:
	@if [ -z "$(CLANG_TIDY)" ]; then \
		echo "Skipping clang-tidy: clang-tidy not found"; \
	else \
		$(MAKE) cmake-cpp-lint; \
		$(CLANG_TIDY) -p "$(CXX_LINT_BUILD_DIR)" $(CXX_TIDY_SOURCES); \
	fi

lint-cpp-cppcheck:
	@if [ -z "$(CPPCHECK)" ]; then \
		echo "Skipping cppcheck: cppcheck not found"; \
	else \
		$(MAKE) cmake-cpp-lint; \
		$(CPPCHECK) --project="$(CXX_LINT_BUILD_DIR)/compile_commands.json" \
			--enable=warning,style,performance,portability \
			--error-exitcode=1 \
			--inline-suppr \
			--suppress=missingIncludeSystem; \
	fi

mypy:
	$(MYPY)

typecheck: mypy

test: test-python test-cpp

test-python:
	$(PYTEST) -q

test-cpp:
	$(CMAKE) -S . -B "$(CXX_TEST_BUILD_DIR)" \
		-DPYDS4_BUILD_PYTHON_EXTENSION=OFF \
		-DPYDS4_BUILD_CXX_TESTS=ON \
		-DPYDS4_USE_FAKE_DS4=ON \
		-DPYDS4_BACKEND=cpu
	$(CMAKE) --build "$(CXX_TEST_BUILD_DIR)" --target pyds4_native_cxx_tests
	$(CTEST) --test-dir "$(CXX_TEST_BUILD_DIR)" --output-on-failure

test-cpp-sanitizers:
	$(CMAKE) -S . -B "$(CXX_SANITIZER_BUILD_DIR)" \
		-DPYDS4_BUILD_PYTHON_EXTENSION=OFF \
		-DPYDS4_BUILD_CXX_TESTS=ON \
		-DPYDS4_USE_FAKE_DS4=ON \
		-DPYDS4_BACKEND=cpu \
		-DPYDS4_ENABLE_SANITIZERS=ON
	$(CMAKE) --build "$(CXX_SANITIZER_BUILD_DIR)" --target pyds4_native_cxx_tests
	$(CTEST) --test-dir "$(CXX_SANITIZER_BUILD_DIR)" --output-on-failure

tests: test

release-tools:
	$(PYTHON) scripts/install_release_tools.py

clean-dist:
	@test -n "$(DIST_DIR)" && test "$(DIST_DIR)" != "." && test "$(DIST_DIR)" != "/" || { \
		echo "Refusing to clean unsafe DIST_DIR=$(DIST_DIR)"; \
		exit 2; \
	}
	$(PYTHON) -c "import shutil; shutil.rmtree('$(DIST_DIR)', ignore_errors=True)"

sdist:
	$(BUILD) --sdist --outdir "$(DIST_DIR)"

wheel:
	$(BUILD) --wheel --outdir "$(DIST_DIR)"

dist:
	$(BUILD) --outdir "$(DIST_DIR)"

release-sdist:
	$(MAKE) release-tools
	$(BUILD) --no-isolation --sdist --outdir "$(DIST_DIR)"

release-wheel:
	@test -n "$(DS4_SOURCE_DIR)" || { \
		echo "DS4_SOURCE_DIR is required for release wheels"; \
		exit 2; \
	}
	$(MAKE) release-tools
	PYDS4_REQUIRE_DS4_SOURCE=1 \
	DS4_SOURCE_DIR="$(DS4_SOURCE_DIR)" \
	DS4_SOURCE_REF="$(DS4_SOURCE_REF)" \
	PYDS4_BACKEND="$(PYDS4_BACKEND)" \
	PYDS4_USE_FAKE_DS4="$(PYDS4_USE_FAKE_DS4)" \
	CUDA_ARCH="$(CUDA_ARCH)" \
	$(BUILD) --no-isolation --wheel --outdir "$(DIST_DIR)" \
		--config-setting=build-dir="$(RELEASE_BUILD_DIR)"

audit-wheel:
	@test "$$(uname -s)" = "Linux" || { \
		echo "audit-wheel requires Linux; run it on a Linux host or inside a Linux container"; \
		exit 2; \
	}
	$(MAKE) release-tools
	$(AUDITWHEEL) show $(WHEEL)

repair-wheel:
	@test -n "$(REPAIRED_DIST_DIR)" && test "$(REPAIRED_DIST_DIR)" != "." && test "$(REPAIRED_DIST_DIR)" != "/" || { \
		echo "Refusing unsafe REPAIRED_DIST_DIR=$(REPAIRED_DIST_DIR)"; \
		exit 2; \
	}
	$(MAKE) audit-wheel
	mkdir -p "$(REPAIRED_DIST_DIR)"
	$(AUDITWHEEL) repair --plat "$(MANYLINUX_PLAT)" --wheel-dir "$(REPAIRED_DIST_DIR)" $(WHEEL)

release-dist: clean-dist
	$(MAKE) release-sdist
	$(MAKE) release-wheel
	$(MAKE) dist-check

dist-check:
	@test -n "$(wildcard $(DIST_DIR)/*)" || { \
		echo "No distributions found in $(DIST_DIR)"; \
		exit 2; \
	}
	$(MAKE) release-tools
	$(TWINE) check $(DIST_DIR)/*

publish: dist-check
	$(TWINE) upload $(TWINE_UPLOAD_ARGS) $(DIST_DIR)/*

release:
	$(MAKE) release-dist
	$(MAKE) publish

version:
	@test -n "$(VERSION)" || { \
		echo "Usage: make version VERSION=X.Y.Z"; \
		exit 2; \
	}
	$(PYTHON) scripts/bump_version.py "$(VERSION)"

pip-check:
	$(PIP) check

wheel-smoke:
	@test -n "$(SMOKE_WHEEL)" || { \
		echo "No wheel matched WHEEL=$(WHEEL)"; \
		exit 2; \
	}
	$(PYTHON) scripts/wheel_smoke.py \
		--wheel "$(SMOKE_WHEEL)" \
		--backend "$(SMOKE_BACKEND)" \
		--expect-available "$(SMOKE_EXPECT_AVAILABLE)" \
		--model "$(SMOKE_MODEL)" \
		--ctx "$(SMOKE_CTX)"
