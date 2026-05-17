# pyds4

`pyds4` is a Python bridge for the [DS4 native inference
engine](https://github.com/antirez/ds4). It exposes import-safe metadata,
backend availability helpers, synchronous engine/session APIs, and an asyncio
facade for token streaming applications.

The package is useful anywhere Python code needs to open a DS4 GGUF, tokenize
chat prompts, run native generation, and stream decoded tokens without binding
the application to a specific framework. [Avalan](https://github.com/avalan-ai/avalan)
also integrates with `pyds4` as a DS4 runtime backend, but Avalan is not
required to use this project.

## Requirements

- Python 3.11 or newer.
- A C++ toolchain supported by CMake and scikit-build-core.
- Upstream DS4 source from <https://github.com/antirez/ds4> when building a
  real native backend.
- macOS arm64 with Metal for `PYDS4_BACKEND=metal`, or Linux with the CUDA
  toolkit for `PYDS4_BACKEND=cuda`.

`PYDS4_BACKEND=cpu` exists for diagnostics and tests. It is not the intended
production path.

## Install

Create and activate a virtual environment, then install the Python package:

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

Without `DS4_SOURCE_DIR` or `PYDS4_USE_FAKE_DS4=1`, the install is
metadata-only. Importing `pyds4` works, but `pyds4.is_backend_available()`
returns `False` because no native DS4 source was compiled into the extension.

To build against upstream DS4, clone DS4 and point `DS4_SOURCE_DIR` at that
checkout:

```bash
git clone https://github.com/antirez/ds4.git /path/to/ds4

DS4_SOURCE_DIR=/path/to/ds4 \
PYDS4_BACKEND=metal \
python -m pip install -e .
```

Select a different backend at build time when needed:

```bash
DS4_SOURCE_DIR=/path/to/ds4 \
PYDS4_BACKEND=cuda \
CUDA_ARCH=90 \
python -m pip install -e .

DS4_SOURCE_DIR=/path/to/ds4 \
PYDS4_BACKEND=cpu \
python -m pip install -e .
```

The build validates the pinned DS4 source revision and required native files.
Metal builds also package DS4's `metal/*.metal` kernels into the wheel and
configure DS4's `DS4_METAL_*_SOURCE` overrides at import time when callers have
not set them.

For wrapper development and CI, build against the deterministic fake DS4 shim:

```bash
PYDS4_USE_FAKE_DS4=1 PYDS4_BACKEND=cpu python -m pip install -e '.[test]'
```

The fake shim lives in `tests/fake_ds4/` and does not require a GGUF, GPU, or
upstream DS4 checkout.

## Metal Examples

On macOS arm64, build the editable package against an upstream DS4 checkout
with the Metal backend:

```bash
git clone https://github.com/antirez/ds4.git /path/to/ds4

DS4_SOURCE_DIR=/path/to/ds4 \
PYDS4_BACKEND=metal \
python -m pip install -e '.[test]'
```

The same build can be run through the Makefile:

```bash
DS4_SOURCE_DIR=/path/to/ds4 \
PYDS4_BACKEND=metal \
make ds4-bridge
```

Verify that the installed package can see the Metal backend:

```bash
python - <<'PY'
import pyds4

print("backend:", pyds4.__ds4_native_backend__)
print("available:", pyds4.is_backend_available("metal"))
print("reason:", pyds4.backend_unavailable_reason("metal"))
PY
```

Run the async streaming example with a DS4-supported GGUF:

```bash
python examples/generate_text_async.py \
  --backend metal \
  --model /path/to/ds4flash.gguf \
  --ctx-size 4096 \
  --max-new-tokens 256 \
  --temperature 0 \
  'Explain LLM distillation in one paragraph.'
```

Run the real-model Metal integration tests:

```bash
PYDS4_MODEL=/path/to/ds4flash.gguf \
PYDS4_BACKEND=metal \
PYDS4_CTX=4096 \
python -m pytest -q tests/test_real_*.py tests/test_async_real_*.py
```

Build and smoke-test a Metal wheel:

```bash
DS4_SOURCE_DIR=/path/to/ds4 \
PYDS4_BACKEND=metal \
make wheel

WHEEL='dist/pyds4-0.1.0-*.whl' \
SMOKE_BACKEND=metal \
SMOKE_EXPECT_AVAILABLE=true \
SMOKE_MODEL=/path/to/ds4flash.gguf \
SMOKE_CTX=4096 \
make wheel-smoke
```

## Test

Install the development dependencies:

```bash
python -m pip install -e '.[test,dev]'
```

For the full wrapper test suite without a real DS4 checkout, install the
fake-native build:

```bash
PYDS4_USE_FAKE_DS4=1 PYDS4_BACKEND=cpu python -m pip install -e '.[test,dev]'
```

Run the Python and direct native C++ test suites:

```bash
make test
```

Useful narrower checks are also available:

```bash
make test-python
make test-cpp
make test-cpp-sanitizers
make lint
```

Real-model integration tests are skipped unless a DS4-supported GGUF is
provided:

```bash
PYDS4_MODEL=/path/to/ds4flash.gguf \
PYDS4_BACKEND=metal \
PYDS4_CTX=4096 \
python -m pytest -q tests/test_real_*.py tests/test_async_real_*.py
```

## Build

Install release tooling:

```bash
python -m pip install -e '.[test,release]'
```

Build a source distribution:

```bash
make sdist
```

Build a wheel for the selected backend:

```bash
DS4_SOURCE_DIR=/path/to/ds4 \
PYDS4_BACKEND=metal \
make wheel
```

Smoke-test a built wheel:

```bash
WHEEL='dist/pyds4-0.1.0-*.whl' \
SMOKE_BACKEND=metal \
SMOKE_EXPECT_AVAILABLE=true \
make wheel-smoke
```

With a real model available, include a one-token native smoke:

```bash
WHEEL='dist/pyds4-0.1.0-*.whl' \
SMOKE_BACKEND=metal \
SMOKE_EXPECT_AVAILABLE=true \
SMOKE_MODEL=/path/to/ds4flash.gguf \
SMOKE_CTX=4096 \
make wheel-smoke
```

## Async Token Streaming

`pyds4.AsyncEngine` owns a synchronous DS4 engine on one worker thread and
serializes all native calls through that owner. `AsyncSession.stream_text()`
is the recommended helper for plain decoded text because it handles UTF-8
decoding, stop-string buffering, EOS suppression, and session advancement on
top of the lower-level `AsyncSession.next_token()` primitive.
`AsyncSession.generate_text()` returns the same decoded text as a complete
string for callers that do not need incremental chunks.

```python
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pyds4


async def stream_ds4_tokens(
    *,
    model_path: str,
    prompt: str,
    backend: pyds4.Backend = pyds4.Backend.METAL,
    ctx_size: int = 4096,
    max_new_tokens: int = 256,
) -> AsyncIterator[str]:
    if not pyds4.is_backend_available(backend.value):
        raise RuntimeError(pyds4.backend_unavailable_reason(backend.value))

    options = pyds4.EngineOptions(
        model_path=model_path,
        backend=backend,
    )
    sampling = pyds4.SamplingOptions(
        temperature=0.7,
        top_k=40,
        top_p=0.95,
        min_p=0.0,
    )

    async with pyds4.AsyncEngine(options) as engine:
        think_mode = pyds4.think_mode_for_context(
            pyds4.ThinkMode.NONE,
            ctx_size,
        )
        prompt_tokens = await engine.encode_chat_prompt(
            system=None,
            prompt=prompt,
            think_mode=think_mode,
        )

        async with await engine.create_session(ctx_size) as session:
            await session.sync(prompt_tokens)
            generation = pyds4.GenerationOptions(
                max_new_tokens=max_new_tokens,
                sampling=sampling,
            )
            async for chunk in session.stream_text(generation):
                yield chunk


async def main() -> None:
    async for chunk in stream_ds4_tokens(
        model_path="/path/to/ds4flash.gguf",
        prompt="Explain LLM distillation in one paragraph.",
    ):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
```

The repository also includes runnable examples:

```bash
python examples/generate_text_async.py \
  --backend metal \
  --model /path/to/ds4flash.gguf \
  --ctx-size 4096 \
  --max-new-tokens 256 \
  'Explain LLM distillation in one paragraph.'
```

## API Overview

- `pyds4.Engine` and `pyds4.Session` provide the synchronous low-level native
  interface.
- `pyds4.AsyncEngine` and `pyds4.AsyncSession` provide an asyncio facade for
  applications that need incremental streaming.
- `AsyncSession.stream_text()` and `AsyncSession.generate_text()` provide
  framework-neutral decoded text generation with stop-string buffering,
  incremental UTF-8 decoding, EOS suppression, and the same
  mutating-cancellation policy as `next_token()`.
- `pyds4.EngineOptions` configures model path, backend, MTP options, threading,
  steering options, and native startup log replay.
- `pyds4.SamplingOptions` configures temperature, top-k, top-p, min-p, and
  seed-based sampling.
- `pyds4.GenerationOptions`, `pyds4.GenerationScoreOptions`, and
  `pyds4.TokenScoreMode` describe framework-neutral generation limits, stop
  strings, EOS handling, decoding, and optional token score details. They do
  not depend on Avalan response or tool objects.
- `pyds4.dsml` provides import-safe, framework-neutral DSML data classes for
  prompt messages, tool calls, parse results, and tool schema normalization.
  These helpers do not import Avalan or native DS4 extension objects.
- `Session.eval_speculative_argmax()` and
  `AsyncSession.eval_speculative_argmax()` expose DS4's greedy MTP speculative
  step when the engine was opened with an MTP model and draft depth greater
  than one. Without that per-engine MTP support, the methods raise a clear
  `Ds4GenerationError` before mutating the session.
- `pyds4.is_backend_available()` and `pyds4.backend_unavailable_reason()`
  report whether the installed wheel can run a requested backend.

## Upstream DS4

This project binds to upstream DS4 and intentionally keeps DS4 source outside
the Python package checkout. Use the upstream repository for DS4 source,
model-download scripts, and native runtime details:

<https://github.com/antirez/ds4>
