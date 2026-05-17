<h1 align="center">pyds4</h1>
<h3 align="center">Python bindings for the DS4 DeepSeek V4 Flash inference engine</h3>

<p align="center">
  <a href="https://github.com/avalan-ai/pyds4/actions/workflows/fake-native.yml"><img src="https://github.com/avalan-ai/pyds4/actions/workflows/fake-native.yml/badge.svg" alt="Tests" /></a>
  <a href="https://pypi.org/project/pyds4/"><img src="https://img.shields.io/pypi/v/pyds4.svg?label=PyPI" alt="PyPI" /></a>
  <img src="https://img.shields.io/pypi/pyversions/pyds4.svg" alt="Python versions" />
  <img src="https://img.shields.io/github/last-commit/avalan-ai/pyds4.svg" alt="Last commit" />
  <a href="https://pypi.org/project/pyds4/"><img src="https://img.shields.io/pypi/l/pyds4.svg" alt="License" /></a>
</p>

`pyds4` is a Python package for running
[DS4](https://github.com/antirez/ds4)-supported DeepSeek V4 Flash GGUF models
from Python. It wraps the DS4 native engine with synchronous and asyncio APIs,
token streaming, chat prompt helpers, token logprobs, DSML tool-call helpers,
session snapshots, and payload-backed disk KV cache helpers.

This is not a generic GGUF runner. It targets the model files and native API
supported by [DS4](https://github.com/antirez/ds4).
[Avalan](https://github.com/avalan-ai/avalan) uses `pyds4` for native DS4
inference, but `pyds4` is usable directly from any Python application.

## Contents

- [Install](#install)
- [Model Files](#model-files)
- [Async Streaming Quickstart](#async-streaming-quickstart)
- [Runnable Examples](#runnable-examples)
- [Tool Use With DSML](#tool-use-with-dsml)
- [Advanced APIs](#advanced-apis)
- [Build From Source](#build-from-source)
- [Benchmark](#benchmark)
- [Test](#test)
- [Release](#release)

## Install

Install the published package from PyPI:

```sh
python -m pip install -U pyds4
```

Check which native backend the installed package was built for:

```sh
python - <<'PY'
import pyds4

backend = pyds4.__ds4_native_backend__
print("pyds4:", pyds4.__version__)
print("backend:", backend)
print("available:", pyds4.is_backend_available(backend))
if not pyds4.is_backend_available(backend):
    print(pyds4.backend_unavailable_reason(backend))
PY
```

`pyds4` requires Python 3.11 or newer. Production targets are macOS arm64 with
Metal and Linux with CUDA. CPU builds exist for diagnostics and tests only.

`pyds4` wheels are built for one selected native backend. If a wheel for your
platform is not available, build from source against a DS4 checkout as shown in
[Build From Source](#build-from-source).

## Model Files

DS4 opens a local GGUF file directly from the filesystem. Use
[DS4](https://github.com/antirez/ds4) to download one of the supported
DeepSeek V4 Flash GGUFs:

```sh
git clone https://github.com/antirez/ds4.git /path/to/ds4
cd /path/to/ds4
./download_model.sh q2-imatrix
```

The DS4 repository documents the available quantizations, memory expectations,
optional MTP model, and current engine limitations. The examples below assume
`/path/to/ds4/ds4flash.gguf`.

## Async Streaming Quickstart

`AsyncEngine` owns DS4 on a single worker thread and serializes native calls
there. For most applications, `AsyncSession.stream_text()` is the right
high-level API: it advances the session, suppresses EOS text, buffers stop
strings, and handles incremental UTF-8 decoding.

```python
import asyncio

import pyds4

MODEL_PATH = "/path/to/ds4/ds4flash.gguf"
CTX_SIZE = 4096


async def main() -> None:
    backend = pyds4.Backend(pyds4.__ds4_native_backend__)
    if not pyds4.is_backend_available(backend.value):
        raise RuntimeError(pyds4.backend_unavailable_reason(backend.value))

    options = pyds4.EngineOptions(
        model_path=MODEL_PATH,
        backend=backend,
        native_log=False,
    )

    async with pyds4.AsyncEngine(options) as engine:
        prompt_tokens = await engine.encode_chat_prompt(
            system="You are concise.",
            prompt="Explain why KV caches matter in one paragraph.",
            think_mode=pyds4.think_mode_for_context(
                pyds4.ThinkMode.NONE,
                CTX_SIZE,
            ),
        )

        async with await engine.create_session(CTX_SIZE) as session:
            await session.sync(prompt_tokens)

            generation = pyds4.GenerationOptions(max_new_tokens=128)
            async for chunk in session.stream_text(generation):
                print(chunk, end="", flush=True)
            print()


asyncio.run(main())
```

Use sampling by passing `SamplingOptions` into `GenerationOptions`:

```python
generation = pyds4.GenerationOptions(
    max_new_tokens=128,
    sampling=pyds4.SamplingOptions(
        temperature=0.7,
        top_k=40,
        top_p=0.95,
        seed=1,
    ),
)
```

If you build your own token loop but want the same stop-string behavior,
`StopStringBuffer` exposes the generic buffering used by `stream_text()`:

```python
buffer = pyds4.StopStringBuffer(["</s>", "STOP"])
for chunk in buffer.push(decoded_token_text):
    print(chunk, end="")
for chunk in buffer.flush():
    print(chunk, end="")
```

## Runnable Examples

The [async example](examples/generate_text_async.py) exposes the common knobs
for backend, context size, sampling, streaming, and thread count:

```sh
python examples/generate_text_async.py \
  --model /path/to/ds4/ds4flash.gguf \
  --backend metal \
  --ctx-size 4096 \
  --max-new-tokens 128 \
  --temperature 0 \
  "Explain LLM distillation in one paragraph."
```

For lower-level control, the [sync example](examples/generate_text.py) shows
the synchronous session loop: `sync()` the prompt, pick `argmax()` or
`sample()`, call `eval()` to advance, then decode
`engine.token_text(token_id)`.

## Tool Use With DSML

DeepSeek V4 Flash tool calls use DSML text. `pyds4` can render prompts with
tool schemas, tokenize rendered DSML chat prompts, parse generated tool calls,
and stream argument-value deltas from a growing tool block. It does not execute
tools; your application owns dispatching the parsed call and appending the tool
result on the next turn.

```python
import pyds4
from pyds4.dsml import (
    DsmlMessage,
    DsmlParseStatus,
    DsmlPrompt,
    DsmlToolCallBufferStatus,
    parse_generated_message,
    render_prompt,
    tool_call_buffer_status,
)

tool_schema = {
    "type": "function",
    "function": {
        "name": "math.calculator",
        "description": "Evaluate a small arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}

rendered = render_prompt(
    DsmlPrompt(
        system_content="Use tools for arithmetic.",
        messages=[DsmlMessage(role="user", content="What is 4 * 7?")],
        tool_schemas=[tool_schema],
    ),
    think_mode=pyds4.ThinkMode.NONE,
)

async def collect_tool_call_text(engine: pyds4.AsyncEngine) -> str:
    prompt_tokens = await engine.tokenize_rendered_chat(rendered)
    generated = ""

    async with await engine.create_session(4096) as session:
        await session.sync(prompt_tokens)
        async for chunk in session.stream_text(
            pyds4.GenerationOptions(max_new_tokens=512),
        ):
            generated += chunk
            if (
                tool_call_buffer_status(generated)
                is DsmlToolCallBufferStatus.CLOSED
            ):
                break

    return generated


async def run_tool_prompt(engine: pyds4.AsyncEngine) -> None:
    generated = await collect_tool_call_text(engine)
    parsed = parse_generated_message(generated)
    if parsed.status is DsmlParseStatus.COMPLETE:
        for call in parsed.calls:
            print(call.name, call.arguments)
```

When continuing after a tool result, render the next prompt with the prior
assistant tool call and a `DsmlMessage(role="tool", content=...)` result so
the DSML transcript stays aligned with DS4's native prompt format.
String parameter rendering escapes every accepted DSML parameter close-marker
variant before it reaches the generated block, while preserving literal entity
text such as `&lt;/parameter>` and `&amp;lt;/parameter>` when parsed back.

## Advanced APIs

### Token Scores

Use `AsyncSession.next_token()` when you need token-level metadata instead of
plain text chunks:

```python
step = await session.next_token(
    decode=True,
    scores=pyds4.GenerationScoreOptions(
        mode=pyds4.TokenScoreMode.TOKEN_LOGPROB_AND_TOP_LOGPROBS,
        top_k=5,
    ),
)

print(step.decoded_text, step.token_logprob)
for score in step.top_logprobs:
    print(score.token_id, score.logprob)
```

The synchronous `Session` exposes the same primitives directly with
`argmax()`, `sample()`, `token_logprob()`, `top_logprobs()`, and `eval()`.

### Snapshots And Disk KV Cache

Sessions can save and restore in-memory snapshots and serialized payloads.
`Ds4DiskKvCache` uses payloads to cache a prompt prefix on disk.

```python
from pathlib import Path

from pyds4.kv_cache import Ds4DiskKvCache

cache = Ds4DiskKvCache(
    Path("~/.cache/pyds4/kv").expanduser(),
    model_namespace="deepseek-v4-flash-q2-imatrix",
    backend=backend,
)

async with await engine.create_session(CTX_SIZE) as session:
    restored = await cache.arestore(session, prompt_tokens, CTX_SIZE)
    if restored.status == "miss" and restored.synced:
        await cache.astore(
            session,
            prompt_tokens,
            CTX_SIZE,
            size_budget_bytes=2_000_000_000,
        )

    text = await session.generate_text(
        pyds4.GenerationOptions(max_new_tokens=128),
    )
```

Store immediately after syncing or restoring the prefix you want to cache.
`Ds4DiskKvCache` caches payload bytes, not snapshots. On a hit, it loads the
payload with `load_payload()`; on a miss or corrupt entry, it calls
`sync(prompt_tokens)` by default. `model_namespace` is caller-defined and
should identify the exact model and configuration. Cache metadata can include
prompt text, and payload bytes are session state, so treat cache directories as
sensitive. `size_budget_bytes` is opt-in and enforced after a payload is
written.

### Progress And Cancellation

`AsyncSession.progress` is an `asyncio.Queue[pyds4.ProgressEvent]` populated
when the native backend reports long-running progress. Cancelling a mutating
async operation poisons that session and closes the native session during
cleanup; create a fresh session after cancellation.

### MTP And Speculative Evaluation

`EngineOptions` accepts `mtp_path`, `mtp_draft_tokens`, and `mtp_margin` for
DS4's optional MTP path. Engine metadata exposes `has_mtp` and
`mtp_draft_tokens`, while sessions expose `eval_speculative_argmax()`. Treat
this as an advanced DS4-specific path and validate it with your target model.

## Build From Source

Source builds need a [DS4](https://github.com/antirez/ds4) checkout from the
repository's default branch, plus CMake and a platform C/C++ toolchain:

```sh
git clone https://github.com/antirez/ds4.git /path/to/ds4
```

Build a Metal package on macOS arm64:

```sh
DS4_SOURCE_DIR=/path/to/ds4 \
PYDS4_BACKEND=metal \
python -m pip install --no-binary=pyds4 pyds4
```

Build a CUDA package on Linux:

```sh
DS4_SOURCE_DIR=/path/to/ds4 \
PYDS4_BACKEND=cuda \
CUDA_ARCH=90 \
python -m pip install --no-binary=pyds4 pyds4
```

To build this checkout and install it into a specific project's virtual
environment, point `PYTHON` at that project's interpreter:

```sh
DS4_SOURCE_DIR=.local/ds4 \
PYDS4_BACKEND=metal \
PYTHON=/path/to/project/.venv/bin/python \
make ds4-bridge
```

If `DS4_SOURCE_DIR` is omitted during a source build, the package remains
import-safe but native inference is unavailable.

For local wrapper development without a real GGUF or GPU, use the deterministic
fake DS4 shim:

```sh
PYDS4_USE_FAKE_DS4=1 PYDS4_BACKEND=cpu \
python -m pip install -e ".[test,dev]"
```

## Benchmark

Benchmark the pyds4 sync path, async primitive path, and async `next_token()`
path:

```sh
python scripts/benchmark_generation.py \
  --model /path/to/ds4/ds4flash.gguf \
  --backend metal \
  --ctx-size 4096 \
  --max-new-tokens 128 \
  --mode pyds4 \
  --json-output /tmp/pyds4-bench.json
```

The benchmark reports open, prompt, warmup, sync, generation and total time,
time to first token, tokens per second, event-loop latency, queue round-trip
latency, and output preview.

## Test

Run the fake-native test suite:

```sh
PYDS4_USE_FAKE_DS4=1 PYDS4_BACKEND=cpu python -m pip install -e ".[test,dev]"
make test
make test-cpp-sanitizers
```

Run real-model integration tests when a supported DS4 GGUF is available:

```sh
PYDS4_MODEL=/path/to/ds4/ds4flash.gguf \
PYDS4_BACKEND=metal \
PYDS4_CTX=4096 \
python -m pytest -q \
  tests/test_real_ds4_integration.py \
  tests/test_async_real_ds4_integration.py
```

Build and smoke-test a wheel:

```sh
DS4_SOURCE_DIR=/path/to/ds4 PYDS4_BACKEND=metal make wheel

WHEEL="dist/pyds4-*.whl" \
SMOKE_BACKEND=metal \
SMOKE_EXPECT_AVAILABLE=true \
SMOKE_MODEL=/path/to/ds4/ds4flash.gguf \
SMOKE_CTX=4096 \
make wheel-smoke
```

## Release

`pyproject.toml` is the version source. For a new release, bump it and commit:

```sh
make version VERSION=1.0.1
```

Build and publish one backend wheel plus the sdist:

```sh
DS4_SOURCE_DIR=/path/to/ds4 PYDS4_BACKEND=metal make release
```

Use `PYDS4_BACKEND=cuda` on a CUDA Linux build host to produce the NVIDIA
wheel. Metal and CUDA wheels can be uploaded for the same `pyds4` version
because they have different platform tags. The GitHub `Release` workflow
builds the sdist, macOS arm64 Metal wheels, and Linux x86_64 CUDA wheels.
The `cuda_arch` workflow input controls the NVIDIA architecture passed to
CMake.

The CUDA wheel is audited in the release workflow but keeps CUDA runtime and
cuBLAS libraries external, so it is built as a `linux_x86_64` wheel. PyPI
rejects that platform tag, so CUDA wheels are attached to the GitHub release;
PyPI receives the sdist and macOS wheels. A strict `auditwheel repair`
currently bundles those NVIDIA libraries and creates an artifact too large for
PyPI's default file limit.
