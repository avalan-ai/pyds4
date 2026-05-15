from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pyds4

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / ".local" / "ds4" / "ds4flash.gguf"
)
DEFAULT_AVALAN_ROOT = Path("/Users/mariano/Code/ai/avalan")
MODE_PYDS4_SYNC = "pyds4-sync"
MODE_PYDS4_ASYNC_PRIMITIVE = "pyds4-async-primitive"
MODE_PYDS4_ASYNC_NEXT_TOKEN = "pyds4-async-next-token"
MODE_AVALAN_STREAM = "avalan-stream"
MODE_AVALAN_NONSTREAM = "avalan-nonstream"
PYDS4_MODES = (
    MODE_PYDS4_SYNC,
    MODE_PYDS4_ASYNC_PRIMITIVE,
    MODE_PYDS4_ASYNC_NEXT_TOKEN,
)
AVALAN_MODES = (MODE_AVALAN_STREAM, MODE_AVALAN_NONSTREAM)
ALL_MODES = (*PYDS4_MODES, *AVALAN_MODES)


@dataclass(slots=True)
class BenchmarkConfig:
    model_path: Path
    backend: pyds4.Backend
    ctx_size: int
    max_new_tokens: int
    prompt: str
    system: str | None
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    seed: int | None
    n_threads: int
    warmup_tokens: int
    queue_probes: int
    loop_probe_interval_ms: float
    avalan_root: Path


@dataclass(slots=True)
class BenchmarkResult:
    mode: str
    generated_tokens: int
    output_bytes: int
    output_preview: str
    open_s: float | None
    prompt_s: float | None
    warmup_s: float | None
    sync_s: float | None
    generation_s: float
    total_s: float
    ttft_s: float | None
    tokens_per_s: float | None
    loop_latency_p95_ms: float | None
    loop_latency_max_ms: float | None
    queue_roundtrip_p50_ms: float | None
    queue_roundtrip_p95_ms: float | None
    queue_calls_per_token: float | None


class LoopLatencyProbe:
    def __init__(self, interval_ms: float) -> None:
        self._interval_s = interval_ms / 1000.0
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self.samples_ms: list[float] = []

    async def __aenter__(self) -> "LoopLatencyProbe":
        self._running = True
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        self._running = False
        task = self._task
        if task is not None:
            await task

    async def _run(self) -> None:
        target = perf_counter() + self._interval_s
        while self._running:
            await asyncio.sleep(max(target - perf_counter(), 0.0))
            now = perf_counter()
            self.samples_ms.append(max(now - target, 0.0) * 1000.0)
            target += self._interval_s


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def p50(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def output_preview(data: bytes | str, *, limit: int = 120) -> str:
    text = (
        data.decode("utf-8", errors="replace")
        if isinstance(data, bytes)
        else data
    )
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def tokens_per_second(tokens: int, generation_s: float) -> float | None:
    if tokens <= 0 or generation_s <= 0:
        return None
    return tokens / generation_s


def engine_options(config: BenchmarkConfig) -> pyds4.EngineOptions:
    return pyds4.EngineOptions(
        model_path=str(config.model_path),
        backend=config.backend,
        n_threads=config.n_threads,
    )


def sampling_options(config: BenchmarkConfig) -> pyds4.SamplingOptions:
    return pyds4.SamplingOptions(
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        min_p=config.min_p,
        seed=config.seed,
    )


def use_sampling(config: BenchmarkConfig) -> bool:
    return config.temperature > 0.0


def sync_candidate(
    session: pyds4.Session,
    config: BenchmarkConfig,
    sampling: pyds4.SamplingOptions,
) -> int:
    if use_sampling(config):
        return session.sample(sampling)
    return session.argmax()


def sync_warmup(
    engine: pyds4.Engine,
    prompt_tokens: list[int],
    config: BenchmarkConfig,
) -> float | None:
    if config.warmup_tokens <= 0:
        return None
    sampling = sampling_options(config)
    start = perf_counter()
    with engine.create_session(config.ctx_size) as session:
        session.sync(prompt_tokens)
        for _ in range(config.warmup_tokens):
            token_id = sync_candidate(session, config, sampling)
            if token_id == engine.eos_token_id:
                break
            session.eval(token_id)
            engine.token_text(token_id)
    return perf_counter() - start


def run_pyds4_sync(config: BenchmarkConfig) -> BenchmarkResult:
    total_start = perf_counter()
    open_start = perf_counter()
    with pyds4.Engine(engine_options(config)) as engine:
        open_s = perf_counter() - open_start

        prompt_start = perf_counter()
        prompt_tokens = engine.encode_chat_prompt(
            system=config.system,
            prompt=config.prompt,
            think_mode=pyds4.ThinkMode.NONE,
        )
        prompt_s = perf_counter() - prompt_start
        warmup_s = sync_warmup(engine, prompt_tokens, config)
        sampling = sampling_options(config)
        eos_token_id = engine.eos_token_id

        with engine.create_session(config.ctx_size) as session:
            sync_start = perf_counter()
            session.sync(prompt_tokens)
            sync_s = perf_counter() - sync_start

            chunks: list[bytes] = []
            first_token_s: float | None = None
            generation_start = perf_counter()
            for _ in range(config.max_new_tokens):
                token_id = sync_candidate(session, config, sampling)
                if token_id == eos_token_id:
                    break
                session.eval(token_id)
                token_bytes = engine.token_text(token_id)
                if first_token_s is None:
                    first_token_s = perf_counter() - generation_start
                chunks.append(token_bytes)

            generation_s = perf_counter() - generation_start

    output = b"".join(chunks)
    total_s = perf_counter() - total_start
    return BenchmarkResult(
        mode=MODE_PYDS4_SYNC,
        generated_tokens=len(chunks),
        output_bytes=len(output),
        output_preview=output_preview(output),
        open_s=open_s,
        prompt_s=prompt_s,
        warmup_s=warmup_s,
        sync_s=sync_s,
        generation_s=generation_s,
        total_s=total_s,
        ttft_s=first_token_s,
        tokens_per_s=tokens_per_second(len(chunks), generation_s),
        loop_latency_p95_ms=None,
        loop_latency_max_ms=None,
        queue_roundtrip_p50_ms=None,
        queue_roundtrip_p95_ms=None,
        queue_calls_per_token=0.0,
    )


async def async_queue_probe(
    engine: pyds4.AsyncEngine,
    probes: int,
) -> tuple[float | None, float | None]:
    if probes <= 0:
        return None, None

    samples_ms: list[float] = []
    for _ in range(probes):
        start = perf_counter()
        await engine.eos_token_id
        samples_ms.append((perf_counter() - start) * 1000.0)
    return p50(samples_ms), percentile(samples_ms, 95.0)


async def async_warmup(
    engine: pyds4.AsyncEngine,
    prompt_tokens: list[int],
    config: BenchmarkConfig,
) -> float | None:
    if config.warmup_tokens <= 0:
        return None

    sampling = sampling_options(config)
    start = perf_counter()
    async with await engine.create_session(config.ctx_size) as session:
        await session.sync(prompt_tokens)
        eos_token_id = await engine.eos_token_id
        for _ in range(config.warmup_tokens):
            token_id = (
                await session.sample(sampling)
                if use_sampling(config)
                else await session.argmax()
            )
            if token_id == eos_token_id:
                break
            await session.eval(token_id)
            await engine.token_text(token_id)
    return perf_counter() - start


async def run_pyds4_async_primitive(
    config: BenchmarkConfig,
) -> BenchmarkResult:
    total_start = perf_counter()
    open_start = perf_counter()
    async with pyds4.AsyncEngine(engine_options(config)) as engine:
        open_s = perf_counter() - open_start
        queue_p50_ms, queue_p95_ms = await async_queue_probe(
            engine,
            config.queue_probes,
        )

        prompt_start = perf_counter()
        prompt_tokens = await engine.encode_chat_prompt(
            system=config.system,
            prompt=config.prompt,
            think_mode=pyds4.ThinkMode.NONE,
        )
        prompt_s = perf_counter() - prompt_start
        warmup_s = await async_warmup(engine, prompt_tokens, config)
        sampling = sampling_options(config)
        eos_token_id = await engine.eos_token_id

        async with await engine.create_session(config.ctx_size) as session:
            sync_start = perf_counter()
            await session.sync(prompt_tokens)
            sync_s = perf_counter() - sync_start

            chunks: list[bytes] = []
            first_token_s: float | None = None
            async with LoopLatencyProbe(
                config.loop_probe_interval_ms
            ) as loop_probe:
                generation_start = perf_counter()
                for _ in range(config.max_new_tokens):
                    token_id = (
                        await session.sample(sampling)
                        if use_sampling(config)
                        else await session.argmax()
                    )
                    if token_id == eos_token_id:
                        break
                    await session.eval(token_id)
                    token_bytes = await engine.token_text(token_id)
                    if first_token_s is None:
                        first_token_s = perf_counter() - generation_start
                    chunks.append(token_bytes)
                generation_s = perf_counter() - generation_start

    output = b"".join(chunks)
    total_s = perf_counter() - total_start
    return BenchmarkResult(
        mode=MODE_PYDS4_ASYNC_PRIMITIVE,
        generated_tokens=len(chunks),
        output_bytes=len(output),
        output_preview=output_preview(output),
        open_s=open_s,
        prompt_s=prompt_s,
        warmup_s=warmup_s,
        sync_s=sync_s,
        generation_s=generation_s,
        total_s=total_s,
        ttft_s=first_token_s,
        tokens_per_s=tokens_per_second(len(chunks), generation_s),
        loop_latency_p95_ms=percentile(loop_probe.samples_ms, 95.0),
        loop_latency_max_ms=max(loop_probe.samples_ms, default=None),
        queue_roundtrip_p50_ms=queue_p50_ms,
        queue_roundtrip_p95_ms=queue_p95_ms,
        queue_calls_per_token=3.0,
    )


async def run_pyds4_async_next_token(
    config: BenchmarkConfig,
) -> BenchmarkResult:
    total_start = perf_counter()
    open_start = perf_counter()
    async with pyds4.AsyncEngine(engine_options(config)) as engine:
        open_s = perf_counter() - open_start
        queue_p50_ms, queue_p95_ms = await async_queue_probe(
            engine,
            config.queue_probes,
        )

        prompt_start = perf_counter()
        prompt_tokens = await engine.encode_chat_prompt(
            system=config.system,
            prompt=config.prompt,
            think_mode=pyds4.ThinkMode.NONE,
        )
        prompt_s = perf_counter() - prompt_start
        warmup_s = await async_warmup(engine, prompt_tokens, config)
        sampling = sampling_options(config) if use_sampling(config) else None

        async with await engine.create_session(config.ctx_size) as session:
            sync_start = perf_counter()
            await session.sync(prompt_tokens)
            sync_s = perf_counter() - sync_start

            chunks: list[bytes] = []
            first_token_s: float | None = None
            async with LoopLatencyProbe(
                config.loop_probe_interval_ms
            ) as loop_probe:
                generation_start = perf_counter()
                for _ in range(config.max_new_tokens):
                    step = await session.next_token(sampling, decode=True)
                    if step.is_eos:
                        break
                    if first_token_s is None:
                        first_token_s = perf_counter() - generation_start
                    if step.token_bytes is not None:
                        chunks.append(step.token_bytes)
                generation_s = perf_counter() - generation_start

    output = b"".join(chunks)
    total_s = perf_counter() - total_start
    return BenchmarkResult(
        mode=MODE_PYDS4_ASYNC_NEXT_TOKEN,
        generated_tokens=len(chunks),
        output_bytes=len(output),
        output_preview=output_preview(output),
        open_s=open_s,
        prompt_s=prompt_s,
        warmup_s=warmup_s,
        sync_s=sync_s,
        generation_s=generation_s,
        total_s=total_s,
        ttft_s=first_token_s,
        tokens_per_s=tokens_per_second(len(chunks), generation_s),
        loop_latency_p95_ms=percentile(loop_probe.samples_ms, 95.0),
        loop_latency_max_ms=max(loop_probe.samples_ms, default=None),
        queue_roundtrip_p50_ms=queue_p50_ms,
        queue_roundtrip_p95_ms=queue_p95_ms,
        queue_calls_per_token=1.0,
    )


def install_avalan_path(avalan_root: Path) -> None:
    src = avalan_root / "src"
    if src.is_dir():
        sys.path.insert(0, str(src))


def avalan_generation_settings(config: BenchmarkConfig) -> Any:
    from avalan.entities import GenerationSettings, ReasoningSettings

    return GenerationSettings(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=use_sampling(config),
        top_k=config.top_k,
        top_p=config.top_p,
        min_p=config.min_p,
        use_async_generator=True,
        reasoning=ReasoningSettings(enabled=False),
    )


def avalan_engine_settings(config: BenchmarkConfig) -> Any:
    from avalan.entities import TransformerEngineSettings

    return TransformerEngineSettings(
        backend_config={
            "ctx_size": config.ctx_size,
            "native_backend": config.backend.value,
            "n_threads": config.n_threads,
        }
    )


async def run_avalan(
    config: BenchmarkConfig, *, streaming: bool
) -> BenchmarkResult:
    install_avalan_path(config.avalan_root)
    from avalan.model.nlp.text.ds4 import Ds4Model

    mode = MODE_AVALAN_STREAM if streaming else MODE_AVALAN_NONSTREAM
    total_start = perf_counter()
    open_start = perf_counter()
    with Ds4Model(
        str(config.model_path),
        avalan_engine_settings(config),
    ) as model:
        open_s = perf_counter() - open_start
        settings = avalan_generation_settings(config)
        chunks: list[str] = []
        first_token_s: float | None = None

        async with LoopLatencyProbe(
            config.loop_probe_interval_ms
        ) as loop_probe:
            generation_start = perf_counter()
            response = await model(
                config.prompt,
                system_prompt=config.system,
                settings=settings,
            )
            prompt_s = perf_counter() - generation_start
            if streaming:
                async for chunk in response:
                    token_text = (
                        chunk.token if hasattr(chunk, "token") else str(chunk)
                    )
                    if token_text:
                        if first_token_s is None:
                            first_token_s = perf_counter() - generation_start
                        chunks.append(token_text)
                output = "".join(chunks)
                generated_tokens = response.output_token_count
            else:
                output = await response.to_str()
                generated_tokens = response.output_token_count
                if output:
                    first_token_s = None
            generation_s = perf_counter() - generation_start

    total_s = perf_counter() - total_start
    return BenchmarkResult(
        mode=mode,
        generated_tokens=generated_tokens,
        output_bytes=len(output.encode("utf-8")),
        output_preview=output_preview(output),
        open_s=open_s,
        prompt_s=prompt_s,
        warmup_s=None,
        sync_s=None,
        generation_s=generation_s,
        total_s=total_s,
        ttft_s=first_token_s,
        tokens_per_s=tokens_per_second(generated_tokens, generation_s),
        loop_latency_p95_ms=percentile(loop_probe.samples_ms, 95.0),
        loop_latency_max_ms=max(loop_probe.samples_ms, default=None),
        queue_roundtrip_p50_ms=None,
        queue_roundtrip_p95_ms=None,
        queue_calls_per_token=1.0,
    )


async def run_mode(
    mode: str,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    if mode == MODE_PYDS4_SYNC:
        return run_pyds4_sync(config)
    if mode == MODE_PYDS4_ASYNC_PRIMITIVE:
        return await run_pyds4_async_primitive(config)
    if mode == MODE_PYDS4_ASYNC_NEXT_TOKEN:
        return await run_pyds4_async_next_token(config)
    if mode == MODE_AVALAN_STREAM:
        return await run_avalan(config, streaming=True)
    if mode == MODE_AVALAN_NONSTREAM:
        return await run_avalan(config, streaming=False)
    raise ValueError(f"Unknown benchmark mode: {mode}")


def parse_modes(raw_modes: list[str], include_avalan: bool) -> list[str]:
    result: list[str] = []
    for raw in raw_modes:
        for mode in raw.split(","):
            normalized = mode.strip()
            if not normalized:
                continue
            if normalized == "all":
                result.extend(ALL_MODES if include_avalan else PYDS4_MODES)
            elif normalized == "pyds4":
                result.extend(PYDS4_MODES)
            elif normalized == "avalan":
                result.extend(AVALAN_MODES)
            elif normalized in ALL_MODES:
                result.append(normalized)
            else:
                choices = ", ".join(("all", "pyds4", "avalan", *ALL_MODES))
                raise ValueError(
                    f"Unknown --mode {normalized!r}; choices: {choices}"
                )

    deduplicated: list[str] = []
    for mode in result:
        if mode not in deduplicated:
            deduplicated.append(mode)
    return deduplicated


def format_seconds(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def format_ms(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def print_results(results: list[BenchmarkResult]) -> None:
    rows = [
        (
            "mode",
            "tok",
            "ttft_s",
            "tok/s",
            "gen_s",
            "open_s",
            "loop_p95_ms",
            "queue_p50_ms",
            "q/token",
        )
    ]
    for result in results:
        rows.append(
            (
                result.mode,
                str(result.generated_tokens),
                format_seconds(result.ttft_s),
                (
                    ""
                    if result.tokens_per_s is None
                    else f"{result.tokens_per_s:.2f}"
                ),
                format_seconds(result.generation_s),
                format_seconds(result.open_s),
                format_ms(result.loop_latency_p95_ms),
                format_ms(result.queue_roundtrip_p50_ms),
                (
                    ""
                    if result.queue_calls_per_token is None
                    else f"{result.queue_calls_per_token:.1f}"
                ),
            )
        )

    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for index, row in enumerate(rows):
        print(
            "  ".join(
                value.ljust(widths[column]) for column, value in enumerate(row)
            )
        )
        if index == 0:
            print("  ".join("-" * width for width in widths))

    print()
    for result in results:
        if result.output_preview:
            print(f"{result.mode}: {result.output_preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DS4 generation paths for pyds4 and Avalan.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--backend",
        choices=[backend.value for backend in pyds4.Backend],
        default=pyds4.__ds4_native_backend__,
    )
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--prompt",
        default="Write one short sentence about reliable software.",
    )
    parser.add_argument("--system", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-threads", type=int, default=0)
    parser.add_argument("--warmup-tokens", type=int, default=0)
    parser.add_argument("--queue-probes", type=int, default=16)
    parser.add_argument("--loop-probe-interval-ms", type=float, default=5.0)
    parser.add_argument(
        "--mode",
        action="append",
        default=None,
        help=(
            "Benchmark mode. Use comma-separated values or repeat the flag. "
            "Choices: all, pyds4, avalan, "
            + ", ".join(ALL_MODES)
        ),
    )
    parser.add_argument(
        "--include-avalan",
        action="store_true",
        help="Include Avalan modes when --mode all is selected.",
    )
    parser.add_argument(
        "--avalan-root", type=Path, default=DEFAULT_AVALAN_ROOT
    )
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BenchmarkConfig:
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    backend = pyds4.Backend(args.backend)
    if not pyds4.is_backend_available(backend.value):
        raise RuntimeError(pyds4.backend_unavailable_reason(backend.value))

    return BenchmarkConfig(
        model_path=model_path,
        backend=backend,
        ctx_size=args.ctx_size,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
        system=args.system,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        seed=args.seed,
        n_threads=args.n_threads,
        warmup_tokens=args.warmup_tokens,
        queue_probes=args.queue_probes,
        loop_probe_interval_ms=args.loop_probe_interval_ms,
        avalan_root=args.avalan_root.expanduser().resolve(),
    )


async def async_main() -> None:
    args = parse_args()
    config = build_config(args)
    modes = parse_modes(args.mode or ["all"], args.include_avalan)

    results: list[BenchmarkResult] = []
    for mode in modes:
        print(f"running {mode}...", flush=True)
        results.append(await run_mode(mode, config))

    print_results(results)
    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps([asdict(result) for result in results], indent=2)
            + "\n",
            encoding="utf-8",
        )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
