from __future__ import annotations

import argparse
import asyncio
import codecs
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pyds4

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / ".local" / "ds4" / "ds4flash.gguf"
)


async def stream_text_bytes(
    *,
    model_path: Path,
    prompt: str,
    system: str | None,
    backend: pyds4.Backend,
    ctx_size: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    seed: int | None,
    n_threads: int,
) -> AsyncIterator[bytes]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    if not pyds4.is_backend_available(backend.value):
        raise RuntimeError(pyds4.backend_unavailable_reason(backend.value))

    options = pyds4.EngineOptions(
        model_path=str(model_path),
        backend=backend,
        n_threads=n_threads,
    )
    sampling = (
        pyds4.SamplingOptions(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            seed=seed,
        )
        if temperature > 0.0
        else None
    )

    async with pyds4.AsyncEngine(options) as engine:
        think_mode = pyds4.think_mode_for_context(
            pyds4.ThinkMode.NONE,
            ctx_size,
        )
        prompt_tokens = await engine.encode_chat_prompt(
            system=system,
            prompt=prompt,
            think_mode=think_mode,
        )

        async with await engine.create_session(ctx_size) as session:
            await session.sync(prompt_tokens)
            for _ in range(max_new_tokens):
                step = await session.next_token(sampling, decode=True)
                if step.is_eos:
                    break
                if step.token_bytes:
                    yield step.token_bytes


async def stream_text(**kwargs: object) -> AsyncIterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    async for token_bytes in stream_text_bytes(**kwargs):
        text = decoder.decode(token_bytes, final=False)
        if text:
            yield text

    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail


async def generate_text(**kwargs: object) -> str:
    chunks = [chunk async for chunk in stream_text_bytes(**kwargs)]
    return b"".join(chunks).decode("utf-8", errors="replace")


async def stream_text_to_stdout(**kwargs: object) -> None:
    async for chunk in stream_text(**kwargs):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a short response with pyds4.AsyncEngine.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Answer with one short word: name one primary color.",
    )
    parser.add_argument("--system", default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--backend",
        choices=[backend.value for backend in pyds4.Backend],
        default=pyds4.__ds4_native_backend__,
    )
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-threads", type=int, default=0)
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print each generated token as soon as it is available.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    kwargs = {
        "model_path": args.model.expanduser(),
        "prompt": args.prompt,
        "system": args.system,
        "backend": pyds4.Backend(args.backend),
        "ctx_size": args.ctx_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "seed": args.seed,
        "n_threads": args.n_threads,
    }

    if args.stream:
        await stream_text_to_stdout(**kwargs)
    else:
        output = await generate_text(**kwargs)
        print(output)


if __name__ == "__main__":
    asyncio.run(main())
