from __future__ import annotations

import argparse
import asyncio
import codecs
import sys
import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pyds4

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / ".local" / "ds4" / "ds4flash.gguf"
)


def stream_text_bytes(
    *,
    model_path: Path,
    prompt: str,
    system: str | None,
    backend: pyds4.Backend,
    ctx_size: int,
    max_new_tokens: int,
    temperature: float,
    n_threads: int,
) -> Iterator[bytes]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    if not pyds4.is_backend_available(backend.value):
        raise RuntimeError(pyds4.backend_unavailable_reason(backend.value))

    options = pyds4.EngineOptions(
        model_path=str(model_path),
        backend=backend,
        n_threads=n_threads,
    )
    sampling = pyds4.SamplingOptions(
        temperature=temperature,
        top_k=40,
        top_p=0.95,
        min_p=0.0,
    )

    with pyds4.Engine(options) as engine:
        think_mode = pyds4.think_mode_for_context(
            pyds4.ThinkMode.NONE,
            ctx_size,
        )
        prompt_tokens = engine.encode_chat_prompt(
            system=system,
            prompt=prompt,
            think_mode=think_mode,
        )

        with engine.create_session(ctx_size) as session:
            session.sync(prompt_tokens)
            for _ in range(max_new_tokens):
                token_id = (
                    session.argmax()
                    if temperature == 0
                    else session.sample(sampling)
                )
                if token_id == engine.eos_token_id:
                    break

                session.eval(token_id)
                yield engine.token_text(token_id)


def generate_text(**kwargs: object) -> str:
    chunks = stream_text_bytes(**kwargs)
    return b"".join(chunks).decode("utf-8", errors="replace")


async def generate_text_async(**kwargs: object) -> str:
    return await asyncio.to_thread(generate_text, **kwargs)


async def stream_text_async(**kwargs: object) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()

    def worker() -> None:
        try:
            for token_bytes in stream_text_bytes(**kwargs):
                loop.call_soon_threadsafe(queue.put_nowait, token_bytes)
        except Exception as error:
            loop.call_soon_threadsafe(queue.put_nowait, error)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    while True:
        item = await queue.get()
        if item is None:
            tail = decoder.decode(b"", final=True)
            if tail:
                yield tail
            return
        if isinstance(item, Exception):
            raise item

        text = decoder.decode(item, final=False)
        if text:
            yield text


def stream_text_to_stdout(**kwargs: object) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    for token_bytes in stream_text_bytes(**kwargs):
        text = decoder.decode(token_bytes, final=False)
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()

    tail = decoder.decode(b"", final=True)
    if tail:
        sys.stdout.write(tail)
    sys.stdout.write("\n")
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a short response with pyds4.",
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
    parser.add_argument("--n-threads", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("sync", "async"),
        default="sync",
        help="Use direct sync calls or run them through asyncio.to_thread.",
    )
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
        "n_threads": args.n_threads,
    }
    if args.stream and args.mode == "async":
        async for chunk in stream_text_async(**kwargs):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
    elif args.stream:
        stream_text_to_stdout(**kwargs)
    elif args.mode == "async":
        output = await generate_text_async(**kwargs)
        print(output)
    else:
        output = generate_text(**kwargs)
        print(output)


if __name__ == "__main__":
    asyncio.run(main())
