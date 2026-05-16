from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from queue import Queue
from threading import Event, Thread
from typing import Any, Generic, TypeVar

from .errors import Ds4Cancelled, Ds4ContextError, Ds4LoadError
from .native import Engine as _SyncEngine
from .native import Session as _SyncSession
from .types import (
    EngineOptions,
    GenerationStep,
    ProgressEvent,
    SamplingOptions,
    ThinkMode,
    TokenScore,
)

_T = TypeVar("_T")


def _set_future_result(
    future: asyncio.Future[_T],
    result: _T,
) -> None:
    if not future.cancelled():
        future.set_result(result)


def _set_future_exception(
    future: asyncio.Future[_T],
    error: BaseException,
) -> None:
    if not future.cancelled():
        future.set_exception(error)


def _validate_bool_option(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")


@dataclass(slots=True)
class _WorkerJob(Generic[_T]):
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[_T]
    func: Callable[[], _T]
    started: Event
    finished: Event
    on_cancelled_after_run: Callable[[], None] | None = None


class _OwnerWorker:
    __slots__ = ("_jobs", "_thread", "_stopped")

    def __init__(self) -> None:
        self._jobs: Queue[_WorkerJob[Any] | None] = Queue()
        self._thread = Thread(
            target=self._run,
            name="pyds4-async-owner",
            daemon=True,
        )
        self._stopped = False
        self._thread.start()

    async def call(
        self,
        func: Callable[[], _T],
        *,
        on_cancelled_after_start: Callable[[], None] | None = None,
        on_cancelled_after_run: Callable[[], None] | None = None,
    ) -> _T:
        if self._stopped:
            raise Ds4LoadError("DS4 async engine is closed.")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[_T] = loop.create_future()
        started = Event()
        finished = Event()
        self._jobs.put(
            _WorkerJob(
                loop=loop,
                future=future,
                func=func,
                started=started,
                finished=finished,
                on_cancelled_after_run=on_cancelled_after_run,
            )
        )
        try:
            return await future
        except asyncio.CancelledError:
            if (
                on_cancelled_after_start is not None
                and started.is_set()
                and not finished.is_set()
            ):
                on_cancelled_after_start()
            raise

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._jobs.put(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            if job.future.cancelled():
                continue

            try:
                job.started.set()
                result = job.func()
            except BaseException as error:
                if job.future.cancelled():
                    self._run_cancel_cleanup(job)
                else:
                    job.loop.call_soon_threadsafe(
                        _set_future_exception,
                        job.future,
                        error,
                    )
            else:
                if job.future.cancelled():
                    self._run_cancel_cleanup(job)
                else:
                    job.loop.call_soon_threadsafe(
                        _set_future_result,
                        job.future,
                        result,
                    )
            finally:
                job.finished.set()

    def _run_cancel_cleanup(self, job: _WorkerJob[Any]) -> None:
        cleanup = job.on_cancelled_after_run
        if cleanup is None:
            return
        try:
            cleanup()
        except BaseException as error:
            job.loop.call_soon_threadsafe(
                job.loop.call_exception_handler,
                {
                    "message": "pyds4 async cancellation cleanup failed",
                    "exception": error,
                },
            )


class AsyncEngine:
    """Async facade that owns a synchronous DS4 engine on one worker thread."""

    __slots__ = (
        "_close_task",
        "_closed",
        "_engine",
        "_open_lock",
        "_options",
        "_worker",
    )

    def __init__(self, options: EngineOptions) -> None:
        self._options = options
        self._worker: _OwnerWorker | None = None
        self._engine: _SyncEngine | None = None
        self._open_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    @classmethod
    async def open(cls, options: EngineOptions) -> "AsyncEngine":
        engine = cls(options)
        await engine._ensure_open()
        return engine

    @property
    def closed(self) -> bool:
        engine = self._engine
        return self._closed or (engine is not None and engine.closed)

    @property
    def routed_quant_bits(self) -> Awaitable[int]:
        return self._call_engine(lambda engine: engine.routed_quant_bits)

    @property
    def has_mtp(self) -> Awaitable[bool]:
        return self._call_engine(lambda engine: engine.has_mtp)

    @property
    def mtp_draft_tokens(self) -> Awaitable[int]:
        return self._call_engine(lambda engine: engine.mtp_draft_tokens)

    @property
    def eos_token_id(self) -> Awaitable[int]:
        return self._call_engine(lambda engine: engine.eos_token_id)

    async def __aenter__(self) -> "AsyncEngine":
        await self._ensure_open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def create_session(self, ctx_size: int) -> "AsyncSession":
        session = await self._call_engine(
            lambda engine: engine.create_session(ctx_size)
        )
        return AsyncSession(self, session)

    async def token_text(self, token_id: int) -> bytes:
        return await self._call_engine(
            lambda engine: engine.token_text(token_id)
        )

    async def tokenize_text(self, text: str) -> list[int]:
        return await self._call_engine(
            lambda engine: engine.tokenize_text(text)
        )

    async def tokenize_rendered_chat(self, text: str) -> list[int]:
        return await self._call_engine(
            lambda engine: engine.tokenize_rendered_chat(text)
        )

    async def chat_begin(self) -> list[int]:
        return await self._call_engine(lambda engine: engine.chat_begin())

    async def chat_append_message(
        self,
        tokens: list[int],
        role: str,
        content: str,
    ) -> None:
        working_tokens: object = (
            list(tokens) if isinstance(tokens, list) else tokens
        )

        await self._call_engine(
            lambda engine: engine.chat_append_message(
                working_tokens,  # type: ignore[arg-type]
                role,
                content,
            )
        )
        if isinstance(working_tokens, list):
            tokens[:] = working_tokens

    async def chat_append_assistant_prefix(
        self,
        tokens: list[int],
        think_mode: ThinkMode | str,
    ) -> None:
        working_tokens: object = (
            list(tokens) if isinstance(tokens, list) else tokens
        )

        await self._call_engine(
            lambda engine: engine.chat_append_assistant_prefix(
                working_tokens,  # type: ignore[arg-type]
                think_mode,
            )
        )
        if isinstance(working_tokens, list):
            tokens[:] = working_tokens

    async def encode_chat_prompt(
        self,
        system: str | None,
        prompt: str,
        think_mode: ThinkMode | str,
    ) -> list[int]:
        return await self._call_engine(
            lambda engine: engine.encode_chat_prompt(
                system, prompt, think_mode
            )
        )

    async def _ensure_open(self) -> None:
        if self._engine is not None:
            return
        if self._closed:
            raise Ds4LoadError("DS4 async engine is closed.")

        async with self._open_lock:
            if self._engine is not None:
                return
            if self._closed:
                raise Ds4LoadError("DS4 async engine is closed.")
            if self._worker is None:
                self._worker = _OwnerWorker()
            self._engine = await self._worker.call(
                lambda: _SyncEngine(self._options)
            )

    async def _call_engine(
        self,
        func: Callable[[_SyncEngine], _T],
        *,
        on_cancelled_after_start: Callable[[], None] | None = None,
        on_cancelled_after_run: Callable[[], None] | None = None,
    ) -> _T:
        await self._ensure_open()
        engine = self._engine
        worker = self._worker
        if engine is None or worker is None:
            raise Ds4LoadError("DS4 async engine is closed.")
        return await worker.call(
            lambda: func(engine),
            on_cancelled_after_start=on_cancelled_after_start,
            on_cancelled_after_run=on_cancelled_after_run,
        )

    async def _close(self) -> None:
        self._closed = True
        engine = self._engine
        worker = self._worker
        self._engine = None
        self._worker = None

        try:
            if engine is not None and worker is not None:
                await worker.call(engine.close)
        finally:
            if worker is not None:
                worker.stop()


class AsyncSession:
    """Async facade for a DS4 session owned by an ``AsyncEngine``."""

    __slots__ = (
        "_closed",
        "_engine",
        "_poisoned",
        "_progress_loop",
        "_progress_queue",
        "_progress_reader_fd",
        "_progress_reader_installed",
        "_progress_wakeup_fd",
        "_session",
    )

    def __init__(
        self,
        engine: AsyncEngine,
        session: _SyncSession,
    ) -> None:
        self._engine = engine
        self._session: _SyncSession | None = session
        self._closed = False
        self._poisoned = False
        self._progress_queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()
        self._progress_loop: asyncio.AbstractEventLoop | None
        try:
            self._progress_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._progress_loop = None
        self._progress_reader_fd = -1
        self._progress_wakeup_fd = -1
        self._progress_reader_installed = False
        self._install_progress_bridge()

    @property
    def closed(self) -> bool:
        session = self._session
        return (
            self._closed or self._poisoned or session is None or session.closed
        )

    @property
    def pos(self) -> Awaitable[int]:
        return self._call_session(lambda session: session.pos)

    @property
    def ctx(self) -> Awaitable[int]:
        return self._call_session(lambda session: session.ctx)

    @property
    def tokens(self) -> Awaitable[list[int]]:
        return self._call_session(lambda session: session.tokens)

    @property
    def progress(self) -> asyncio.Queue[ProgressEvent]:
        return self._progress_queue

    async def __aenter__(self) -> "AsyncSession":
        self._require_session()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        session = self._session
        self._close_progress_bridge()
        if session is None:
            self._closed = True
            return

        self._closed = True
        if session.closed or self._engine.closed:
            self._session = None
            return

        try:
            await asyncio.shield(
                self._engine._call_engine(lambda _: session.close())
            )
        finally:
            self._session = None

    async def sync(self, prompt_tokens: list[int] | tuple[int, ...]) -> None:
        prompt_snapshot: list[int] | tuple[int, ...] = (
            tuple(prompt_tokens)
            if isinstance(prompt_tokens, list)
            else prompt_tokens
        )
        await self._call_session(
            lambda session: session.sync(prompt_snapshot),
            mutating=True,
        )

    async def eval(self, token_id: int) -> None:
        await self._call_session(
            lambda session: session.eval(token_id),
            mutating=True,
        )

    async def argmax(self) -> int:
        return await self._call_session(lambda session: session.argmax())

    async def argmax_excluding(self, token_id: int) -> int:
        return await self._call_session(
            lambda session: session.argmax_excluding(token_id)
        )

    async def sample(self, options: SamplingOptions) -> int:
        return await self._call_session(
            lambda session: session.sample(options),
            mutating=True,
        )

    async def token_logprob(self, token_id: int) -> float:
        return await self._call_session(
            lambda session: session.token_logprob(token_id)
        )

    async def top_logprobs(self, k: int) -> list[TokenScore]:
        return await self._call_session(
            lambda session: session.top_logprobs(k)
        )

    async def eval_speculative_argmax(
        self,
        first_token: int,
        max_tokens: int,
        eos_token_id: int,
    ) -> list[int]:
        return await self._call_session(
            lambda session: session.eval_speculative_argmax(
                first_token,
                max_tokens,
                eos_token_id,
            ),
            mutating=True,
        )

    async def next_token(
        self,
        options: SamplingOptions | None = None,
        *,
        advance: bool = True,
        decode: bool = False,
        stop_on_eos: bool = True,
        exclude_token_id: int | None = None,
    ) -> GenerationStep:
        _validate_bool_option("advance", advance)
        _validate_bool_option("decode", decode)
        _validate_bool_option("stop_on_eos", stop_on_eos)
        if options is not None and exclude_token_id is not None:
            raise ValueError(
                "exclude_token_id cannot be used with sampling options."
            )

        session = self._require_session()

        def run_step(engine: _SyncEngine) -> GenerationStep:
            if options is None:
                if exclude_token_id is None:
                    token_id = session.argmax()
                else:
                    token_id = session.argmax_excluding(exclude_token_id)
            else:
                token_id = session.sample(options)

            is_eos = token_id == engine.eos_token_id
            should_advance = advance and not (stop_on_eos and is_eos)
            if should_advance:
                session.eval(token_id)

            token_bytes = None
            if decode and not (stop_on_eos and is_eos):
                token_bytes = engine.token_text(token_id)

            return GenerationStep(
                token_id=token_id,
                is_eos=is_eos,
                advanced=should_advance,
                token_bytes=token_bytes,
            )

        try:
            if not advance and options is None:
                return await self._engine._call_engine(run_step)

            return await self._engine._call_engine(
                run_step,
                on_cancelled_after_start=self._poison_after_cancel,
                on_cancelled_after_run=session.close,
            )
        finally:
            self._drain_progress_events_to_queue()

    async def rewind(self, pos: int) -> None:
        await self._call_session(
            lambda session: session.rewind(pos),
            mutating=True,
        )

    async def save_snapshot(self) -> bytes:
        return await self._call_session(
            lambda session: session.save_snapshot()
        )

    async def load_snapshot(self, snapshot: bytes) -> None:
        await self._call_session(
            lambda session: session.load_snapshot(snapshot),
            mutating=True,
        )

    async def save_payload(self) -> bytes:
        return await self._call_session(lambda session: session.save_payload())

    async def load_payload(self, payload: bytes) -> None:
        await self._call_session(
            lambda session: session.load_payload(payload),
            mutating=True,
        )

    async def invalidate(self) -> None:
        await self._call_session(
            lambda session: session.invalidate(),
            mutating=True,
        )

    async def _call_session(
        self,
        func: Callable[[_SyncSession], _T],
        *,
        mutating: bool = False,
    ) -> _T:
        session = self._require_session()
        try:
            if not mutating:
                return await self._engine._call_engine(lambda _: func(session))

            return await self._engine._call_engine(
                lambda _: func(session),
                on_cancelled_after_start=self._poison_after_cancel,
                on_cancelled_after_run=session.close,
            )
        finally:
            self._drain_progress_events_to_queue()

    def _require_session(self) -> _SyncSession:
        if self._poisoned:
            raise Ds4Cancelled(
                "DS4 async session was cancelled during a mutating operation."
            )
        session = self._session
        if self._closed or session is None or session.closed:
            raise Ds4ContextError("DS4 async session is closed.")
        return session

    def _poison_after_cancel(self) -> None:
        self._poisoned = True

    def _install_progress_bridge(self) -> None:
        session = self._session
        loop = self._progress_loop
        if session is None or loop is None or os.name != "posix":
            return
        if not hasattr(session, "_set_progress_wakeup_fd") or not hasattr(
            session,
            "_drain_progress_events",
        ):
            return

        reader_fd = -1
        wakeup_fd = -1
        try:
            reader_fd, wakeup_fd = os.pipe()
            os.set_blocking(reader_fd, False)
            os.set_blocking(wakeup_fd, False)
            loop.add_reader(reader_fd, self._drain_progress_notifications)
            self._progress_reader_fd = reader_fd
            self._progress_wakeup_fd = wakeup_fd
            self._progress_reader_installed = True
            session._set_progress_wakeup_fd(wakeup_fd)
        except Exception as error:
            loop.call_exception_handler(
                {
                    "message": "pyds4 async progress bridge setup failed",
                    "exception": error,
                }
            )
            if self._progress_reader_installed:
                loop.remove_reader(reader_fd)
                self._progress_reader_installed = False
            for fd in (reader_fd, wakeup_fd):
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            self._progress_reader_fd = -1
            self._progress_wakeup_fd = -1

    def _close_progress_bridge(self) -> None:
        self._drain_progress_events_to_queue()
        session = self._session
        if session is not None and hasattr(session, "_set_progress_wakeup_fd"):
            try:
                session._set_progress_wakeup_fd(-1)
            except Exception:
                pass

        loop = self._progress_loop
        if (
            loop is not None
            and self._progress_reader_installed
            and self._progress_reader_fd >= 0
        ):
            loop.remove_reader(self._progress_reader_fd)
        self._progress_reader_installed = False

        for name in ("_progress_reader_fd", "_progress_wakeup_fd"):
            fd = getattr(self, name)
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, name, -1)

    def _drain_progress_notifications(self) -> None:
        fd = self._progress_reader_fd
        if fd < 0:
            return

        while True:
            try:
                if not os.read(fd, 4096):
                    break
            except BlockingIOError:
                break
            except OSError:
                return

        self._drain_progress_events_to_queue()

    def _drain_progress_events_to_queue(self) -> None:
        session = self._session
        if session is None or not hasattr(session, "_drain_progress_events"):
            return
        try:
            events = session._drain_progress_events()
        except Exception as error:
            loop = self._progress_loop
            if loop is not None:
                loop.call_exception_handler(
                    {
                        "message": "pyds4 async progress drain failed",
                        "exception": error,
                    }
                )
            return
        for event in events:
            self._progress_queue.put_nowait(event)


__all__ = ["AsyncEngine", "AsyncSession", "GenerationStep", "ProgressEvent"]
