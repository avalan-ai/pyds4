from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from time import time
from typing import Any

from ._metadata import C_INT_MAX, DS4_COMMIT
from ._version import __ds4_native_backend__, __version__

DS4_KV_CACHE_VERSION = 1


def _validate_str(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value


def _validate_nonempty_str(name: str, value: str) -> str:
    value = _validate_str(name, value)
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return value


def _validate_optional_str(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _validate_str(name, value)


def _validate_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _validate_int(
    name: str,
    value: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")
    return value


def _validate_float(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}.")
    return result


def _validate_payload_file(value: str) -> str:
    value = _validate_nonempty_str("payload_file", value)
    if Path(value).name != value:
        raise ValueError("payload_file must be a file name.")
    return value


def _validate_optional_size_budget(
    value: int | None,
) -> int | None:
    if value is None:
        return None
    return _validate_int("size_budget_bytes", value, minimum=0)


def _validate_bytes(name: str, value: object) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes.")
    return value


def _normalize_backend(value: object | None) -> str:
    if value is None:
        return __ds4_native_backend__
    backend = getattr(value, "value", value)
    if not isinstance(backend, str):
        raise TypeError("backend must be a string.")
    return _validate_nonempty_str("backend", backend)


def _normalize_token_ids(prompt_tokens: Iterable[int]) -> tuple[int, ...]:
    if isinstance(prompt_tokens, (bytes, str)):
        raise TypeError("prompt_tokens must be an iterable of token ids.")
    try:
        tokens = tuple(prompt_tokens)
    except TypeError as error:
        raise TypeError(
            "prompt_tokens must be an iterable of token ids."
        ) from error
    for token in tokens:
        _validate_int(
            "prompt_tokens item",
            token,
            minimum=0,
            maximum=C_INT_MAX,
        )
    return tokens


def _token_digest(prompt_tokens: tuple[int, ...]) -> str:
    payload = json.dumps(
        prompt_tokens,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _key_digest(
    *,
    model_namespace: str,
    ctx_size: int,
    token_count: int,
    token_sha256: str,
) -> str:
    payload = {
        "ctx_size": ctx_size,
        "model_namespace": model_namespace,
        "token_count": token_count,
        "token_sha256": token_sha256,
    }
    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _method(target: object, name: str) -> Any:
    method = getattr(target, name, None)
    if not callable(method):
        raise TypeError(f"session must provide a callable {name}() method.")
    return method


def _call_sync(method: Any, *args: object) -> object:
    result = method(*args)
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if callable(close):
            close()
        raise TypeError(
            "session method returned an awaitable; use arestore() or "
            "astore() with async sessions."
        )
    return result


async def _call_async(method: Any, *args: object) -> object:
    result = method(*args)
    if inspect.isawaitable(result):
        return await result
    return result


def _error_text(error: BaseException) -> str:
    message = str(error)
    if message:
        return message
    return type(error).__name__


@dataclass(frozen=True, slots=True)
class Ds4KvCacheEntry:
    """Describe the deterministic file locations for one cache key."""

    key: str
    metadata_path: Path
    payload_path: Path
    token_sha256: str

    def __post_init__(self) -> None:
        _validate_nonempty_str("key", self.key)
        object.__setattr__(self, "metadata_path", Path(self.metadata_path))
        object.__setattr__(self, "payload_path", Path(self.payload_path))
        _validate_nonempty_str("token_sha256", self.token_sha256)


@dataclass(frozen=True, slots=True)
class Ds4KvCacheRestoreResult:
    """Describe a cache restore attempt for one prompt prefix."""

    status: str
    entry: Ds4KvCacheEntry
    metadata: "Ds4KvCacheMetadata | None" = None
    restored: bool = False
    synced: bool = False
    error: str | None = None
    warning: str | None = None

    def __post_init__(self) -> None:
        _validate_nonempty_str("status", self.status)
        if not isinstance(self.entry, Ds4KvCacheEntry):
            raise TypeError("entry must be a Ds4KvCacheEntry instance.")
        if self.metadata is not None and not isinstance(
            self.metadata,
            Ds4KvCacheMetadata,
        ):
            raise TypeError(
                "metadata must be a Ds4KvCacheMetadata instance or None."
            )
        _validate_bool("restored", self.restored)
        _validate_bool("synced", self.synced)
        _validate_optional_str("error", self.error)
        _validate_optional_str("warning", self.warning)


@dataclass(frozen=True, slots=True)
class Ds4KvCacheStoreResult:
    """Describe a cache store attempt for one prompt prefix."""

    status: str
    entry: Ds4KvCacheEntry
    metadata: "Ds4KvCacheMetadata | None" = None
    stored: bool = False
    evicted_entries: tuple[Ds4KvCacheEntry, ...] = ()
    error: str | None = None
    warning: str | None = None

    def __post_init__(self) -> None:
        _validate_nonempty_str("status", self.status)
        if not isinstance(self.entry, Ds4KvCacheEntry):
            raise TypeError("entry must be a Ds4KvCacheEntry instance.")
        if self.metadata is not None and not isinstance(
            self.metadata,
            Ds4KvCacheMetadata,
        ):
            raise TypeError(
                "metadata must be a Ds4KvCacheMetadata instance or None."
            )
        _validate_bool("stored", self.stored)
        for entry in self.evicted_entries:
            if not isinstance(entry, Ds4KvCacheEntry):
                raise TypeError(
                    "evicted_entries items must be Ds4KvCacheEntry objects."
                )
        _validate_optional_str("error", self.error)
        _validate_optional_str("warning", self.warning)


@dataclass(frozen=True, slots=True)
class Ds4KvCacheMetadata:
    """Store portable DS4 disk KV cache entry metadata."""

    version: int
    key: str
    model_namespace: str
    pyds4_version: str
    ds4_commit: str
    backend: str
    ctx_size: int
    token_count: int
    token_sha256: str
    payload_file: str
    payload_size: int = 0
    rendered_prompt: str | None = None
    hit_count: int = 0
    created_at: float = 0.0
    accessed_at: float = 0.0

    def __post_init__(self) -> None:
        _validate_int("version", self.version, minimum=1)
        _validate_nonempty_str("key", self.key)
        _validate_nonempty_str("model_namespace", self.model_namespace)
        _validate_nonempty_str("pyds4_version", self.pyds4_version)
        _validate_nonempty_str("ds4_commit", self.ds4_commit)
        _validate_nonempty_str("backend", self.backend)
        _validate_int("ctx_size", self.ctx_size, minimum=1, maximum=C_INT_MAX)
        _validate_int("token_count", self.token_count, minimum=0)
        _validate_nonempty_str("token_sha256", self.token_sha256)
        object.__setattr__(
            self,
            "payload_file",
            _validate_payload_file(self.payload_file),
        )
        _validate_int("payload_size", self.payload_size, minimum=0)
        _validate_optional_str("rendered_prompt", self.rendered_prompt)
        _validate_int("hit_count", self.hit_count, minimum=0)
        _validate_float("created_at", self.created_at, minimum=0.0)
        _validate_float("accessed_at", self.accessed_at, minimum=0.0)

    @classmethod
    def from_json_dict(
        cls,
        value: Mapping[str, object],
    ) -> "Ds4KvCacheMetadata":
        """Return validated metadata loaded from a JSON object."""
        if not isinstance(value, Mapping):
            raise TypeError("metadata must be a JSON object.")

        rendered_prompt = value.get("rendered_prompt")
        if rendered_prompt is not None and not isinstance(
            rendered_prompt,
            str,
        ):
            raise TypeError("rendered_prompt must be a string or null.")

        return cls(
            version=_metadata_int(value, "version"),
            key=_metadata_str(value, "key"),
            model_namespace=_metadata_str(value, "model_namespace"),
            pyds4_version=_metadata_str(value, "pyds4_version"),
            ds4_commit=_metadata_str(value, "ds4_commit"),
            backend=_metadata_str(value, "backend"),
            ctx_size=_metadata_int(value, "ctx_size"),
            token_count=_metadata_int(value, "token_count"),
            token_sha256=_metadata_str(value, "token_sha256"),
            payload_file=_metadata_str(value, "payload_file"),
            payload_size=_metadata_int(value, "payload_size"),
            rendered_prompt=rendered_prompt,
            hit_count=_metadata_int(value, "hit_count"),
            created_at=_metadata_float(value, "created_at"),
            accessed_at=_metadata_float(value, "accessed_at"),
        )

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable metadata object."""
        return {
            "accessed_at": self.accessed_at,
            "backend": self.backend,
            "created_at": self.created_at,
            "ctx_size": self.ctx_size,
            "ds4_commit": self.ds4_commit,
            "hit_count": self.hit_count,
            "key": self.key,
            "model_namespace": self.model_namespace,
            "payload_file": self.payload_file,
            "payload_size": self.payload_size,
            "pyds4_version": self.pyds4_version,
            "rendered_prompt": self.rendered_prompt,
            "token_count": self.token_count,
            "token_sha256": self.token_sha256,
            "version": self.version,
        }


class Ds4DiskKvCache:
    """Manage deterministic DS4 payload cache keys and metadata files."""

    __slots__ = (
        "_backend",
        "_cache_version",
        "_directory",
        "_ds4_commit",
        "_model_namespace",
        "_pyds4_version",
    )

    def __init__(
        self,
        directory: str | Path,
        model_namespace: str,
        *,
        backend: object | None = None,
        cache_version: int = DS4_KV_CACHE_VERSION,
        pyds4_version: str = __version__,
        ds4_commit: str = DS4_COMMIT,
    ) -> None:
        self._directory = Path(directory)
        self._model_namespace = _validate_nonempty_str(
            "model_namespace",
            model_namespace,
        )
        self._backend = _normalize_backend(backend)
        self._cache_version = _validate_int(
            "cache_version",
            cache_version,
            minimum=1,
        )
        self._pyds4_version = _validate_nonempty_str(
            "pyds4_version",
            pyds4_version,
        )
        self._ds4_commit = _validate_nonempty_str("ds4_commit", ds4_commit)

    @property
    def directory(self) -> Path:
        """Return the cache directory."""
        return self._directory

    @property
    def model_namespace(self) -> str:
        """Return the model namespace used for key derivation."""
        return self._model_namespace

    @property
    def backend(self) -> str:
        """Return the backend recorded in metadata."""
        return self._backend

    def entry_for(
        self,
        prompt_tokens: Iterable[int],
        ctx_size: int,
    ) -> Ds4KvCacheEntry:
        """Return deterministic paths for a token-prefix cache entry."""
        tokens = _normalize_token_ids(prompt_tokens)
        ctx = _validate_int("ctx_size", ctx_size, minimum=1, maximum=C_INT_MAX)
        return self._entry_for_tokens(tokens, ctx)

    def _entry_for_tokens(
        self,
        prompt_tokens: tuple[int, ...],
        ctx_size: int,
    ) -> Ds4KvCacheEntry:
        token_sha256 = _token_digest(prompt_tokens)
        key = _key_digest(
            model_namespace=self._model_namespace,
            ctx_size=ctx_size,
            token_count=len(prompt_tokens),
            token_sha256=token_sha256,
        )
        return Ds4KvCacheEntry(
            key=key,
            metadata_path=self._directory / f"{key}.json",
            payload_path=self._directory / f"{key}.payload",
            token_sha256=token_sha256,
        )

    def metadata_for(
        self,
        prompt_tokens: Iterable[int],
        ctx_size: int,
        *,
        rendered_prompt: str | None = None,
        payload_size: int = 0,
        created_at: float | None = None,
        accessed_at: float | None = None,
        hit_count: int = 0,
    ) -> Ds4KvCacheMetadata:
        """Return metadata for a cache entry without reading payload bytes."""
        tokens = _normalize_token_ids(prompt_tokens)
        ctx = _validate_int("ctx_size", ctx_size, minimum=1, maximum=C_INT_MAX)
        entry = self._entry_for_tokens(tokens, ctx)
        now = time()
        created = now if created_at is None else created_at
        accessed = created if accessed_at is None else accessed_at
        return Ds4KvCacheMetadata(
            version=self._cache_version,
            key=entry.key,
            model_namespace=self._model_namespace,
            pyds4_version=self._pyds4_version,
            ds4_commit=self._ds4_commit,
            backend=self._backend,
            ctx_size=ctx,
            token_count=len(tokens),
            token_sha256=entry.token_sha256,
            payload_file=entry.payload_path.name,
            payload_size=payload_size,
            rendered_prompt=rendered_prompt,
            hit_count=hit_count,
            created_at=created,
            accessed_at=accessed,
        )

    def metadata_matches(
        self,
        metadata: Ds4KvCacheMetadata,
        prompt_tokens: Iterable[int],
        ctx_size: int,
    ) -> bool:
        """Return whether metadata matches this cache and token prefix."""
        if not isinstance(metadata, Ds4KvCacheMetadata):
            raise TypeError("metadata must be a Ds4KvCacheMetadata instance.")

        tokens = _normalize_token_ids(prompt_tokens)
        entry = self.entry_for(tokens, ctx_size)
        return (
            metadata.version == self._cache_version
            and metadata.key == entry.key
            and metadata.model_namespace == self._model_namespace
            and metadata.pyds4_version == self._pyds4_version
            and metadata.ds4_commit == self._ds4_commit
            and metadata.backend == self._backend
            and metadata.ctx_size == ctx_size
            and metadata.token_count == len(tokens)
            and metadata.token_sha256 == entry.token_sha256
            and metadata.payload_file == entry.payload_path.name
        )

    def read_metadata(
        self,
        entry_or_path: Ds4KvCacheEntry | str | Path,
    ) -> Ds4KvCacheMetadata | None:
        """Read and validate metadata, returning ``None`` for bad entries."""
        path = (
            entry_or_path.metadata_path
            if isinstance(entry_or_path, Ds4KvCacheEntry)
            else Path(entry_or_path)
        )
        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return None
            return Ds4KvCacheMetadata.from_json_dict(raw)
        except (
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None

    def write_metadata(self, metadata: Ds4KvCacheMetadata) -> Path:
        """Write metadata JSON and return the metadata path."""
        if not isinstance(metadata, Ds4KvCacheMetadata):
            raise TypeError("metadata must be a Ds4KvCacheMetadata instance.")

        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{metadata.key}.json"
        path.write_text(
            json.dumps(metadata.to_json_dict(), sort_keys=True),
            encoding="utf-8",
        )
        return path

    def restore(
        self,
        session: object,
        prompt_tokens: Iterable[int],
        ctx_size: int,
        *,
        enabled: bool = True,
        sync_on_miss: bool = True,
    ) -> Ds4KvCacheRestoreResult:
        """Restore a session payload or synchronize live tokens on a miss."""
        _validate_bool("enabled", enabled)
        _validate_bool("sync_on_miss", sync_on_miss)
        tokens = _normalize_token_ids(prompt_tokens)
        ctx = _validate_int("ctx_size", ctx_size, minimum=1, maximum=C_INT_MAX)
        entry = self._entry_for_tokens(tokens, ctx)

        if not enabled:
            synced = self._sync_on_miss(session, tokens, sync_on_miss)
            return Ds4KvCacheRestoreResult(
                status="disabled",
                entry=entry,
                synced=synced,
            )

        metadata = self.read_metadata(entry)
        miss_error: str | None = None
        if metadata is not None and self.metadata_matches(
            metadata,
            tokens,
            ctx,
        ):
            payload_path = self._directory / metadata.payload_file
            try:
                payload = payload_path.read_bytes()
                if len(payload) != metadata.payload_size:
                    raise ValueError(
                        "cache payload size does not match metadata."
                    )
                _call_sync(_method(session, "load_payload"), payload)
            except Exception as error:
                miss_error = _error_text(error)
            else:
                restored_metadata = replace(
                    metadata,
                    hit_count=metadata.hit_count + 1,
                    accessed_at=time(),
                    payload_size=len(payload),
                )
                warning = self._try_write_metadata(restored_metadata)
                return Ds4KvCacheRestoreResult(
                    status="hit",
                    entry=entry,
                    metadata=restored_metadata,
                    restored=True,
                    warning=warning,
                )
        elif metadata is not None:
            miss_error = "cache metadata does not match request."

        synced = self._sync_on_miss(session, tokens, sync_on_miss)
        return Ds4KvCacheRestoreResult(
            status="miss",
            entry=entry,
            metadata=metadata,
            synced=synced,
            error=miss_error,
        )

    async def arestore(
        self,
        session: object,
        prompt_tokens: Iterable[int],
        ctx_size: int,
        *,
        enabled: bool = True,
        sync_on_miss: bool = True,
    ) -> Ds4KvCacheRestoreResult:
        """Async restore variant for sessions with awaitable payload APIs."""
        _validate_bool("enabled", enabled)
        _validate_bool("sync_on_miss", sync_on_miss)
        tokens = _normalize_token_ids(prompt_tokens)
        ctx = _validate_int("ctx_size", ctx_size, minimum=1, maximum=C_INT_MAX)
        entry = self._entry_for_tokens(tokens, ctx)

        if not enabled:
            synced = await self._async_sync_on_miss(
                session,
                tokens,
                sync_on_miss,
            )
            return Ds4KvCacheRestoreResult(
                status="disabled",
                entry=entry,
                synced=synced,
            )

        metadata = self.read_metadata(entry)
        miss_error: str | None = None
        if metadata is not None and self.metadata_matches(
            metadata,
            tokens,
            ctx,
        ):
            payload_path = self._directory / metadata.payload_file
            try:
                payload = payload_path.read_bytes()
                if len(payload) != metadata.payload_size:
                    raise ValueError(
                        "cache payload size does not match metadata."
                    )
                await _call_async(_method(session, "load_payload"), payload)
            except Exception as error:
                miss_error = _error_text(error)
            else:
                restored_metadata = replace(
                    metadata,
                    hit_count=metadata.hit_count + 1,
                    accessed_at=time(),
                    payload_size=len(payload),
                )
                warning = self._try_write_metadata(restored_metadata)
                return Ds4KvCacheRestoreResult(
                    status="hit",
                    entry=entry,
                    metadata=restored_metadata,
                    restored=True,
                    warning=warning,
                )
        elif metadata is not None:
            miss_error = "cache metadata does not match request."

        synced = await self._async_sync_on_miss(session, tokens, sync_on_miss)
        return Ds4KvCacheRestoreResult(
            status="miss",
            entry=entry,
            metadata=metadata,
            synced=synced,
            error=miss_error,
        )

    def store(
        self,
        session: object,
        prompt_tokens: Iterable[int],
        ctx_size: int,
        *,
        rendered_prompt: str | None = None,
        enabled: bool = True,
        size_budget_bytes: int | None = None,
        raise_on_error: bool = False,
    ) -> Ds4KvCacheStoreResult:
        """Save a session payload and metadata, returning write failures."""
        _validate_optional_str("rendered_prompt", rendered_prompt)
        _validate_bool("enabled", enabled)
        budget = _validate_optional_size_budget(size_budget_bytes)
        _validate_bool("raise_on_error", raise_on_error)
        tokens = _normalize_token_ids(prompt_tokens)
        ctx = _validate_int("ctx_size", ctx_size, minimum=1, maximum=C_INT_MAX)
        entry = self._entry_for_tokens(tokens, ctx)

        if not enabled:
            return Ds4KvCacheStoreResult(status="disabled", entry=entry)

        try:
            payload = _validate_bytes(
                "save_payload result",
                _call_sync(_method(session, "save_payload")),
            )
            existing = self.read_metadata(entry)
            now = time()
            created_at = now
            hit_count = 0
            if existing is not None and self.metadata_matches(
                existing,
                tokens,
                ctx,
            ):
                created_at = existing.created_at
                hit_count = existing.hit_count

            metadata = self.metadata_for(
                tokens,
                ctx,
                rendered_prompt=rendered_prompt,
                payload_size=len(payload),
                created_at=created_at,
                accessed_at=now,
                hit_count=hit_count,
            )
            self._write_payload_and_metadata(entry, metadata, payload)
            evicted_entries = self.evict(budget) if budget is not None else ()
            return Ds4KvCacheStoreResult(
                status="stored",
                entry=entry,
                metadata=metadata,
                stored=True,
                evicted_entries=evicted_entries,
            )
        except Exception as error:
            if raise_on_error:
                raise
            return Ds4KvCacheStoreResult(
                status="error",
                entry=entry,
                error=_error_text(error),
            )

    async def astore(
        self,
        session: object,
        prompt_tokens: Iterable[int],
        ctx_size: int,
        *,
        rendered_prompt: str | None = None,
        enabled: bool = True,
        size_budget_bytes: int | None = None,
        raise_on_error: bool = False,
    ) -> Ds4KvCacheStoreResult:
        """Async store variant for sessions with awaitable payload APIs."""
        _validate_optional_str("rendered_prompt", rendered_prompt)
        _validate_bool("enabled", enabled)
        budget = _validate_optional_size_budget(size_budget_bytes)
        _validate_bool("raise_on_error", raise_on_error)
        tokens = _normalize_token_ids(prompt_tokens)
        ctx = _validate_int("ctx_size", ctx_size, minimum=1, maximum=C_INT_MAX)
        entry = self._entry_for_tokens(tokens, ctx)

        if not enabled:
            return Ds4KvCacheStoreResult(status="disabled", entry=entry)

        try:
            payload = _validate_bytes(
                "save_payload result",
                await _call_async(_method(session, "save_payload")),
            )
            existing = self.read_metadata(entry)
            now = time()
            created_at = now
            hit_count = 0
            if existing is not None and self.metadata_matches(
                existing,
                tokens,
                ctx,
            ):
                created_at = existing.created_at
                hit_count = existing.hit_count

            metadata = self.metadata_for(
                tokens,
                ctx,
                rendered_prompt=rendered_prompt,
                payload_size=len(payload),
                created_at=created_at,
                accessed_at=now,
                hit_count=hit_count,
            )
            self._write_payload_and_metadata(entry, metadata, payload)
            evicted_entries = self.evict(budget) if budget is not None else ()
            return Ds4KvCacheStoreResult(
                status="stored",
                entry=entry,
                metadata=metadata,
                stored=True,
                evicted_entries=evicted_entries,
            )
        except Exception as error:
            if raise_on_error:
                raise
            return Ds4KvCacheStoreResult(
                status="error",
                entry=entry,
                error=_error_text(error),
            )

    def evict(self, size_budget_bytes: int) -> tuple[Ds4KvCacheEntry, ...]:
        """Evict least-useful entries until payload bytes fit the budget."""
        budget = _validate_int(
            "size_budget_bytes",
            size_budget_bytes,
            minimum=0,
        )
        if not self._directory.exists():
            return ()

        candidates: list[tuple[Ds4KvCacheMetadata, Ds4KvCacheEntry, int]] = []
        for metadata_path in self._directory.glob("*.json"):
            metadata = self.read_metadata(metadata_path)
            if metadata is None or not self._metadata_belongs_to_cache(
                metadata
            ):
                continue
            payload_path = self._directory / metadata.payload_file
            try:
                payload_size = payload_path.stat().st_size
            except OSError:
                continue
            candidates.append(
                (
                    metadata,
                    Ds4KvCacheEntry(
                        key=metadata.key,
                        metadata_path=metadata_path,
                        payload_path=payload_path,
                        token_sha256=metadata.token_sha256,
                    ),
                    payload_size,
                )
            )

        total_size = sum(payload_size for _, _, payload_size in candidates)
        if total_size <= budget:
            return ()

        evicted: list[Ds4KvCacheEntry] = []
        candidates.sort(
            key=lambda item: (
                item[0].hit_count,
                item[0].accessed_at,
                item[0].created_at,
                item[0].key,
            )
        )
        for _, entry, payload_size in candidates:
            if total_size <= budget:
                break
            self._remove_entry_files(entry)
            total_size -= payload_size
            evicted.append(entry)
        return tuple(evicted)

    def _metadata_belongs_to_cache(
        self,
        metadata: Ds4KvCacheMetadata,
    ) -> bool:
        return (
            metadata.version == self._cache_version
            and metadata.model_namespace == self._model_namespace
            and metadata.pyds4_version == self._pyds4_version
            and metadata.ds4_commit == self._ds4_commit
            and metadata.backend == self._backend
        )

    def _sync_on_miss(
        self,
        session: object,
        prompt_tokens: tuple[int, ...],
        enabled: bool,
    ) -> bool:
        if not enabled:
            return False
        _call_sync(_method(session, "sync"), list(prompt_tokens))
        return True

    async def _async_sync_on_miss(
        self,
        session: object,
        prompt_tokens: tuple[int, ...],
        enabled: bool,
    ) -> bool:
        if not enabled:
            return False
        await _call_async(_method(session, "sync"), list(prompt_tokens))
        return True

    def _try_write_metadata(
        self,
        metadata: Ds4KvCacheMetadata,
    ) -> str | None:
        try:
            self.write_metadata(metadata)
        except OSError as error:
            return _error_text(error)
        return None

    def _write_payload_and_metadata(
        self,
        entry: Ds4KvCacheEntry,
        metadata: Ds4KvCacheMetadata,
        payload: bytes,
    ) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        payload_tmp_path = entry.payload_path.with_name(
            f"{entry.payload_path.name}.tmp"
        )
        metadata_tmp_path = entry.metadata_path.with_name(
            f"{entry.metadata_path.name}.tmp"
        )
        try:
            payload_tmp_path.write_bytes(payload)
            metadata_tmp_path.write_text(
                json.dumps(metadata.to_json_dict(), sort_keys=True),
                encoding="utf-8",
            )
            payload_tmp_path.replace(entry.payload_path)
            metadata_tmp_path.replace(entry.metadata_path)
        finally:
            for path in (payload_tmp_path, metadata_tmp_path):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    @staticmethod
    def _remove_entry_files(entry: Ds4KvCacheEntry) -> None:
        for path in (entry.payload_path, entry.metadata_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _metadata_str(metadata: Mapping[str, object], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str):
        raise TypeError(f"metadata {key!r} must be a string.")
    return value


def _metadata_int(metadata: Mapping[str, object], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"metadata {key!r} must be an integer.")
    return value


def _metadata_float(metadata: Mapping[str, object], key: str) -> float:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"metadata {key!r} must be a number.")
    return float(value)


__all__ = [
    "DS4_KV_CACHE_VERSION",
    "Ds4DiskKvCache",
    "Ds4KvCacheEntry",
    "Ds4KvCacheMetadata",
    "Ds4KvCacheRestoreResult",
    "Ds4KvCacheStoreResult",
]
