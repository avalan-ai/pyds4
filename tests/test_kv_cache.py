from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from pyds4.kv_cache import (
    DS4_KV_CACHE_VERSION,
    Ds4DiskKvCache,
    Ds4KvCacheEntry,
    Ds4KvCacheMetadata,
)


class PayloadSession:
    def __init__(
        self,
        *,
        payload: bytes = b"payload",
        fail_load: bool = False,
    ) -> None:
        self.payload = payload
        self.fail_load = fail_load
        self.loaded_payloads: list[bytes] = []
        self.saved = 0
        self.synced_tokens: list[int] | None = None
        self.live_tokens = [42]

    def sync(self, prompt_tokens: list[int]) -> None:
        self.synced_tokens = list(prompt_tokens)
        self.live_tokens = list(prompt_tokens)

    def load_payload(self, payload: bytes) -> None:
        self.loaded_payloads.append(payload)
        if self.fail_load:
            raise RuntimeError("payload rejected")
        self.live_tokens = [99]

    def save_payload(self) -> bytes:
        self.saved += 1
        return self.payload


class AsyncPayloadSession:
    def __init__(
        self,
        *,
        payload: bytes = b"payload",
        fail_load: bool = False,
    ) -> None:
        self.payload = payload
        self.fail_load = fail_load
        self.loaded_payloads: list[bytes] = []
        self.saved = 0
        self.synced_tokens: list[int] | None = None
        self.live_tokens = [42]

    async def sync(self, prompt_tokens: list[int]) -> None:
        self.synced_tokens = list(prompt_tokens)
        self.live_tokens = list(prompt_tokens)

    async def load_payload(self, payload: bytes) -> None:
        self.loaded_payloads.append(payload)
        if self.fail_load:
            raise RuntimeError("payload rejected")
        self.live_tokens = [99]

    async def save_payload(self) -> bytes:
        self.saved += 1
        return self.payload


def write_cache_entry(
    cache: Ds4DiskKvCache,
    prompt_tokens: list[int],
    ctx_size: int,
    payload: bytes,
    *,
    hit_count: int = 0,
    accessed_at: float = 1.0,
) -> Ds4KvCacheEntry:
    entry = cache.entry_for(prompt_tokens, ctx_size)
    entry.payload_path.parent.mkdir(parents=True, exist_ok=True)
    entry.payload_path.write_bytes(payload)
    metadata = cache.metadata_for(
        prompt_tokens,
        ctx_size,
        payload_size=len(payload),
        created_at=accessed_at,
        accessed_at=accessed_at,
        hit_count=hit_count,
    )
    cache.write_metadata(metadata)
    return entry


def test_kv_cache_same_tokens_same_key_and_text_is_metadata_only(
    tmp_path: Path,
) -> None:
    cache = Ds4DiskKvCache(
        tmp_path,
        "model-a",
        backend="metal",
    )

    entry = cache.entry_for([1, 2, 3], 4096)
    same_entry = cache.entry_for((token for token in [1, 2, 3]), 4096)
    first_metadata = cache.metadata_for(
        [1, 2, 3],
        4096,
        rendered_prompt="<prompt-a>",
        created_at=1.0,
    )
    second_metadata = cache.metadata_for(
        [1, 2, 3],
        4096,
        rendered_prompt="<prompt-b>",
        created_at=1.0,
    )

    assert entry.key == same_entry.key
    assert first_metadata.key == entry.key
    assert second_metadata.key == entry.key
    assert first_metadata.rendered_prompt == "<prompt-a>"
    assert second_metadata.rendered_prompt == "<prompt-b>"


def test_kv_cache_key_changes_for_prefix_context_or_namespace(
    tmp_path: Path,
) -> None:
    cache = Ds4DiskKvCache(tmp_path / "a", "model-a", backend="metal")
    other_namespace = Ds4DiskKvCache(
        tmp_path / "a",
        "model-b",
        backend="metal",
    )

    keys = {
        cache.entry_for([1, 2, 3], 4096).key,
        cache.entry_for([1, 3, 2], 4096).key,
        cache.entry_for([1, 2], 4096).key,
        cache.entry_for([1, 2, 3], 8192).key,
        other_namespace.entry_for([1, 2, 3], 4096).key,
    }

    assert len(keys) == 5


def test_kv_cache_metadata_round_trips_without_payload_bytes(
    tmp_path: Path,
) -> None:
    cache = Ds4DiskKvCache(tmp_path, "model-a", backend="metal")
    metadata = cache.metadata_for(
        [4, 5, 6],
        4096,
        rendered_prompt="<rendered>",
        payload_size=123,
        created_at=1.25,
        accessed_at=2.5,
        hit_count=3,
    )
    entry = cache.entry_for([4, 5, 6], 4096)

    path = cache.write_metadata(metadata)
    loaded = cache.read_metadata(entry)

    assert path == entry.metadata_path
    assert loaded == metadata
    assert not entry.payload_path.exists()
    assert loaded is not None
    assert cache.metadata_matches(loaded, [4, 5, 6], 4096) is True


@pytest.mark.parametrize(
    "prompt_tokens",
    [
        [1, True],
        [1, -1],
        [1, 2**31],
        [1, "2"],
        "1,2",
        object(),
    ],
)
def test_kv_cache_rejects_non_token_values(
    tmp_path: Path,
    prompt_tokens: object,
) -> None:
    cache = Ds4DiskKvCache(tmp_path, "model-a", backend="metal")

    with pytest.raises((TypeError, ValueError), match="prompt_tokens"):
        cache.entry_for(prompt_tokens, 4096)  # type: ignore[arg-type]


@pytest.mark.parametrize("ctx_size", [0, -1, True, 2**31])
def test_kv_cache_rejects_invalid_context_sizes(
    tmp_path: Path,
    ctx_size: object,
) -> None:
    cache = Ds4DiskKvCache(tmp_path, "model-a", backend="metal")

    with pytest.raises((TypeError, ValueError), match="ctx_size"):
        cache.entry_for([1, 2], ctx_size)  # type: ignore[arg-type]


def test_kv_cache_corrupt_metadata_is_ignored(tmp_path: Path) -> None:
    cache = Ds4DiskKvCache(tmp_path, "model-a", backend="metal")
    entry = cache.entry_for([1], 16)
    tmp_path.mkdir(exist_ok=True)

    entry.metadata_path.write_text("{", encoding="utf-8")
    assert cache.read_metadata(entry) is None

    entry.metadata_path.write_text("[]", encoding="utf-8")
    assert cache.read_metadata(entry) is None

    entry.metadata_path.write_text(
        '{"key":"missing-fields"}',
        encoding="utf-8",
    )
    assert cache.read_metadata(entry) is None

    unsafe_key_metadata = cache.metadata_for([1], 16).to_json_dict()
    unsafe_key_metadata["key"] = "../escape"
    entry.metadata_path.write_text(
        json.dumps(unsafe_key_metadata),
        encoding="utf-8",
    )
    assert cache.read_metadata(entry) is None


def test_kv_cache_mismatched_metadata_invalidates_entry(
    tmp_path: Path,
) -> None:
    cache = Ds4DiskKvCache(tmp_path, "model-a", backend="metal")
    metadata = cache.metadata_for([1, 2, 3], 4096, created_at=1.0)
    different_model = Ds4DiskKvCache(tmp_path, "model-b", backend="metal")
    different_backend = Ds4DiskKvCache(tmp_path, "model-a", backend="cuda")

    assert cache.metadata_matches(metadata, [1, 2, 3], 4096) is True
    assert cache.metadata_matches(metadata, [1, 2, 3], 8192) is False
    assert different_model.metadata_matches(metadata, [1, 2, 3], 4096) is False
    assert different_backend.metadata_matches(metadata, [1, 2, 3], 4096) is (
        False
    )
    assert (
        cache.metadata_matches(
            replace(metadata, version=DS4_KV_CACHE_VERSION + 1),
            [1, 2, 3],
            4096,
        )
        is False
    )


def test_kv_cache_metadata_rejects_invalid_payload_file() -> None:
    with pytest.raises(ValueError, match="payload_file"):
        Ds4KvCacheMetadata(
            version=DS4_KV_CACHE_VERSION,
            key="key",
            model_namespace="model-a",
            pyds4_version="1.0.1",
            ds4_commit="commit",
            backend="metal",
            ctx_size=4096,
            token_count=1,
            token_sha256="sha",
            payload_file="../entry.payload",
        )


@pytest.mark.parametrize("key", ["../escape", "nested/key", r"nested\key"])
def test_kv_cache_metadata_rejects_path_like_keys(key: str) -> None:
    with pytest.raises(ValueError, match="key"):
        Ds4KvCacheMetadata(
            version=DS4_KV_CACHE_VERSION,
            key=key,
            model_namespace="model-a",
            pyds4_version="1.0.1",
            ds4_commit="commit",
            backend="metal",
            ctx_size=4096,
            token_count=1,
            token_sha256="sha",
            payload_file="entry.payload",
        )


def test_kv_cache_restore_hit_loads_payload_and_updates_metadata(
    tmp_path: Path,
) -> None:
    cache = Ds4DiskKvCache(tmp_path, "model-a", backend="metal")
    entry = write_cache_entry(cache, [1, 2, 3], 16, b"cached-payload")
    session = PayloadSession()

    result = cache.restore(session, [1, 2, 3], 16)
    metadata = cache.read_metadata(entry)

    assert result.status == "hit"
    assert result.restored is True
    assert result.synced is False
    assert result.warning is None
    assert session.loaded_payloads == [b"cached-payload"]
    assert session.synced_tokens is None
    assert metadata is not None
    assert metadata.hit_count == 1


def test_kv_cache_hit_restores_fake_native_session(tmp_path: Path) -> None:
    pyds4 = pytest.importorskip("pyds4")
    native = pytest.importorskip("pyds4._native")
    if not getattr(native, "__ds4_fake_native__", False):
        pytest.skip("requires a PYDS4_USE_FAKE_DS4 build")

    native.fake_reset_counters()
    cache = Ds4DiskKvCache(tmp_path, "model-a", backend="cpu")

    options = pyds4.EngineOptions(model_path="model.gguf", backend="cpu")
    with pyds4.Engine(options) as engine:
        source = engine.create_session(64)
        target = engine.create_session(64)
        source.sync([1, 2, 3])

        store_result = cache.store(source, [1, 2, 3], 64)
        target.sync([9])
        restore_result = cache.restore(target, [1, 2, 3], 64)

        assert store_result.status == "stored"
        assert restore_result.status == "hit"
        assert target.tokens == [1, 2, 3]

        source.close()
        target.close()

    counters = native.fake_counters()
    assert counters["save_payload_calls"] == 1
    assert counters["load_payload_calls"] == 1


def test_kv_cache_restore_miss_falls_back_to_live_sync(
    tmp_path: Path,
) -> None:
    cache = Ds4DiskKvCache(tmp_path, "model-a", backend="metal")
    session = PayloadSession()

    result = cache.restore(session, [4, 5], 16)

    assert result.status == "miss"
    assert result.restored is False
    assert result.synced is True
    assert session.loaded_payloads == []
    assert session.synced_tokens == [4, 5]


def test_kv_cache_store_saves_payload_and_metadata(
    tmp_path: Path,
) -> None:
    cache = Ds4DiskKvCache(tmp_path, "model-a", backend="metal")
    session = PayloadSession(payload=b"fresh-payload")

    result = cache.store(
        session,
        [7, 8],
        16,
        rendered_prompt="<rendered>",
    )
    metadata = cache.read_metadata(result.entry)

    assert result.status == "stored"
    assert result.stored is True
    assert result.error is None
    assert result.entry.payload_path.read_bytes() == b"fresh-payload"
    assert metadata is not None
    assert metadata.rendered_prompt == "<rendered>"
    assert metadata.payload_size == len(b"fresh-payload")
    assert session.saved == 1


def test_kv_cache_budget_eviction_removes_least_useful_entries(
    tmp_path: Path,
) -> None:
    cache = Ds4DiskKvCache(tmp_path, "model-a", backend="metal")
    evictable = write_cache_entry(
        cache,
        [1],
        16,
        b"11111",
        hit_count=0,
        accessed_at=1.0,
    )
    survivor = write_cache_entry(
        cache,
        [2],
        16,
        b"22222",
        hit_count=2,
        accessed_at=1.0,
    )
    session = PayloadSession(payload=b"33333")

    result = cache.store(session, [3], 16, size_budget_bytes=10)

    assert result.status == "stored"
    assert [entry.key for entry in result.evicted_entries] == [evictable.key]
    assert not evictable.payload_path.exists()
    assert not evictable.metadata_path.exists()
    assert survivor.payload_path.exists()
    assert result.entry.payload_path.exists()


def test_kv_cache_eviction_skips_mismatched_metadata_file_names(
    tmp_path: Path,
) -> None:
    cache = Ds4DiskKvCache(tmp_path, "model-a", backend="metal")
    metadata = cache.metadata_for(
        [1],
        16,
        payload_size=5,
        created_at=1.0,
        accessed_at=1.0,
    )

    mismatched_payload_path = tmp_path / "other.payload"
    mismatched_payload_path.write_bytes(b"other")
    mismatched_payload_metadata = replace(
        metadata,
        payload_file=mismatched_payload_path.name,
    )
    cache.write_metadata(mismatched_payload_metadata)

    orphan_payload_path = tmp_path / metadata.payload_file
    orphan_payload_path.write_bytes(b"orphan")
    orphan_metadata_path = tmp_path / "orphan.json"
    orphan_metadata_path.write_text(
        json.dumps(metadata.to_json_dict(), sort_keys=True),
        encoding="utf-8",
    )

    evicted = cache.evict(0)

    assert evicted == ()
    assert mismatched_payload_path.exists()
    assert orphan_payload_path.exists()
    assert (tmp_path / f"{metadata.key}.json").exists()
    assert orphan_metadata_path.exists()


def test_kv_cache_disabled_restore_and_store_leave_no_files(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache = Ds4DiskKvCache(cache_dir, "model-a", backend="metal")
    restore_session = PayloadSession()
    store_session = PayloadSession()

    restore_result = cache.restore(
        restore_session,
        [1],
        16,
        enabled=False,
    )
    store_result = cache.store(store_session, [1], 16, enabled=False)

    assert restore_result.status == "disabled"
    assert restore_result.synced is True
    assert restore_session.synced_tokens == [1]
    assert store_result.status == "disabled"
    assert store_result.stored is False
    assert not cache_dir.exists()
    assert store_session.saved == 0


def test_kv_cache_corrupt_payload_is_returned_as_miss(
    tmp_path: Path,
) -> None:
    cache = Ds4DiskKvCache(tmp_path, "model-a", backend="metal")
    entry = write_cache_entry(cache, [1], 16, b"good")
    metadata = cache.read_metadata(entry)
    assert metadata is not None
    cache.write_metadata(replace(metadata, payload_size=99))
    session = PayloadSession()

    result = cache.restore(session, [1], 16)

    assert result.status == "miss"
    assert result.error == "cache payload size does not match metadata."
    assert session.loaded_payloads == []
    assert session.synced_tokens == [1]


def test_kv_cache_restore_failure_does_not_prevent_live_sync(
    tmp_path: Path,
) -> None:
    cache = Ds4DiskKvCache(tmp_path, "model-a", backend="metal")
    write_cache_entry(cache, [1, 2], 16, b"bad-payload")
    session = PayloadSession(fail_load=True)

    result = cache.restore(session, [1, 2], 16)

    assert result.status == "miss"
    assert result.error == "payload rejected"
    assert session.loaded_payloads == [b"bad-payload"]
    assert session.synced_tokens == [1, 2]


def test_kv_cache_async_restore_and_store_support_awaitable_sessions(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        cache = Ds4DiskKvCache(tmp_path, "model-a", backend="metal")
        store_session = AsyncPayloadSession(payload=b"async-payload")

        store_result = await cache.astore(
            store_session,
            [1, 2],
            16,
            rendered_prompt="<rendered>",
        )

        restore_session = AsyncPayloadSession()
        restore_result = await cache.arestore(restore_session, [1, 2], 16)
        metadata = cache.read_metadata(restore_result.entry)

        assert store_result.status == "stored"
        assert store_result.stored is True
        assert store_session.saved == 1
        assert restore_result.status == "hit"
        assert restore_result.restored is True
        assert restore_session.loaded_payloads == [b"async-payload"]
        assert restore_session.synced_tokens is None
        assert metadata is not None
        assert metadata.rendered_prompt == "<rendered>"
        assert metadata.hit_count == 1

    asyncio.run(scenario())


def test_kv_cache_async_restore_failure_falls_back_to_awaited_sync(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        cache = Ds4DiskKvCache(tmp_path, "model-a", backend="metal")
        write_cache_entry(cache, [1, 2], 16, b"bad-payload")
        session = AsyncPayloadSession(fail_load=True)

        result = await cache.arestore(session, [1, 2], 16)

        assert result.status == "miss"
        assert result.error == "payload rejected"
        assert session.loaded_payloads == [b"bad-payload"]
        assert session.synced_tokens == [1, 2]

    asyncio.run(scenario())


def test_kv_cache_sync_helpers_reject_awaitable_sessions(
    tmp_path: Path,
) -> None:
    cache = Ds4DiskKvCache(tmp_path, "model-a", backend="metal")
    write_cache_entry(cache, [1], 16, b"payload")
    session = AsyncPayloadSession()

    restore_result = cache.restore(session, [1], 16, sync_on_miss=False)
    store_result = cache.store(session, [1], 16)

    assert restore_result.status == "miss"
    assert restore_result.error is not None
    assert "use arestore() or astore()" in restore_result.error
    assert store_result.status == "error"
    assert store_result.error is not None
    assert "use arestore() or astore()" in store_result.error


def test_kv_cache_disk_write_failure_is_non_fatal_by_default(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "cache-file"
    cache_path.write_text("not a directory", encoding="utf-8")
    cache = Ds4DiskKvCache(cache_path, "model-a", backend="metal")
    session = PayloadSession(payload=b"fresh-payload")

    result = cache.store(session, [1], 16)

    assert result.status == "error"
    assert result.stored is False
    assert result.error is not None
    assert session.saved == 1
    assert session.live_tokens == [42]


def test_kv_cache_import_does_not_import_native_extension_objects() -> None:
    env = {
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import sys; "
                "import pyds4.kv_cache as kv_cache; "
                "print(hasattr(kv_cache, 'Ds4DiskKvCache')); "
                "print('pyds4._native' in sys.modules)"
            ),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == ["True", "False"]
