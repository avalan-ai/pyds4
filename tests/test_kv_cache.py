from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from pyds4.kv_cache import (
    DS4_KV_CACHE_VERSION,
    Ds4DiskKvCache,
    Ds4KvCacheMetadata,
)


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
    assert cache.metadata_matches(
        replace(metadata, version=DS4_KV_CACHE_VERSION + 1),
        [1, 2, 3],
        4096,
    ) is False


def test_kv_cache_metadata_rejects_invalid_payload_file() -> None:
    with pytest.raises(ValueError, match="payload_file"):
        Ds4KvCacheMetadata(
            version=DS4_KV_CACHE_VERSION,
            key="key",
            model_namespace="model-a",
            pyds4_version="0.1.0",
            ds4_commit="commit",
            backend="metal",
            ctx_size=4096,
            token_count=1,
            token_sha256="sha",
            payload_file="../entry.payload",
        )


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
