from __future__ import annotations

from pathlib import Path

import pyds4
from pyds4._metal import (
    METAL_SOURCE_ENV,
    configure_metal_source_paths,
)


def _write_metal_sources(package_dir: Path) -> Path:
    metal_dir = package_dir / "metal"
    metal_dir.mkdir()
    for filename in METAL_SOURCE_ENV.values():
        (metal_dir / filename).write_text("// test source\n", encoding="utf-8")
    return metal_dir


def test_configure_metal_source_paths_sets_packaged_paths(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pyds4"
    package_dir.mkdir()
    metal_dir = _write_metal_sources(package_dir)
    environ: dict[str, str] = {}

    configured = configure_metal_source_paths(
        backend="metal",
        package_paths=[str(package_dir)],
        environ=environ,
    )

    assert configured == metal_dir
    assert environ == {
        env_name: str(metal_dir / filename)
        for env_name, filename in METAL_SOURCE_ENV.items()
    }


def test_configure_metal_source_paths_does_not_override_existing_env(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pyds4"
    package_dir.mkdir()
    _write_metal_sources(package_dir)
    first_env_name = next(iter(METAL_SOURCE_ENV))
    environ = {first_env_name: "/custom/source.metal"}

    configure_metal_source_paths(
        backend="metal",
        package_paths=[str(package_dir)],
        environ=environ,
    )

    assert environ[first_env_name] == "/custom/source.metal"


def test_configure_metal_source_paths_ignores_non_metal_backend(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pyds4"
    package_dir.mkdir()
    _write_metal_sources(package_dir)
    environ: dict[str, str] = {}

    assert (
        configure_metal_source_paths(
            backend="cpu",
            package_paths=[str(package_dir)],
            environ=environ,
        )
        is None
    )
    assert environ == {}


def test_public_import_configures_metal_without_source_tree_assumption() -> (
    None
):
    if pyds4.__ds4_native_backend__ != "metal":
        return

    configured = configure_metal_source_paths()
    if configured is not None:
        assert all(
            (configured / filename).is_file()
            for filename in METAL_SOURCE_ENV.values()
        )
