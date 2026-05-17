from __future__ import annotations

import re
import sys
from pathlib import Path

VERSION_PATTERN = re.compile(
    r"^[0-9]+\.[0-9]+\.[0-9]+"
    r"(?:(?:a|b|rc)[0-9]+)?"
    r"(?:\.post[0-9]+)?"
    r"(?:\.dev[0-9]+)?$"
)

FILES_WITH_VERSION = (
    Path("pyproject.toml"),
    Path("src/pyds4/_native.cpp"),
    Path("tests/test_kv_cache.py"),
    Path("tests/test_metadata.py"),
)


def _current_version() -> str:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version = "([^"]+)"$', pyproject)
    if match is None:
        raise SystemExit("Could not find project.version in pyproject.toml.")
    return match.group(1)


def _replace_version(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    updated = text.replace(old, new)
    if updated == text:
        raise SystemExit(f"{path} did not contain version {old!r}.")
    path.write_text(updated, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py X.Y.Z", file=sys.stderr)
        return 2

    new_version = sys.argv[1]
    if VERSION_PATTERN.fullmatch(new_version) is None:
        print(
            f"Invalid version {new_version!r}; expected a PEP 440 X.Y.Z-style "
            "release version.",
            file=sys.stderr,
        )
        return 2

    old_version = _current_version()
    if old_version == new_version:
        print(f"pyds4 is already at {new_version}.")
        return 0

    for path in FILES_WITH_VERSION:
        _replace_version(path, old_version, new_version)

    print(f"Bumped pyds4 from {old_version} to {new_version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
