from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


def _dependencies() -> list[str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies: list[str] = []

    for requirement in pyproject["build-system"]["requires"]:
        dependencies.append(requirement)

    optional = pyproject["project"]["optional-dependencies"]
    for extra in ("test", "release"):
        for requirement in optional[extra]:
            dependencies.append(requirement)

    return list(dict.fromkeys(dependencies))


def main() -> int:
    dependencies = _dependencies()
    return subprocess.call(
        [sys.executable, "-m", "pip", "install", *dependencies]
    )


if __name__ == "__main__":
    raise SystemExit(main())
