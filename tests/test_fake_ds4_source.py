from __future__ import annotations

import re
from pathlib import Path

import pyds4


def test_fake_ds4_source_declares_and_implements_required_symbols() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fake_dir = repo_root / "tests" / "fake_ds4"
    header = (fake_dir / "ds4.h").read_text(encoding="utf-8")
    source = (fake_dir / "ds4.c").read_text(encoding="utf-8")
    wrapper = (repo_root / "src" / "pyds4" / "_native.cpp").read_text(
        encoding="utf-8"
    )

    wrapper_symbols = set(re.findall(r"\b(ds4_[A-Za-z0-9_]+)\s*\(", wrapper))
    required_symbols = set(pyds4.REQUIRED_C_SYMBOLS)
    missing_from_metadata = wrapper_symbols - required_symbols

    assert not missing_from_metadata, (
        "wrapper-used DS4 symbols missing from REQUIRED_C_SYMBOLS: "
        f"{sorted(missing_from_metadata)}"
    )

    for symbol in pyds4.REQUIRED_C_SYMBOLS:
        pattern = re.compile(rf"\b{re.escape(symbol)}\s*\(")
        assert pattern.search(header), f"{symbol} missing from fake ds4.h"
        assert pattern.search(source), f"{symbol} missing from fake ds4.c"
