from __future__ import annotations

from typing import Any

NATIVE_BACKEND: str
VERSION: str

def __getattr__(name: str) -> Any: ...
