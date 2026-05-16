from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from numbers import Real

from ._metadata import C_INT_MAX


class Backend(StrEnum):
    """Name the native DS4 execution backend."""

    METAL = "metal"
    CUDA = "cuda"
    CPU = "cpu"


class ThinkMode(StrEnum):
    """Name the DS4 reasoning prompt mode."""

    NONE = "none"
    HIGH = "high"
    MAX = "max"


def _validate_int(
    name: str,
    value: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")


def _validate_str(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")


def _validate_optional_str(name: str, value: str | None) -> None:
    if value is None:
        return
    _validate_str(name, value)


def _validate_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")


def _validate_optional_int(
    name: str,
    value: int | None,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    if value is None:
        return
    _validate_int(name, value, minimum=minimum, maximum=maximum)


def _validate_str_tuple(name: str, value: Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, (bytes, str)) or not isinstance(value, Iterable):
        raise TypeError(f"{name} must be an iterable of strings.")
    result = tuple(value)
    for item in result:
        _validate_str(f"{name} item", item)
    return result


def _validate_real(
    name: str,
    value: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number.")
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum:g}.")


@dataclass(frozen=True, slots=True)
class EngineOptions:
    """Store native DS4 engine-open options."""

    model_path: str
    backend: Backend = Backend.METAL
    mtp_path: str | None = None
    n_threads: int = 0
    mtp_draft_tokens: int = 0
    mtp_margin: float = 0.0
    directional_steering_file: str | None = None
    directional_steering_attn: float = 0.0
    directional_steering_ffn: float = 0.0
    warm_weights: bool = False
    quality: bool = False
    native_log: bool = True

    def __post_init__(self) -> None:
        _validate_str("model_path", self.model_path)
        object.__setattr__(self, "backend", Backend(self.backend))
        _validate_optional_str("mtp_path", self.mtp_path)
        _validate_int(
            "n_threads",
            self.n_threads,
            minimum=0,
            maximum=C_INT_MAX,
        )
        _validate_int(
            "mtp_draft_tokens",
            self.mtp_draft_tokens,
            minimum=0,
            maximum=C_INT_MAX,
        )
        _validate_real("mtp_margin", self.mtp_margin)
        _validate_optional_str(
            "directional_steering_file",
            self.directional_steering_file,
        )
        _validate_real(
            "directional_steering_attn",
            self.directional_steering_attn,
        )
        _validate_real(
            "directional_steering_ffn",
            self.directional_steering_ffn,
        )
        _validate_bool("warm_weights", self.warm_weights)
        _validate_bool("quality", self.quality)
        _validate_bool("native_log", self.native_log)


@dataclass(frozen=True, slots=True)
class SamplingOptions:
    """Store DS4 token sampling options."""

    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        _validate_real("temperature", self.temperature, minimum=0.0)
        _validate_int("top_k", self.top_k, minimum=0, maximum=C_INT_MAX)
        _validate_real("top_p", self.top_p, minimum=0.0, maximum=1.0)
        _validate_real("min_p", self.min_p, minimum=0.0, maximum=1.0)
        _validate_optional_int("seed", self.seed)


@dataclass(frozen=True, slots=True)
class Ds4Capabilities:
    """Report import-safe DS4 runtime capabilities for this pyds4 build."""

    backend: str
    ds4_commit: str
    ds4_api_version: int | None
    required_symbols: tuple[str, ...]
    available_backends: tuple[str, ...]
    snapshots: bool
    payloads: bool
    logprobs: bool
    top_logprobs: bool
    progress: bool
    mtp: bool
    speculative_eval: bool

    def __post_init__(self) -> None:
        _validate_str("backend", self.backend)
        _validate_str("ds4_commit", self.ds4_commit)
        _validate_optional_int("ds4_api_version", self.ds4_api_version)
        object.__setattr__(
            self,
            "required_symbols",
            _validate_str_tuple("required_symbols", self.required_symbols),
        )
        object.__setattr__(
            self,
            "available_backends",
            _validate_str_tuple("available_backends", self.available_backends),
        )
        _validate_bool("snapshots", self.snapshots)
        _validate_bool("payloads", self.payloads)
        _validate_bool("logprobs", self.logprobs)
        _validate_bool("top_logprobs", self.top_logprobs)
        _validate_bool("progress", self.progress)
        _validate_bool("mtp", self.mtp)
        _validate_bool("speculative_eval", self.speculative_eval)


@dataclass(frozen=True, slots=True)
class TokenScore:
    """Describe one token's native DS4 next-token log probability."""

    token_id: int
    logprob: float

    def __post_init__(self) -> None:
        _validate_int(
            "token_id",
            self.token_id,
            minimum=0,
            maximum=C_INT_MAX,
        )
        _validate_real("logprob", self.logprob)


@dataclass(frozen=True, slots=True)
class GenerationStep:
    """Describe one candidate token selected by an async generation helper."""

    token_id: int
    is_eos: bool
    advanced: bool
    token_bytes: bytes | None = None


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Describe one DS4 session progress notification."""

    event: str
    current: int
    total: int
