from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
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


class TokenScoreMode(StrEnum):
    """Name the token score detail requested during generation."""

    NONE = "none"
    TOKEN_LOGPROB = "token_logprob"
    TOP_LOGPROBS = "top_logprobs"
    TOKEN_LOGPROB_AND_TOP_LOGPROBS = "token_logprob_and_top_logprobs"


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


def _validate_optional_bytes(name: str, value: bytes | None) -> None:
    if value is None:
        return
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes or None.")


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


def _validate_optional_real(name: str, value: float | None) -> None:
    if value is None:
        return
    _validate_real(name, value)


def _validate_stop_strings(
    value: str | list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    result: tuple[str, ...]
    if isinstance(value, str):
        result = (value,)
    elif isinstance(value, (list, tuple)):
        result = tuple(value)
    else:
        raise TypeError(
            "stop_strings must be a string, list, or tuple of strings."
        )

    for item in result:
        _validate_str("stop_strings item", item)
        if not item:
            raise ValueError("stop_strings items must not be empty.")
    return result


def _validate_token_score_tuple(
    name: str,
    value: Iterable["TokenScore"],
) -> tuple["TokenScore", ...]:
    if isinstance(value, (bytes, str)) or not isinstance(value, Iterable):
        raise TypeError(f"{name} must be an iterable of TokenScore objects.")
    result = tuple(value)
    for item in result:
        if not isinstance(item, TokenScore):
            raise TypeError(f"{name} items must be TokenScore objects.")
    return result


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
class GenerationScoreOptions:
    """Configure optional token score details for generation steps."""

    mode: TokenScoreMode = TokenScoreMode.NONE
    top_k: int = 0

    def __post_init__(self) -> None:
        try:
            mode = TokenScoreMode(self.mode)
        except ValueError as error:
            raise ValueError(
                f"Unsupported token score mode {self.mode!r}."
            ) from error
        object.__setattr__(self, "mode", mode)
        _validate_int("top_k", self.top_k, minimum=0, maximum=C_INT_MAX)

        requests_top_logprobs = mode in {
            TokenScoreMode.TOP_LOGPROBS,
            TokenScoreMode.TOKEN_LOGPROB_AND_TOP_LOGPROBS,
        }
        if requests_top_logprobs and self.top_k == 0:
            raise ValueError(
                "top_k must be positive when top logprobs are requested."
            )
        if not requests_top_logprobs and self.top_k != 0:
            raise ValueError(
                "top_k must be 0 unless top logprobs are requested."
            )


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Configure framework-neutral DS4 token generation behavior."""

    max_new_tokens: int = 1
    sampling: SamplingOptions | None = None
    stop_strings: str | list[str] | tuple[str, ...] = ()
    stop_on_eos: bool = True
    decode: bool = False
    scores: GenerationScoreOptions = field(
        default_factory=GenerationScoreOptions
    )
    advance: bool = True

    def __post_init__(self) -> None:
        _validate_int(
            "max_new_tokens",
            self.max_new_tokens,
            minimum=0,
            maximum=C_INT_MAX,
        )
        if self.sampling is not None and not isinstance(
            self.sampling,
            SamplingOptions,
        ):
            raise TypeError(
                "sampling must be a SamplingOptions instance or None."
            )
        object.__setattr__(
            self,
            "stop_strings",
            _validate_stop_strings(self.stop_strings),
        )
        _validate_bool("stop_on_eos", self.stop_on_eos)
        _validate_bool("decode", self.decode)
        if not isinstance(self.scores, GenerationScoreOptions):
            raise TypeError(
                "scores must be a GenerationScoreOptions instance."
            )
        _validate_bool("advance", self.advance)


@dataclass(frozen=True, slots=True)
class GenerationStep:
    """Describe one candidate token selected by an async generation helper."""

    token_id: int
    is_eos: bool
    advanced: bool
    token_bytes: bytes | None = None
    decoded_text: str | None = None
    token_logprob: float | None = None
    top_logprobs: tuple[TokenScore, ...] = ()

    def __post_init__(self) -> None:
        _validate_int(
            "token_id",
            self.token_id,
            minimum=0,
            maximum=C_INT_MAX,
        )
        _validate_bool("is_eos", self.is_eos)
        _validate_bool("advanced", self.advanced)
        _validate_optional_bytes("token_bytes", self.token_bytes)
        _validate_optional_str("decoded_text", self.decoded_text)
        _validate_optional_real("token_logprob", self.token_logprob)
        object.__setattr__(
            self,
            "top_logprobs",
            _validate_token_score_tuple("top_logprobs", self.top_logprobs),
        )


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Describe one DS4 session progress notification."""

    event: str
    current: int
    total: int
