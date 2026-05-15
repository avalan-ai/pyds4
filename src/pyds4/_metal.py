from __future__ import annotations

import os
from collections.abc import Iterable, MutableMapping
from pathlib import Path

from ._version import __ds4_native_backend__

METAL_SOURCE_ENV = {
    "DS4_METAL_FLASH_ATTN_SOURCE": "flash_attn.metal",
    "DS4_METAL_DENSE_SOURCE": "dense.metal",
    "DS4_METAL_MOE_SOURCE": "moe.metal",
    "DS4_METAL_DSV4_HC_SOURCE": "dsv4_hc.metal",
    "DS4_METAL_UNARY_SOURCE": "unary.metal",
    "DS4_METAL_DSV4_KV_SOURCE": "dsv4_kv.metal",
    "DS4_METAL_DSV4_ROPE_SOURCE": "dsv4_rope.metal",
    "DS4_METAL_DSV4_MISC_SOURCE": "dsv4_misc.metal",
    "DS4_METAL_ARGSORT_SOURCE": "argsort.metal",
    "DS4_METAL_CPY_SOURCE": "cpy.metal",
    "DS4_METAL_CONCAT_SOURCE": "concat.metal",
    "DS4_METAL_GET_ROWS_SOURCE": "get_rows.metal",
    "DS4_METAL_SUM_ROWS_SOURCE": "sum_rows.metal",
    "DS4_METAL_SOFTMAX_SOURCE": "softmax.metal",
    "DS4_METAL_REPEAT_SOURCE": "repeat.metal",
    "DS4_METAL_GLU_SOURCE": "glu.metal",
    "DS4_METAL_NORM_SOURCE": "norm.metal",
    "DS4_METAL_BIN_SOURCE": "bin.metal",
    "DS4_METAL_SET_ROWS_SOURCE": "set_rows.metal",
}


def _complete_metal_source_dir(package_paths: Iterable[str]) -> Path | None:
    for package_path in package_paths:
        metal_dir = Path(package_path) / "metal"
        if all(
            (metal_dir / filename).is_file()
            for filename in METAL_SOURCE_ENV.values()
        ):
            return metal_dir
    return None


def configure_metal_source_paths(
    *,
    backend: str = __ds4_native_backend__,
    package_paths: Iterable[str] | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> Path | None:
    """Point DS4's Metal runtime at packaged kernel source files.

    Upstream DS4 loads Metal kernels from `metal/*.metal` or from
    DS4_METAL_*_SOURCE overrides during engine open. Wheels are imported from
    arbitrary working directories, so a Metal build must seed those overrides
    without requiring callers to run from a DS4 checkout.
    """
    if backend != "metal":
        return None
    if environ is None:
        environ = os.environ
    if package_paths is None:
        import pyds4

        package_paths = pyds4.__path__
    metal_dir = _complete_metal_source_dir(package_paths)
    if metal_dir is None:
        return None
    for env_name, filename in METAL_SOURCE_ENV.items():
        environ.setdefault(env_name, str(metal_dir / filename))
    return metal_dir
