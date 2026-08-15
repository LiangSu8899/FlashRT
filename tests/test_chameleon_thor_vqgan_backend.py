from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

try:
    from flash_rt.frontends.torch.chameleon_thor import ChameleonTorchFrontendThor
except ModuleNotFoundError as exc:
    root = Path(__file__).resolve().parents[1]
    build_dir = Path(os.environ.get("FLASHRT_BUILD_DIR", root / "build"))
    cache = build_dir / "CMakeCache.txt"
    chameleon_enabled = cache.is_file() and (
        "FLASHRT_ENABLE_CHAMELEON:BOOL=ON" in cache.read_text(errors="replace")
    )
    if chameleon_enabled or exc.name != "flash_rt.flash_rt_kernels":
        raise
    pytest.skip("flash_rt_kernels is not built", allow_module_level=True)


def test_chameleon_trt_vqgan_is_opt_in_by_default():
    sig = inspect.signature(ChameleonTorchFrontendThor.__init__)
    assert sig.parameters["use_trt_vqgan"].default is False


def test_chameleon_fa4_attn_is_opt_in_by_default():
    sig = inspect.signature(ChameleonTorchFrontendThor.__init__)
    assert sig.parameters["use_fa4_attn"].default is None
    os.environ.pop("FLASHRT_CHAMELEON_FA4_ATTN", None)
    assert bool(os.environ.get("FLASHRT_CHAMELEON_FA4_ATTN", "0") in ("1", "true", "on")) is False
