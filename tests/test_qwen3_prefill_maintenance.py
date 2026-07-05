"""Static maintenance guardrails for Qwen3 prefill kernel integration."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_qwen3_fast_prefill_paths_are_default_with_escape_hatches():
    src = _read("flash_rt/frontends/torch/qwen3_rtx.py")
    attn = _read("flash_rt/hardware/rtx/attn_backend_qwen3.py")

    assert "FLASH_RT_QWEN3_RESID_FUSE', '1'" in src
    assert "FLASH_RT_QWEN3_SWIGLU_FOLD', '1'" in src
    assert 'FLASH_RT_QWEN3_FP8_FMHA", "1"' in attn
    assert "FLASH_RT_QWEN3_NO_RESID_FUSE" in src


def test_qwen3_fp8_attention_not_gated_by_motus_object():
    cmake = _read("CMakeLists.txt")
    motus_obj = re.search(
        r"add_library\(motus_aux_sm120_obj OBJECT(?P<body>.*?)\)\n",
        cmake,
        re.S,
    )
    assert motus_obj is not None
    assert "sage2_attn_f8_raw.cu" not in motus_obj.group("body")
    assert "fmha_fp8_causal_gqa_sm120.cu" not in motus_obj.group("body")

    qwen3_obj = re.search(
        r"add_library\(qwen3_prefill_sm120_obj OBJECT(?P<body>.*?)\)\n",
        cmake,
        re.S,
    )
    assert qwen3_obj is not None
    assert "sage2_attn_f8_raw.cu" in qwen3_obj.group("body")
    assert "fmha_fp8_causal_gqa_sm120.cu" in qwen3_obj.group("body")
    assert "ENABLE_QWEN3_FP8_PREFILL_ATTN=1" in cmake


def test_dev_probe_bindings_are_explicitly_gated():
    cmake = _read("CMakeLists.txt")
    bindings = _read("csrc/bindings.cpp")

    assert "option(FLASHRT_ENABLE_SM120_DEV_KERNELS" in cmake
    assert "FLASHRT_HAVE_SM120_NVFP4_DEV=1" in cmake

    for symbol in (
        "fp4_w4a16_tilesweep_sm120_bf16out",
        "fp4_normfold_probe_sm120_bf16out",
        "fp4_normfold_pq_probe_sm120",
    ):
        idx = bindings.index(symbol)
        guard = bindings.rfind("#ifdef FLASHRT_HAVE_SM120_NVFP4_DEV", 0, idx)
        end = bindings.rfind("#endif", 0, idx)
        assert guard > end


def test_kernel_sources_are_not_in_both_core_and_gated_lists():
    cmake = _read("CMakeLists.txt")

    for source in (
        "csrc/kernels/qwen36_misc.cu",
        "csrc/kernels/silu_mul_to_nvfp4_swizzled.cu",
        "csrc/kernels/fp4_swiglu_compact_sm120.cu",
    ):
        assert cmake.count(source) == 1
