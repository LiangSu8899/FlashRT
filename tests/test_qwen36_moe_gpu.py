"""Optional first-light test for Qwen3.6-35B-A3B on SM120.

Run with:

    FLASHRT_QWEN36_MOE_CKPT_DIR=/models/Qwen3.6-35B-A3B \
    PYTHONPATH=. pytest -q tests/test_qwen36_moe_gpu.py
"""

from __future__ import annotations

import importlib
import os

import pytest


CKPT = os.environ.get("FLASHRT_QWEN36_MOE_CKPT_DIR")


def _require_sm120():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("the kernelized qwen3_5_moe path requires SM120")
    try:
        kernels = importlib.import_module("flash_rt.flash_rt_kernels")
    except ImportError as exc:
        pytest.skip(f"FlashRT CUDA extension is unavailable: {exc}")
    if not hasattr(kernels, "moe_weighted_sum_sm120_bf16"):
        pytest.skip("FlashRT was built without qwen3_5_moe kernels")
    return torch, kernels


def _weighted_sum(kernels, d_dn, rows, weights, out):
    import torch

    status = kernels.moe_weighted_sum_sm120_bf16(
        d_dn.data_ptr(),
        rows.data_ptr(),
        weights.data_ptr(),
        out.data_ptr(),
        1,
        weights.numel(),
        d_dn.shape[1],
        d_dn.shape[1],
        torch.cuda.current_stream().cuda_stream,
    )
    assert status == 0


def test_qwen35moe_weighted_sum_is_graph_deterministic():
    torch, kernels = _require_sm120()

    generator = torch.Generator(device="cuda").manual_seed(7)
    d_dn = torch.randn(
        8, 2048, dtype=torch.bfloat16, device="cuda",
        generator=generator)
    rows = torch.arange(8, dtype=torch.int32, device="cuda")
    weights = torch.softmax(
        torch.randn(8, dtype=torch.float32, device="cuda",
                    generator=generator),
        dim=0,
    ).contiguous()
    reference = weights @ d_dn.float()
    out = torch.empty(2048, dtype=torch.float32, device="cuda")

    _weighted_sum(kernels, d_dn, rows, weights, out)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    assert (out - reference).abs().max().item() <= 5e-7

    eager_results = []
    for _ in range(20):
        _weighted_sum(kernels, d_dn, rows, weights, out)
        eager_results.append(out.clone())
    torch.cuda.synchronize()
    assert all(torch.equal(eager_results[0], result)
               for result in eager_results[1:])

    graph_out = torch.empty_like(out)
    _weighted_sum(kernels, d_dn, rows, weights, graph_out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _weighted_sum(kernels, d_dn, rows, weights, graph_out)

    graph_results = []
    for _ in range(20):
        graph.replay()
        graph_results.append(graph_out.clone())
    torch.cuda.synchronize()
    assert torch.isfinite(graph_results[0]).all()
    assert all(torch.equal(graph_results[0], result)
               for result in graph_results[1:])
    assert torch.equal(graph_results[0], eager_results[0])


@pytest.mark.skipif(
    not CKPT,
    reason="set FLASHRT_QWEN36_MOE_CKPT_DIR for the real-model test",
)
def test_qwen36_moe_first_light_matches_hf_golden():
    torch, _ = _require_sm120()

    from flash_rt.frontends.torch.qwen36_moe_rtx import (
        Qwen36MoeTextFrontendRtx,
    )

    frontend = Qwen36MoeTextFrontendRtx(
        CKPT,
        device="cuda:0",
        max_seq=128,
        kernelized=True,
        quant_scope="experts",
    )
    messages = [{
        "role": "user",
        "content": "Write a Python function that merges two sorted lists.",
    }]
    prompt = frontend.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    frontend.set_prompt(prompt)

    with torch.no_grad():
        logits = frontend.infer()
    assert logits.shape == (1, 20, 248320)
    assert torch.isfinite(logits).all()

    # Official Transformers BF16 greedy output for the prompt above.
    golden = [
        8160, 579, 264, 7047, 1817, 25, 271, 16,
        13, 220, 2972, 15771, 2598, 279, 2570, 5952,
    ]
    with torch.no_grad():
        generations = [
            frontend.generate(max_new_tokens=len(golden))
            for _ in range(8)
        ]
    assert generations == [golden] * len(generations)
