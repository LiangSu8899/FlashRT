"""Optional first-light test for Qwen3.6-35B-A3B on SM120.

Run with:

    FLASHRT_QWEN36_MOE_CKPT_DIR=/models/Qwen3.6-35B-A3B \
    PYTHONPATH=. pytest -q tests/test_qwen36_moe_gpu.py
"""

from __future__ import annotations

import os

import pytest


CKPT = os.environ.get("FLASHRT_QWEN36_MOE_CKPT_DIR")

pytestmark = pytest.mark.skipif(
    not CKPT,
    reason="set FLASHRT_QWEN36_MOE_CKPT_DIR for the real-model test",
)


def test_qwen36_moe_first_light_matches_hf_golden():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("the kernelized qwen3_5_moe path requires SM120")

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
