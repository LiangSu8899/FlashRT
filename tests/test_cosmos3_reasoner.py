"""Cosmos3 Reasoner request contracts and Thor kernel correctness."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from flash_rt.models.cosmos3_reasoner.pipeline_thor import (
    ATTN_SPLITS,
    HD,
    HID,
    IMAGE_TOKEN_ID,
    NH,
    NKV,
    VIDEO_TOKEN_ID,
    CosmosReasonerThor,
    _require_reasoner_kernels,
)


def _bare_engine(*, max_seq: int = 32, max_new: int = 8) -> CosmosReasonerThor:
    engine = object.__new__(CosmosReasonerThor)
    engine.max_seq = max_seq
    engine.max_new = max_new
    engine.w = {
        "model.visual.embeddings.patch_embedding.weight": torch.empty(1, 768),
    }
    return engine


@pytest.mark.parametrize("quant", ["bf16", "fp4"])
def test_reasoner_required_symbols_fail_before_model_setup(quant: str):
    with pytest.raises(RuntimeError, match=r"FLASHRT_ENABLE_COSMOS3_REASONER=ON") as exc:
        _require_reasoner_kernels(SimpleNamespace(), quant)
    message = str(exc.value)
    if quant == "bf16":
        assert "cosmos3_reasoner_rope_kv_bf16" in message
        assert "cosmos3_reasoner_decode_attn_bf16" in message
    else:
        assert "cosmos3_reasoner_gemv_w4a16_bf16" in message
        assert "cosmos3_reasoner_rope_kv_fp8_bf16" in message
        assert "cosmos3_reasoner_decode_attn_fp8kv_bf16" in message


def test_reasoner_constructor_checks_symbols_before_checkpoint(monkeypatch):
    import flash_rt

    fake = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "flash_rt.flash_rt_kernels", fake)
    monkeypatch.setattr(flash_rt, "flash_rt_kernels", fake, raising=False)
    with pytest.raises(RuntimeError, match=r"FLASHRT_ENABLE_COSMOS3_REASONER=ON"):
        CosmosReasonerThor("/path/that/does/not/exist", quant="fp4")


def test_reasoner_text_request_bounds():
    engine = _bare_engine(max_seq=12, max_new=4)
    ids = torch.tensor([1, 2, 3], dtype=torch.long)
    assert engine._validate_request(ids, None, None, False, None) == (3, 4)
    assert engine._validate_request(ids, None, None, False, 1) == (3, 1)
    assert engine._validate_request(torch.ones(8, dtype=torch.long), None, None, False, 4) == (8, 4)

    for value in (0, -1, 5, True, 1.5, "2"):
        with pytest.raises(ValueError, match="max_new_tokens"):
            engine._validate_request(ids, None, None, False, value)
    with pytest.raises(ValueError, match="at least one"):
        engine._validate_request(torch.empty(0, dtype=torch.long), None, None, False, 1)
    with pytest.raises(ValueError, match="rank-1"):
        engine._validate_request(ids.view(1, -1), None, None, False, 1)
    with pytest.raises(ValueError, match="dtype=torch.long"):
        engine._validate_request(ids.int(), None, None, False, 1)
    with pytest.raises(ValueError, match="exceeds max_seq"):
        engine._validate_request(torch.ones(10, dtype=torch.long), None, None, False, 3)


@pytest.mark.parametrize(
    ("is_video", "token"),
    [(False, IMAGE_TOKEN_ID), (True, VIDEO_TOKEN_ID)],
)
def test_reasoner_visual_request_contract(is_video: bool, token: int):
    engine = _bare_engine()
    ids = torch.tensor([1, token, 2], dtype=torch.long)
    grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
    pixels = torch.randn(4, 768)
    assert engine._validate_request(ids, pixels, grid, is_video, 2) == (3, 2)

    with pytest.raises(ValueError, match="provided together"):
        engine._validate_request(ids, pixels, None, is_video, 2)
    with pytest.raises(ValueError, match="provided together"):
        engine._validate_request(ids, None, grid, is_video, 2)
    with pytest.raises(ValueError, match="no visual input"):
        engine._validate_request(ids, None, None, is_video, 2)
    with pytest.raises(ValueError, match="rows do not match"):
        engine._validate_request(ids, pixels[:3], grid, is_video, 2)
    with pytest.raises(ValueError, match="width must be 768"):
        engine._validate_request(ids, pixels[:, :16], grid, is_video, 2)
    with pytest.raises(ValueError, match="integer tensor"):
        engine._validate_request(ids, pixels, grid.float(), is_video, 2)
    with pytest.raises(ValueError, match="divisible by merge"):
        bad_grid = torch.tensor([[1, 1, 4]], dtype=torch.long)
        engine._validate_request(ids, pixels, bad_grid, is_video, 2)
    with pytest.raises(ValueError, match="requires 1 merged vision tokens"):
        engine._validate_request(torch.tensor([1, 2]), pixels, grid, is_video, 2)


def _reasoner_kernels():
    if not torch.cuda.is_available():
        pytest.skip("Reasoner kernel canary requires CUDA")
    try:
        import flash_rt.flash_rt_kernels as fvk
    except Exception as exc:  # pragma: no cover - build dependent
        pytest.skip(f"flash_rt_kernels is unavailable: {exc}")
    required = (
        "cosmos3_reasoner_rope_kv_bf16",
        "cosmos3_reasoner_rope_kv_fp8_bf16",
        "cosmos3_reasoner_decode_attn_bf16",
        "cosmos3_reasoner_decode_attn_fp8kv_bf16",
        "cosmos3_reasoner_gemv_w4a16_bf16",
    )
    if any(not hasattr(fvk, name) for name in required):
        pytest.skip("Cosmos3 Reasoner kernels are not enabled in this build")
    return fvk


def test_reasoner_graph_capture_precedes_prompt_prefill():
    if not torch.cuda.is_available():
        pytest.skip("generate ordering test requires CUDA")
    engine = _bare_engine(max_seq=8, max_new=1)
    engine.quant = "test"
    engine.use_graph = True
    engine.eos_token_id = 2
    engine.w["embed_tokens.weight"] = torch.randn(
        32, 4, device="cuda", dtype=torch.bfloat16
    )
    events: list[str] = []

    def ensure_graph():
        events.append("capture")

    def forward_prefill(_embeds, _cos, _sin, _start, *, causal):
        assert causal is True
        events.append("prefill")
        logits = torch.zeros(1, 32, device="cuda")
        logits[0, 3] = 1
        return logits

    engine._ensure_graph = ensure_graph
    engine._forward = forward_prefill
    engine._cos_sin = lambda pos: (pos.to(torch.bfloat16), pos.to(torch.bfloat16))
    out, _ = engine.generate(torch.tensor([1], dtype=torch.long), max_new_tokens=1)
    assert out == [3]
    assert events == ["capture", "prefill"]


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(
        actual.float().reshape(1, -1), expected.float().reshape(1, -1)
    ))


@pytest.mark.parametrize("fp8_cache", [False, True])
def test_reasoner_rope_kv_matches_torch(fp8_cache: bool):
    fvk = _reasoner_kernels()
    torch.manual_seed(7)
    device = "cuda"
    max_seq, pos, slot = 16, 5, 7
    q = (torch.randn(NH, HD, device=device) * 0.25).to(torch.bfloat16)
    k = (torch.randn(NKV, HD, device=device) * 0.25).to(torch.bfloat16)
    v = (torch.randn(NKV, HD, device=device) * 0.25).to(torch.bfloat16)
    angles = torch.randn(max_seq, HD // 2, device=device)
    emb = torch.cat((angles, angles), dim=-1)
    cos = emb.cos().to(torch.bfloat16)
    sin = emb.sin().to(torch.bfloat16)
    d_pos = torch.tensor([pos], dtype=torch.long, device=device)
    d_slot = torch.tensor([slot], dtype=torch.long, device=device)
    q_out = torch.empty_like(q)
    cache_dtype = torch.uint8 if fp8_cache else torch.bfloat16
    kc = torch.zeros(max_seq, NKV, HD, dtype=cache_dtype, device=device)
    vc = torch.zeros_like(kc)
    stream = torch.cuda.current_stream().cuda_stream
    name = "cosmos3_reasoner_rope_kv_fp8_bf16" if fp8_cache else "cosmos3_reasoner_rope_kv_bf16"
    getattr(fvk, name)(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), cos.data_ptr(), sin.data_ptr(),
        d_pos.data_ptr(), d_slot.data_ptr(), q_out.data_ptr(), kc.data_ptr(), vc.data_ptr(),
        NH, NKV, stream,
    )
    torch.cuda.synchronize()

    def rope_float(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[:, half:], x[:, :half]), dim=-1)
        return x.float() * cos[pos].float() + rotated.float() * sin[pos].float()

    def rope_ref(x: torch.Tensor) -> torch.Tensor:
        return rope_float(x).to(torch.bfloat16)

    assert torch.equal(q_out, rope_ref(q))
    k_ref = rope_ref(k)
    if fp8_cache:
        k_actual = kc.view(torch.float8_e4m3fn)[slot].float()
        v_actual = vc.view(torch.float8_e4m3fn)[slot].float()
        assert torch.equal(k_actual, rope_float(k).to(torch.float8_e4m3fn).float())
        assert torch.equal(v_actual, v.to(torch.float8_e4m3fn).float())
    else:
        assert torch.equal(kc[slot], k_ref)
        assert torch.equal(vc[slot], v)


@pytest.mark.parametrize("fp8_cache", [False, True])
def test_reasoner_decode_attention_matches_torch(fp8_cache: bool):
    fvk = _reasoner_kernels()
    torch.manual_seed(11)
    device = "cuda"
    length = 73
    q = (torch.randn(NH, HD, device=device) * 0.2).to(torch.bfloat16)
    k = (torch.randn(length, NKV, HD, device=device) * 0.2).to(torch.bfloat16)
    v = (torch.randn(length, NKV, HD, device=device) * 0.2).to(torch.bfloat16)
    if fp8_cache:
        kc = k.to(torch.float8_e4m3fn).view(torch.uint8)
        vc = v.to(torch.float8_e4m3fn).view(torch.uint8)
        k_ref = kc.view(torch.float8_e4m3fn).float()
        v_ref = vc.view(torch.float8_e4m3fn).float()
        name = "cosmos3_reasoner_decode_attn_fp8kv_bf16"
    else:
        kc, vc = k, v
        k_ref, v_ref = k.float(), v.float()
        name = "cosmos3_reasoner_decode_attn_bf16"
    d_len = torch.tensor([length], dtype=torch.int32, device=device)
    out = torch.empty(NH, HD, dtype=torch.bfloat16, device=device)
    pacc = torch.empty(NH, ATTN_SPLITS, HD, dtype=torch.float32, device=device)
    pml = torch.empty(NH, ATTN_SPLITS, 2, dtype=torch.float32, device=device)
    scale = HD ** -0.5
    getattr(fvk, name)(
        q.data_ptr(), kc.data_ptr(), vc.data_ptr(), d_len.data_ptr(), out.data_ptr(),
        pacc.data_ptr(), pml.data_ptr(), NH, NKV, scale,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    group = NH // NKV
    kh = k_ref.permute(1, 0, 2).repeat_interleave(group, dim=0)
    vh = v_ref.permute(1, 0, 2).repeat_interleave(group, dim=0)
    scores = torch.einsum("hd,hld->hl", q.float(), kh) * scale
    ref = torch.einsum("hl,hld->hd", scores.softmax(dim=-1), vh).to(torch.bfloat16)
    assert _cosine(out, ref) >= 0.999
    assert torch.allclose(out.float(), ref.float(), atol=0.015, rtol=0.03)


def test_reasoner_w4a16_gemv_matches_dequantized_reference():
    fvk = _reasoner_kernels()
    torch.manual_seed(19)
    device = "cuda"
    n, k = 37, 256
    weight = torch.randn(n, k, device=device) * 0.15
    activation = torch.randn(k, device=device).to(torch.bfloat16)
    blocks = weight.view(n, k // 16, 16)
    scales = (blocks.abs().amax(dim=-1) / 6.0).clamp(min=1e-8)
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device)
    normalized = blocks / scales[..., None]
    indices = (normalized.abs()[..., None] - levels).abs().argmin(dim=-1)
    codes = (indices + (normalized < 0) * 8).to(torch.uint8).view(n, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    scales_bf16 = scales.to(torch.bfloat16).contiguous()
    out = torch.empty(n, dtype=torch.bfloat16, device=device)
    fvk.cosmos3_reasoner_gemv_w4a16_bf16(
        packed.data_ptr(), scales_bf16.data_ptr(), activation.data_ptr(), out.data_ptr(),
        n, k, torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    signed_levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=device,
    )
    dequant = signed_levels[codes.long()].view(n, k // 16, 16) * scales_bf16.float()[..., None]
    ref = (dequant.view(n, k) @ activation.float()).to(torch.bfloat16)
    assert _cosine(out, ref) >= 0.9999
    assert torch.allclose(out.float(), ref.float(), atol=0.125, rtol=0.02)
