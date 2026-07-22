"""Cosmos3-Edge native kernel canaries."""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 epilogue canary requires CUDA")
def test_cosmos3_edge_fp4_relu2_epilogue_matches_split_chain():
    try:
        fp4 = importlib.import_module("flash_rt.flash_rt_fp4")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_fp4 not importable: {exc}")
    required = (
        "cosmos3_edge_fp4_gemm_relu2_fp4out",
        "quantize_fp4_dynamic_sfa_mse_fp16",
    )
    if any(not hasattr(fp4, name) for name in required):
        pytest.skip("Cosmos3-Edge fused FP4 epilogue is not built")

    m, n, k = 128, 256, 256
    torch.manual_seed(8675309)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.2
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
    stream = torch.cuda.current_stream().cuda_stream

    def packed(rows: int, cols: int):
        data = torch.empty(rows, cols // 2, device="cuda", dtype=torch.uint8)
        sfa = torch.empty(fp4.sfa_size_bytes(rows, cols, False), device="cuda", dtype=torch.uint8)
        return data, sfa

    xp, xs = packed(m, k)
    wp = torch.empty(n, k // 2, device="cuda", dtype=torch.uint8)
    ws = torch.empty(fp4.sfa_size_bytes(n, k, True), device="cuda", dtype=torch.uint8)
    assert fp4.quantize_fp4_dynamic_sfa_fp16(
        x.data_ptr(), xp.data_ptr(), xs.data_ptr(), m, k, False, stream) == 0
    assert fp4.quantize_fp4_dynamic_sfa_mse_fp16(
        w.data_ptr(), wp.data_ptr(), ws.data_ptr(), n, k, True, stream) == 0

    split16 = torch.empty(m, n, device="cuda", dtype=torch.float16)
    split_p, split_s = packed(m, n)
    fused_p, fused_s = packed(m, n)
    assert fp4.cutlass_fp4_gemm_variant(
        3, xp.data_ptr(), xs.data_ptr(), wp.data_ptr(), ws.data_ptr(),
        split16.data_ptr(), m, n, k, 1.0, 0.0, stream) == 0
    fp4.cosmos3_edge_relu2_fp4_sfa_fp16(
        split16.data_ptr(), split_p.data_ptr(), split_s.data_ptr(), m, n, stream)
    assert fp4.cosmos3_edge_fp4_gemm_relu2_fp4out(
        xp.data_ptr(), xs.data_ptr(), wp.data_ptr(), ws.data_ptr(),
        fused_p.data_ptr(), fused_s.data_ptr(), m, n, k, stream) == 0
    torch.cuda.synchronize()

    # Validate the actual interface contract: both packed outputs and their
    # generated SFA must be consumable by the same downstream FP4 GEMM. Exact
    # codes differ because the fused path activates FP32 accumulators while the
    # split path rounds through FP16 first.
    down_n = 128
    down_w = torch.randn(down_n, n, device="cuda", dtype=torch.float16) * 0.02
    down_wp = torch.empty(down_n, n // 2, device="cuda", dtype=torch.uint8)
    down_ws = torch.empty(
        fp4.sfa_size_bytes(down_n, n, True), device="cuda", dtype=torch.uint8)
    assert fp4.quantize_fp4_dynamic_sfa_mse_fp16(
        down_w.data_ptr(), down_wp.data_ptr(), down_ws.data_ptr(),
        down_n, n, True, stream) == 0
    split_out = torch.empty(m, down_n, device="cuda", dtype=torch.float16)
    fused_out = torch.empty_like(split_out)
    assert fp4.cutlass_fp4_gemm_variant(
        3, split_p.data_ptr(), split_s.data_ptr(), down_wp.data_ptr(), down_ws.data_ptr(),
        split_out.data_ptr(), m, down_n, n, 1.0, 0.0, stream) == 0
    assert fp4.cutlass_fp4_gemm_variant(
        3, fused_p.data_ptr(), fused_s.data_ptr(), down_wp.data_ptr(), down_ws.data_ptr(),
        fused_out.data_ptr(), m, down_n, n, 1.0, 0.0, stream) == 0
    torch.cuda.synchronize()
    cosine = F.cosine_similarity(split_out.float().flatten(), fused_out.float().flatten(), dim=0)
    assert cosine.item() > 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="relu2 kernel canary requires CUDA")
@pytest.mark.parametrize("numel", [10000, 10001])
def test_cosmos3_edge_relu2_inplace_bf16_matches_torch(numel: int):
    try:
        fvk = importlib.import_module("flash_rt.flash_rt_kernels")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")
    if not hasattr(fvk, "relu2_inplace_bf16"):
        pytest.skip("relu2_inplace_bf16 binding is not built")

    torch.manual_seed(numel)
    x = torch.randn(numel, device="cuda", dtype=torch.bfloat16) * 3
    expected = torch.relu(x).square()
    actual = x.clone()
    fvk.relu2_inplace_bf16(
        actual.data_ptr(),
        actual.numel(),
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="qk norm+RoPE kernel canary requires CUDA")
def test_cosmos3_edge_qk_norm_rope_bf16_matches_two_step_kernels():
    try:
        fvk = importlib.import_module("flash_rt.flash_rt_kernels")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")
    if not hasattr(fvk, "cosmos3_edge_qk_norm_rope_bf16"):
        pytest.skip("cosmos3_edge_qk_norm_rope_bf16 binding not built")

    rows, q_heads, k_heads, head_dim = 17, 16, 8, 128
    torch.manual_seed(1234)
    q = torch.randn(rows, q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(rows, k_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    q_w = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    k_w = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(rows, head_dim, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn(rows, head_dim, device="cuda", dtype=torch.bfloat16)

    # Reference is the fp64 math (per-head RMSNorm -> rotate-half RoPE). The
    # fused kernel keeps fp32 through the whole chain and rounds to bf16 once,
    # so it must sit within bf16-output rounding of the exact math (it is
    # closer to it than the two-step norm->bf16->rope kernel chain).
    def _ref64(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x64 = x.double()
        rms = torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + 1e-5)
        n = x64 * rms * w.double()
        half = head_dim // 2
        rot = torch.cat([-n[..., half:], n[..., :half]], dim=-1)
        return n * cos.double().unsqueeze(1) + rot * sin.double().unsqueeze(1)

    q_expected = _ref64(q, q_w)
    k_expected = _ref64(k, k_w)
    stream = torch.cuda.current_stream().cuda_stream

    q_actual = torch.empty_like(q)
    k_actual = torch.empty_like(k)
    fvk.cosmos3_edge_qk_norm_rope_bf16(
        q.data_ptr(),
        k.data_ptr(),
        q_w.data_ptr(),
        k_w.data_ptr(),
        cos.data_ptr(),
        sin.data_ptr(),
        q_actual.data_ptr(),
        k_actual.data_ptr(),
        rows,
        q_heads,
        k_heads,
        head_dim,
        head_dim,
        1e-5,
        stream,
    )
    torch.cuda.synchronize()

    q_max = (q_actual.double() - q_expected).abs().max().item()
    k_max = (k_actual.double() - k_expected).abs().max().item()
    q_cos = F.cosine_similarity(q_actual.double().flatten(), q_expected.flatten(), dim=0).item()
    k_cos = F.cosine_similarity(k_actual.double().flatten(), k_expected.flatten(), dim=0).item()
    assert q_max <= 0.09  # bf16 output ulp at these randn magnitudes
    assert k_max <= 0.09
    assert q_cos > 0.99999
    assert k_cos > 0.99999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 conv3d canary requires CUDA")
def test_cosmos3_edge_fp8_conv3d_v17_binding_matches_torch_small_shape():
    try:
        fvk = importlib.import_module("flash_rt.flash_rt_kernels")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")
    missing = [
        name
        for name in (
            "fp8_conv3d_v17_ndhwc_bf16out",
            "bf16_ndhwc_to_ncdhw_transpose",
        )
        if not hasattr(fvk, name)
    ]
    if missing:
        pytest.skip(f"required FP8 conv3d binding(s) not built: {missing}")

    b, t_cache, t_new, h, w, ci, co = 1, 2, 3, 4, 5, 32, 8
    torch.manual_seed(1357)
    cache = (torch.randn(b, t_cache, h, w, ci, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    new = (torch.randn(b, t_new, h, w, ci, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    weight = (torch.randn(co, 3, 3, 3, ci, device="cuda") * 0.08).to(torch.float8_e4m3fn)
    bias = torch.randn(co, device="cuda", dtype=torch.bfloat16) * 0.05
    alpha = 0.75

    y_ndhwc = torch.empty(b, t_new, h, w, co, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.current_stream().cuda_stream
    rc = fvk.fp8_conv3d_v17_ndhwc_bf16out(
        cache.data_ptr(),
        new.data_ptr(),
        weight.data_ptr(),
        y_ndhwc.data_ptr(),
        bias.data_ptr(),
        b,
        t_cache,
        t_new,
        h,
        w,
        ci,
        co,
        alpha,
        stream,
    )
    assert rc == 0
    actual = torch.empty(b, co, t_new, h, w, device="cuda", dtype=torch.bfloat16)
    rc = fvk.bf16_ndhwc_to_ncdhw_transpose(
        y_ndhwc.data_ptr(),
        actual.data_ptr(),
        b,
        co,
        t_new,
        h,
        w,
        stream,
    )
    assert rc == 0
    torch.cuda.synchronize()

    x_ref = torch.cat(
        [
            cache.float().permute(0, 4, 1, 2, 3),
            new.float().permute(0, 4, 1, 2, 3),
        ],
        dim=2,
    )
    w_ref = weight.float().permute(0, 4, 1, 2, 3).contiguous()
    expected = F.conv3d(x_ref, w_ref, bias=None, padding=(0, 1, 1))
    expected = (expected * alpha + bias.float().view(1, co, 1, 1, 1)).to(torch.bfloat16)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 conv3d canary requires CUDA")
def test_cosmos3_edge_bf16_conv3d_v0_binding_matches_torch_small_shape():
    try:
        fvk = importlib.import_module("flash_rt.flash_rt_kernels")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")
    missing = [
        name
        for name in (
            "bf16_conv3d_v0_ndhwc_bf16out",
            "bf16_ndhwc_to_ncdhw_transpose",
        )
        if not hasattr(fvk, name)
    ]
    if missing:
        pytest.skip(f"required BF16 conv3d binding(s) not built: {missing}")

    b, t_cache, t_new, h, w, ci, co = 1, 2, 3, 4, 5, 32, 16
    torch.manual_seed(2468)
    cache = (torch.randn(b, t_cache, h, w, ci, device="cuda") * 0.25).to(torch.bfloat16)
    new = (torch.randn(b, t_new, h, w, ci, device="cuda") * 0.25).to(torch.bfloat16)
    weight = (torch.randn(co, 3, 3, 3, ci, device="cuda") * 0.08).to(torch.bfloat16)
    bias = torch.randn(co, device="cuda", dtype=torch.bfloat16) * 0.05
    alpha = 0.75

    y_ndhwc = torch.empty(b, t_new, h, w, co, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.current_stream().cuda_stream
    rc = fvk.bf16_conv3d_v0_ndhwc_bf16out(
        cache.data_ptr(),
        new.data_ptr(),
        weight.data_ptr(),
        y_ndhwc.data_ptr(),
        bias.data_ptr(),
        b,
        t_cache,
        t_new,
        h,
        w,
        ci,
        co,
        alpha,
        stream,
    )
    assert rc == 0
    actual = torch.empty(b, co, t_new, h, w, device="cuda", dtype=torch.bfloat16)
    rc = fvk.bf16_ndhwc_to_ncdhw_transpose(
        y_ndhwc.data_ptr(),
        actual.data_ptr(),
        b,
        co,
        t_new,
        h,
        w,
        stream,
    )
    assert rc == 0
    torch.cuda.synchronize()

    x_ref = torch.cat(
        [
            cache.permute(0, 4, 1, 2, 3),
            new.permute(0, 4, 1, 2, 3),
        ],
        dim=2,
    )
    w_ref = weight.permute(0, 4, 1, 2, 3).contiguous()
    expected = F.conv3d(x_ref, w_ref, bias=None, padding=(0, 1, 1))
    expected = (expected.float() * alpha + bias.float().view(1, co, 1, 1, 1)).to(torch.bfloat16)

    diff = (actual.float() - expected.float()).abs()
    assert diff.max().item() <= 0.03125
    assert F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item() > 0.9999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flat velocity fill kernel canary requires CUDA")
def test_cosmos3_edge_fill_flat_velocity_bf16_matches_torch():
    try:
        fvk = importlib.import_module("flash_rt.flash_rt_kernels")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")
    if not hasattr(fvk, "cosmos3_edge_fill_flat_velocity_bf16"):
        pytest.skip("cosmos3_edge_fill_flat_velocity_bf16 binding is not built")

    flat_dim = 1201920
    action_numel = 60 * 64
    torch.manual_seed(4321)
    action = torch.randn(action_numel, device="cuda", dtype=torch.bfloat16)
    actual = torch.empty(flat_dim, device="cuda", dtype=torch.bfloat16)
    expected = torch.zeros_like(actual)
    expected[-action_numel:] = action

    fvk.cosmos3_edge_fill_flat_velocity_bf16(
        action.data_ptr(),
        actual.data_ptr(),
        actual.numel(),
        action.numel(),
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="action bias tail kernel canary requires CUDA")
def test_cosmos3_edge_add_bias_zero_action_tail_bf16_matches_torch():
    try:
        fvk = importlib.import_module("flash_rt.flash_rt_kernels")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")
    if not hasattr(fvk, "cosmos3_edge_add_bias_zero_action_tail_bf16"):
        pytest.skip("cosmos3_edge_add_bias_zero_action_tail_bf16 binding is not built")

    rows, cols, valid_cols = 60, 64, 9
    torch.manual_seed(9876)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(cols, device="cuda", dtype=torch.bfloat16)
    expected = x.clone()
    expected.add_(bias)
    expected[:, valid_cols:] = 0
    actual = x.clone()

    fvk.cosmos3_edge_add_bias_zero_action_tail_bf16(
        actual.data_ptr(),
        bias.data_ptr(),
        rows,
        cols,
        valid_cols,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="row scatter/gather kernel canary requires CUDA")
def test_cosmos3_edge_row_scatter_gather_bf16_matches_torch():
    try:
        fvk = importlib.import_module("flash_rt.flash_rt_kernels")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")
    missing = [
        name
        for name in (
            "cosmos3_edge_scatter_rows_bf16",
            "cosmos3_edge_gather_rows_bf16",
        )
        if not hasattr(fvk, name)
    ]
    if missing:
        pytest.skip(f"required kernel binding(s) not built: {missing}")

    rows, hidden, full_rows = 60, 2048, 128
    torch.manual_seed(2468)
    row_indices = torch.randperm(full_rows, device="cuda", dtype=torch.int64)[:rows].contiguous()
    src = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    dst = torch.randn(full_rows, hidden, device="cuda", dtype=torch.bfloat16)
    expected_dst = dst.clone()
    expected_dst[row_indices] = src
    actual_dst = dst.clone()
    stream = torch.cuda.current_stream().cuda_stream

    fvk.cosmos3_edge_scatter_rows_bf16(
        src.data_ptr(),
        actual_dst.data_ptr(),
        row_indices.data_ptr(),
        rows,
        hidden,
        stream,
    )
    gathered = torch.empty_like(src)
    fvk.cosmos3_edge_gather_rows_bf16(
        actual_dst.data_ptr(),
        gathered.data_ptr(),
        row_indices.data_ptr(),
        rows,
        hidden,
        stream,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual_dst, expected_dst)
    assert torch.equal(gathered, src)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="action tail copy kernel canary requires CUDA")
def test_cosmos3_edge_copy_action_tail_f32_to_bf16_matches_torch():
    try:
        fvk = importlib.import_module("flash_rt.flash_rt_kernels")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")
    if not hasattr(fvk, "cosmos3_edge_copy_action_tail_f32_to_bf16"):
        pytest.skip("cosmos3_edge_copy_action_tail_f32_to_bf16 binding is not built")

    flat_dim = 1201920
    action_numel = 60 * 64
    torch.manual_seed(1357)
    flat_noise = torch.randn(flat_dim, device="cuda", dtype=torch.float32)
    expected = flat_noise[-action_numel:].to(dtype=torch.bfloat16)
    actual = torch.empty(action_numel, device="cuda", dtype=torch.bfloat16)

    fvk.cosmos3_edge_copy_action_tail_f32_to_bf16(
        flat_noise.data_ptr(),
        actual.data_ptr(),
        flat_noise.numel(),
        action_numel,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="action bias+timestep kernel canary requires CUDA")
def test_cosmos3_edge_add_action_bias_timestep_bf16_matches_torch():
    try:
        fvk = importlib.import_module("flash_rt.flash_rt_kernels")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")
    if not hasattr(fvk, "cosmos3_edge_add_action_bias_timestep_bf16"):
        pytest.skip("cosmos3_edge_add_action_bias_timestep_bf16 binding is not built")

    rows, hidden = 60, 2048
    torch.manual_seed(8642)
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    static_bias = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    timestep = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    expected = x.clone()
    expected.add_(static_bias)
    expected.add_(timestep.expand(rows, hidden))
    actual = x.clone()

    fvk.cosmos3_edge_add_action_bias_timestep_bf16(
        actual.data_ptr(),
        static_bias.data_ptr(),
        timestep.data_ptr(),
        rows,
        hidden,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 add kernel canary requires CUDA")
@pytest.mark.parametrize("shape", [(60, 2048), (257,)])
def test_cosmos3_edge_add_bf16_matches_torch(shape: tuple[int, ...]):
    try:
        fvk = importlib.import_module("flash_rt.flash_rt_kernels")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")
    if not hasattr(fvk, "cosmos3_edge_add_bf16"):
        pytest.skip("cosmos3_edge_add_bf16 binding is not built")

    torch.manual_seed(sum(shape))
    a = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    expected = a + b
    actual = torch.empty_like(expected)

    fvk.cosmos3_edge_add_bf16(
        a.data_ptr(),
        b.data_ptr(),
        actual.data_ptr(),
        actual.numel(),
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


def _avgdown3d_reference(
    x: torch.Tensor,
    *,
    out_channels: int,
    factor_t: int,
    factor_s: int,
    group_size: int,
) -> torch.Tensor:
    pad_t = (factor_t - x.shape[2] % factor_t) % factor_t
    x_pad = torch.nn.functional.pad(x, (0, 0, 0, 0, pad_t, 0))
    b, c, t, h, w = x_pad.shape
    factor = factor_t * factor_s * factor_s
    return (
        x_pad.view(
            b,
            c,
            t // factor_t,
            factor_t,
            h // factor_s,
            factor_s,
            w // factor_s,
            factor_s,
        )
        .permute(0, 1, 3, 5, 7, 2, 4, 6)
        .contiguous()
        .view(b, c * factor, t // factor_t, h // factor_s, w // factor_s)
        .view(b, out_channels, group_size, t // factor_t, h // factor_s, w // factor_s)
        .mean(dim=2)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AvgDown3D kernel canary requires CUDA")
@pytest.mark.parametrize(
    ("shape", "out_channels", "factor_t", "factor_s", "group_size"),
    [
        ((1, 160, 5, 8, 12), 160, 1, 2, 4),
        ((1, 160, 5, 8, 12), 320, 2, 2, 4),
        ((1, 320, 6, 6, 8), 640, 2, 2, 4),
        ((1, 640, 3, 4, 6), 640, 1, 1, 1),
    ],
)
def test_cosmos3_edge_avgdown3d_bf16_matches_torch(
    shape: tuple[int, ...],
    out_channels: int,
    factor_t: int,
    factor_s: int,
    group_size: int,
):
    try:
        fvk = importlib.import_module("flash_rt.flash_rt_kernels")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")
    if not hasattr(fvk, "cosmos3_edge_avgdown3d_bf16"):
        pytest.skip("cosmos3_edge_avgdown3d_bf16 binding is not built")

    torch.manual_seed(sum(shape) + out_channels + factor_t * 17 + factor_s)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    expected = _avgdown3d_reference(
        x,
        out_channels=out_channels,
        factor_t=factor_t,
        factor_s=factor_s,
        group_size=group_size,
    )
    actual = torch.empty_like(expected)

    fvk.cosmos3_edge_avgdown3d_bf16(
        x.data_ptr(),
        actual.data_ptr(),
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
        out_channels,
        factor_t,
        factor_s,
        group_size,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="UniPC step kernel canary requires CUDA")
def test_cosmos3_edge_unipc_step_f32_bf16_matches_torch_formula():
    try:
        fvk = importlib.import_module("flash_rt.flash_rt_kernels")
    except Exception as exc:  # pragma: no cover - build/env dependent
        pytest.skip(f"flash_rt_kernels not importable: {exc}")
    if not hasattr(fvk, "cosmos3_edge_unipc_step_f32_bf16"):
        pytest.skip("cosmos3_edge_unipc_step_f32_bf16 binding is not built")

    torch.manual_seed(9753)
    numel = 4099
    sample = torch.randn(numel, device="cuda", dtype=torch.float32)
    velocity = torch.randn(numel, device="cuda", dtype=torch.bfloat16)
    prev_m1 = torch.randn(numel, device="cuda", dtype=torch.float32)
    prev_m2 = torch.randn(numel, device="cuda", dtype=torch.float32)
    prev_last = torch.randn(numel, device="cuda", dtype=torch.float32)
    sigma = 0.713
    coeffs = {
        "c_sample": 0.0,
        "c_last": 0.41,
        "c_prev_m1": -0.37,
        "c_prev_m2": 0.08,
        "c_curr_m": 0.88,
        "p_sample": 0.73,
        "p_curr_m": -0.22,
        "p_prev_m1": 0.19,
    }
    current_m = sample - sigma * velocity
    corrected = (
        coeffs["c_last"] * prev_last
        + coeffs["c_prev_m1"] * prev_m1
        + coeffs["c_prev_m2"] * prev_m2
        + coeffs["c_curr_m"] * current_m
    )
    expected_next = coeffs["p_sample"] * corrected + coeffs["p_curr_m"] * current_m + coeffs["p_prev_m1"] * prev_m1

    actual_next = sample.clone()
    actual_m = torch.empty_like(sample)
    actual_last = torch.empty_like(sample)
    fvk.cosmos3_edge_unipc_step_f32_bf16(
        actual_next.data_ptr(),
        velocity.data_ptr(),
        prev_m1.data_ptr(),
        prev_m2.data_ptr(),
        prev_last.data_ptr(),
        actual_next.data_ptr(),
        actual_m.data_ptr(),
        actual_last.data_ptr(),
        numel,
        sigma,
        2,
        2,
        coeffs["c_sample"],
        coeffs["c_last"],
        coeffs["c_prev_m1"],
        coeffs["c_prev_m2"],
        coeffs["c_curr_m"],
        coeffs["p_sample"],
        coeffs["p_curr_m"],
        coeffs["p_prev_m1"],
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    assert torch.allclose(actual_m, current_m, rtol=0, atol=1e-6)
    assert torch.allclose(actual_last, corrected, rtol=0, atol=1e-6)
    assert torch.allclose(actual_next, expected_next, rtol=0, atol=1e-6)
