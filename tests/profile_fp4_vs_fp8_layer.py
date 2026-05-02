"""Per-kernel profiling of a single Pi0.5 encoder layer at M=968.

Each kernel/kernel-group is captured in its own CUDA Graph and replayed to
get clean steady-state timing.

Goal: understand exactly where FP4-layer time vs FP8-layer time differs,
find leftover overhead to remove.
"""
from __future__ import annotations
import numpy as np
import torch

import flash_rt.flash_rt_fp4 as fvk_fp4
import flash_rt.flash_rt_kernels as fvk


def bench_graph(make_fn, iters=1000, warmup=50):
    """make_fn(stream_int) -> callable(): run one iter on that stream.
    Returns microseconds per iter."""
    s = torch.cuda.Stream()
    s_int = s.cuda_stream
    fn = make_fn(s_int)
    with torch.cuda.stream(s):
        for _ in range(warmup): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        g.capture_begin()
        fn()
        g.capture_end()
    torch.cuda.synchronize()
    e1 = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
    e1.record()
    for _ in range(iters): g.replay()
    e2.record(); e2.synchronize()
    return e1.elapsed_time(e2) * 1000 / iters


def main():
    # Shapes from Pi0.5 encoder
    Se, De, He = 968, 2048, 8192
    NH, HD = 8, 256
    total_keys = 2048  # typical

    # Allocate buffers (contiguous)
    device = 'cuda'
    torch.manual_seed(0)

    x = (torch.randn(Se, De, dtype=torch.float16, device=device) * 0.3).contiguous()
    fg = (torch.randn(Se, De, dtype=torch.float16, device=device) * 0.3).contiguous()
    x_fp8 = torch.empty(Se * De, dtype=torch.uint8, device=device)
    descale = torch.tensor([0.4], dtype=torch.float32, device=device)
    descale_ptr = descale.data_ptr()

    # FP8 weights (random fp8 bytes + scale)
    qkv_w_fp8 = torch.randint(0, 200, (2560, De), dtype=torch.uint8, device=device)
    o_w_fp8   = torch.randint(0, 200, (De, De),   dtype=torch.uint8, device=device)
    gu_w_fp8  = torch.randint(0, 200, (2 * He, De), dtype=torch.uint8, device=device)
    dw_fp8    = torch.randint(0, 200, (De, He),   dtype=torch.uint8, device=device)
    qkv = torch.empty(Se, 2560, dtype=torch.float16, device=device)
    gate = torch.empty(Se, 2 * He, dtype=torch.float16, device=device)
    hid_fp8_buf = torch.empty(Se * He, dtype=torch.uint8, device=device)
    fg_out = torch.empty(Se, De, dtype=torch.float16, device=device)

    # FP4 weights (fake) - using quant_weight utility
    from flash_rt.executors.fp4_utils import quant_weight_nvfp4, FP4ActScratch
    gu_fp4 = quant_weight_nvfp4(torch.randn(2 * He, De, dtype=torch.float16, device=device).contiguous())
    dw_fp4 = quant_weight_nvfp4(torch.randn(De, He, dtype=torch.float16, device=device).contiguous())
    sc_gu = FP4ActScratch(Se, De, device)
    sc_dn = FP4ActScratch(Se, He, device)
    x_normed = torch.empty(Se, De, dtype=torch.float16, device=device)
    hid_fp16 = torch.empty(Se, He, dtype=torch.float16, device=device)

    # ═══════════════════════════════════════════════
    # FP8 sub-kernels
    # ═══════════════════════════════════════════════
    print("── Per-kernel timing (in CUDA Graph, M=968) ──")

    # FP8 path pieces (relevant to Gate+Up and Down GEMM region)
    def _fp8_resrms(st):
        return lambda: fvk.residual_add_rms_norm_fp8_noweight_fp16(
            x.data_ptr(), fg.data_ptr(), x_fp8.data_ptr(), Se, De, descale_ptr, st)
    def _fp8_gu(st):
        # Fake alpha
        return lambda: fvk.cutlass_fp8_t1(x_fp8.data_ptr(), gu_w_fp8.data_ptr(), gate.data_ptr(),
                                           Se, 2*He, De, 1.0, 0.0, st)
    def _fp8_silu(st):
        return lambda: fvk.gate_geglu_merged_fp8_fp16(gate.data_ptr(), hid_fp8_buf.data_ptr(), Se, He, descale_ptr, st)
    def _fp8_down(st):
        return lambda: fvk.cutlass_fp8_wide(hid_fp8_buf.data_ptr(), dw_fp8.data_ptr(), fg_out.data_ptr(),
                                             Se, De, He, 1.0, 0.0, st)

    fp8_resrms = bench_graph(_fp8_resrms)
    fp8_gu     = bench_graph(_fp8_gu)
    fp8_silu   = bench_graph(_fp8_silu)
    fp8_down   = bench_graph(_fp8_down)
    fp8_total  = fp8_resrms + fp8_gu + fp8_silu + fp8_down
    print(f"\nFP8 layer Gate+Up path (sum of kernels):")
    print(f"  res+rms→fp8        : {fp8_resrms:6.2f}μs")
    print(f"  cutlass_fp8_t1 GU  : {fp8_gu:6.2f}μs")
    print(f"  silu_mul→fp8       : {fp8_silu:6.2f}μs")
    print(f"  cutlass_fp8_wide D : {fp8_down:6.2f}μs")
    print(f"  ─ Total             : {fp8_total:6.2f}μs")

    # ═══════════════════════════════════════════════
    # FP4 fused pipeline (current)
    # ═══════════════════════════════════════════════
    variant_gu = 8  # V8 wide-N+K (newest empirical best)
    variant_dn = 1  # V1 cluster2x1x1

    def _fp4_f3(st):
        return lambda: fvk_fp4.residual_add_rms_norm_fp4_sfa_fp16(
            x.data_ptr(), fg.data_ptr(),
            sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(), Se, De, st)
    def _fp4_gu(st):
        return lambda: fvk_fp4.cutlass_fp4_gemm_variant(
            variant_gu,
            sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
            gu_fp4['packed'].data_ptr(), gu_fp4['sfb'].data_ptr(),
            gate.data_ptr(),
            Se, 2*He, De, 1.0, 0.0, st)
    def _fp4_f4(st):
        return lambda: fvk_fp4.gate_geglu_fp4_sfa_v2_fp16(
            gate.data_ptr(), sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(), Se, He, st)
    def _fp4_down(st):
        return lambda: fvk_fp4.cutlass_fp4_gemm_variant(
            variant_dn,
            sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
            dw_fp4['packed'].data_ptr(), dw_fp4['sfb'].data_ptr(),
            fg_out.data_ptr(),
            Se, De, He, 1.0, 0.0, st)

    fp4_f3   = bench_graph(_fp4_f3)
    fp4_gu   = bench_graph(_fp4_gu)
    fp4_f4   = bench_graph(_fp4_f4)
    fp4_down = bench_graph(_fp4_down)
    fp4_total = fp4_f3 + fp4_gu + fp4_f4 + fp4_down
    print(f"\nFP4 layer Gate+Up path (sum of fused kernels):")
    print(f"  F3 res+rms+fp4+SFA : {fp4_f3:6.2f}μs")
    print(f"  fp4 GEMM V{variant_gu} GU      : {fp4_gu:6.2f}μs")
    print(f"  F4 silu+fp4+SFA    : {fp4_f4:6.2f}μs")
    print(f"  fp4 GEMM V{variant_dn} Down    : {fp4_down:6.2f}μs")
    print(f"  ─ Total             : {fp4_total:6.2f}μs")

    print(f"\nΔ FP4 vs FP8 per layer: {fp4_total - fp8_total:+.2f}μs")
    print(f"Breakdown:")
    print(f"  pre-GU:  FP4 {fp4_f3:5.2f}μs vs FP8 {fp8_resrms:5.2f}μs = {fp4_f3-fp8_resrms:+.2f}μs")
    print(f"  GU GEMM: FP4 {fp4_gu:5.2f}μs vs FP8 {fp8_gu:5.2f}μs = {fp4_gu-fp8_gu:+.2f}μs")
    print(f"  silu:    FP4 {fp4_f4:5.2f}μs vs FP8 {fp8_silu:5.2f}μs = {fp4_f4-fp8_silu:+.2f}μs")
    print(f"  Down GEMM: FP4 {fp4_down:5.2f}μs vs FP8 {fp8_down:5.2f}μs = {fp4_down-fp8_down:+.2f}μs")

    # ═══════════════════════════════════════════════
    # All FP4 GEMM variants sweep at the actual shapes
    # ═══════════════════════════════════════════════
    print(f"\n── FP4 GEMM variant sweep @ real shapes ──")
    for vi in range(fvk_fp4.cutlass_fp4_gemm_num_variants()):
        def _gu_v(st, v=vi):
            return lambda: fvk_fp4.cutlass_fp4_gemm_variant(
                v, sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
                gu_fp4['packed'].data_ptr(), gu_fp4['sfb'].data_ptr(),
                gate.data_ptr(), Se, 2*He, De, 1.0, 0.0, st)
        def _dn_v(st, v=vi):
            return lambda: fvk_fp4.cutlass_fp4_gemm_variant(
                v, sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
                dw_fp4['packed'].data_ptr(), dw_fp4['sfb'].data_ptr(),
                fg_out.data_ptr(), Se, De, He, 1.0, 0.0, st)
        name = fvk_fp4.cutlass_fp4_gemm_variant_name(vi)
        gu_t = bench_graph(_gu_v, iters=200, warmup=20)
        dn_t = bench_graph(_dn_v, iters=200, warmup=20)
        print(f"  V{vi} ({name}): Gate+Up {gu_t:6.2f}μs  Down {dn_t:6.2f}μs")

    # 3-layer total expected saving
    print(f"\n── Per-infer expected (3 FP4 layers) ──")
    print(f"  FP8 total (3 layers): {3*fp8_total:.1f}μs")
    print(f"  FP4 total (3 layers): {3*fp4_total:.1f}μs")
    print(f"  Δ (theoretical): {3*(fp4_total - fp8_total):+.1f}μs = {3*(fp4_total - fp8_total)/1000:+.3f}ms")

if __name__ == '__main__':
    main()
