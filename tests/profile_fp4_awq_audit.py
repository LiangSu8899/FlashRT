"""Per-kernel latency audit for Pi0.5 NVFP4 + AWQ encoder FFN path.

Extends profile_fp4_vs_fp8_layer.py with:
  - AWQ fused kernels (F3+mul, F4 v2+mul) timing at production shape
  - 10-variant GEMM sweep at the actual Gate+Up and Down shapes
  - Best-variant lookup per shape (input for P6 per-layer variant pick)

Reads no model weights; uses random fp16 inputs and quant_weight_nvfp4 to
exercise the kernel paths in CUDA Graph replay (steady-state timing).

Usage:
    python3 tests/profile_fp4_awq_audit.py
"""
from __future__ import annotations
import torch

import flash_rt.flash_rt_fp4 as fvk_fp4
import flash_rt.flash_rt_kernels as fvk
from flash_rt.executors.fp4_utils import FP4ActScratch, quant_weight_nvfp4


def bench_graph(make_fn, iters=1000, warmup=50):
    s = torch.cuda.Stream()
    fn = make_fn(s.cuda_stream)
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        g.capture_begin()
        fn()
        g.capture_end()
    torch.cuda.synchronize()
    e1 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)
    e1.record()
    for _ in range(iters):
        g.replay()
    e2.record()
    e2.synchronize()
    return e1.elapsed_time(e2) * 1000.0 / iters


def main():
    torch.manual_seed(0)
    device = "cuda"
    Se, De, He = 968, 2048, 8192

    # buffers
    x = (torch.randn(Se, De, dtype=torch.float16, device=device) * 0.3).contiguous()
    fg = (torch.randn(Se, De, dtype=torch.float16, device=device) * 0.3).contiguous()
    x_fp8 = torch.empty(Se * De, dtype=torch.uint8, device=device)
    descale = torch.tensor([0.4], dtype=torch.float32, device=device)
    descale_ptr = descale.data_ptr()

    gu_w_fp8 = torch.randint(0, 200, (2 * He, De), dtype=torch.uint8, device=device)
    dw_fp8 = torch.randint(0, 200, (De, He), dtype=torch.uint8, device=device)
    qkv_w_fp8 = torch.randint(0, 200, (2560, De), dtype=torch.uint8, device=device)
    o_w_fp8 = torch.randint(0, 200, (De, De), dtype=torch.uint8, device=device)
    gate = torch.empty(Se, 2 * He, dtype=torch.float16, device=device)
    qkv = torch.empty(Se, 2560, dtype=torch.float16, device=device)
    fg_out = torch.empty(Se, De, dtype=torch.float16, device=device)
    hid_fp8_buf = torch.empty(Se * He, dtype=torch.uint8, device=device)
    o_fp8 = torch.empty(Se * De, dtype=torch.uint8, device=device)
    attn_out = torch.empty(Se * De, dtype=torch.float16, device=device)

    # FP4 weights
    gu_fp4 = quant_weight_nvfp4(
        torch.randn(2 * He, De, dtype=torch.float16, device=device).contiguous())
    dw_fp4 = quant_weight_nvfp4(
        torch.randn(De, He, dtype=torch.float16, device=device).contiguous())
    sc_gu = FP4ActScratch(Se, De, device)
    sc_dn = FP4ActScratch(Se, He, device)

    # AWQ inv_s buffers
    inv_s_gu = torch.ones(De, dtype=torch.float16, device=device)
    inv_s_dn = torch.ones(He, dtype=torch.float16, device=device)

    print(f"=== Pi0.5 NVFP4 AWQ per-kernel audit @ Se={Se} D={De} H={He} ===\n")

    # ── FP8 reference path ──
    print("[FP8 baseline path, full layer non-attn portion]")
    timings_fp8 = {}
    timings_fp8["rms_qkv→fp8"] = bench_graph(lambda st: lambda: fvk.rms_norm_fp8_noweight_fp16(
        x.data_ptr(), x_fp8.data_ptr(), Se, De, descale_ptr, st))
    timings_fp8["GEMM_QKV"] = bench_graph(lambda st: lambda: fvk.cutlass_fp8_sq(
        x_fp8.data_ptr(), qkv_w_fp8.data_ptr(), qkv.data_ptr(),
        Se, 2560, De, 1.0, 0.0, st))
    timings_fp8["quant_attn→fp8"] = bench_graph(lambda st: lambda: fvk.quantize_fp8_static_fp16(
        attn_out.data_ptr(), o_fp8.data_ptr(), descale_ptr, Se * De, st))
    timings_fp8["GEMM_O"] = bench_graph(lambda st: lambda: fvk.cutlass_fp8_sq(
        o_fp8.data_ptr(), o_w_fp8.data_ptr(), fg_out.data_ptr(),
        Se, De, De, 1.0, 0.0, st))
    timings_fp8["res+rms→fp8 (gu)"] = bench_graph(lambda st: lambda: fvk.residual_add_rms_norm_fp8_noweight_fp16(
        x.data_ptr(), fg.data_ptr(), x_fp8.data_ptr(), Se, De, descale_ptr, st))
    timings_fp8["GEMM_GU(t1)"] = bench_graph(lambda st: lambda: fvk.cutlass_fp8_t1(
        x_fp8.data_ptr(), gu_w_fp8.data_ptr(), gate.data_ptr(),
        Se, 2 * He, De, 1.0, 0.0, st))
    timings_fp8["silu_mul→fp8"] = bench_graph(lambda st: lambda: fvk.gate_geglu_merged_fp8_fp16(
        gate.data_ptr(), hid_fp8_buf.data_ptr(), Se, He, descale_ptr, st))
    timings_fp8["GEMM_Down(wide)"] = bench_graph(lambda st: lambda: fvk.cutlass_fp8_wide(
        hid_fp8_buf.data_ptr(), dw_fp8.data_ptr(), fg_out.data_ptr(),
        Se, De, He, 1.0, 0.0, st))
    timings_fp8["res+rms→fp8 (exit)"] = bench_graph(lambda st: lambda: fvk.residual_add_rms_norm_fp8_noweight_fp16(
        x.data_ptr(), fg.data_ptr(), x_fp8.data_ptr(), Se, De, descale_ptr, st))
    fp8_total = sum(timings_fp8.values())
    for k, v in timings_fp8.items():
        print(f"  {k:24s}: {v:7.2f} μs")
    print(f"  {'─ FP8 layer total':24s}: {fp8_total:7.2f} μs\n")

    # ── FP4 fused path (no AWQ) ──
    print("[FP4 fused path (no AWQ)]")
    timings_fp4 = {}
    timings_fp4["F3 res+rms+fp4+SFA"] = bench_graph(lambda st: lambda: fvk_fp4.residual_add_rms_norm_fp4_sfa_fp16(
        x.data_ptr(), fg.data_ptr(),
        sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(), Se, De, st))
    timings_fp4["F3 v2 (1tpb=block)"] = bench_graph(lambda st: lambda: fvk_fp4.residual_add_rms_norm_fp4_sfa_v2_fp16(
        x.data_ptr(), fg.data_ptr(),
        sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(), Se, De, st))
    timings_fp4["GEMM_GU V8"] = bench_graph(lambda st: lambda: fvk_fp4.cutlass_fp4_gemm_variant(
        8, sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
        gu_fp4['packed'].data_ptr(), gu_fp4['sfb'].data_ptr(),
        gate.data_ptr(), Se, 2 * He, De, 1.0, 0.0, st))
    timings_fp4["F4 v2 silu+fp4+SFA"] = bench_graph(lambda st: lambda: fvk_fp4.gate_geglu_fp4_sfa_v2_fp16(
        gate.data_ptr(), sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(), Se, He, st))
    timings_fp4["GEMM_Down V1"] = bench_graph(lambda st: lambda: fvk_fp4.cutlass_fp4_gemm_variant(
        1, sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
        dw_fp4['packed'].data_ptr(), dw_fp4['sfb'].data_ptr(),
        fg_out.data_ptr(), Se, De, He, 1.0, 0.0, st))
    fp4_total = sum(timings_fp4.values())
    for k, v in timings_fp4.items():
        print(f"  {k:24s}: {v:7.2f} μs")
    print(f"  {'─ FP4 (FFN portion)':24s}: {fp4_total:7.2f} μs\n")

    # ── FP4 + AWQ fused path ──
    print("[FP4 + AWQ fused path (production)]")
    timings_awq = {}
    timings_awq["F3+mul res+rms+inv+fp4+SFA"] = bench_graph(lambda st: lambda: fvk_fp4.residual_add_rms_norm_mul_fp4_sfa_fp16(
        x.data_ptr(), fg.data_ptr(), inv_s_gu.data_ptr(),
        sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(), Se, De, st))
    timings_awq["GEMM_GU V8"] = timings_fp4["GEMM_GU V8"]
    # P1 path comparison kernels
    if hasattr(fvk_fp4, "cutlass_fp4_gemm_fp4out"):
        # P1 split GU: 2× fp4out GEMM (S, H, D)
        gate_p4 = torch.empty(Se, He // 2, dtype=torch.uint8, device=device)
        up_p4   = torch.empty(Se, He // 2, dtype=torch.uint8, device=device)
        gate_sfa = torch.empty(fvk_fp4.sfa_size_bytes(Se, He, False), dtype=torch.uint8, device=device)
        up_sfa   = torch.empty(fvk_fp4.sfa_size_bytes(Se, He, False), dtype=torch.uint8, device=device)
        # Half weights (separate gate / up [H, D])
        gw_fp4 = quant_weight_nvfp4(torch.randn(He, De, dtype=torch.float16, device=device).contiguous())
        uw_fp4 = quant_weight_nvfp4(torch.randn(He, De, dtype=torch.float16, device=device).contiguous())
        timings_awq["[P1] fp4out gate (S,H,D)"] = bench_graph(lambda st: lambda: fvk_fp4.cutlass_fp4_gemm_fp4out(
            sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
            gw_fp4['packed'].data_ptr(), gw_fp4['sfb'].data_ptr(),
            gate_p4.data_ptr(), gate_sfa.data_ptr(),
            Se, He, De, st))
        timings_awq["[P1] fp4out up   (S,H,D)"] = bench_graph(lambda st: lambda: fvk_fp4.cutlass_fp4_gemm_fp4out(
            sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
            uw_fp4['packed'].data_ptr(), uw_fp4['sfb'].data_ptr(),
            up_p4.data_ptr(), up_sfa.data_ptr(),
            Se, He, De, st))
        timings_awq["[P1] geglu_two (no AWQ)"] = bench_graph(lambda st: lambda: fvk_fp4.geglu_two_fp4_to_fp4(
            gate_p4.data_ptr(), gate_sfa.data_ptr(),
            up_p4.data_ptr(),   up_sfa.data_ptr(),
            sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
            Se, He, st))
        if hasattr(fvk_fp4, "geglu_two_mul_fp4_to_fp4"):
            timings_awq["[P1] geglu_two+mul (AWQ)"] = bench_graph(lambda st: lambda: fvk_fp4.geglu_two_mul_fp4_to_fp4(
                gate_p4.data_ptr(), gate_sfa.data_ptr(),
                up_p4.data_ptr(),   up_sfa.data_ptr(),
                inv_s_dn.data_ptr(),
                sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
                Se, He, st))
    timings_awq["F4 v2+mul silu+inv+fp4+SFA"] = bench_graph(lambda st: lambda: fvk_fp4.gate_geglu_mul_fp4_sfa_v2_fp16(
        gate.data_ptr(), inv_s_dn.data_ptr(),
        sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(), Se, He, st))
    timings_awq["GEMM_Down V1"] = timings_fp4["GEMM_Down V1"]
    awq_total = sum(timings_awq.values())
    for k, v in timings_awq.items():
        print(f"  {k:28s}: {v:7.2f} μs")
    print(f"  {'─ AWQ (FFN portion)':28s}: {awq_total:7.2f} μs\n")

    # ── Δ summary ──
    pre_fp8 = timings_fp8["res+rms→fp8 (gu)"] + timings_fp8["GEMM_GU(t1)"] + \
              timings_fp8["silu_mul→fp8"] + timings_fp8["GEMM_Down(wide)"]
    print(f"[Δ FFN portion only — comparable kernel groups]")
    print(f"  FP8  res+GU+silu+Down : {pre_fp8:7.2f} μs")
    print(f"  FP4  fused 4 kernels  : {fp4_total:7.2f} μs   Δ {fp4_total-pre_fp8:+.2f} μs")
    print(f"  AWQ  fused 4 kernels  : {awq_total:7.2f} μs   Δ {awq_total-pre_fp8:+.2f} μs")
    print(f"  AWQ overhead vs FP4   :                 {awq_total-fp4_total:+.2f} μs\n")

    # ── 10-variant sweep at GU shape ──
    print(f"[FP4 GEMM variant sweep @ Gate+Up shape (M={Se} N={2*He} K={De})]")
    gu_results = []
    for vi in range(fvk_fp4.cutlass_fp4_gemm_num_variants()):
        try:
            t = bench_graph(lambda st, v=vi: lambda: fvk_fp4.cutlass_fp4_gemm_variant(
                v, sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
                gu_fp4['packed'].data_ptr(), gu_fp4['sfb'].data_ptr(),
                gate.data_ptr(), Se, 2 * He, De, 1.0, 0.0, st),
                iters=300, warmup=20)
            name = fvk_fp4.cutlass_fp4_gemm_variant_name(vi)
            gu_results.append((vi, t, name))
            print(f"  V{vi}: {t:7.2f} μs   ({name})")
        except Exception as e:
            print(f"  V{vi}: SKIP ({e})")
    gu_best = min(gu_results, key=lambda r: r[1])
    print(f"  → BEST: V{gu_best[0]} @ {gu_best[1]:.2f} μs\n")

    # ── 10-variant sweep at Down shape ──
    print(f"[FP4 GEMM variant sweep @ Down shape (M={Se} N={De} K={He})]")
    dn_results = []
    for vi in range(fvk_fp4.cutlass_fp4_gemm_num_variants()):
        try:
            t = bench_graph(lambda st, v=vi: lambda: fvk_fp4.cutlass_fp4_gemm_variant(
                v, sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
                dw_fp4['packed'].data_ptr(), dw_fp4['sfb'].data_ptr(),
                fg_out.data_ptr(), Se, De, He, 1.0, 0.0, st),
                iters=300, warmup=20)
            name = fvk_fp4.cutlass_fp4_gemm_variant_name(vi)
            dn_results.append((vi, t, name))
            print(f"  V{vi}: {t:7.2f} μs   ({name})")
        except Exception as e:
            print(f"  V{vi}: SKIP ({e})")
    dn_best = min(dn_results, key=lambda r: r[1])
    print(f"  → BEST: V{dn_best[0]} @ {dn_best[1]:.2f} μs\n")

    # ── Best-variant impact projection ──
    saving_gu = (timings_fp4["GEMM_GU V8"] - gu_best[1]) * 18
    saving_dn = (timings_fp4["GEMM_Down V1"] - dn_best[1]) * 18
    print(f"[P6 per-layer best-variant projection (full 18 FP4 layers)]")
    print(f"  GU   V8={timings_fp4['GEMM_GU V8']:.2f} → V{gu_best[0]}={gu_best[1]:.2f}   "
          f"saving = {saving_gu:+.1f} μs/infer = {saving_gu/1000:+.3f} ms")
    print(f"  Down V1={timings_fp4['GEMM_Down V1']:.2f} → V{dn_best[0]}={dn_best[1]:.2f}   "
          f"saving = {saving_dn:+.1f} μs/infer = {saving_dn/1000:+.3f} ms")
    print(f"  Total potential                            "
          f"= {(saving_gu+saving_dn)/1000:+.3f} ms\n")

    # ── 18-layer total (production) projection ──
    print(f"[Full 18-layer FFN projection (production)]")
    print(f"  FP8 18 × ({pre_fp8:.0f}μs)         = {18*pre_fp8/1000:.2f} ms")
    print(f"  FP4 18 × ({fp4_total:.0f}μs)         = {18*fp4_total/1000:.2f} ms   "
          f"Δ {18*(fp4_total-pre_fp8)/1000:+.2f} ms")
    print(f"  AWQ 18 × ({awq_total:.0f}μs)         = {18*awq_total/1000:.2f} ms   "
          f"Δ {18*(awq_total-pre_fp8)/1000:+.2f} ms")


if __name__ == "__main__":
    main()
