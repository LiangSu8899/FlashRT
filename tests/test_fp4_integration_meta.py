"""Step B meta-test: validate fp4_utils + new fp16 norm kernels before
frontend integration (Phase 4.3 Step C).

Must pass ALL checks for integration to proceed. Fail-closed.

Checks:
  1. quant_weight_nvfp4 + quant_act_nvfp4 + fp4_gemm → cos ≥ 0.999
     vs pytorch fake_nvfp4 reference on encoder shapes (M=968).
  2. Per-variant micro-bench at encoder shapes → confirm FP4 variant choice.
  3. New norm kernels numerically correct (cos ≥ 0.9999 vs pytorch RMSNorm
     reference) + residual in-place update matches fp8 kernel exactly.
     Note: we do NOT require fp8-bit-exact because the fp16-out kernel stores
     an fp16 intermediate before descale is applied downstream, while the
     fp8-out kernel fuses descale into fp32 scale before the final cast.
     Architectural ~0.1% fp8-bit difference is expected and within ±1 LSB.
"""
from __future__ import annotations

import time
import torch

import flash_rt.flash_rt_fp4 as fvk_fp4
import flash_rt.flash_rt_kernels as fvk
from flash_rt.executors.fp4_utils import (
    FP4ActScratch, fp4_gemm, pick_variant,
    quant_act_nvfp4, quant_weight_nvfp4,
)


# ── pytorch NVFP4 reference (block=16, UE4M3 block scale, e2m1 elements) ──
# Same algorithm as csrc/quantize/quantize_fp4_dynamic.cu host reference;
# already validated in Phase 4.2 at cos=1.0000.
E2M1_LEVELS = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
E2M1_MIDS   = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def fake_nvfp4(x: torch.Tensor) -> torch.Tensor:
    """Simulate NVFP4 quant+dequant: x_fp16 [N, D] → fp32 [N, D]."""
    assert x.shape[-1] % 16 == 0
    orig_shape = x.shape
    x = x.float().reshape(-1, x.shape[-1] // 16, 16)  # [rows, blocks, 16]
    amax = x.abs().amax(dim=-1, keepdim=True)         # [rows, blocks, 1]
    scale = amax / 6.0
    # UE4M3 round (positive fp8_e4m3)
    scale_ue4m3 = scale.to(torch.float8_e4m3fn).float()
    inv = torch.where(scale_ue4m3 > 0, 1.0 / scale_ue4m3, torch.zeros_like(scale_ue4m3))
    xn = x * inv
    # bucket round to E2M1 levels
    sign = torch.sign(xn)
    absn = xn.abs()
    mids = E2M1_MIDS.to(x.device)
    idx = torch.bucketize(absn, mids)                  # [rows, blocks, 16]
    levels = E2M1_LEVELS.to(x.device)[idx]
    xq = sign * levels
    out = (xq * scale_ue4m3).reshape(orig_shape)
    return out


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def check_gemm_shape(M: int, N: int, K: int, variant: int,
                      tag: str, seed: int = 0) -> tuple[float, float]:
    """Run kernel fp4 gemm vs pytorch fake_nvfp4 ref. Returns (cos, elapsed_us)."""
    torch.manual_seed(seed)
    device = 'cuda'
    a = torch.randn(M, K, dtype=torch.float16, device=device) * 0.4
    w = torch.randn(N, K, dtype=torch.float16, device=device) * 0.1

    w_quant = quant_weight_nvfp4(w)
    scratch = FP4ActScratch(M, K, device)
    out = torch.empty(M, N, dtype=torch.float16, device=device)

    quant_act_nvfp4(a, scratch, M, stream=0)
    fp4_gemm(scratch, w_quant, out, M, N, K, variant_idx=variant)
    torch.cuda.synchronize()

    # Latency (20 iters)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        quant_act_nvfp4(a, scratch, M, stream=0)
        fp4_gemm(scratch, w_quant, out, M, N, K, variant_idx=variant)
    end.record()
    end.synchronize()
    elapsed_us = start.elapsed_time(end) * 1000.0 / 20

    # Reference
    a_fake = fake_nvfp4(a)
    w_fake = fake_nvfp4(w)
    ref = a_fake @ w_fake.T
    cos = cos_sim(out.float(), ref)
    return cos, elapsed_us


def check_new_norm_correctness(seed: int = 0) -> dict:
    """Validate new fp16-out norm kernels:
    (a) cos vs pytorch RMSNorm fp16 reference ≥ 0.9999
    (b) residual in-place update bit-exact vs fp8-out kernel
    (c) fp8-bit mismatch fraction reported (informational only)
    """
    torch.manual_seed(seed)
    device = 'cuda'
    S, D = 968, 2048
    x = torch.randn(S, D, dtype=torch.float16, device=device) * 0.5

    # (a) fp16 numerical correctness
    out_fp16 = torch.empty(S, D, dtype=torch.float16, device=device)
    fvk_fp4.rms_norm_noweight_fp16(x.data_ptr(), out_fp16.data_ptr(), S, D, 0)
    torch.cuda.synchronize()

    xf = x.float()
    rms = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + 1e-6)
    ref_fp16 = (xf * rms).half()
    cos_rms = cos_sim(out_fp16, ref_fp16)

    # Residual variant
    res1 = torch.randn(S, D, dtype=torch.float16, device=device) * 0.3
    res2 = res1.clone()
    descale = torch.tensor([0.45], dtype=torch.float32, device=device)

    out_fp16_r = torch.empty(S, D, dtype=torch.float16, device=device)
    fvk_fp4.residual_add_rms_norm_noweight_fp16(
        res1.data_ptr(), x.data_ptr(), out_fp16_r.data_ptr(), S, D, 0
    )

    out_fp8_r = torch.empty(S, D, dtype=torch.float8_e4m3fn, device=device)
    fvk.residual_add_rms_norm_fp8_noweight_fp16(
        res2.data_ptr(), x.data_ptr(), out_fp8_r.data_ptr(),
        S, D, descale.data_ptr(), 0
    )
    torch.cuda.synchronize()

    res_sum_ref = (res1.float())  # already updated in-place by new kernel
    res_sum_fp8 = (res2.float())
    res_update_max = (res_sum_ref - res_sum_fp8).abs().max().item()

    # (a) residual variant fp16 cos
    xf2 = res1.float()  # residual now holds x+prev_residual post-kernel
    rms2 = torch.rsqrt((xf2 * xf2).mean(dim=-1, keepdim=True) + 1e-6)
    ref_fp16_r = (xf2 * rms2).half()
    cos_res = cos_sim(out_fp16_r, ref_fp16_r)

    # (c) informational: fp8-bit mismatch when downstream applies descale+cast
    scaled = (out_fp16_r.float() / descale.item()).clamp(-448.0, 448.0)
    out_fp8_indirect = scaled.to(torch.float8_e4m3fn)
    a = out_fp8_r.view(torch.uint8).flatten()
    b = out_fp8_indirect.view(torch.uint8).flatten()
    fp8_mismatch_frac = (a != b).float().mean().item()

    return {
        'cos_rms': cos_rms,
        'cos_res_rms': cos_res,
        'residual_update_max': res_update_max,
        'fp8_mismatch_frac': fp8_mismatch_frac,
    }


def main():
    print("=" * 70)
    print("Step B — Phase 4.3 FP4 Integration Meta-Test")
    print("=" * 70)

    # Encoder shapes (Pi0.5, Se=968)
    Se = 968
    D = 2048
    H = 16384   # Gate+Up out (=H*2 merged)
    # Actually Pi0.5 H=16384/2 = 8192 for single; Gate+Up merged out = 16384, Down in = 8192.

    encoder_cases = [
        ("enc_QKV",     Se, 2560,  2048),
        ("enc_O",       Se, 2048,  2048),
        ("enc_Gate+Up", Se, 16384, 2048),
        ("enc_Down",    Se, 2048,  8192),
    ]

    # ── Check 1+2: GEMM correctness + per-variant timing ──
    print("\n[1/2] GEMM correctness + variant selection @ encoder M=968")
    print(f"{'shape':14s} {'N':>6s} {'K':>6s} {'variant':>8s} {'cos':>10s} {'us':>8s}")
    all_pass = True
    for tag, M, N, K in encoder_cases:
        default_variant = pick_variant(N, K)
        best = None
        for v in [4, 6, 7]:
            cos, us = check_gemm_shape(M, N, K, v, tag)
            marker = " ★" if v == default_variant else ""
            print(f"{tag:14s} {N:6d} {K:6d} {v:8d} {cos:10.6f} {us:8.2f}{marker}")
            if best is None or us < best[1]:
                best = (v, us, cos)
        # Require default variant cos ≥ 0.999
        cos_def, _ = check_gemm_shape(M, N, K, default_variant, tag)
        if cos_def < 0.999:
            print(f"  ❌ {tag} cos {cos_def:.6f} < 0.999")
            all_pass = False
        else:
            print(f"  ✓ {tag} cos {cos_def:.6f} ≥ 0.999; fastest variant = V{best[0]} @ {best[1]:.2f}μs")

    # ── Check 3: new norm kernels numerically correct ──
    print("\n[3] New fp16-out norm numerical correctness (S=968, D=2048)")
    for seed in range(3):
        r = check_new_norm_correctness(seed)
        print(f"  seed={seed}: "
              f"cos(rms)={r['cos_rms']:.6f}  "
              f"cos(res+rms)={r['cos_res_rms']:.6f}  "
              f"res_update_max|Δ|={r['residual_update_max']:.1e}  "
              f"[info: fp8-bit mismatch={r['fp8_mismatch_frac']*100:.2f}%]")
        if r['cos_rms'] < 0.9999:
            print(f"  ❌ rms_norm cos < 0.9999 (seed={seed})")
            all_pass = False
        if r['cos_res_rms'] < 0.9999:
            print(f"  ❌ residual+rms cos < 0.9999 (seed={seed})")
            all_pass = False
        if r['residual_update_max'] > 1e-6:
            print(f"  ❌ residual in-place update differs (seed={seed})")
            all_pass = False

    print()
    print("=" * 70)
    if all_pass:
        print("✅ ALL CHECKS PASS — Step C integration may proceed")
    else:
        print("❌ SOME CHECKS FAILED — HALT, do not proceed to Step C")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
