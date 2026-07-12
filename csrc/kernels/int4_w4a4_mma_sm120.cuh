// SPDX-License-Identifier: Apache-2.0
//
// Tensor-core INT4 (E0M3) W4A4 GEMV for sm_120 — M=1 full-N decode
// primitive plus the matching INT4 block-scale quantizer. Twin of
// fp4_w4a4_mma_sm120.{cu,cuh} with a uniform-grid element format.
//
// ── How INT4 works on sm_120 ──────────────────────────────────────
// ptxas only accepts `e2m1` element types for
// `mma.sync.kind::mxf4nvf4.block_scale`, but the SASS encoding of
// OMMA.SF.16864 carries the operand element format in instruction
// bits 78 (A) / 79 (B): 0 = E2M1, 1 = E0M3 (uniform INT4, codebook
// -7..7, sign-magnitude nibbles). The bits are undocumented and
// invisible to nvdisasm; throughput is identical to E2M1 (verified
// on RTX 5090: 2027 TFLOPS for E2M1xE2M1, INT4xINT4 and mixed).
//
// Therefore every kernel in this TU is COMPILED as e2m1 and MUST be
// post-processed with tools/patch_int4_omma_sm120.py, which flips
// bits 78/79 of every OMMA.SF instruction inside kernels whose
// mangled name contains "int4_". Loading this TU without the patch
// step yields E2M1 semantics on INT4-encoded data (i.e. garbage) —
// the launcher cannot detect this, so builds must treat the patch
// as part of the link step. A runtime canary is provided
// (int4_w4a4_sm120_codebook_canary) that returns 0 only if the
// running SASS actually decodes the INT4 codebook.
//
// ── Data formats ──────────────────────────────────────────────────
// Elements: 4-bit sign-magnitude, nibble = (sign<<3) | mag, mag 0..7,
//   value = (-1)^sign * mag. Packed 2/byte, low nibble = even k.
// Scale factors: UE4M3 per 16 elements (identical layout and swizzle
//   to the NVFP4 path — nvfp4_sf_linear_to_swizzled scheme), with an
//   fp32 global scale per tensor: value = nibble * ue4m3(sf) * g.
//   Quantizer uses g = amax / (7 * 448) (INT4_MAX * UE4M3_MAX).
//
// ── llama.cpp / GGUF note ─────────────────────────────────────────
// GGML Q4 formats are uniform grids, so they map onto E0M3 without
// the E2M1 regrid loss:
//   Q4_0 (block 32, fp16 scale d, q in -8..7, w = q*d): mag -8 clamps
//     to -7 (one code point), d re-encodes into ue4m3*g two-level.
//   Q4_K (block 256/16, 6-bit scales+mins): asymmetric; fold the min
//     into a per-block bias channel or re-center offline, scales map
//     per-16 directly.
// Conversion is offline/host-side; this TU only defines the runtime
// GEMV + activation quantizer.
//
// Wiring note: when this TU is added to flash_rt_kernels, the build
// must (1) run the OMMA patch tool on the produced .so as a
// post-build step and (2) add the exported symbols to the frontend
// fail-fast list + a smoke test, per repo convention.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace gemm {

// y[N] = alpha * sum_k dq(A[k]) * dq(B[n,k])   (M = 1)
//
// A_packed: (K/2,)  uint8   INT4 activation row (packed nibbles)
// B_packed: (N, K/2) uint8  INT4 weights, row-major over N
// SFA:      swizzled UE4M3 SF table for the single A row
// SFB:      swizzled UE4M3 SF table for N weight rows
// alpha:    caller folds gA * gB (the two fp32 global scales) here
// Constraints: K % 64 == 0, N % 8 == 0.
int int4_w4a4_mma_sm120_full_n_bf16out(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    int          N,
    int          K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream);

// bf16 (rows, K) row-major → INT4 nibbles (rows, K/2) + swizzled
// UE4M3 SF table. `global_scale` is a device fp32 pointer (from
// int4_weight_global_scale_sm120 below, or any caller-provided
// scalar). SF table byte size = ceil(rows/128) * ceil(K/64) * 512.
// Serves both weights (rows = N, offline) and activations (rows = 1).
int int4_quantize_bf16_sm120(
    const void*  x_bf16,
    void*        out_packed,
    void*        out_sf_swizzled,
    const void*  global_scale,   // device float*
    int          rows,
    int          K,
    cudaStream_t stream);

// amax(|x|) / (7 * 448) over a bf16 tensor → device float.
// `scale_out` must be pre-zeroed (atomicMax accumulation).
int int4_global_scale_bf16_sm120(
    const void*  x_bf16,
    void*        scale_out,      // device float*
    long long    numel,
    cudaStream_t stream);

// Runtime canary: runs one 16x8x64 MMA with a known INT4 pattern and
// checks the output against the INT4 codebook. Returns 0 if the SASS
// was patched (true E0M3 decode), 1 if it still decodes as E2M1
// (patch step missing), negative on launch error. Synchronizes.
int int4_w4a4_sm120_codebook_canary(cudaStream_t stream);

}  // namespace gemm
}  // namespace flash_rt
