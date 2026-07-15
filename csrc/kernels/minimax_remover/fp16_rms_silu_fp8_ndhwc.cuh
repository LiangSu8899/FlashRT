// ================================================================
// flash_rt — MiniMax-Remover fused RMSNorm+SiLU with FP8 quantize
// (channels-last NDHWC).
//
// Three entry points, all built on the same warp-per-spatial template:
//
//  1. fp16_rms_silu_amax_ndhwc
//     Fused norm+silu+amax.  Reads fp16 x, writes fp16 y AND
//     accumulates |y| into a device-side amax buffer via atomicMax.
//     Saves one full read of y compared to (norm+silu) → separate amax.
//
//  2. fp16_rms_silu_quant_fp8_ndhwc
//     Fused norm+silu+quantize-to-FP8.  Reads fp16 x, reads a
//     pre-computed amax from device memory, quantizes and writes
//     fp8 y.  Does NOT write fp16 output — eliminates the fp16
//     intermediate entirely when the consumer only needs FP8.
//
//  3. fp16_rms_silu_amax_quant_fp8_ndhwc
//     2-pass launcher combining (1) and (2): pass 1 computes
//     norm+silu+amax (no fp16 write); pass 2 re-reads x, computes
//     norm+silu+quant.  Produces ONLY fp8 output + scale.
//
// Use case: in WanResidualBlock the pattern is
//   x → norm → silu → conv1(quant→FP8→MMA)
// Fusing norm+silu+amax+quant eliminates the fp16 intermediate
// between norm and conv, saving one full read+write per layer.
// ================================================================
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

// (1) Fused norm+silu+amax → fp16 output + device amax.
//     amax_buf must be zeroed by the caller before the first call.
//     Multiple calls accumulate (atomicMax).
int fp16_rms_silu_amax_ndhwc(
    const void* x_fp16,
    const void* gamma_fp16,
    const void* bias_fp16,
    void* y_fp16,
    void* amax_buf,           // device float *, caller-zeroed
    int B, int C, int T, int H, int W,
    float eps,
    cudaStream_t stream);

// (2) Fused norm+silu+quant → fp8 output (reads pre-computed amax).
//     No fp16 output is written.
int fp16_rms_silu_quant_fp8_ndhwc(
    const void* x_fp16,
    const void* gamma_fp16,
    const void* bias_fp16,
    void* y_fp8,              // __nv_fp8_e4m3 *
    const void* amax_buf,     // device float *, pre-computed
    void* scale_out,          // device float *, may be nullptr
    int B, int C, int T, int H, int W,
    float eps,
    cudaStream_t stream);

// (3) 2-pass: norm+silu+amax+quant → fp8 output + scale.
//     amax_buf is a caller-provided scratch (1 float).
//     No fp16 output.
int fp16_rms_silu_amax_quant_fp8_ndhwc(
    const void* x_fp16,
    const void* gamma_fp16,
    const void* bias_fp16,
    void* y_fp8,
    void* scale_out,
    void* amax_buf,
    int B, int C, int T, int H, int W,
    float eps,
    cudaStream_t stream);

// (3b) Same as (3) but does NOT zero amax_buf before pass 1.
//      For running-max mode: caller seeds amax_buf with the historical
//      running max; pass 1 atomicMax-accumulates the current output's
//      amax; pass 2 quantizes with max(historical, current).
int fp16_rms_silu_amax_quant_fp8_ndhwc_nozero(
    const void* x_fp16,
    const void* gamma_fp16,
    const void* bias_fp16,
    void* y_fp8,
    void* scale_out,
    void* amax_buf,
    int B, int C, int T, int H, int W,
    float eps,
    cudaStream_t stream);

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt
