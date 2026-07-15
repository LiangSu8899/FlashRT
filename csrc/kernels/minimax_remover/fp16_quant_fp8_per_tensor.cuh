// ================================================================
// flash_rt — MiniMax-Remover FP8 per-tensor activation quantize.
//
// Fused 2-pass launcher:
//   Pass 1: parallel amax reduction (block-local → atomicMax).
//   Pass 2: read device-resident amax, compute scale, quantize.
//
// No host sync — scale stays on device.  Works on any contiguous
// fp16 buffer (the caller passes a channels-last 3D tensor whose
// physical memory is NDHWC, and the fp8 output preserves that order).
// ================================================================
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

// Quantize x_fp16 [n elements] → y_fp8, writing per-tensor scale to
// scale_out (device float).  amax_buf is a scratch device float
// (1 element) used for the inter-pass reduction.
//
// Returns 0 on success, negative on error.
int fp16_quant_fp8_per_tensor(
    const void* x_fp16, void* y_fp8,
    void* scale_out, void* amax_buf,
    int n, cudaStream_t stream);

// amax only: grid-stride reduction with atomicMax into amax_buf.
// Caller MUST zero amax_buf before the first call.  Multiple calls
// accumulate (so amax over two tensors = call on each sequentially).
int amax_fp16(
    const void* x_fp16, void* amax_buf,
    int n, cudaStream_t stream);

// Quantize only: reads pre-computed amax from amax_buf, computes
// scale = max(amax,0)/448, writes scale to scale_out, quantizes.
int quantize_fp16_fp8_with_amax(
    const void* x_fp16, void* y_fp8,
    const void* amax_buf, void* scale_out,
    int n, cudaStream_t stream);

// Dual quantize: quantizes TWO fp16 buffers with the SAME shared
// amax in a single kernel launch.  Saves one launch vs calling
// quantize_fp16_fp8_with_amax twice.  Writes one shared scale.
int quantize_fp16_fp8_with_amax_dual(
    const void* x1_fp16, void* y1_fp8, int n1,
    const void* x2_fp16, void* y2_fp8, int n2,
    const void* amax_buf, void* scale_out,
    cudaStream_t stream);

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt
