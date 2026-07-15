// ================================================================
// flash_rt_minimax_remover — fused FFN epilogue kernel declarations.
//
// bias_gelu_quant_fp16_fp8 : fp16 GEMM-out + bias → tanh-gelu → fp8
// bias_quant_fp16_fp8      : fp16 GEMM-out + bias → fp8 (identity)
//
// Both eliminate the intermediate fp16 round-trips of the separate
// add_bias_fp16 + (gelu_inplace) + quantize_fp8 sequence.
// ================================================================
#ifndef FLASHRT_KERNELS_MINIMAX_REMOVER_FP16_BIAS_GELU_QUANT_FP8_CUH
#define FLASHRT_KERNELS_MINIMAX_REMOVER_FP16_BIAS_GELU_QUANT_FP8_CUH

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

// Fused: bias-add + tanh-gelu + quantise → fp8 e4m3.
//   gemm_out : [M*N] fp16, raw FP8-GEMM output (no bias yet)
//   bias     : [N]   fp16
//   out      : [M*N] fp8 e4m3  (the pre-quantised input for the next Linear)
//   d_scale  : float, the NEXT linear's act_scale (quantise divides by it)
// Returns 0 on success, <0 on invalid args.
int bias_gelu_quant_fp16_fp8(
    const void* gemm_out, const void* bias,
    void* out, const float* d_scale,
    int M, int N, cudaStream_t stream);

// Fused: bias-add + quantise → fp8 e4m3 (identity activation).
int bias_quant_fp16_fp8(
    const void* gemm_out, const void* bias,
    void* out, const float* d_scale,
    int M, int N, cudaStream_t stream);

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt

#endif  // FLASHRT_KERNELS_MINIMAX_REMOVER_FP16_BIAS_GELU_QUANT_FP8_CUH
