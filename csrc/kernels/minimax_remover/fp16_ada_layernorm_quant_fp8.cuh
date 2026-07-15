// ================================================================
// flash_rt_minimax_remover — fused adaLN + FP8 quantise kernel.
//
// Single-kernel fusion of the FP8 attention/FFN entry path:
//   (1) FP32-statistics LayerNorm across D
//         mean = mean(x[s,:])            // fp32 reduce
//         var  = mean((x - mean)^2)      // fp32 reduce
//         rstd = 1 / sqrt(var + eps)
//   (2) adaLN modulation (fp32 scale/shift from temb.float()):
//         y = (x - mean) * rstd * (1 + scale[d]) + shift[d]
//   (3) Per-tensor FP8 e4m3 quantise (static act_scale from the FP8
//       Linear that will consume this output):
//         y_fp8 = clip(y / act_scale, ±448) cast to fp8_e4m3fn
//
// Replaces the 3-kernel path in the FP8 transformer block entry:
//   ada_layernorm_fp16_io  →  quantize_fp8_static_fp16  →  fp8_gemm
// Eliminates one full [S,D] fp16 read-modify-write on the LayerNorm
// output.  The output tensor is the pre-quantised input of the next
// FP8 Linear, so its .forward_from_fp8() (or gemm_from_fp8_ext for
// Q/K/V shared-scale) can skip its own activation quantise entirely.
//
// Layout:
//   x        : [S, D] fp16, contiguous row-major
//   scale    : [D]    fp32 (from temb.float().chunk(6))
//   shift    : [D]    fp32
//   act_scale: [1]    fp32 device scalar (target Linear's descale factor)
//   out      : [S, D] fp8_e4m3fn contiguous row-major
//
// Grid: one CUDA block per row.  Each block reduces over D in fp32.
// ================================================================
#ifndef FLASHRT_KERNELS_MINIMAX_REMOVER_FP16_ADA_LN_QUANT_FP8_CUH
#define FLASHRT_KERNELS_MINIMAX_REMOVER_FP16_ADA_LN_QUANT_FP8_CUH

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

// Fused fp32-stat LayerNorm + adaLN modulation + per-tensor fp8 quantise.
// Returns 0 on success, negative on invalid args.
int fp16_ada_layernorm_quant_fp8(
    const void* x_fp16,
    const void* scale_fp32,
    const void* shift_fp32,
    const void* act_scale_fp32,
    void* out_fp8,
    int S, int D, float eps,
    cudaStream_t stream);

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt

#endif  // FLASHRT_KERNELS_MINIMAX_REMOVER_FP16_ADA_LN_QUANT_FP8_CUH
