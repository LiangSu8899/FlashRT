#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

// Fused FP16 NCDHW RMSNorm for Wan VAE (MiniMax-Remover).
// fp16 in/out, fp32 statistics — no dtype cast.
//
// Computes: y = (x / rms(x)) * gamma + bias
//   where rms(x) = sqrt(sum_c(x_c^2) / C + eps)
// which is algebraically identical to WanRMS_norm's
//   F.normalize(x, dim=1) * sqrt(C) * gamma + bias.
//
// x_fp16:     [B, C, T, H, W] fp16, NCDHW layout (C stride = T*H*W).
// gamma_fp16: [C] fp16 affine weight.
// bias_fp16:  [C] fp16 or nullptr (zero bias).
// y_fp16:     [B, C, T, H, W] fp16 output.
int fp16_rms_norm_ncdhw(
    const void* x_fp16,
    const void* gamma_fp16,
    const void* bias_fp16,
    void* y_fp16,
    int B, int C, int T, int H, int W,
    float eps,
    cudaStream_t stream);

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt
