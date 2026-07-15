#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

// Fused FP16 NCDHW RMSNorm + SiLU for Wan VAE (MiniMax-Remover) residual
// blocks, where every WanRMS_norm is immediately followed by a SiLU
// activation. Fusing the two ops into one kernel pass eliminates one full
// tensor read + write (the intermediate norm output) and one kernel launch
// per site (~522 sites/decode).
//
// Computes: y = silu( (x / rms(x)) * gamma + bias )
//   where rms(x) = sqrt(sum_c(x_c^2) / C + eps)
//   and   silu(v) = v / (1 + exp(-v))
// which equals WanRMS_norm.forward(x) followed by F.silu.
//
// fp16 in/out, fp32 statistics + activation -- NO dtype cast.
//
// x_fp16:     [B, C, T, H, W] fp16, NCDHW layout (C stride = T*H*W).
// gamma_fp16: [C] fp16 affine weight.
// bias_fp16:  [C] fp16 or nullptr (zero bias).
// y_fp16:     [B, C, T, H, W] fp16 output.
int fp16_rms_silu_ncdhw(
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
