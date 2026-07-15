// SPDX-License-Identifier: Apache-2.0
// Fused FP16 channels-last (NDHWC) RMSNorm for MiniMax-Remover VAE.
//
// When the VAE pipeline runs in channels-last 3D (NDHWC) memory format,
// cuDNN's conv3d skips the nchwToNhwc/nhwcToNchw conversion kernels
// (~287 ms / decode).  This kernel keeps the norm output in NDHWC so
// the format is preserved end-to-end.
//
// In NDHWC, the C values for each spatial position (b, t, h, w) are
// CONTIGUOUS in memory — making the reduction over C much more cache-
// friendly than the NCDHW variant (where C values are strided by
// T*H*W).
//
// One warp (32 threads) per spatial position.  Each thread handles
// C/32 values, then a warp-level __shfl reduction computes the RMS.
// No shared memory required.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

// FP16 channels-last RMSNorm.
// x_fp16:     [B, C, T, H, W] stored as NDHWC (C contiguous, stride 1).
// gamma_fp16: [C] or [C, 1, 1] (logical shape varies; data is just [C]).
// bias_fp16:  [C] or nullptr.
// y_fp16:     [B, C, T, H, W] NDHWC output.
// Computes: y = F.normalize(x, dim=1) * sqrt(C) * gamma + bias
//          = (x / rms(x)) * gamma + bias   (per-spatial-position)
int fp16_rms_norm_ndhwc(
    const void* x_fp16,
    const void* gamma_fp16,
    const void* bias_fp16,
    void* y_fp16,
    int B, int C, int T, int H, int W,
    float eps,
    cudaStream_t stream);

// FP16 channels-last RMSNorm + SiLU (fused).
// Computes: y = silu( (x / rms(x)) * gamma + bias )
int fp16_rms_silu_ndhwc(
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
