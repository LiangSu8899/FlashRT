// ================================================================
// flash_rt — MiniMax-Remover FP8 Conv3d (implicit-GEMM, NDHWC,
// fp16 output).
//
// Hand-rolled implicit-GEMM FP8 e4m3 conv3d fprop with on-the-fly
// im2col index computation (no intermediate matrix materialization).
// Adapted from the motus v17 kernel with three key changes:
//
//   1. fp16 output  (VAE must stay fp16 — no bf16 cast).
//   2. fp16 bias.
//   3. Per-output-channel alpha vector (act_scale × per-channel
//      weight_scale) for higher PSNR than per-tensor scaling.
//
// Specialised for 3×3×3 causal conv with stride/dilation/groups = 1,
// spatial pad = 1, temporal causal cache (T_cache = 2, T_new ≥ 1).
// ================================================================
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

// Fused FP8 e4m3 conv3d fprop (implicit GEMM, NDHWC layout).
//
//   cache_x_fp8 : [N, T_cache, H, W, Ci]  fp8_e4m3  (may be zero-filled)
//   new_x_fp8   : [N, T_new,   H, W, Ci]  fp8_e4m3
//   w_fp8       : [Co, 3, 3, 3, Ci]       fp8_e4m3
//   y_fp16      : [N, T_new, H, W, Co]    fp16  (channels-last 3D physical)
//   bias_fp16   : [Co] fp16  (may be nullptr)
//   alpha_vec   : [Co] float (per-channel dequant: act_scale×w_scale[co],
//                  may be nullptr → alpha = 1.0)
//
// Returns 0 on success, negative on error.
int fp8_conv3d_mm_ndhwc_fp16out(
    const void* cache_x_fp8,
    const void* new_x_fp8,
    const void* w_fp8,
    void* y_fp16,
    const void* bias_fp16,
    const void* alpha_vec,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    cudaStream_t stream);

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt
