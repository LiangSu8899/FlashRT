#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

// Fused RMSNorm + RoPE + per-warp int8 quantization (for Q).
// Input:  x [B*S, H*Dd] fp16, in-place layout [B, S, H, Dd].
// Output: out_int8 [B*S, H*Dd] int8 (same layout as x).
//         scale [B, H, num_scale_groups] fp32
//           where num_scale_groups = ceil(S / WARPQ) * (BLKQ / WARPQ)
//           For BLKQ=128, WARPQ=32: num_scale_groups = ceil(S/128) * 4
// Each scale covers WARPQ=32 consecutive tokens for one head.
// rstd_buf: caller-owned scratch of B*S floats for the per-token rstd
//           reduction. Pass nullptr to use a stream-ordered transient
//           allocation (slower; for one-off / non-hot-path callers).
int fp16_rmsnorm_rope_quant_int8_q(
    const void* x_fp16,         // [B*S, H*Dd] fp16 input
    const void* weight_fp16,    // [H*Dd] fp16 norm weight
    const void* bias_fp16,      // [H*Dd] fp16 Q-proj bias, or nullptr (fused pre-norm)
    const void* cos_fp32,       // [S, Dd/2] fp32
    const void* sin_fp32,       // [S, Dd/2] fp32
    void* out_int8,             // [B*S, H*Dd] int8 output
    void* scale_fp32,           // [B, H, ceil(S/128)*4] fp32
    int B, int S, int H, int Dd,
    float eps, float sm_scale,
    void* rstd_buf,             // [B*S] fp32 caller-owned scratch, or nullptr
    cudaStream_t stream);

// Fused RMSNorm + RoPE + per-block int8 quantization + smooth_k (for K).
// Input:  x [B*S, H*Dd] fp16.
// Output: out_int8 [B*S, H*Dd] int8.
//         scale [B, H, ceil(S/64)] fp32
// Each scale covers BLKK=64 consecutive tokens for one head.
// smooth_k: subtract per-head mean (km) before quantization.
// rstd_buf: caller-owned scratch of B*S floats (see _q above); nullptr
//           falls back to a stream-ordered transient allocation.
int fp16_rmsnorm_rope_quant_int8_k(
    const void* x_fp16,         // [B*S, H*Dd] fp16 input
    const void* weight_fp16,    // [H*Dd] fp16 norm weight
    const void* bias_fp16,      // [H*Dd] fp16 K-proj bias, or nullptr (fused pre-norm)
    const void* cos_fp32,       // [S, Dd/2] fp32
    const void* sin_fp32,       // [S, Dd/2] fp32
    const void* km_fp16,        // [B, 1, H, Dd] fp16 key mean (smooth_k)
    void* out_int8,             // [B*S, H*Dd] int8 output
    void* scale_fp32,           // [B, H, ceil(S/64)] fp32
    int B, int S, int H, int Dd,
    float eps, float sm_scale,
    void* rstd_buf,             // [B*S] fp32 caller-owned scratch, or nullptr
    cudaStream_t stream);

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt
