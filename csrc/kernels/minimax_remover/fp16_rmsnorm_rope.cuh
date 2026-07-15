// ================================================================
// flash_rt_minimax_remover — fused RMSNorm + interleaved RoPE kernel.
//
// The MiniMax-Remover attention entry does three separate passes over
// each Q/K tensor between the QKV Linear and the attention kernel:
//
//   1. rms_norm_fp32stat(q, norm_q.weight, eps)   # fp32-stat RMSNorm
//        - reads q, computes per-token RMS across D, writes normalised
//   2. .view(B, S, H, Dd)                          # zero-cost reshape
//   3. rope_apply_bshd(q, cos, sin)                # in-place interleaved RoPE
//
// This kernel fuses (1) and (3) into a single pass:
//   * Per-token fp32 RMS reduction over D (D = H * Dd).
//   * Per-element affine (weight[d] broadcast across [B*S] rows).
//   * Interleaved RoPE on adjacent (2k, 2k+1) fp16 pairs, with
//     per-head cos/sin tables shared across heads (cos/sin depend
//     only on the sequence index and the pair index within Dd).
//
// Layout assumptions (matches _kernels.py):
//   x        : [B*S*H, Dd] fp16 contiguous (equivalent to [B,S,H,Dd])
//   weight   : [D]  fp16  (RMSNorm affine, D = H * Dd)
//   cos, sin : [S, Dd/2] fp32
//   eps      : LayerNorm epsilon
//
// The RMS is computed over the *full* D per token (i.e. jointly over
// all heads), matching qk_norm="rms_norm_across_heads".  Grid: one
// block per token (B*S), each block reduces D in fp32 then applies
// RoPE across the H heads sequentially.
// ================================================================
#ifndef FLASHRT_KERNELS_MINIMAX_REMOVER_FP16_RMSNORM_ROPE_CUH
#define FLASHRT_KERNELS_MINIMAX_REMOVER_FP16_RMSNORM_ROPE_CUH

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

// Fused RMSNorm(fp32 stats + fp16 affine) + interleaved RoPE.
// x is treated as [B*S, H, Dd] fp16 contiguous.  Returns 0 on success.
int fp16_rmsnorm_rope_bshd(
    void* x_fp16,               // in/out [B*S, D]  D = H*Dd
    const void* weight_fp16,    // [D]
    const void* cos_fp32,       // [S, Dd/2]
    const void* sin_fp32,       // [S, Dd/2]
    int B, int S, int H, int Dd,
    float eps,
    cudaStream_t stream);

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt

#endif  // FLASHRT_KERNELS_MINIMAX_REMOVER_FP16_RMSNORM_ROPE_CUH
