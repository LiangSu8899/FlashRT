// SPDX-License-Identifier: Apache-2.0
//
// WY-chain q/k L2-norm + pack + per-chunk gate cumsum, v2 launch plan.
// The math is a verbatim transcription of the packaged
// norm_cumsum_pack_qk pair (same block reduction, same rounding); the
// v2 is the cumsum's launch: the packaged kernel walks the whole
// prompt from a single 64-thread block, while the per-64-token chunks
// are independent by construction (the accumulator resets at every
// chunk boundary), so this plan gives each (chunk, head) its own lane
// — bit-exact per chunk, parallel across them. Additive.
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// Fixed 16 k-head / 48 v-head / 128 head-dim / 64-chunk family (the
// same constants the packaged fast arm serves). q16/k16: (S, 16, 128)
// bf16. q16_l2/k16_l2: same shape. q_pack_hv: (NT, 48, 64, 128).
// k_pack_hk: (NT, 16, 64, 128). g: (S, 48) bf16 -> g_cumsum (S, 48).
// Returns 0 on success.
int gdn_wy_norm_cumsum_pack_qk_v2_bf16(
    const void* q16, const void* k16, const void* g, void* q16_l2,
    void* k16_l2, void* q_pack_hv, void* k_pack_hk, void* g_cumsum,
    int S, cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
