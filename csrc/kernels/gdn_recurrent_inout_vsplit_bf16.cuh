// SPDX-License-Identifier: Apache-2.0
//
// Gated-delta recurrent decode step, V-split launch plan.
//
// The recurrence gives every value column its own thread: that thread
// holds state[:, v] in registers and reduces over K entirely on its
// own, so the only cross-thread work in the step is the q/k L2 norm.
// The packaged kernel still launches one block per head — 48 blocks on
// a 170-SM part, a quarter of the machine — because it ties the block
// width to the 128 value columns. This entry splits the columns across
// blocks instead (one warp per 32-column slice), keeping the total
// thread count identical while spreading it over 4x the blocks.
//
// The per-column arithmetic is transcribed unchanged, so the state and
// output a column receives are bit-identical to the packaged kernel's.
// The L2 norm reduces over a warp rather than a 128-thread block, so
// its summation order differs — a fp32 rounding difference on the q/k
// scale, judged by the model's own arbiter band, not claimed bitwise.
// Additive.
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// q/k/v: (B, H, 128) bf16. g/beta: (B, H) bf16. state_in/out:
// (B, H, 128, 128) bf16 (may alias). out: (B, H, 128) bf16.
// head_dim must be 128. Returns 0 on success.
int gdn_recurrent_inout_vsplit_bf16(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, const void* state_in, void* state_out, void* out,
    int B, int num_v_heads, int head_dim, bool use_qk_l2norm,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
