// SPDX-License-Identifier: Apache-2.0
//
// Gated-delta recurrent decode step, streaming-column form.
//
// The recurrence gives every value column its own thread: that thread
// holds state[:, v] and reduces over K entirely on its own. The
// packaged kernel keeps that column in a 128-entry per-thread array,
// which no thread can hold — it spills, and the step pays DRAM for
// state it believes is resident. This entry streams the column
// instead.
//
// The whole kernel is bit-identical to the packaged one — same block
// reduction for the q/k norms, same per-column arithmetic in the same
// order. The only change is that the column no longer lives in a
// per-thread array: 128 registers on top of everything else is past
// what a thread holds, so that array spills to local memory and the
// step reads its own state through DRAM anyway. Streaming the column
// in two passes costs the same traffic on the first and hits cache on
// the second (a head's slice is 32KB), while the register budget
// drops far enough for the scheduler to hide the latency. Additive.
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// q/k/v: (B, H, 128) bf16. g/beta: (B, H) bf16. state_in/out:
// (B, H, 128, 128) bf16 (may alias). out: (B, H, 128) bf16.
// head_dim must be 128. Returns 0 on success.
int gdn_recurrent_inout_stream_bf16(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, const void* state_in, void* state_out, void* out,
    int B, int num_v_heads, int head_dim, bool use_qk_l2norm,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
