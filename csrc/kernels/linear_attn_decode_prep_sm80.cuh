// SPDX-License-Identifier: Apache-2.0
//
// Preparation for a gated-delta linear-attention step: the convolved QKV
// stream taken apart into per-head views, and the decay pair the recurrence
// needs, in one launch.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// Split a convolved QKV row into per-head q/k/v and compute the recurrence's
// gate pair, in a single launch.
//
//   conv_out (S, 2 * k_heads * head_k + v_heads * head_v)
//   a, b     (S, a_stride) and (S, b_stride), first ``v_heads`` used
//   q, k     (S, v_heads, head_k)   -- key heads broadcast to value heads
//   v        (S, v_heads, head_v)
//   g, beta  (S, v_heads)
//
// Query and key are published with fewer heads than value and are repeated,
// so q and k come out at the value head count: the recurrence has one state
// per value head and wants all three at that width.
//
//   beta = sigmoid(b)
//   g    = neg_exp_a_log[h] * softplus(a + dt_bias[h])
//
// The two halves have nothing to do with each other, which is the reason they
// share a launch rather than a kernel: at one row neither fills the part on
// its own, and a launch is not free.
//
// ``v_heads`` must be a multiple of ``k_heads``. Returns 0, or -1 if the
// geometry does not satisfy that.
int linear_attn_split_broadcast_gate_bf16(
    const void* conv_out,
    const void* a,
    const void* b,
    const float* neg_exp_a_log,
    const float* dt_bias,
    void* q, void* k, void* v,
    void* g, void* beta,
    int S, int k_heads, int v_heads, int head_k, int head_v,
    int a_stride, int b_stride,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
