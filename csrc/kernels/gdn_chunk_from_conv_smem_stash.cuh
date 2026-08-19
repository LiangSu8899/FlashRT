// SPDX-License-Identifier: Apache-2.0
//
// Gated-delta chunk core with per-row state stash — the spec-verify
// arm. Identical recurrence to the plain from-conv chunk kernel (same
// per-row bf16 state requantisation, same reduction order), with the
// post-row state additionally written to a stash slab. A rejected
// speculative round then rolls back by *selecting* the stash row at
// the accepted length instead of re-driving the state sublayers: the
// stash row is bit-equal to what a re-advance over that prefix would
// have stored. Additive: new file + new entry point.
#pragma once
#include <cuda_runtime.h>
namespace flash_rt {
namespace gdn {
// conv_out (S, (2*Hk+Hv)*128) packed q|k|v rows from the conv update;
// a/b (S, *_stride) raw gating projections; state (Hv, 128, 128) bf16,
// carried in place; out (S, Hv, 128); stash (S, Hv, 128, 128) bf16 —
// row s holds the carried state *after* row s. head_dim must be 128.
void gdn_chunk_from_conv_smem_h_stash_bf16(
    const void* conv_out, const void* a, const void* b,
    const float* neg_exp_A_log, const float* dt_bias, void* state,
    void* out, void* stash, int S, int num_v_heads, int num_k_heads,
    int head_dim, int a_stride, int b_stride, bool use_qk_l2norm,
    cudaStream_t stream);
}  // namespace gdn
}  // namespace flash_rt
