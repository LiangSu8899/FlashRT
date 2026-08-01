// SPDX-License-Identifier: Apache-2.0
//
// Everything between a fused QKV projection and the attention itself, in one
// launch: the per-head split, the per-head norms, the rotary embedding over
// the part of the head that carries it, and the write into the key/value
// cache.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// Stage one fused QKV row for attention.
//
//   qkv       (S, q_heads * head_dim * (gate ? 2 : 1) + 2 * kv_heads * head_dim)
//   q_norm_w  (head_dim)      applied per query head
//   k_norm_w  (head_dim)      applied per key head
//   cos, sin  (positions, rope_dim / 2)
//   q_out     (S, q_heads, head_dim)
//   gate_out  (S, q_heads * head_dim), or null when the projection has no gate
//   k_cache   (capacity, kv_heads, head_dim)   written at ``pos + row``
//   v_cache   (capacity, kv_heads, head_dim)
//
// When the projection carries an output gate it is interleaved per head --
// each head's slice is its query followed by its gate -- and the gate comes
// out as one contiguous row, which is the form the attention epilogue reads.
//
// Only the first ``rope_dim`` of each head is rotated and the rest is carried
// through, so ``rope_dim`` may be less than ``head_dim``; the table holds
// ``rope_dim / 2`` entries per position because the second half of a rotation
// repeats the first.
//
// ``pos_device``, when not null, supplies the position instead of ``pos``.
// A captured graph replays the addresses it was captured with, so a step whose
// position lives on the device can be captured once and replayed for every
// token; a position passed by value could not.
//
// Returns 0, or -1 if the geometry is not one this decodes.
int attn_qkv_norm_rope_write_bf16(
    const void* qkv,
    const void* q_norm_w,
    const void* k_norm_w,
    const void* cos, const void* sin,
    void* q_out, void* gate_out,
    void* k_cache, void* v_cache,
    int S, int pos, const int* pos_device,
    int q_heads, int kv_heads, int head_dim, int rope_dim,
    bool has_gate, float eps,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
