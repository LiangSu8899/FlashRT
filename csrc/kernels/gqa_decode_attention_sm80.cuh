// SPDX-License-Identifier: Apache-2.0
//
// Attention for one query position against a cache, with grouped key/value
// heads and an optional output gate applied where the result is produced.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// out = softmax(q . K^T * scale) V, for a single query row.
//
//   q        (q_heads, head_dim)
//   k_cache  (capacity, kv_heads, head_dim)
//   v_cache  (capacity, kv_heads, head_dim)
//   gate     (q_heads * head_dim), or null
//   out      (q_heads * head_dim)
//
// Query heads are grouped onto key heads: head h reads key head
// ``h / (q_heads / kv_heads)``. Nothing is expanded to make that happen -- at
// one query row the cache is the whole read, and repeating it to match the
// query heads would multiply the only cost there is.
//
// The softmax runs online, so the scores are never held: a step at position t
// reads the cache once and keeps a running maximum, weight and accumulator.
//
// ``gate``, when given, is applied as ``out * sigmoid(gate)`` in the epilogue,
// where the result is already in registers.
//
// ``seq_len_device``, when not null, supplies the length instead of
// ``seq_len``, so a captured graph can be replayed as the sequence grows.
//
// Returns 0, or -1 if the geometry is not one this decodes.
int gqa_decode_attention_bf16(
    const void* q,
    const void* k_cache, const void* v_cache,
    const void* gate,
    void* out,
    int seq_len, const int* seq_len_device,
    int q_heads, int kv_heads, int head_dim,
    float scale,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
