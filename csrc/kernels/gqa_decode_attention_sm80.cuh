// SPDX-License-Identifier: Apache-2.0
//
// Attention for one query position against a cache, with grouped key/value
// heads and an optional output gate applied where the result is produced.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// out = softmax(q . K^T * scale) V, causally, for ``q_rows`` query rows.
//
//   q        (q_rows, q_heads, head_dim)
//   k_cache  (capacity, kv_heads, head_dim)
//   v_cache  (capacity, kv_heads, head_dim)
//   gate     (q_rows, q_heads * head_dim), or null
//   out      (q_rows, q_heads, head_dim)
//
// The query rows are the last ``q_rows`` of the cache, so row r attends to
// everything up to and including position ``seq_len - q_rows + r``. At one
// row that is the whole cache, which is what a decode step wants; at many it
// is the causal mask a prompt wants, without a mask being built.
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
// ``head_dim`` is a power of two from 32 to 1024: a lane holds the dimensions
// congruent to it modulo the warp width, and that count is a compile-time
// bound because it sizes the accumulator.
//
// Returns 0, or -1 if the geometry is not one this decodes.
int gqa_decode_attention_bf16(
    const void* q,
    const void* k_cache, const void* v_cache,
    const void* gate,
    void* out,
    int seq_len, const int* seq_len_device,
    int q_heads, int kv_heads, int head_dim,
    float scale, int q_rows,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
