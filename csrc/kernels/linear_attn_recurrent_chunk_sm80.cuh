// SPDX-License-Identifier: Apache-2.0
//
// The gated-delta recurrence over a chunk of positions, in one launch, with
// the state kept in float32 throughout.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// Run the recurrence over ``S`` consecutive positions.
//
//   q, k    (S, heads, head_dim)   already broadcast to the value head count
//   v       (S, heads, head_dim)
//   g, beta (S, heads)
//   state   (heads, head_dim, head_dim)  float32, read and written
//   out     (S, heads, head_dim)
//
// Per position: the state decays by exp(g), the part of the value already
// remembered is subtracted, the remainder is written in at rate beta, and the
// output is the state read through the query.
//
// The recurrence is sequential and cannot be widened, but it does not have to
// be re-entered: a thread owns one column of the state and keeps it in
// registers across the whole chunk, so a chunk costs one launch and one pass
// over the state rather than one of each per position. That matters for a
// prompt, where the alternative is a launch per position per layer -- for a
// few hundred positions the launches alone outweigh the arithmetic.
//
// The state stays float32 for the same reason it does in the single-position
// kernel: it is multiplied by a decay every step and never re-derived, so
// rounding it per step is a drift that runs for as long as the sequence does.
// A prompt and the tokens after it therefore see the same state, which they
// would not if the prompt used a lower-precision path.
//
// ``head_dim`` is 64 or 128. Returns 0, or -1 for a geometry not implemented.
int linear_attn_recurrent_chunk_f32state_bf16(
    const void* q, const void* k, const void* v,
    const void* g, const void* beta,
    void* state, void* out,
    int S, int heads, int head_k, int head_v,
    bool use_qk_l2norm,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
