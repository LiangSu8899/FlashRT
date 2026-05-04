// SPDX-License-Identifier: Apache-2.0
//
// Gated DeltaNet (linear attention) kernels for Qwen3.6 — Phase 3.3 / 3.4.
// Implements the same math as transformers/models/qwen3_5/modeling_qwen3_5.py
// torch_recurrent_gated_delta_rule and torch_chunk_gated_delta_rule, fused
// into single-launch kernels for SM120a / RTX 5090.
//
// Qwen3.6 layer config (linear-attn):
//   * num_v_heads = 48, num_k_heads = 16   (Q/K broadcast 3x)
//   * head_k_dim  = head_v_dim = 128
//   * use_qk_l2norm_in_kernel = True
//   * scale = 1 / sqrt(head_k_dim) applied to Q
//
// State per layer: (B, num_v_heads, head_k_dim, head_v_dim) bf16.
// Per-token operations are O(NH_v * HD_k * HD_v) = ~786 K bf16 ops/layer.
//
// All ops accumulate in fp32 internally; state stored as bf16 on disk
// (fp32 state was prototyped but bf16 cos vs HF ref >= 0.999 in practice).

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// Single-token decode using recurrent state. Updates ``state`` in place
// and writes the per-head output to ``out``.
//
// Tensor layouts (all bf16 row-major):
//   q     : (B, num_v_heads, head_k_dim)        already broadcast from
//                                                num_k_heads via repeat_interleave
//   k     : (B, num_v_heads, head_k_dim)        same broadcast
//   v     : (B, num_v_heads, head_v_dim)
//   g     : (B, num_v_heads)                    log-decay gate
//   beta  : (B, num_v_heads)                    sigmoid'd update rate
//   state : (B, num_v_heads, head_k_dim, head_v_dim)  in/out
//   out   : (B, num_v_heads, head_v_dim)
//
// Constraints: head_k_dim == head_v_dim == 128 (Qwen3.6); kernel is
// templated on it for register-allocation efficiency, but exposes a
// runtime check for correctness.
//
// If ``use_qk_l2norm`` is true, q and k are L2-normalized along the
// head dim before the recurrent update (matches HF's
// use_qk_l2norm_in_kernel=True).
void gated_deltanet_recurrent_qwen36_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    void*       state,
    void*       out,
    int B, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream);

// In/out-state variant: reads col from state_in, writes updated col
// to state_out (different buffer). Caller chains state_in[k+1] :=
// state_out[k] to support per-step state save without an extra
// .copy_(state_save, state) launch per step.
void gated_deltanet_recurrent_inout_qwen36_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    const void* state_in,
    void*       state_out,
    void*       out,
    int B, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
