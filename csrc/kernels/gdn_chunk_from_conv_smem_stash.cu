// SPDX-License-Identifier: Apache-2.0
//
// Per-row-stash arm of the gated-delta from-conv chunk core. The
// recurrence, gating math, reduction order, and per-row bf16 state
// requantisation are the plain kernel's, verbatim — the only addition
// is the stash write after each row's update, quantised exactly like
// the carried state, so stash row s is bit-equal to the final state a
// re-advance over rows 0..s would store. One block per value head,
// state resident in shared memory across the row walk.
#include "gdn_chunk_from_conv_smem_stash.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace gdn {
namespace {

constexpr float kEps = 1e-6f;
constexpr int kHD = 128;

template <int HD>
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int warp = t >> 5;
  for (int off = 16; off > 0; off >>= 1)
    val += __shfl_down_sync(0xffffffffu, val, off);
  if (lane == 0) smem[warp] = val;
  __syncthreads();
  if (warp == 0) {
    val = (t < HD / 32) ? smem[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
      val += __shfl_down_sync(0xffffffffu, val, off);
    if (lane == 0) smem[0] = val;
  }
  __syncthreads();
  return smem[0];
}

template <int HD>
__global__ void gdn_chunk_from_conv_smem_stash_kernel(
    const __nv_bfloat16* __restrict__ conv_out,
    const __nv_bfloat16* __restrict__ a_in,
    const __nv_bfloat16* __restrict__ b_in,
    const float* __restrict__ neg_exp_A_log,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ out_,
    __nv_bfloat16* __restrict__ stash,
    int S,
    int num_v_heads,
    int num_k_heads,
    int a_stride,
    int b_stride,
    bool use_qk_l2norm)
{
  static_assert(HD == 128, "HD must be 128 for this host family");
  const int h = blockIdx.x;
  const int t = threadIdx.x;
  if (t >= HD) return;

  extern __shared__ float smem[];
  float* state_s = smem;
  float* qs = state_s + HD * HD;
  float* ks = qs + HD;
  float* scratch = ks + HD;
  float* gate_values = scratch + 32;

  const size_t state_h_off = (size_t)h * HD * HD;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state_s[i * HD + t] = static_cast<float>(
        state[state_h_off + (size_t)i * HD + t]);
  }
  __syncthreads();

  const int broadcast = num_v_heads / num_k_heads;
  const int src_h = h / broadcast;
  const int qk_width = num_k_heads * HD;
  const int row_width = (2 * num_k_heads + num_v_heads) * HD;
  for (int s = 0; s < S; ++s) {
    const size_t row = static_cast<size_t>(s) * row_width;
    const size_t out_off = ((size_t)s * num_v_heads + h) * HD + t;
    qs[t] = static_cast<float>(conv_out[row + src_h * HD + t]);
    ks[t] = static_cast<float>(conv_out[row + qk_width + src_h * HD + t]);
    __syncthreads();

    if (use_qk_l2norm) {
      float q_sq = qs[t] * qs[t];
      float k_sq = ks[t] * ks[t];
      q_sq = block_reduce_sum<HD>(q_sq, scratch);
      // barrier between reduce calls sharing scratch: without it warp 0
      // begins the second reduce's scratch writes while a slower warp
      // still reads the first result (the plain kernel's receipt)
      __syncthreads();
      k_sq = block_reduce_sum<HD>(k_sq, scratch);
      const float q_inv = rsqrtf(q_sq + kEps);
      const float k_inv = rsqrtf(k_sq + kEps);
      qs[t] *= q_inv;
      ks[t] *= k_inv;
      __syncthreads();
    }

    qs[t] *= rsqrtf(static_cast<float>(HD));
    __syncthreads();

    if (t == 0) {
      const float av =
          static_cast<float>(a_in[s * a_stride + h]) + dt_bias[h];
      const float sp = log1pf(__expf(av));
      const float g_log = static_cast<float>(
          __float2bfloat16(neg_exp_A_log[h] * sp));
      gate_values[0] = __expf(g_log);
      const float bv = static_cast<float>(b_in[s * b_stride + h]);
      gate_values[1] = static_cast<float>(
          __float2bfloat16(1.0f / (1.0f + __expf(-bv))));
    }
    __syncthreads();
    const float g_t = gate_values[0];
    const float beta_t = gate_values[1];

    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      state_s[i * HD + t] *= g_t;
    }

    float kv_mem = 0.0f;
    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      kv_mem = fmaf(state_s[i * HD + t], ks[i], kv_mem);
    }

    const float v_t =
        static_cast<float>(conv_out[row + 2 * qk_width + h * HD + t]);
    const float delta = (v_t - kv_mem) * beta_t;

    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      state_s[i * HD + t] =
          fmaf(ks[i], delta, state_s[i * HD + t]);
    }

    float out_t = 0.0f;
    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      out_t = fmaf(state_s[i * HD + t], qs[i], out_t);
    }
    out_[out_off] = __float2bfloat16(out_t);

    // the stash write IS the carried requantisation: row s records
    // bf16(state after row s), exactly the value a re-advance ending
    // here would store, and exactly the value row s+1 resumes from
    const size_t stash_off =
        (((size_t)s * num_v_heads + h)) * HD * HD;
    #pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      const __nv_bfloat16 q16 = __float2bfloat16(state_s[i * HD + t]);
      stash[stash_off + (size_t)i * HD + t] = q16;
      state_s[i * HD + t] = static_cast<float>(q16);
    }
    __syncthreads();
  }

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state[state_h_off + (size_t)i * HD + t] =
        __float2bfloat16(state_s[i * HD + t]);
  }
}

}  // namespace

void gdn_chunk_from_conv_smem_h_stash_bf16(
    const void* conv_out, const void* a, const void* b,
    const float* neg_exp_A_log, const float* dt_bias, void* state,
    void* out, void* stash, int S, int num_v_heads, int num_k_heads,
    int head_dim, int a_stride, int b_stride, bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (S <= 0 || num_v_heads <= 0) return;
  if (head_dim != kHD) return;
  dim3 grid(num_v_heads, 1);
  dim3 block(kHD);
  constexpr size_t kSmemBytes =
      (kHD * kHD + 2 * kHD + 34) * sizeof(float);
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(
        gdn_chunk_from_conv_smem_stash_kernel<kHD>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSmemBytes));
    attr_set = true;
  }
  gdn_chunk_from_conv_smem_stash_kernel<kHD><<<
      grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(conv_out),
      reinterpret_cast<const __nv_bfloat16*>(a),
      reinterpret_cast<const __nv_bfloat16*>(b),
      neg_exp_A_log, dt_bias,
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(out),
      reinterpret_cast<__nv_bfloat16*>(stash),
      S, num_v_heads, num_k_heads, a_stride, b_stride, use_qk_l2norm);
}

}  // namespace gdn
}  // namespace flash_rt
