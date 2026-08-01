// SPDX-License-Identifier: Apache-2.0
//
// See linear_attn_recurrent_chunk_sm80.cuh. SM80-family facilities only.
//
// A block owns one head; thread t owns column t of that head's state, which
// is the whole of why this parallelises: the memory read and the output are
// both contractions along the other axis, so a thread needs no one else's
// column for either. Only the two normalizations are shared, and they are
// over the position's query and key rather than over the state.

#include "kernels/linear_attn_recurrent_chunk_sm80.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {
namespace {

constexpr float kEps = 1e-6f;

template <int HD>
__device__ __forceinline__ float block_sum(float value, float* scratch) {
  for (int offset = 16; offset; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (!lane) scratch[warp] = value;
  __syncthreads();
  constexpr int kWarps = HD / 32;
  value = (threadIdx.x < kWarps) ? scratch[threadIdx.x] : 0.0f;
#pragma unroll
  for (int offset = kWarps / 2; offset; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  if (!threadIdx.x) scratch[kWarps] = value;
  __syncthreads();
  return scratch[kWarps];
}

template <int HD>
__global__ void recurrent_chunk_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __nv_bfloat16* __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    float* __restrict__ state,
    __nv_bfloat16* __restrict__ out,
    int S, int heads, bool use_qk_l2norm) {
  const int head = blockIdx.x;
  const int t = threadIdx.x;

  __shared__ float shared[2 * HD + 64];
  float* query = shared;
  float* key = shared + HD;
  float* scratch = shared + 2 * HD;

  // The column this thread owns, read once for the whole chunk.
  float column[HD];
  const size_t base = static_cast<size_t>(head) * HD * HD;
#pragma unroll 16
  for (int i = 0; i < HD; ++i) column[i] = state[base + i * HD + t];

  for (int position = 0; position < S; ++position) {
    const size_t row =
        (static_cast<size_t>(position) * heads + head) * HD + t;
    query[t] = __bfloat162float(q_in[row]);
    key[t] = __bfloat162float(k_in[row]);
    __syncthreads();

    if (use_qk_l2norm) {
      const float q_sum = block_sum<HD>(query[t] * query[t], scratch);
      __syncthreads();
      const float k_sum = block_sum<HD>(key[t] * key[t], scratch);
      query[t] *= rsqrtf(q_sum + kEps);
      key[t] *= rsqrtf(k_sum + kEps);
      __syncthreads();
    }
    query[t] *= rsqrtf(static_cast<float>(HD));
    __syncthreads();

    const size_t gate = static_cast<size_t>(position) * heads + head;
    const float decay = __expf(__bfloat162float(g_in[gate]));
    const float rate = __bfloat162float(beta_in[gate]);

    float remembered = 0.0f;
#pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      column[i] *= decay;
      remembered = fmaf(column[i], key[i], remembered);
    }
    const float delta =
        (__bfloat162float(v_in[row]) - remembered) * rate;

    float result = 0.0f;
#pragma unroll 16
    for (int i = 0; i < HD; ++i) {
      column[i] = fmaf(key[i], delta, column[i]);
      result = fmaf(column[i], query[i], result);
    }
    out[row] = __float2bfloat16(result);
    // The next position overwrites the staged query and key.
    __syncthreads();
  }

#pragma unroll 16
  for (int i = 0; i < HD; ++i) state[base + i * HD + t] = column[i];
}

}  // namespace

int linear_attn_recurrent_chunk_f32state_bf16(
    const void* q, const void* k, const void* v,
    const void* g, const void* beta,
    void* state, void* out,
    int S, int heads, int head_k, int head_v,
    bool use_qk_l2norm,
    cudaStream_t stream) {
  if (S <= 0) return 0;
  if (heads <= 0 || head_k != head_v) return -1;

  const auto launch = [&](auto kernel, int width) {
    kernel<<<heads, width, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<const __nv_bfloat16*>(g),
        reinterpret_cast<const __nv_bfloat16*>(beta),
        reinterpret_cast<float*>(state),
        reinterpret_cast<__nv_bfloat16*>(out),
        S, heads, use_qk_l2norm);
  };
  switch (head_k) {
    case 64:  launch(recurrent_chunk_kernel<64>, 64);   break;
    case 128: launch(recurrent_chunk_kernel<128>, 128); break;
    default: return -1;
  }
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
