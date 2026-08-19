// SPDX-License-Identifier: Apache-2.0
//
// Gated-delta recurrent decode step, streaming-column form. See header.
#include "kernels/gdn_recurrent_inout_stream_bf16.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {
namespace {

constexpr int kHD = 128;
constexpr float kEps = 1e-6f;

// the packaged kernel's block reduction, transcribed: splitting the
// value columns across smaller blocks would reduce the q/k norms over
// a warp instead, and that changes their fp32 summation order. The
// measured win is not in the split — it is in not spilling the state
// column — so the block width stays 128 and the result stays bitwise
__device__ __forceinline__ float block_reduce_sum(float val,
                                                  float* smem) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    val += __shfl_xor_sync(0xffffffffu, val, off);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = val;
  __syncthreads();
  if (warp == 0) {
    val = (lane < (kHD / 32)) ? smem[lane] : 0.0f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      val += __shfl_xor_sync(0xffffffffu, val, off);
    if (lane == 0) smem[0] = val;
  }
  __syncthreads();
  return smem[0];
}

__global__ void recurrent_stream_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __nv_bfloat16* __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    const __nv_bfloat16* __restrict__ state_in,
    __nv_bfloat16* __restrict__ state_out,
    __nv_bfloat16* __restrict__ out_,
    int num_v_heads, bool use_qk_l2norm) {
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int t = threadIdx.x;            // this thread's value column
  if (t >= kHD) return;

  const size_t hv_off = ((size_t)b * num_v_heads + h) * kHD;

  __shared__ float smem[2 * kHD + 32];
  float* sq = smem;
  float* sk = smem + kHD;
  float* scratch = smem + 2 * kHD;
  sq[t] = static_cast<float>(q_in[hv_off + t]);
  sk[t] = static_cast<float>(k_in[hv_off + t]);
  __syncthreads();

  if (use_qk_l2norm) {
    float q_sq = block_reduce_sum(sq[t] * sq[t], scratch);
    __syncthreads();
    float k_sq = block_reduce_sum(sk[t] * sk[t], scratch);
    sq[t] *= rsqrtf(q_sq + kEps);
    sk[t] *= rsqrtf(k_sq + kEps);
    __syncthreads();
  }
  sq[t] *= rsqrtf(static_cast<float>(kHD));
  __syncthreads();

  const float g_t =
      __expf(static_cast<float>(g_in[b * num_v_heads + h]));
  const float beta_t =
      static_cast<float>(beta_in[b * num_v_heads + h]);

  // Two streaming passes rather than a 128-entry per-thread array.
  // That array is 128 registers on top of everything else, past what
  // a thread can hold, so the column it is meant to keep resident
  // spills to local memory and the step reads its own state twice
  // through DRAM anyway. Reading state twice explicitly costs the
  // same traffic on the first pass and hits cache on the second (a
  // block's slice is 32KB), while the register budget drops far
  // enough for the scheduler to hide the latency. Per-column
  // arithmetic and its order are unchanged.
  const size_t state_h_off = hv_off * kHD;
  float kv_mem = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < kHD; ++i) {
    const float c = static_cast<float>(
        state_in[state_h_off + (size_t)i * kHD + t]) * g_t;
    kv_mem = fmaf(c, sk[i], kv_mem);
  }

  const float v_t = static_cast<float>(v_in[hv_off + t]);
  const float delta = (v_t - kv_mem) * beta_t;

  float out_t = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < kHD; ++i) {
    const float c = fmaf(
        sk[i], delta,
        static_cast<float>(
            state_in[state_h_off + (size_t)i * kHD + t]) * g_t);
    state_out[state_h_off + (size_t)i * kHD + t] =
        __float2bfloat16(c);
    out_t = fmaf(c, sq[i], out_t);
  }
  out_[hv_off + t] = __float2bfloat16(out_t);
}

}  // namespace

int gdn_recurrent_inout_stream_bf16(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, const void* state_in, void* state_out, void* out,
    int B, int num_v_heads, int head_dim, bool use_qk_l2norm,
    cudaStream_t stream) {
  if (!q || !k || !v || !g || !beta || !state_in || !state_out || !out)
    return 1;
  if (head_dim != kHD || B <= 0 || num_v_heads <= 0) return 2;
  dim3 grid(num_v_heads, B);
  recurrent_stream_kernel<<<grid, kHD, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(state_in),
      reinterpret_cast<__nv_bfloat16*>(state_out),
      reinterpret_cast<__nv_bfloat16*>(out), num_v_heads,
      use_qk_l2norm);
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace kernels
}  // namespace flash_rt
