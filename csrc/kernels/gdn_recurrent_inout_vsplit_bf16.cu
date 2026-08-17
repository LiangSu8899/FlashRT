// SPDX-License-Identifier: Apache-2.0
//
// Gated-delta recurrent decode step, V-split launch plan. See header.
#include "kernels/gdn_recurrent_inout_vsplit_bf16.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {
namespace {

constexpr int kHD = 128;
constexpr int kCols = 32;          // value columns per block (one warp)
constexpr float kEps = 1e-6f;

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

__global__ void recurrent_vsplit_kernel(
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
  const int vb = blockIdx.y;            // which 32-column slice
  const int lane = threadIdx.x;         // 0..31
  const int t = vb * kCols + lane;      // this thread's value column

  const size_t hv_off = ((size_t)blockIdx.z * num_v_heads + h) * kHD;

  // q/k stage in registers: each lane owns 4 of the 128 entries, and
  // the L2 norms reduce across the warp
  float qs[4], ks[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = j * kCols + lane;
    qs[j] = static_cast<float>(q_in[hv_off + i]);
    ks[j] = static_cast<float>(k_in[hv_off + i]);
  }
  if (use_qk_l2norm) {
    float q_sq = 0.f, k_sq = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      q_sq += qs[j] * qs[j];
      k_sq += ks[j] * ks[j];
    }
    const float q_inv = rsqrtf(warp_sum(q_sq) + kEps);
    const float k_inv = rsqrtf(warp_sum(k_sq) + kEps);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      qs[j] *= q_inv;
      ks[j] *= k_inv;
    }
  }
  const float qscale = rsqrtf(static_cast<float>(kHD));
  #pragma unroll
  for (int j = 0; j < 4; ++j) qs[j] *= qscale;

  // broadcast the staged vectors so every lane sees all 128 entries
  // in the packaged kernel's index order
  __shared__ float sq[kHD], sk[kHD];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    sq[j * kCols + lane] = qs[j];
    sk[j * kCols + lane] = ks[j];
  }
  __syncwarp();

  const float g_t =
      __expf(static_cast<float>(g_in[blockIdx.z * num_v_heads + h]));
  const float beta_t =
      static_cast<float>(beta_in[blockIdx.z * num_v_heads + h]);

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

int gdn_recurrent_inout_vsplit_bf16(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, const void* state_in, void* state_out, void* out,
    int B, int num_v_heads, int head_dim, bool use_qk_l2norm,
    cudaStream_t stream) {
  if (!q || !k || !v || !g || !beta || !state_in || !state_out || !out)
    return 1;
  if (head_dim != kHD || B <= 0 || num_v_heads <= 0) return 2;
  dim3 grid(num_v_heads, kHD / kCols, B);
  recurrent_vsplit_kernel<<<grid, kCols, 0, stream>>>(
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
