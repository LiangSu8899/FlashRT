// SPDX-License-Identifier: Apache-2.0
//
// WY-chain q/k L2-norm + pack + per-chunk gate cumsum, v2 launch plan.
// See header. Norm kernel body and reduction transcribed verbatim from
// the packaged fast arm; the cumsum keeps the packaged serial order
// inside each 64-token chunk (bit-exact) and parallelizes across the
// independent (chunk, head) pairs.
#include "kernels/gdn_wy_norm_cumsum_pack_qk_v2.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {
namespace {

constexpr int kHD = 128;
constexpr int kQHeads = 16;
constexpr int kVHeads = 48;
constexpr int kWyChunk = 64;
constexpr float kEps = 1e-6f;

template <int HD>
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
  for (int off = 16; off > 0; off >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, off);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = val;
  __syncthreads();
  if (warp == 0) {
    val = (lane < (HD / 32)) ? smem[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      val += __shfl_xor_sync(0xffffffff, val, off);
    }
    if (lane == 0) smem[0] = val;
  }
  __syncthreads();
  return smem[0];
}

__global__ void norm_qk_v2_kernel(
    const __nv_bfloat16* __restrict__ q16,
    const __nv_bfloat16* __restrict__ k16,
    __nv_bfloat16* __restrict__ q16_l2,
    __nv_bfloat16* __restrict__ k16_l2,
    __nv_bfloat16* __restrict__ q_pack_hv,
    __nv_bfloat16* __restrict__ k_pack_hk,
    int S, int num_k_heads, int num_v_heads, int head_group_size) {
  const int t = threadIdx.x;
  const int h = blockIdx.x;
  const int s = blockIdx.y;
  if (t >= kHD || h >= num_k_heads || s >= S) return;

  __shared__ float scratch[32];
  const size_t off = (static_cast<size_t>(s) * num_k_heads + h) * kHD + t;
  const float qv = static_cast<float>(q16[off]);
  const float kv = static_cast<float>(k16[off]);
  float q_sq = qv * qv;
  float k_sq = kv * kv;
  q_sq = block_reduce_sum<kHD>(q_sq, scratch);
  __syncthreads();
  k_sq = block_reduce_sum<kHD>(k_sq, scratch);
  __syncthreads();
  const float q_inv = rsqrtf(q_sq + kEps);
  const float k_inv = rsqrtf(k_sq + kEps);
  const __nv_bfloat16 q_norm = __float2bfloat16(qv * q_inv);
  const __nv_bfloat16 k_norm = __float2bfloat16(kv * k_inv);
  q16_l2[off] = q_norm;
  k16_l2[off] = k_norm;
  if (k_pack_hk != nullptr) {
    const int chunk = s / kWyChunk;
    const int tt = s - chunk * kWyChunk;
    k_pack_hk[
        ((static_cast<size_t>(chunk) * num_k_heads + h) * kWyChunk + tt)
        * kHD + t] = k_norm;
  }
  if (q_pack_hv != nullptr) {
    const int chunk = s / kWyChunk;
    const int tt = s - chunk * kWyChunk;
    #pragma unroll
    for (int r = 0; r < head_group_size; ++r) {
      const int vh = h * head_group_size + r;
      q_pack_hv[
          ((static_cast<size_t>(chunk) * num_v_heads + vh) * kWyChunk + tt)
          * kHD + t] = q_norm;
    }
  }
}

// one thread per (chunk, head): the packaged serial order inside the
// chunk, chunks in parallel (the accumulator resets at every boundary)
__global__ void cumsum_g_v2_kernel(
    const __nv_bfloat16* __restrict__ g,
    __nv_bfloat16* __restrict__ g_cumsum,
    int S, int num_v_heads, int num_chunks) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int h = idx % num_v_heads;
  const int chunk = idx / num_v_heads;
  if (chunk >= num_chunks) return;
  const int s0 = chunk * kWyChunk;
  const int s1 = min(s0 + kWyChunk, S);
  float acc = 0.0f;
  for (int s = s0; s < s1; ++s) {
    const size_t off = static_cast<size_t>(s) * num_v_heads + h;
    acc += static_cast<float>(g[off]);
    g_cumsum[off] = __float2bfloat16(acc);
  }
}

}  // namespace

int gdn_wy_norm_cumsum_pack_qk_v2_bf16(
    const void* q16, const void* k16, const void* g, void* q16_l2,
    void* k16_l2, void* q_pack_hv, void* k_pack_hk, void* g_cumsum,
    int S, cudaStream_t stream) {
  if (!q16 || !k16 || !g || !q16_l2 || !k16_l2 || !g_cumsum) return 1;
  if (S <= 0) return 2;
  norm_qk_v2_kernel<<<dim3(kQHeads, S), kHD, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q16),
      reinterpret_cast<const __nv_bfloat16*>(k16),
      reinterpret_cast<__nv_bfloat16*>(q16_l2),
      reinterpret_cast<__nv_bfloat16*>(k16_l2),
      reinterpret_cast<__nv_bfloat16*>(q_pack_hv),
      reinterpret_cast<__nv_bfloat16*>(k_pack_hk),
      S, kQHeads, kVHeads, kVHeads / kQHeads);
  const int chunks = (S + kWyChunk - 1) / kWyChunk;
  const int total = chunks * kVHeads;
  cumsum_g_v2_kernel<<<(total + 127) / 128, 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<__nv_bfloat16*>(g_cumsum),
      S, kVHeads, chunks);
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace kernels
}  // namespace flash_rt
