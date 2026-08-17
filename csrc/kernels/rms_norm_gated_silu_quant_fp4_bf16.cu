// SPDX-License-Identifier: Apache-2.0
//
// Fused RMSNorm + weight + silu(gate) + NVFP4 quantize. See header.
#include "kernels/rms_norm_gated_silu_quant_fp4_bf16.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace {

constexpr int kDim = 128;

__device__ __forceinline__ int rgs_sfa_offset_128x64(
    int row, int k, int dim) {
  const int row_block = row >> 7;
  const int row_in_block = row & 127;
  const int k_block = k >> 6;
  const int k_in_block = k & 63;
  const int k_blocks = (dim + 63) >> 6;
  return row_block * k_blocks * 512 + k_block * 512 +
      (row_in_block & 31) * 16 + (row_in_block >> 5) * 4 +
      (k_in_block >> 4);
}

__device__ __forceinline__ uint8_t rgs_fp32_to_e2m1(float x) {
    uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
    float ax = fabsf(x);
    uint8_t mant;
    if      (ax <= 0.25f) mant = 0u;
    else if (ax <= 0.75f) mant = 1u;
    else if (ax <= 1.25f) mant = 2u;
    else if (ax <= 1.75f) mant = 3u;
    else if (ax <= 2.5f)  mant = 4u;
    else if (ax <= 3.5f)  mant = 5u;
    else if (ax <= 5.0f)  mant = 6u;
    else                  mant = 7u;
    return sign | mant;
}

__global__ void rms_norm_gated_silu_quant_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    uint2* __restrict__ packed,
    uint8_t* __restrict__ sfa,
    int M, int D, float eps) {
  const int m = blockIdx.x;
  const int t = threadIdx.x;
  if (m >= M || t >= kDim) return;

  const size_t row_off = (size_t)m * kDim + t;
  const float xv = __bfloat162float(x[row_off]);
  const float gv = __bfloat162float(gate[row_off]);

  // block-reduce sum-of-squares, transcribed from the packaged kernel
  float sq = xv * xv;
  for (int off = 16; off > 0; off >>= 1)
    sq += __shfl_xor_sync(0xffffffff, sq, off);
  __shared__ float warp_sq[4];
  __shared__ float reduced;
  const int lane = t & 31;
  const int warp = t >> 5;
  if (lane == 0) warp_sq[warp] = sq;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < 4) ? warp_sq[lane] : 0.0f;
    v += __shfl_xor_sync(0xffffffff, v, 1);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    if (lane == 0) reduced = v;
  }
  __syncthreads();

  const float rms_inv = rsqrtf(reduced / static_cast<float>(kDim) + eps);
  const float wv = __bfloat162float(weight[t]);
  const __nv_bfloat16 norm_bf = __float2bfloat16(xv * rms_inv);
  const __nv_bfloat16 weighted_bf =
      __float2bfloat16(wv * __bfloat162float(norm_bf));
  const float silu_g = gv / (1.0f + __expf(-gv));
  const __nv_bfloat16 out_bf =
      __float2bfloat16(__bfloat162float(weighted_bf) * silu_g);
  out[row_off] = out_bf;

  // the consumer reads this row as one (1, M*128) activation: its
  // 16-element quantize blocks tile a head's lanes exactly, so the
  // eight blocks of this row quantize here, in the production path's
  // own arithmetic
  __shared__ float vals[kDim];
  vals[t] = __bfloat162float(out_bf);
  __syncthreads();
  if (t >= kDim / 16) return;
  const int blk = t;                       // 0..7
  const int base = blk * 16;
  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float a = fabsf(vals[base + i]);
    if (a > amax) amax = a;
  }
  float desired = amax / 6.f;
  if (desired < 1e-12f) desired = 1e-12f;
  __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
  const float bs_dq = static_cast<float>(bs_q);
  const int col = m * kDim + base;         // column in the (1, D) row
  sfa[rgs_sfa_offset_128x64(0, col, D)] =
      *reinterpret_cast<uint8_t*>(&bs_q);
  const float inv_bs = 1.f / bs_dq;
  uint2 o;
  uint8_t* ob = reinterpret_cast<uint8_t*>(&o);
  #pragma unroll
  for (int p = 0; p < 8; ++p) {
    const uint8_t lo = rgs_fp32_to_e2m1(vals[base + 2 * p] * inv_bs);
    const uint8_t hi = rgs_fp32_to_e2m1(vals[base + 2 * p + 1] * inv_bs);
    ob[p] = static_cast<uint8_t>(lo | (hi << 4));
  }
  packed[(size_t)m * (kDim / 16) + blk] = o;
}

}  // namespace

int rms_norm_gated_silu_quant_fp4_bf16(
    const void* x, const void* gate, const void* weight, void* out,
    void* packed, void* sfa, int M, int dim, float eps,
    cudaStream_t stream) {
  if (!x || !gate || !weight || !out || !packed || !sfa) return 1;
  if (dim != kDim || M <= 0) return 2;
  rms_norm_gated_silu_quant_kernel<<<M, kDim, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<const __nv_bfloat16*>(gate),
      reinterpret_cast<const __nv_bfloat16*>(weight),
      reinterpret_cast<__nv_bfloat16*>(out),
      reinterpret_cast<uint2*>(packed),
      reinterpret_cast<uint8_t*>(sfa), M, M * kDim, eps);
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace kernels
}  // namespace flash_rt
