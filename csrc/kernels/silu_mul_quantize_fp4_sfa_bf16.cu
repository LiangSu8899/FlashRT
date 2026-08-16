// SPDX-License-Identifier: Apache-2.0
//
// Fused SwiGLU activation + NVFP4 quantize producer. See header for the
// contract. The quantize stage is a verbatim transcription of the
// production bf16 quantize kernel (same scale selection, rounding table
// and SFA layout); the activation stage rounds its fp32 product through
// bf16 so the quantizer sees the same value class the split chain fed
// it — measured bit-exact against (silu·mul kernel -> quantize kernel)
// across M in {1, 7, 2044} at H=17408.
#include "kernels/silu_mul_quantize_fp4_sfa_bf16.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace flash_rt {
namespace fp4 {
namespace {

__device__ __forceinline__ int smq_sfa_offset_128x64(
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

__device__ __forceinline__ uint8_t smq_fp32_to_e2m1(float x) {
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

__global__ void kernel_silu_mul_quantize_fp4_sfa_bf16(
    const int4* __restrict__ src,      // bf16 (N, 2H) as int4 chunks
    uint2* __restrict__ dst_packed,    // (N, H/2) bytes as uint2
    uint8_t* __restrict__ dst_sfa,
    int N, int H8) {                   // H8 = H/8 int4 chunks per half
  const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  const int n_blocks = H8 >> 1;        // 16 output elements per block
  if (row >= N || block_idx >= n_blocks) return;

  const long row_off = (long)row * (H8 << 1);
  const int4 g0 = src[row_off + 2 * block_idx];
  const int4 g1 = src[row_off + 2 * block_idx + 1];
  const int4 u0 = src[row_off + H8 + 2 * block_idx];
  const int4 u1 = src[row_off + H8 + 2 * block_idx + 1];
  const __nv_bfloat16* gh0 = reinterpret_cast<const __nv_bfloat16*>(&g0);
  const __nv_bfloat16* gh1 = reinterpret_cast<const __nv_bfloat16*>(&g1);
  const __nv_bfloat16* uh0 = reinterpret_cast<const __nv_bfloat16*>(&u0);
  const __nv_bfloat16* uh1 = reinterpret_cast<const __nv_bfloat16*>(&u1);

  float vals[16];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float g = __bfloat162float(gh0[i]);
    const float u = __bfloat162float(uh0[i]);
    vals[i] = __bfloat162float(
        __float2bfloat16((g / (1.f + expf(-g))) * u));
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float g = __bfloat162float(gh1[i]);
    const float u = __bfloat162float(uh1[i]);
    vals[8 + i] = __bfloat162float(
        __float2bfloat16((g / (1.f + expf(-g))) * u));
  }

  float amax = 0.f;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float a = fabsf(vals[i]);
    if (a > amax) amax = a;
  }

  float desired = amax / 6.f;
  if (desired < 1e-12f) desired = 1e-12f;
  __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
  const float bs_dq = static_cast<float>(bs_q);

  const int H = H8 << 3;
  dst_sfa[smq_sfa_offset_128x64(row, block_idx * 16, H)] =
      *reinterpret_cast<uint8_t*>(&bs_q);

  const float inv_bs = 1.f / bs_dq;
  uint2 out;
  uint8_t* ob = reinterpret_cast<uint8_t*>(&out);
  #pragma unroll
  for (int p = 0; p < 8; ++p) {
    const uint8_t lo = smq_fp32_to_e2m1(vals[2 * p] * inv_bs);
    const uint8_t hi = smq_fp32_to_e2m1(vals[2 * p + 1] * inv_bs);
    ob[p] = static_cast<uint8_t>(lo | (hi << 4));
  }
  dst_packed[(long)row * n_blocks + block_idx] = out;
}

}  // namespace

int silu_mul_quantize_fp4_sfa_bf16(
    const void* merged_bf16, void* dst_packed, void* dst_sfa,
    int N, int H, cudaStream_t stream) {
  if (!merged_bf16 || !dst_packed || !dst_sfa) return 1;
  if (N <= 0 || H <= 0 || (H % 16) != 0) return 2;
  if ((reinterpret_cast<uintptr_t>(merged_bf16) & 15) ||
      (reinterpret_cast<uintptr_t>(dst_packed) & 7)) return 3;
  const int n_blocks = H / 16;
  const int threads = 128;
  dim3 grid((n_blocks + threads - 1) / threads, N);
  kernel_silu_mul_quantize_fp4_sfa_bf16<<<grid, threads, 0, stream>>>(
      reinterpret_cast<const int4*>(merged_bf16),
      reinterpret_cast<uint2*>(dst_packed),
      reinterpret_cast<uint8_t*>(dst_sfa),
      N, H / 8);
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace fp4
}  // namespace flash_rt
