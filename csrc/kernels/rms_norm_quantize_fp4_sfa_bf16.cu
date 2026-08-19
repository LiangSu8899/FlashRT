// SPDX-License-Identifier: Apache-2.0
//
// Fused (1+w)-form RMSNorm + NVFP4 quantize producer. See header. One
// block per row: the fp32 row is staged in shared memory (one global
// read), reduced for mean(x^2), then each thread quantizes 16-element
// blocks with the production quantize path verbatim while also writing
// the normed bf16 row.
#include "kernels/rms_norm_quantize_fp4_sfa_bf16.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace flash_rt {
namespace fp4 {
namespace {

__device__ __forceinline__ int rnq_sfa_offset_128x64(
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

__device__ __forceinline__ uint8_t rnq_fp32_to_e2m1(float x) {
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

template <int THREADS>
__global__ void kernel_rms_norm_quantize_fp4_sfa_bf16(
    const int4* __restrict__ x,        // bf16 (N, D) as int4 (8 elems)
    const __nv_bfloat16* __restrict__ w,
    float eps,
    int4* __restrict__ normed,         // bf16 (N, D) as int4
    uint2* __restrict__ dst_packed,
    uint8_t* __restrict__ dst_sfa,
    int N, int D) {
  __shared__ float red[THREADS / 32];
  const int row = blockIdx.x;
  if (row >= N) return;
  const int D8 = D >> 3;
  const int4* xr = x + (size_t)row * D8;

  // pass 1: vectorized sum of squares (the row stays L2-resident for
  // pass 2 — no shared-memory staging, so occupancy is thread-bound)
  float ssq = 0.f;
  for (int i = threadIdx.x; i < D8; i += THREADS) {
    const int4 raw = xr[i];
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float v = __bfloat162float(h[j]);
      ssq += v * v;
    }
  }
  #pragma unroll
  for (int o = 16; o; o >>= 1)
    ssq += __shfl_down_sync(0xffffffffu, ssq, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ssq;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = (threadIdx.x < THREADS / 32) ? red[threadIdx.x] : 0.f;
    #pragma unroll
    for (int o = 16; o; o >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  const float rstd = rsqrtf(red[0] / (float)D + eps);

  // pass 2: one 16-element block per iteration — vectorized re-read
  // (L2), norm, vectorized bf16 store, production quantize path
  const int n_blocks = D / 16;
  const int4* wr = reinterpret_cast<const int4*>(w);
  for (int b = threadIdx.x; b < n_blocks; b += THREADS) {
    const int4 r0 = xr[2 * b], r1 = xr[2 * b + 1];
    const int4 w0 = wr[2 * b], w1 = wr[2 * b + 1];
    const __nv_bfloat16* h0 = reinterpret_cast<const __nv_bfloat16*>(&r0);
    const __nv_bfloat16* h1 = reinterpret_cast<const __nv_bfloat16*>(&r1);
    const __nv_bfloat16* g0 = reinterpret_cast<const __nv_bfloat16*>(&w0);
    const __nv_bfloat16* g1 = reinterpret_cast<const __nv_bfloat16*>(&w1);
    float vals[16];
    int4 nb0, nb1;
    __nv_bfloat16* nh0 = reinterpret_cast<__nv_bfloat16*>(&nb0);
    __nv_bfloat16* nh1 = reinterpret_cast<__nv_bfloat16*>(&nb1);
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float y = __bfloat162float(h0[i]) * rstd *
          (1.f + __bfloat162float(g0[i]));
      nh0[i] = __float2bfloat16(y);
      vals[i] = __bfloat162float(nh0[i]);
      const float a = fabsf(vals[i]);
      if (a > amax) amax = a;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float y = __bfloat162float(h1[i]) * rstd *
          (1.f + __bfloat162float(g1[i]));
      nh1[i] = __float2bfloat16(y);
      vals[8 + i] = __bfloat162float(nh1[i]);
      const float a = fabsf(vals[8 + i]);
      if (a > amax) amax = a;
    }
    normed[(size_t)row * D8 + 2 * b] = nb0;
    normed[(size_t)row * D8 + 2 * b + 1] = nb1;
    float desired = amax / 6.f;
    if (desired < 1e-12f) desired = 1e-12f;
    __nv_fp8_e4m3 bs_q = __nv_fp8_e4m3(fmaxf(desired, 0.f));
    const float bs_dq = static_cast<float>(bs_q);
    dst_sfa[rnq_sfa_offset_128x64(row, b * 16, D)] =
        *reinterpret_cast<uint8_t*>(&bs_q);
    const float inv_bs = 1.f / bs_dq;
    uint2 out;
    uint8_t* ob = reinterpret_cast<uint8_t*>(&out);
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
      const uint8_t lo = rnq_fp32_to_e2m1(vals[2 * p] * inv_bs);
      const uint8_t hi = rnq_fp32_to_e2m1(vals[2 * p + 1] * inv_bs);
      ob[p] = static_cast<uint8_t>(lo | (hi << 4));
    }
    dst_packed[(size_t)row * n_blocks + b] = out;
  }
}

}  // namespace

int rms_norm_quantize_fp4_sfa_bf16(
    const void* x_bf16, const void* w_bf16, float eps, void* normed_bf16,
    void* dst_packed, void* dst_sfa, int N, int D, cudaStream_t stream) {
  if (!x_bf16 || !w_bf16 || !normed_bf16 || !dst_packed || !dst_sfa)
    return 1;
  if (N <= 0 || D <= 0 || (D % 16) != 0 || D > 8192) return 2;
  if ((reinterpret_cast<uintptr_t>(x_bf16) & 15) ||
      (reinterpret_cast<uintptr_t>(w_bf16) & 15) ||
      (reinterpret_cast<uintptr_t>(normed_bf16) & 15)) return 3;
  constexpr int THREADS = 256;
  kernel_rms_norm_quantize_fp4_sfa_bf16<THREADS>
      <<<N, THREADS, 0, stream>>>(
      reinterpret_cast<const int4*>(x_bf16),
      reinterpret_cast<const __nv_bfloat16*>(w_bf16), eps,
      reinterpret_cast<int4*>(normed_bf16),
      reinterpret_cast<uint2*>(dst_packed),
      reinterpret_cast<uint8_t*>(dst_sfa), N, D);
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace fp4
}  // namespace flash_rt
