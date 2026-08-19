// SPDX-License-Identifier: Apache-2.0
//
// Chunk-parallel causal conv1d update with per-thread step batching
// and GQA split outputs. See header for the contract.
#include "kernels/causal_conv1d_update_steps_gqa_bf16.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {
namespace {

constexpr int kConvDim = 10240;
constexpr int kK = 4;
constexpr int kSteps = 8;
constexpr int kThreads = 128;

__device__ __forceinline__ float conv_silu(float x) {
  // matches the packaged kernel: x * sigmoid(x) via the fast exp
  return x / (1.0f + __expf(-x));
}

__global__ void conv1d_steps_gqa_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ q16,
    __nv_bfloat16* __restrict__ k16,
    __nv_bfloat16* __restrict__ v48,
    int S, bool apply_silu) {
  const int c = blockIdx.x * kThreads + threadIdx.x;
  if (c >= kConvDim) return;
  const int s0 = blockIdx.y * kSteps;

  // tap weights and bias for this channel
  float wv[kK];
  #pragma unroll
  for (int i = 0; i < kK; ++i)
    wv[i] = static_cast<float>(w[c * kK + i]);
  const float bv = (bias != nullptr) ? static_cast<float>(bias[c]) : 0.f;

  // rolling window xv[0..2] = taps t = s-3..s-1, xv[3] = t = s
  float xv[kK];
  #pragma unroll
  for (int i = 0; i < kK - 1; ++i) {
    const int t = s0 - (kK - 1) + i;
    float v = 0.f;
    if (t >= 0) {
      v = static_cast<float>(x[(size_t)t * kConvDim + c]);
    } else if (t >= -(kK - 1)) {
      v = static_cast<float>(state[c * (kK - 1) + (t + kK - 1)]);
    }
    xv[i] = v;
  }

  #pragma unroll
  for (int j = 0; j < kSteps; ++j) {
    const int s = s0 + j;
    if (s >= S) return;
    xv[kK - 1] = static_cast<float>(x[(size_t)s * kConvDim + c]);
    // same accumulation order as the packaged kernel: bias first,
    // then taps in ascending-t order
    float acc = bv;
    #pragma unroll
    for (int i = 0; i < kK; ++i)
      acc = fmaf(xv[i], wv[i], acc);
    if (apply_silu) acc = conv_silu(acc);
    const __nv_bfloat16 y = __float2bfloat16(acc);
    if (c < 2048) {
      q16[(size_t)s * 2048 + c] = y;
    } else if (c < 4096) {
      k16[(size_t)s * 2048 + (c - 2048)] = y;
    } else {
      v48[(size_t)s * 6144 + (c - 4096)] = y;
    }
    #pragma unroll
    for (int i = 0; i < kK - 1; ++i)
      xv[i] = xv[i + 1];
  }
}

}  // namespace

int causal_conv1d_update_steps_gqa_bf16(
    const void* x, const void* w, const void* bias, const void* state,
    void* q16, void* k16, void* v48, int S, bool apply_silu,
    cudaStream_t stream) {
  if (!x || !w || !state || !q16 || !k16 || !v48) return 1;
  if (S <= 0) return 2;
  dim3 grid(kConvDim / kThreads, (S + kSteps - 1) / kSteps);
  conv1d_steps_gqa_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<const __nv_bfloat16*>(w),
      reinterpret_cast<const __nv_bfloat16*>(bias),
      reinterpret_cast<const __nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(q16),
      reinterpret_cast<__nv_bfloat16*>(k16),
      reinterpret_cast<__nv_bfloat16*>(v48), S, apply_silu);
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace kernels
}  // namespace flash_rt
