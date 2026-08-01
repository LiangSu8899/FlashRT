// SPDX-License-Identifier: Apache-2.0
//
// See bf16_matvec_narrow_sm80.cuh. SM80-family facilities only.
//
// A block owns one output row and its threads divide the contraction. The row
// is read as eight values at a time so the loads are the widest the type
// allows, and the partial sums are reduced first within a warp by shuffles
// and then across warps through a few words of shared memory.

#include "kernels/bf16_matvec_narrow_sm80.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kChunk = 8;               // bf16 values per 16-byte load

__global__ void matvec_narrow_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int K) {
  const int row = blockIdx.x;
  const int chunks = K / kChunk;
  const auto* row_base =
      reinterpret_cast<const int4*>(weight + static_cast<size_t>(row) * K);
  const auto* x_base = reinterpret_cast<const int4*>(x);

  float sum = 0.0f;
  for (int chunk = threadIdx.x; chunk < chunks; chunk += kThreads) {
    const int4 w = row_base[chunk];
    const int4 v = x_base[chunk];
    const auto* wv = reinterpret_cast<const __nv_bfloat16*>(&w);
    const auto* xv = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
    for (int i = 0; i < kChunk; ++i) {
      sum = fmaf(__bfloat162float(wv[i]), __bfloat162float(xv[i]), sum);
    }
  }

  for (int offset = 16; offset; offset >>= 1) {
    sum += __shfl_xor_sync(0xffffffffu, sum, offset);
  }
  __shared__ float partial[kWarps];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (!lane) partial[warp] = sum;
  __syncthreads();
  if (threadIdx.x) return;
  float total = 0.0f;
#pragma unroll
  for (int i = 0; i < kWarps; ++i) total += partial[i];
  out[row] = __float2bfloat16(total);
}

}  // namespace

int bf16_matvec_narrow_bf16(
    const void* x, const void* weight, void* out,
    int N, int K, cudaStream_t stream) {
  if (N <= 0) return 0;
  if (K <= 0 || (K % kChunk)) return -1;

  matvec_narrow_kernel<<<N, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<const __nv_bfloat16*>(weight),
      reinterpret_cast<__nv_bfloat16*>(out), K);
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
