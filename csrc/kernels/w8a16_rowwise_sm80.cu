// SPDX-License-Identifier: Apache-2.0
//
// Weight-only INT8 with a per-output-channel scale. SM80-family facilities
// only.
//
// This serves the output projection of a model whose embedding table is tied
// to it, which for a small model is a large share of what a token reads: at
// 4B with a 248k vocabulary the table is 1.18 GiB of a 3.05 GiB step, so
// halving it is worth more than anything left in the backbone.
//
// The row is read sixteen values at a time and the scale applies once, at the
// end, to the whole dot product -- it is a property of the output channel, not
// of any part of the sum.

#include "kernels/w8a16_rowwise_sm80.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kUnroll = 4;              // sixteen-byte loads in flight
constexpr int kPad = 2;                 // odd word stride across lanes
constexpr int kChunk = 16;              // int8 per vector load

__device__ __forceinline__ float dot16(
    const int4 packed, const __nv_bfloat16* __restrict__ x) {
  const int8_t* values = reinterpret_cast<const int8_t*>(&packed);
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < kChunk; ++j) {
    acc = fmaf(static_cast<float>(values[j]), __bfloat162float(x[j]), acc);
  }
  return acc;
}

__global__ void rowwise_matvec_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int8_t* __restrict__ weight,
    const __half* __restrict__ scale,
    __nv_bfloat16* __restrict__ out,
    int N, int K) {
  extern __shared__ __nv_bfloat16 x_sh[];
  constexpr int kStride = kChunk + kPad;
  const int chunks = K / kChunk;

  for (int index = threadIdx.x; index < K; index += kThreads) {
    x_sh[(index / kChunk) * kStride + (index % kChunk)] = x[index];
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (row >= N) return;
  const int4* row_weight =
      reinterpret_cast<const int4*>(weight + static_cast<size_t>(row) * K);

  float acc = 0.0f;
  int chunk = lane;
  const int step = 32 * kUnroll;
  for (; chunk + 32 * (kUnroll - 1) < chunks; chunk += step) {
    int4 staged[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      staged[u] = row_weight[chunk + 32 * u];
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      acc += dot16(staged[u], x_sh + (chunk + 32 * u) * kStride);
    }
  }
  for (; chunk < chunks; chunk += 32) {
    acc += dot16(row_weight[chunk], x_sh + chunk * kStride);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_xor_sync(0xffffffffu, acc, offset);
  }
  if (lane == 0) {
    out[row] = __float2bfloat16_rn(acc * __half2float(scale[row]));
  }
}

__global__ void rowwise_gather_kernel(
    const int64_t* __restrict__ ids,
    const int8_t* __restrict__ weight,
    const __half* __restrict__ scale,
    __nv_bfloat16* __restrict__ out,
    int rows, int table_rows, int K) {
  const int row = blockIdx.y;
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || column >= K) return;
  const int64_t index = ids[row];
  const size_t destination = static_cast<size_t>(row) * K + column;
  if (index < 0 || index >= table_rows) {
    // A sampled id and a table are two different things to be wrong about;
    // reading past the table would make the second look like the first.
    out[destination] = __float2bfloat16(0.0f);
    return;
  }
  const size_t source = static_cast<size_t>(index) * K + column;
  out[destination] = __float2bfloat16_rn(
      static_cast<float>(weight[source]) * __half2float(scale[index]));
}

}  // namespace

int w8a16_rowwise_matvec_bf16(
    const void* x, const void* weight, const void* scale, void* out,
    int N, int K, cudaStream_t stream) {
  if (!x || !weight || !scale || !out) return 1;
  if (N <= 0 || K <= 0) return 2;
  if (K % kChunk) return 3;
  const size_t shared =
      static_cast<size_t>(K / kChunk) * (kChunk + kPad)
      * sizeof(__nv_bfloat16);
  rowwise_matvec_kernel<<<(N + kWarps - 1) / kWarps, kThreads, shared,
                          stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<const int8_t*>(weight),
      reinterpret_cast<const __half*>(scale),
      reinterpret_cast<__nv_bfloat16*>(out), N, K);
  return 0;
}

int int8_rowwise_gather_bf16(
    const void* ids, const void* weight, const void* scale, void* out,
    int rows, int table_rows, int K, cudaStream_t stream) {
  if (!ids || !weight || !scale || !out) return 1;
  if (rows <= 0 || table_rows <= 0 || K <= 0) return 2;
  const dim3 block(256);
  const dim3 grid((K + block.x - 1) / block.x, rows);
  rowwise_gather_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const int64_t*>(ids),
      reinterpret_cast<const int8_t*>(weight),
      reinterpret_cast<const __half*>(scale),
      reinterpret_cast<__nv_bfloat16*>(out), rows, table_rows, K);
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
