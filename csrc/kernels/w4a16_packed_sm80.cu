// SPDX-License-Identifier: Apache-2.0
//
// Weight-only 4-bit GEMV/GEMM over an int32-packed, group-scaled layout.
// SM80-family facilities only.
//
// At batch one this is bound by reading the weight, so the shape of the work
// is decided by two things and neither is arithmetic:
//
// A warp owns several rows. One row per warp leaves a single vector load in
// flight, and at K=2560 with groups of 32 the group loop runs twice per lane,
// so nothing pipelines. R rows give R loads outstanding, and R is picked per
// shape so the product lands near eight.
//
// The staged activation is padded. A group of 32 bf16 is 16 banks, so lanes
// two apart would collide on every group. Two elements of pad make the
// per-group stride odd in words, which is conflict-free across the warp for
// every group size here.

#include "kernels/w4a16_packed_sm80.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace flash_rt {
namespace kernels {
namespace {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kValuesPerWord = 8;
constexpr int kPad = 2;                // keeps the per-group word stride odd

// One word: eight values against eight staged activations.
//
// The nibbles are not extracted one at a time. Masking the word four ways puts
// value p and value p+4 in the two halves of one register, and OR-ing the bf16
// pattern for 128 turns each half into the number 128 + n outright -- bf16 has
// a seven-bit mantissa, so a four-bit value lands in it exactly. Twelve
// instructions produce eight values where extracting them separately took
// something over thirty, and on a part with few SMs that difference is the
// whole bandwidth: the same kernel structure over int8, whose decode is one
// multiply, reaches 96 GB/s where this reached 40.
//
// The 128 is not subtracted here, and neither is the format's own offset of
// eight. Both are constant across a group, so
//
//     sum (n_i - 8) x_i  =  sum (128 + n_i) x_i  -  136 sum x_i
//
// and the right-hand sum is over the activation alone: computed once while
// staging and shared by every row. What is left in this loop is a conversion
// and a multiply-add.
//
// The accumulation stays in fp32 deliberately. Doing it in bf16 pairs would
// halve the instruction count again, but the terms here are all near 128 and
// the result is their difference from 136 sum x -- eight mantissa bits do not
// survive that cancellation.
__device__ __forceinline__ float word_dot(
    uint32_t word, const __nv_bfloat162* __restrict__ pairs) {
  float dot = 0.0f;
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    // Low half is value p, high half is value p + 4.
    const uint32_t bits = ((word >> (4 * p)) & 0x000F000Fu) | 0x43004300u;
    __nv_bfloat162 decoded;
    memcpy(&decoded, &bits, sizeof(decoded));
    const float2 value = __bfloat1622float2(decoded);
    const float2 activation = __bfloat1622float2(pairs[p]);
    dot = fmaf(value.x, activation.x, dot);
    dot = fmaf(value.y, activation.y, dot);
  }
  return dot;
}

// What the folded offsets cost the group: 136 for the 128 the pattern carries
// and the 8 the format does.
constexpr float kFoldedOffset = 136.0f;

template <int G>
__device__ __forceinline__ float group_dot(
    const uint32_t* __restrict__ words,
    const __nv_bfloat16* __restrict__ activation, float activation_sum) {
  const __nv_bfloat162* pairs =
      reinterpret_cast<const __nv_bfloat162*>(activation);
  float dot = 0.0f;
#pragma unroll
  for (int w = 0; w < G / kValuesPerWord; ++w) {
    dot += word_dot(words[w], pairs + w * 4);
  }
  return dot - kFoldedOffset * activation_sum;
}

// Staging writes value i of a word to slot (i % 4) * 2 + i / 4, so the pair a
// mask produces -- value p beside value p + 4 -- is one aligned read.
__device__ __forceinline__ int staged_slot(int index) {
  const int word = index / kValuesPerWord;
  const int within = index % kValuesPerWord;
  return word * kValuesPerWord + (within % 4) * 2 + within / 4;
}

template <int R, int G>
__global__ void packed_matvec_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint32_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ out,
    int N, int K) {
  extern __shared__ __nv_bfloat16 x_sh[];
  constexpr int kStride = G + kPad;
  constexpr int kWords = G / kValuesPerWord;
  const int groups = K / G;
  // The folded offsets need one sum per group, computed here and read by
  // every row -- the whole point of folding them out of the row loop.
  float* group_sum = reinterpret_cast<float*>(x_sh + groups * kStride);

  for (int index = threadIdx.x; index < K; index += kThreads) {
    x_sh[(index / G) * kStride + staged_slot(index % G)] = x[index];
  }
  __syncthreads();
  for (int g = threadIdx.x; g < groups; g += kThreads) {
    float total = 0.0f;
    const __nv_bfloat16* values = x_sh + g * kStride;
#pragma unroll 8
    for (int i = 0; i < G; ++i) total += __bfloat162float(values[i]);
    group_sum[g] = total;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row_base = (blockIdx.x * kWarps + warp) * R;
  if (row_base >= N) return;

  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.0f;

  for (int group = lane; group < groups; group += 32) {
    const __nv_bfloat16* activation = x_sh + group * kStride;
    uint32_t words[R][kWords];
    __nv_bfloat16 scales[R];
    // Every row's load is issued before any is consumed, so R reads are
    // outstanding rather than one.
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = row_base + r;
      const bool live = row < N;
      const size_t base =
          (static_cast<size_t>(row) * groups + group) * kWords;
#pragma unroll
      for (int w = 0; w < kWords; ++w) {
        words[r][w] = live ? packed[base + w] : 0u;
      }
      scales[r] = live
          ? scale[static_cast<size_t>(row) * groups + group]
          : __float2bfloat16(0.0f);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      acc[r] = fmaf(group_dot<G>(words[r], activation, group_sum[group]),
                    __bfloat162float(scales[r]), acc[r]);
    }
  }

#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = row_base + r;
      if (row < N) out[row] = __float2bfloat16_rn(acc[r]);
    }
  }
}

// The batched form. One block per (row tile, activation row): the weight read
// dominates and blockIdx.y walks the activations.
template <int G>
__global__ void packed_gemm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint32_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ out,
    int M, int N, int K) {
  extern __shared__ __nv_bfloat16 x_sh[];
  constexpr int kStride = G + kPad;
  constexpr int kWords = G / kValuesPerWord;
  const int groups = K / G;
  const int row_of_x = blockIdx.y;
  // The folded offsets need one sum per group, computed here and read by
  // every row -- the whole point of folding them out of the row loop.
  float* group_sum = reinterpret_cast<float*>(x_sh + groups * kStride);

  for (int index = threadIdx.x; index < K; index += kThreads) {
    x_sh[(index / G) * kStride + staged_slot(index % G)] = x[static_cast<size_t>(row_of_x) * K + index];
  }
  __syncthreads();
  for (int g = threadIdx.x; g < groups; g += kThreads) {
    float total = 0.0f;
    const __nv_bfloat16* values = x_sh + g * kStride;
#pragma unroll 8
    for (int i = 0; i < G; ++i) total += __bfloat162float(values[i]);
    group_sum[g] = total;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (row >= N) return;

  float acc = 0.0f;
  for (int group = lane; group < groups; group += 32) {
    acc = fmaf(
        group_dot<G>(
            packed + (static_cast<size_t>(row) * groups + group) * kWords,
            x_sh + group * kStride, group_sum[group]),
        __bfloat162float(scale[static_cast<size_t>(row) * groups + group]),
        acc);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_xor_sync(0xffffffffu, acc, offset);
  }
  if (lane == 0) {
    out[static_cast<size_t>(row_of_x) * N + row] = __float2bfloat16_rn(acc);
  }
}

int validate(const void* x, const void* packed, const void* scale,
             const void* out, int N, int K, int group) {
  if (!x || !packed || !scale || !out) return 1;
  if (N <= 0 || K <= 0) return 2;
  if (group != 32 && group != 64 && group != 128) return 3;
  if (K % group) return 4;
  return 0;
}

size_t shared_bytes(int K, int group) {
  const size_t groups = static_cast<size_t>(K / group);
  return groups * (group + kPad) * sizeof(__nv_bfloat16)
      + groups * sizeof(float);
}

// Rows per warp so that rows x (groups / 32) lands near eight loads in
// flight: a shape that already pipelines eight deep wants one row.
int rows_per_warp(int K, int group) {
  const int steps = (K / group) / 32;
  return steps >= 8 ? 1 : (steps >= 4 ? 2 : (steps >= 2 ? 4 : 8));
}

}  // namespace

int w4a16_packed_matvec_bf16(
    const void* x, const void* packed, const void* scale, void* out,
    int N, int K, int group, cudaStream_t stream) {
  const int bad = validate(x, packed, scale, out, N, K, group);
  if (bad) return bad;

  const auto* x_ptr = reinterpret_cast<const __nv_bfloat16*>(x);
  const auto* packed_ptr = reinterpret_cast<const uint32_t*>(packed);
  const auto* scale_ptr = reinterpret_cast<const __nv_bfloat16*>(scale);
  auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out);
  const size_t shared = shared_bytes(K, group);
  const int rows = rows_per_warp(K, group);

#define FLASHRT_DISPATCH(ROWS, GROUP)                                         \
  packed_matvec_kernel<ROWS, GROUP>                                           \
      <<<(N + kWarps * (ROWS) - 1) / (kWarps * (ROWS)), kThreads, shared,     \
         stream>>>(x_ptr, packed_ptr, scale_ptr, out_ptr, N, K)
#define FLASHRT_DISPATCH_ROWS(GROUP)                                          \
  switch (rows) {                                                             \
    case 1: FLASHRT_DISPATCH(1, GROUP); break;                                \
    case 2: FLASHRT_DISPATCH(2, GROUP); break;                                \
    case 4: FLASHRT_DISPATCH(4, GROUP); break;                                \
    default: FLASHRT_DISPATCH(8, GROUP); break;                               \
  }

  switch (group) {
    case 32: FLASHRT_DISPATCH_ROWS(32); break;
    case 64: FLASHRT_DISPATCH_ROWS(64); break;
    default: FLASHRT_DISPATCH_ROWS(128); break;
  }
#undef FLASHRT_DISPATCH_ROWS
#undef FLASHRT_DISPATCH
  return 0;
}

int w4a16_packed_gemm_bf16(
    const void* x, const void* packed, const void* scale, void* out,
    int M, int N, int K, int group, cudaStream_t stream) {
  const int bad = validate(x, packed, scale, out, N, K, group);
  if (bad) return bad;
  if (M <= 0) return 5;
  if (M > 65535) return 6;
  if (M == 1) {
    return w4a16_packed_matvec_bf16(x, packed, scale, out, N, K, group,
                                    stream);
  }

  const dim3 grid((N + kWarps - 1) / kWarps, M);
  const auto* x_ptr = reinterpret_cast<const __nv_bfloat16*>(x);
  const auto* packed_ptr = reinterpret_cast<const uint32_t*>(packed);
  const auto* scale_ptr = reinterpret_cast<const __nv_bfloat16*>(scale);
  auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out);
  const size_t shared = shared_bytes(K, group);

  switch (group) {
    case 32:
      packed_gemm_kernel<32><<<grid, kThreads, shared, stream>>>(
          x_ptr, packed_ptr, scale_ptr, out_ptr, M, N, K);
      break;
    case 64:
      packed_gemm_kernel<64><<<grid, kThreads, shared, stream>>>(
          x_ptr, packed_ptr, scale_ptr, out_ptr, M, N, K);
      break;
    default:
      packed_gemm_kernel<128><<<grid, kThreads, shared, stream>>>(
          x_ptr, packed_ptr, scale_ptr, out_ptr, M, N, K);
      break;
  }
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
