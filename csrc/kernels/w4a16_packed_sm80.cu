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
#include <cstdlib>
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
    uint32_t word, const float* __restrict__ activation) {
  float dot = 0.0f;
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    // Low half is value p, high half is value p + 4.
    const uint32_t bits = ((word >> (4 * p)) & 0x000F000Fu) | 0x43004300u;
    __nv_bfloat162 decoded;
    memcpy(&decoded, &bits, sizeof(decoded));
    const float2 value = __bfloat1622float2(decoded);
    dot = fmaf(value.x, activation[2 * p], dot);
    dot = fmaf(value.y, activation[2 * p + 1], dot);
  }
  return dot;
}

// One word of weight, decoded once and left in registers.
//
// The batched form contracts a decoded word against every row of an
// activation tile, so the decode is hoisted out of that loop the same way the
// activation's conversion is hoisted out of the row loop above.
__device__ __forceinline__ void decode_word(uint32_t word, float* out) {
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const uint32_t bits = ((word >> (4 * p)) & 0x000F000Fu) | 0x43004300u;
    __nv_bfloat162 decoded;
    memcpy(&decoded, &bits, sizeof(decoded));
    const float2 value = __bfloat1622float2(decoded);
    out[2 * p] = value.x;
    out[2 * p + 1] = value.y;
  }
}

// One word of activation, converted once and left in registers.
//
// The rows a warp owns all contract against the same activation, so
// converting it inside the row loop converts it once per row: at four rows
// paired with their gate partners, the same eight values were converted eight
// times. Hoisting it here costs eight registers and removes the rest.
__device__ __forceinline__ void stage_word(
    const __nv_bfloat16* __restrict__ activation, int word, float* out) {
  const __nv_bfloat162* pairs =
      reinterpret_cast<const __nv_bfloat162*>(activation) + word * 4;
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const float2 value = __bfloat1622float2(pairs[p]);
    out[2 * p] = value.x;
    out[2 * p + 1] = value.y;
  }
}

// What the folded offsets cost the group: 136 for the 128 the pattern carries
// and the 8 the format does.
constexpr float kFoldedOffset = 136.0f;

template <int G>
__device__ __forceinline__ float group_dot(
    const uint32_t* __restrict__ words,
    const __nv_bfloat16* __restrict__ activation, float activation_sum) {
  float dot = 0.0f;
#pragma unroll
  for (int w = 0; w < G / kValuesPerWord; ++w) {
    float staged[kValuesPerWord];
    stage_word(activation, w, staged);
    dot += word_dot(words[w], staged);
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

// silu(g) * u, in fp32 where the accumulation already is.
__device__ __forceinline__ float gated(float g, float u) {
  return (g / (1.0f + __expf(-g))) * u;
}

template <int R, int G, bool kGated>
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
  // When gated, a warp owns a row of the first half and its partner in the
  // second, so the rows it covers are half the output it produces.
  const int rows_out = kGated ? N / 2 : N;
  const int row_base = (blockIdx.x * kWarps + warp) * R;
  if (row_base >= rows_out) return;
  const int partner = kGated ? N / 2 : 0;

  float acc[R];
  float acc_up[kGated ? R : 1];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.0f;
  if (kGated) {
#pragma unroll
    for (int r = 0; r < R; ++r) acc_up[r] = 0.0f;
  }

  for (int group = lane; group < groups; group += 32) {
    const __nv_bfloat16* activation = x_sh + group * kStride;
    uint32_t words[R][kWords];
    __nv_bfloat16 scales[R];
    // Every row's load is issued before any is consumed, so R reads are
    // outstanding rather than one.
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = row_base + r;
      const bool live = row < rows_out;
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
    // The partner rows a gate needs, read the same way. Loading them here
    // rather than in a second pass lets both streams contract against one
    // conversion of the activation.
    uint32_t up_words[kGated ? R : 1][kWords];
    __nv_bfloat16 up_scales[kGated ? R : 1];
    if (kGated) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int row = row_base + r + partner;
        const bool live = row_base + r < rows_out;
        const size_t base =
            (static_cast<size_t>(row) * groups + group) * kWords;
#pragma unroll
        for (int w = 0; w < kWords; ++w) {
          up_words[r][w] = live ? packed[base + w] : 0u;
        }
        up_scales[r] = live
            ? scale[static_cast<size_t>(row) * groups + group]
            : __float2bfloat16(0.0f);
      }
    }

    // One conversion of the activation, every row of the tile against it.
    float part[R];
    float part_up[kGated ? R : 1];
#pragma unroll
    for (int r = 0; r < R; ++r) part[r] = 0.0f;
    if (kGated) {
#pragma unroll
      for (int r = 0; r < R; ++r) part_up[r] = 0.0f;
    }
#pragma unroll
    for (int w = 0; w < kWords; ++w) {
      float staged[kValuesPerWord];
      stage_word(activation, w, staged);
#pragma unroll
      for (int r = 0; r < R; ++r) part[r] += word_dot(words[r][w], staged);
      if (kGated) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
          part_up[r] += word_dot(up_words[r][w], staged);
        }
      }
    }

    const float folded = kFoldedOffset * group_sum[group];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      acc[r] = fmaf(part[r] - folded, __bfloat162float(scales[r]), acc[r]);
      if (kGated) {
        acc_up[r] = fmaf(part_up[r] - folded,
                         __bfloat162float(up_scales[r]), acc_up[r]);
      }
    }
  }

#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], offset);
      if (kGated) acc_up[r] += __shfl_xor_sync(0xffffffffu, acc_up[r], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = row_base + r;
      if (row >= rows_out) continue;
      out[row] = __float2bfloat16_rn(
          kGated ? gated(acc[r], acc_up[r]) : acc[r]);
    }
  }
}

// The batched form.
//
// A weight is read once per activation row when a block owns one row, which is
// what this used to do: a prompt of two hundred and fifty-six positions then
// cost two hundred and fifty-six reads of the model, and time to the first
// token was linear in the prompt with the slope of a decode step. Batching
// bought nothing.
//
// So a block owns a tile of activation rows and the contraction is walked in
// chunks: the chunk of every row in the tile is staged in shared memory, and
// the weight for that chunk is read once and used against all of them. The
// weight is then read once per tile rather than once per row, and the decode
// -- which is a fifth of the kernel's time, measured by removing it -- is paid
// once per tile as well.
//
// The chunk is thirty-two groups so that every lane of a warp has a group to
// take, and the tile is as many rows as that leaves shared memory for.
template <int TM, int G>
__global__ void packed_gemm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint32_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ out,
    int M, int N, int K) {
  constexpr int kWords = G / kValuesPerWord;
  constexpr int kChunkGroups = 32;            // one group per lane
  constexpr int kStride = G + 1;              // odd, so lanes miss each other
  const int groups = K / G;

  // Staged as float, not bfloat16. In the single-row form the conversion is
  // paid once and bfloat16 halves the shared memory; here every row of the
  // tile contracts against the same values, so bfloat16 would pay the
  // conversion once per row -- sixteen times the decode it was hoisted to
  // save. Measured that way round it was twice as slow.
  extern __shared__ float tile_sh[];             // TM * kChunkGroups * kStride
  float* group_sum =                          // TM * kChunkGroups
      tile_sh + TM * kChunkGroups * kStride;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int base_row = blockIdx.y * TM;
  const int rows_here = min(TM, M - base_row);
  const int row = blockIdx.x * kWarps + warp;

  float acc[TM];
#pragma unroll
  for (int m = 0; m < TM; ++m) acc[m] = 0.0f;

  for (int first = 0; first < groups; first += kChunkGroups) {
    const int here = min(kChunkGroups, groups - first);

    // Stage this chunk of every row in the tile. Read in order, written to
    // the slots the decode will want them in.
    // Every row of the tile is staged, the last one repeated where the tile
    // runs past the end. The loops that consume it are then bounded at
    // compile time -- with a runtime bound the accumulators are indexed by a
    // variable and land in local memory, and every multiply-add becomes a
    // load and a store. Measured that way the batched form was slower than
    // doing the rows one at a time.
    for (int index = threadIdx.x; index < TM * here * G; index += kThreads) {
      const int within = index % G;
      const int group = (index / G) % here;
      const int m = index / (G * here);
      const int source = min(base_row + m, M - 1);
      tile_sh[(m * kChunkGroups + group) * kStride + within] =
          __bfloat162float(
              x[(static_cast<size_t>(source) * K) + (first + group) * G
                + within]);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < TM * here; index += kThreads) {
      const int group = index % here;
      const int m = index / here;
      const float* values = tile_sh + (m * kChunkGroups + group) * kStride;
      float total = 0.0f;
#pragma unroll 8
      for (int i = 0; i < G; ++i) total += values[i];
      group_sum[m * kChunkGroups + group] = total;
    }
    __syncthreads();

    if (row < N && lane < here) {
      const int group = first + lane;
      uint32_t words[kWords];
#pragma unroll
      for (int w = 0; w < kWords; ++w) {
        words[w] = packed[(static_cast<size_t>(row) * groups + group) * kWords
                          + w];
      }
      const float step = __bfloat162float(
          scale[static_cast<size_t>(row) * groups + group]);
      // One decode of the weight, every row of the tile against it.
      float partial[TM];
#pragma unroll
      for (int m = 0; m < TM; ++m) partial[m] = 0.0f;
#pragma unroll
      for (int w = 0; w < kWords; ++w) {
        float decoded[kValuesPerWord];
        decode_word(words[w], decoded);
#pragma unroll
        for (int m = 0; m < TM; ++m) {
          const float* staged =
              tile_sh + (m * kChunkGroups + lane) * kStride + w * kValuesPerWord;
          // Summed within the word and then added, which is the association
          // the single-row form uses. A prompt and the tokens after it go
          // through different kernels, and it is worth their agreeing to the
          // bit rather than to a tolerance.
          float dot = 0.0f;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            dot = fmaf(decoded[2 * i], staged[i], dot);
            dot = fmaf(decoded[2 * i + 1], staged[i + 4], dot);
          }
          partial[m] += dot;
        }
      }
#pragma unroll
      for (int m = 0; m < TM; ++m) {
        acc[m] = fmaf(
            partial[m] - kFoldedOffset * group_sum[m * kChunkGroups + lane],
            step, acc[m]);
      }
    }
    __syncthreads();
  }

  if (row >= N) return;
#pragma unroll
  for (int m = 0; m < TM; ++m) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], offset);
    }
  }
  if (lane) return;
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    if (m < rows_here) {
      out[static_cast<size_t>(base_row + m) * N + row] =
          __float2bfloat16_rn(acc[m]);
    }
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
// Rows per warp, chosen from how many multiprocessors there are and
// overridable so it can be swept on a part without rebuilding.
//
// Every block stages the whole activation before it reads any weight, so that
// work is paid once per block and the redundancy is proportional to how many
// blocks there are -- which is inversely proportional to the rows a warp
// takes. On a part with few multiprocessors the redundancy decides: measured
// on an eight-multiprocessor part, the widest contraction went from 47 to 72
// GB/s between one row and eight, because its activation was being staged by
// three hundred and twenty blocks instead of forty.
//
// On a part with many, the opposite decides: taking more rows leaves too few
// blocks to fill it, and on a hundred-and-seventy-multiprocessor part one row
// measured fastest for most shapes. So this is not a property of the shape,
// which is what it used to be keyed on.
int rows_per_warp(int K, int group, bool gated) {
  static const int forced = [] {
    const char* value = getenv("FLASHRT_W4A16_ROWS");
    return value ? atoi(value) : 0;
  }();
  if (forced == 1 || forced == 2 || forced == 4 || forced == 8) return forced;
  static const int processors = [] {
    int count = 0;
    cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, 0);
    return count;
  }();
  // A gated warp carries two streams of weight against one activation, so it
  // reaches the same amount of work at fewer rows. Two, measured on two
  // eight-multiprocessor parts that disagree about almost everything else:
  // on one it was the best of the four tried and on the other within two per
  // cent of the best. Every other shape wanted eight on both.
  if (processors > 0 && processors <= 24) return gated ? 2 : 8;
  const int steps = (K / group) / 32;
  const int rows = steps >= 8 ? 1 : (steps >= 4 ? 2 : (steps >= 2 ? 4 : 8));
  return gated ? 1 : rows;
}

}  // namespace

namespace {

template <bool kGated>
int launch_matvec(
    const void* x, const void* packed, const void* scale, void* out,
    int N, int K, int group, cudaStream_t stream) {
  const int bad = validate(x, packed, scale, out, N, K, group);
  if (bad) return bad;
  if (kGated && (N & 1)) return 7;
  const int rows_out = kGated ? N / 2 : N;

  const auto* x_ptr = reinterpret_cast<const __nv_bfloat16*>(x);
  const auto* packed_ptr = reinterpret_cast<const uint32_t*>(packed);
  const auto* scale_ptr = reinterpret_cast<const __nv_bfloat16*>(scale);
  auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out);
  const size_t shared = shared_bytes(K, group);
  const int rows = rows_per_warp(K, group, kGated);

#define FLASHRT_DISPATCH(ROWS, GROUP)                                         \
  packed_matvec_kernel<ROWS, GROUP, kGated>                                   \
      <<<(rows_out + kWarps * (ROWS) - 1) / (kWarps * (ROWS)), kThreads,      \
         shared, stream>>>(x_ptr, packed_ptr, scale_ptr, out_ptr, N, K)
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

}  // namespace

int w4a16_packed_matvec_bf16(
    const void* x, const void* packed, const void* scale, void* out,
    int N, int K, int group, cudaStream_t stream) {
  return launch_matvec<false>(x, packed, scale, out, N, K, group, stream);
}

int w4a16_packed_matvec_gated_bf16(
    const void* x, const void* packed, const void* scale, void* out,
    int N, int K, int group, cudaStream_t stream) {
  return launch_matvec<true>(x, packed, scale, out, N, K, group, stream);
}

int w4a16_packed_gemm_bf16(
    const void* x, const void* packed, const void* scale, void* out,
    int M, int N, int K, int group, cudaStream_t stream) {
  const int bad = validate(x, packed, scale, out, N, K, group);
  if (bad) return bad;
  if (M <= 0) return 5;
  if (M == 1) {
    return w4a16_packed_matvec_bf16(x, packed, scale, out, N, K, group,
                                    stream);
  }

  // As many activation rows per block as shared memory holds a chunk of.
  constexpr int kChunkGroups = 32;
  const size_t per_row =
      static_cast<size_t>(kChunkGroups) * (group + 1) * sizeof(float)
      + static_cast<size_t>(kChunkGroups) * sizeof(float);
  const size_t budget = 40u << 10;
  int tile = 16;
  while (tile > 1 && per_row * tile > budget) tile >>= 1;

  const auto* x_ptr = reinterpret_cast<const __nv_bfloat16*>(x);
  const auto* packed_ptr = reinterpret_cast<const uint32_t*>(packed);
  const auto* scale_ptr = reinterpret_cast<const __nv_bfloat16*>(scale);
  auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out);

#define FLASHRT_GEMM_DISPATCH(TILE, GROUP)                                    \
  packed_gemm_kernel<TILE, GROUP>                                             \
      <<<dim3((N + kWarps - 1) / kWarps, (M + (TILE)-1) / (TILE)), kThreads,  \
         per_row * (TILE), stream>>>(x_ptr, packed_ptr, scale_ptr, out_ptr,   \
                                     M, N, K)

#define FLASHRT_GEMM_TILE(GROUP)                                              \
  switch (tile) {                                                             \
    case 1: FLASHRT_GEMM_DISPATCH(1, GROUP); break;                           \
    case 2: FLASHRT_GEMM_DISPATCH(2, GROUP); break;                           \
    case 4: FLASHRT_GEMM_DISPATCH(4, GROUP); break;                           \
    case 8: FLASHRT_GEMM_DISPATCH(8, GROUP); break;                           \
    default: FLASHRT_GEMM_DISPATCH(16, GROUP); break;                         \
  }

  switch (group) {
    case 32: FLASHRT_GEMM_TILE(32); break;
    case 64: FLASHRT_GEMM_TILE(64); break;
    default: FLASHRT_GEMM_TILE(128); break;
  }
#undef FLASHRT_GEMM_TILE
#undef FLASHRT_GEMM_DISPATCH
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
