// SPDX-License-Identifier: Apache-2.0
//
// See gqa_decode_attention_sm80.cuh. SM80-family facilities only.
//
// A block owns one query head and its warps divide the cache between them:
// warp w takes positions w, w + warps, w + 2 * warps, and keeps its own
// running maximum, weight and accumulator. Each warp finishes with a partial
// softmax over its own positions, and the block combines those at the end --
// the same rescaling an online softmax does within a warp, applied once
// across them.
//
// A lane holds the accumulator for the head dimensions congruent to it modulo
// the warp width, which is also how it reads the cache, so consecutive lanes
// read consecutive values.

#include "kernels/gqa_decode_attention_sm80.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {
namespace {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;

// Values per lane, which is head_dim / 32. A template parameter rather than an
// argument because the accumulator is indexed by it: as a runtime bound the
// array lands in local memory, and the accumulator is read and written once
// per cached position.
template <int SLOTS>
__global__ void gqa_decode_attention_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ out,
    int seq_len, const int* __restrict__ seq_len_device,
    int q_heads, int kv_heads, int head_dim, int group, float scale,
    int q_rows) {
  extern __shared__ float shared[];
  float* query = shared;                        // head_dim
  float* partial = query + head_dim;            // kWarps * head_dim
  float* warp_max = partial + kWarps * head_dim;   // kWarps
  float* warp_weight = warp_max + kWarps;          // kWarps

  const int head = blockIdx.x;
  const int row = blockIdx.y;
  const int key_head = head / group;
  // Row r is at cache position seq_len - q_rows + r, and attends to
  // everything up to and including itself.
  const int length =
      (seq_len_device ? *seq_len_device : seq_len) - q_rows + row + 1;

  const __nv_bfloat16* q_row =
      q + (static_cast<size_t>(row) * q_heads + head) * head_dim;
  for (int i = threadIdx.x; i < head_dim; i += kThreads) {
    query[i] = __bfloat162float(q_row[i]);
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  float accumulator[SLOTS];
#pragma unroll
  for (int j = 0; j < SLOTS; ++j) accumulator[j] = 0.0f;
  float running_max = -INFINITY;
  float running_weight = 0.0f;

  const size_t head_offset = static_cast<size_t>(key_head) * head_dim;
  const size_t stride = static_cast<size_t>(kv_heads) * head_dim;
  for (int position = warp; position < length; position += kWarps) {
    const __nv_bfloat16* key =
        k_cache + static_cast<size_t>(position) * stride + head_offset;
    float score = 0.0f;
#pragma unroll
    for (int j = 0; j < SLOTS; ++j) {
      const int index = lane + (j << 5);
      score = fmaf(query[index], __bfloat162float(key[index]), score);
    }
    for (int offset = 16; offset; offset >>= 1) {
      score += __shfl_xor_sync(0xffffffffu, score, offset);
    }
    score *= scale;

    const float next_max = fmaxf(running_max, score);
    const float rescale = __expf(running_max - next_max);
    const float weight = __expf(score - next_max);
    running_max = next_max;
    running_weight = running_weight * rescale + weight;

    const __nv_bfloat16* value =
        v_cache + static_cast<size_t>(position) * stride + head_offset;
#pragma unroll
    for (int j = 0; j < SLOTS; ++j) {
      const int index = lane + (j << 5);
      accumulator[j] =
          fmaf(weight, __bfloat162float(value[index]), accumulator[j] * rescale);
    }
  }

#pragma unroll
  for (int j = 0; j < SLOTS; ++j) {
    partial[warp * head_dim + lane + (j << 5)] = accumulator[j];
  }
  if (!lane) {
    warp_max[warp] = running_max;
    warp_weight[warp] = running_weight;
  }
  __syncthreads();

  // One rescaling across warps, then the division the softmax was deferring.
  float total_max = -INFINITY;
  for (int w = 0; w < kWarps; ++w) total_max = fmaxf(total_max, warp_max[w]);
  float total_weight = 0.0f;
  for (int w = 0; w < kWarps; ++w) {
    total_weight += warp_weight[w] * __expf(warp_max[w] - total_max);
  }
  const float inverse = 1.0f / total_weight;

  const size_t row_offset =
      (static_cast<size_t>(row) * q_heads + head) * head_dim;
  __nv_bfloat16* out_row = out + row_offset;
  const __nv_bfloat16* gate_row = gate ? gate + row_offset : nullptr;
  for (int i = threadIdx.x; i < head_dim; i += kThreads) {
    float value = 0.0f;
    for (int w = 0; w < kWarps; ++w) {
      value = fmaf(partial[w * head_dim + i],
                   __expf(warp_max[w] - total_max), value);
    }
    value *= inverse;
    if (gate_row) {
      const float g = __bfloat162float(gate_row[i]);
      value *= 1.0f / (1.0f + __expf(-g));
    }
    out_row[i] = __float2bfloat16(value);
  }
}

}  // namespace

int gqa_decode_attention_bf16(
    const void* q,
    const void* k_cache, const void* v_cache,
    const void* gate,
    void* out,
    int seq_len, const int* seq_len_device,
    int q_heads, int kv_heads, int head_dim,
    float scale, int q_rows,
    cudaStream_t stream) {
  if (q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads) return -1;
  if (head_dim <= 0 || (head_dim & 31)) return -1;
  if (q_rows <= 0) return -1;
  if (!seq_len_device && seq_len < q_rows) return -1;

  const size_t shared =
      (static_cast<size_t>(head_dim) * (kWarps + 1) + 2 * kWarps) *
      sizeof(float);
  const dim3 grid(q_heads, q_rows);
  const auto launch = [&](auto kernel) {
    kernel<<<grid, kThreads, shared, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(k_cache),
        reinterpret_cast<const __nv_bfloat16*>(v_cache),
        reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<__nv_bfloat16*>(out),
        seq_len, seq_len_device, q_heads, kv_heads, head_dim,
        q_heads / kv_heads, scale, q_rows);
  };
  switch (head_dim >> 5) {
    case 2:  launch(gqa_decode_attention_kernel<2>);  break;
    case 4:  launch(gqa_decode_attention_kernel<4>);  break;
    case 8:  launch(gqa_decode_attention_kernel<8>);  break;
    case 16: launch(gqa_decode_attention_kernel<16>); break;
    case 32: launch(gqa_decode_attention_kernel<32>); break;
    default: return -1;
  }
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
