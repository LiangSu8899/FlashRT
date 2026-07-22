// Cosmos3-Reasoner decode attention: single-query GQA flash-decode.
//
// q [NH, HD] bf16 attends over the bf16 KV cache prefix [len, NKV, HD]
// (len is read from a device int32 so a captured CUDA graph handles growing
// sequences without a fixed-window mask). One block per query head, online
// softmax in fp32, coalesced K/V row reads. NH=16, NKV=8 (group 2), HD=128.

#include "cosmos3_reasoner_attn.cuh"

#include <cuda_bf16.h>
#include <math.h>

namespace flash_rt::kernels {
namespace {

constexpr int HD = 128;
constexpr int TS = 4;   // rows per iteration (one per warp)
constexpr int SPLITS = 20;  // KV splits per head (fills Thor SMs on long KV)

// Phase 1: each block covers one (head, split) KV chunk; emits an
// unnormalized partial accumulator plus its (m, l) softmax state.
__global__ void reasoner_decode_attn_split_kernel(
    const __nv_bfloat16* __restrict__ q,        // [NH, HD]
    const __nv_bfloat16* __restrict__ k_cache,  // [max_seq, NKV, HD]
    const __nv_bfloat16* __restrict__ v_cache,  // [max_seq, NKV, HD]
    const int* __restrict__ len_ptr,            // valid rows
    float* __restrict__ part_acc,               // [NH, SPLITS, HD]
    float* __restrict__ part_ml,                // [NH, SPLITS, 2]
    int nkv,
    float scale)
{
  const int head = blockIdx.x;
  const int split = blockIdx.y;
  const int kv = head / (gridDim.x / nkv);
  const int tid = threadIdx.x;      // 0..127
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int len = *len_ptr;
  const int chunk = (len + SPLITS - 1) / SPLITS;
  const int s_begin = split * chunk;
  const int s_end = min(len, s_begin + chunk);

  __shared__ float sh_q[HD];
  __shared__ float sh_score[TS];

  sh_q[tid] = __bfloat162float(q[head * HD + tid]);
  __syncthreads();

  float m = -INFINITY;
  float l = 0.0f;
  float acc = 0.0f;  // this thread's output dim = tid

  for (int s0 = s_begin; s0 < s_end; s0 += TS) {
    const int s = s0 + warp;
    float score = -INFINITY;
    if (s < s_end) {
      const __nv_bfloat16* krow = k_cache + ((size_t)s * nkv + kv) * HD;
      float d = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        d += sh_q[lane * 4 + j] * __bfloat162float(krow[lane * 4 + j]);
      }
#pragma unroll
      for (int off = 16; off; off >>= 1) {
        d += __shfl_xor_sync(0xffffffffu, d, off);
      }
      score = d * scale;
    }
    if (lane == 0) {
      sh_score[warp] = score;
    }
    __syncthreads();

    float tile_max = sh_score[0];
#pragma unroll
    for (int t = 1; t < TS; ++t) {
      tile_max = fmaxf(tile_max, sh_score[t]);
    }
    const float m_new = fmaxf(m, tile_max);
    const float corr = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
    float p[TS];
    float p_sum = 0.0f;
#pragma unroll
    for (int t = 0; t < TS; ++t) {
      p[t] = (sh_score[t] == -INFINITY) ? 0.0f : __expf(sh_score[t] - m_new);
      p_sum += p[t];
    }
    acc *= corr;
#pragma unroll
    for (int t = 0; t < TS; ++t) {
      const int s_t = s0 + t;
      if (p[t] != 0.0f) {
        acc += p[t] * __bfloat162float(v_cache[((size_t)s_t * nkv + kv) * HD + tid]);
      }
    }
    l = l * corr + p_sum;
    m = m_new;
    __syncthreads();
  }
  part_acc[((size_t)head * SPLITS + split) * HD + tid] = acc;
  if (tid == 0) {
    part_ml[((size_t)head * SPLITS + split) * 2 + 0] = m;
    part_ml[((size_t)head * SPLITS + split) * 2 + 1] = l;
  }
}

// Phase 2: combine the SPLITS partials per head.
__global__ void reasoner_decode_attn_reduce_kernel(
    const float* __restrict__ part_acc,  // [NH, SPLITS, HD]
    const float* __restrict__ part_ml,   // [NH, SPLITS, 2]
    __nv_bfloat16* __restrict__ out)     // [NH, HD]
{
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float sh_m[SPLITS];
  __shared__ float sh_l[SPLITS];
  if (tid < SPLITS) {
    sh_m[tid] = part_ml[((size_t)head * SPLITS + tid) * 2 + 0];
    sh_l[tid] = part_ml[((size_t)head * SPLITS + tid) * 2 + 1];
  }
  __syncthreads();
  float m_star = -INFINITY;
#pragma unroll
  for (int i = 0; i < SPLITS; ++i) {
    m_star = fmaxf(m_star, sh_m[i]);
  }
  float l_star = 0.0f;
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < SPLITS; ++i) {
    const float w = (sh_m[i] == -INFINITY) ? 0.0f : __expf(sh_m[i] - m_star);
    l_star += sh_l[i] * w;
    acc += part_acc[((size_t)head * SPLITS + i) * HD + tid] * w;
  }
  out[head * HD + tid] = __float2bfloat16(l_star > 0.0f ? acc / l_star : 0.0f);
}

}  // namespace

void cosmos3_reasoner_decode_attn_bf16(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const int* len_ptr,
    __nv_bfloat16* out,
    float* part_acc,
    float* part_ml,
    int num_heads,
    int num_kv_heads,
    float scale,
    cudaStream_t stream)
{
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr || len_ptr == nullptr ||
      out == nullptr || part_acc == nullptr || part_ml == nullptr) {
    return;
  }
  dim3 grid(num_heads, SPLITS);
  reasoner_decode_attn_split_kernel<<<grid, HD, 0, stream>>>(
      q, k_cache, v_cache, len_ptr, part_acc, part_ml, num_kv_heads, scale);
  reasoner_decode_attn_reduce_kernel<<<num_heads, HD, 0, stream>>>(part_acc, part_ml, out);
}

}  // namespace flash_rt::kernels
