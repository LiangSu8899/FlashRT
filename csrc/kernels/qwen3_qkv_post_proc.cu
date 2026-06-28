// SPDX-License-Identifier: Apache-2.0
//
// Fused q_norm/k_norm + RoPE + Q_buf/KV cache write.
// See qwen3_qkv_post_proc.cuh for design notes.

#include "qwen3_qkv_post_proc.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int HEAD_DIM = 128;
constexpr int HALF = HEAD_DIM / 2;        // 64
constexpr int THREADS = HEAD_DIM;         // 1 thread per head_dim element
constexpr int N_WARPS = THREADS / 32;     // 4

// Block-wide sum reduction (4 warps × 32 lanes).
//
// First reduces within each warp via __shfl_xor_sync, then aggregates
// across warps via a 4-element smem scratch + final warp shuffle.
__device__ __forceinline__ float block_sum_4warp(float v, float* smem4) {
  // Intra-warp reduction.
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, off);
  }
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  if (lane == 0) smem4[wid] = v;
  __syncthreads();
  // Final warp reduces the 4 partial sums.
  if (wid == 0) {
    float t = (lane < N_WARPS) ? smem4[lane] : 0.f;
    #pragma unroll
    for (int off = 2; off > 0; off >>= 1) {
      t += __shfl_xor_sync(0xffffffff, t, off);
    }
    if (lane == 0) smem4[0] = t;
  }
  __syncthreads();
  return smem4[0];
}

// Q kernel: gridDim.x = n_q_heads, blockDim.x = HEAD_DIM (128).
__global__ void q_norm_rope_qstage_kernel(
    const __nv_bfloat16* __restrict__ q_pre,      // (n_q, 128)
    const __nv_bfloat16* __restrict__ q_norm_w,   // (128,)
    const __nv_bfloat16* __restrict__ cos_v,      // (64,)
    const __nv_bfloat16* __restrict__ sin_v,      // (64,)
    __nv_bfloat16* __restrict__ q_buf,            // (n_q, 128)
    int n_q,
    float eps) {
  int head = blockIdx.x;
  if (head >= n_q) return;
  int tid = threadIdx.x;

  __shared__ float s_normed[HEAD_DIM];
  __shared__ float s_smem4[N_WARPS];

  const __nv_bfloat16* q_row = q_pre + head * HEAD_DIM;
  float v = __bfloat162float(q_row[tid]);
  float w = __bfloat162float(q_norm_w[tid]);

  // Sum-of-squares reduction across the 128 threads.
  float sq = v * v;
  float sum_sq = block_sum_4warp(sq, s_smem4);
  float rstd = rsqrtf(sum_sq / float(HEAD_DIM) + eps);

  // Apply RMSNorm + weight.
  float normed = v * rstd * w;
  s_normed[tid] = normed;
  __syncthreads();

  // Apply RoPE (full rotary; rotary_dim = head_dim).
  // Pair index: tid < half pairs with (tid + half), and rotate_half
  // uses negation on the lo half.
  float partner;
  float c, sn;
  if (tid < HALF) {
    partner = s_normed[tid + HALF];
    c = __bfloat162float(cos_v[tid]);
    sn = __bfloat162float(sin_v[tid]);
    // x_out = normed * cos - partner * sin
    float out = normed * c - partner * sn;
    q_buf[head * HEAD_DIM + tid] = __float2bfloat16(out);
  } else {
    partner = s_normed[tid - HALF];
    int half_idx = tid - HALF;
    c = __bfloat162float(cos_v[half_idx]);
    sn = __bfloat162float(sin_v[half_idx]);
    // x_out = normed * cos + partner * sin
    float out = normed * c + partner * sn;
    q_buf[head * HEAD_DIM + tid] = __float2bfloat16(out);
  }
}

// K kernel: gridDim.x = n_kv_heads, blockDim.x = HEAD_DIM (128).
// Same RoPE path as Q. ALSO writes V[head, tid] to V_cache (V is just
// copied — no norm, no RoPE).
__global__ void k_norm_rope_kvwrite_kernel(
    const __nv_bfloat16* __restrict__ k_pre,      // (n_kv, 128)
    const __nv_bfloat16* __restrict__ v_pre,      // (n_kv, 128)
    const __nv_bfloat16* __restrict__ k_norm_w,   // (128,)
    const __nv_bfloat16* __restrict__ cos_v,      // (64,)
    const __nv_bfloat16* __restrict__ sin_v,      // (64,)
    __nv_bfloat16* __restrict__ k_cache_dst,      // base of (n_kv, 128)
    __nv_bfloat16* __restrict__ v_cache_dst,      // base of (n_kv, 128)
    int n_kv,
    float eps) {
  int head = blockIdx.x;
  if (head >= n_kv) return;
  int tid = threadIdx.x;

  __shared__ float s_normed[HEAD_DIM];
  __shared__ float s_smem4[N_WARPS];

  const __nv_bfloat16* k_row = k_pre + head * HEAD_DIM;
  float v = __bfloat162float(k_row[tid]);
  float w = __bfloat162float(k_norm_w[tid]);

  float sq = v * v;
  float sum_sq = block_sum_4warp(sq, s_smem4);
  float rstd = rsqrtf(sum_sq / float(HEAD_DIM) + eps);

  float normed = v * rstd * w;
  s_normed[tid] = normed;
  __syncthreads();

  // Apply RoPE → write to K_cache slot.
  float partner, c, sn;
  if (tid < HALF) {
    partner = s_normed[tid + HALF];
    c = __bfloat162float(cos_v[tid]);
    sn = __bfloat162float(sin_v[tid]);
    float out = normed * c - partner * sn;
    k_cache_dst[head * HEAD_DIM + tid] = __float2bfloat16(out);
  } else {
    partner = s_normed[tid - HALF];
    int half_idx = tid - HALF;
    c = __bfloat162float(cos_v[half_idx]);
    sn = __bfloat162float(sin_v[half_idx]);
    float out = normed * c + partner * sn;
    k_cache_dst[head * HEAD_DIM + tid] = __float2bfloat16(out);
  }

  // V is just copied (no norm, no RoPE).
  v_cache_dst[head * HEAD_DIM + tid] = v_pre[head * HEAD_DIM + tid];
}

// Device-position variant: writes K/V into K_cache[*cur_pos] / V_cache[*cur_pos]
// where cur_pos is read from device memory, so a single captured graph serves
// every decode position (the host bumps *cur_pos before each replay). row_elems
// = elements between consecutive position slots (n_kv * HEAD_DIM). Same rope/norm
// math as k_norm_rope_kvwrite_kernel.
__global__ void k_norm_rope_kvwrite_devpos_kernel(
    const __nv_bfloat16* __restrict__ k_pre,
    const __nv_bfloat16* __restrict__ v_pre,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const __nv_bfloat16* __restrict__ cos_v,
    const __nv_bfloat16* __restrict__ sin_v,
    __nv_bfloat16* __restrict__ k_cache_base,
    __nv_bfloat16* __restrict__ v_cache_base,
    const int* __restrict__ cur_pos,
    int row_elems,
    int n_kv,
    float eps) {
  int head = blockIdx.x;
  if (head >= n_kv) return;
  int tid = threadIdx.x;
  const size_t slot = (size_t)(*cur_pos) * row_elems;
  __nv_bfloat16* k_cache_dst = k_cache_base + slot;
  __nv_bfloat16* v_cache_dst = v_cache_base + slot;

  __shared__ float s_normed[HEAD_DIM];
  __shared__ float s_smem4[N_WARPS];

  const __nv_bfloat16* k_row = k_pre + head * HEAD_DIM;
  float v = __bfloat162float(k_row[tid]);
  float w = __bfloat162float(k_norm_w[tid]);
  float sum_sq = block_sum_4warp(v * v, s_smem4);
  float rstd = rsqrtf(sum_sq / float(HEAD_DIM) + eps);
  float normed = v * rstd * w;
  s_normed[tid] = normed;
  __syncthreads();

  float partner, c, sn;
  if (tid < HALF) {
    partner = s_normed[tid + HALF];
    c = __bfloat162float(cos_v[tid]);
    sn = __bfloat162float(sin_v[tid]);
    k_cache_dst[head * HEAD_DIM + tid] = __float2bfloat16(normed * c - partner * sn);
  } else {
    partner = s_normed[tid - HALF];
    int half_idx = tid - HALF;
    c = __bfloat162float(cos_v[half_idx]);
    sn = __bfloat162float(sin_v[half_idx]);
    k_cache_dst[head * HEAD_DIM + tid] = __float2bfloat16(normed * c + partner * sn);
  }
  v_cache_dst[head * HEAD_DIM + tid] = v_pre[head * HEAD_DIM + tid];
}

// ── Prefill (S>1) batched variants ──
// One block per (row, head); grid = (n_heads, S). Read strided q/k/v from
// the fused QKV output (in_row_stride = qkv_N), per-row cos/sin (stride
// HALF), write to contiguous Q_buf / K_cache / V_cache (out/cache row
// stride passed in). Same norm+RoPE math as the decode kernels; folds the
// per-layer rms_norm + multi-op RoPE + Q/K/V copies into 2 launches.
__global__ void q_norm_rope_qstage_prefill_kernel(
    const __nv_bfloat16* __restrict__ q_pre,      // (S, *) strided
    const __nv_bfloat16* __restrict__ q_norm_w,   // (128,)
    const __nv_bfloat16* __restrict__ cos_v,      // (S, 64)
    const __nv_bfloat16* __restrict__ sin_v,      // (S, 64)
    __nv_bfloat16* __restrict__ q_buf,            // (S, n_q*128)
    int n_q, int S,
    int in_row_stride, int out_row_stride,
    float eps) {
  int head = blockIdx.x;
  int row = blockIdx.y;
  if (head >= n_q || row >= S) return;
  int tid = threadIdx.x;

  __shared__ float s_normed[HEAD_DIM];
  __shared__ float s_smem4[N_WARPS];

  const __nv_bfloat16* q_row = q_pre + (size_t)row * in_row_stride + head * HEAD_DIM;
  const __nv_bfloat16* cos_r = cos_v + (size_t)row * HALF;
  const __nv_bfloat16* sin_r = sin_v + (size_t)row * HALF;
  float v = __bfloat162float(q_row[tid]);
  float w = __bfloat162float(q_norm_w[tid]);
  float rstd = rsqrtf(block_sum_4warp(v * v, s_smem4) / float(HEAD_DIM) + eps);
  float normed = v * rstd * w;
  s_normed[tid] = normed;
  __syncthreads();

  __nv_bfloat16* dst = q_buf + (size_t)row * out_row_stride + head * HEAD_DIM;
  if (tid < HALF) {
    float partner = s_normed[tid + HALF];
    float c = __bfloat162float(cos_r[tid]);
    float sn = __bfloat162float(sin_r[tid]);
    dst[tid] = __float2bfloat16(normed * c - partner * sn);
  } else {
    float partner = s_normed[tid - HALF];
    int hi = tid - HALF;
    float c = __bfloat162float(cos_r[hi]);
    float sn = __bfloat162float(sin_r[hi]);
    dst[tid] = __float2bfloat16(normed * c + partner * sn);
  }
}

__global__ void k_norm_rope_kvwrite_prefill_kernel(
    const __nv_bfloat16* __restrict__ k_pre,      // (S, *) strided
    const __nv_bfloat16* __restrict__ v_pre,      // (S, *) strided
    const __nv_bfloat16* __restrict__ k_norm_w,   // (128,)
    const __nv_bfloat16* __restrict__ cos_v,      // (S, 64)
    const __nv_bfloat16* __restrict__ sin_v,      // (S, 64)
    __nv_bfloat16* __restrict__ k_cache,          // (S, n_kv*128)
    __nv_bfloat16* __restrict__ v_cache,          // (S, n_kv*128)
    int n_kv, int S,
    int in_row_stride, int cache_row_stride,
    float eps) {
  int head = blockIdx.x;
  int row = blockIdx.y;
  if (head >= n_kv || row >= S) return;
  int tid = threadIdx.x;

  __shared__ float s_normed[HEAD_DIM];
  __shared__ float s_smem4[N_WARPS];

  const __nv_bfloat16* k_row = k_pre + (size_t)row * in_row_stride + head * HEAD_DIM;
  const __nv_bfloat16* v_row = v_pre + (size_t)row * in_row_stride + head * HEAD_DIM;
  const __nv_bfloat16* cos_r = cos_v + (size_t)row * HALF;
  const __nv_bfloat16* sin_r = sin_v + (size_t)row * HALF;
  float v = __bfloat162float(k_row[tid]);
  float w = __bfloat162float(k_norm_w[tid]);
  float rstd = rsqrtf(block_sum_4warp(v * v, s_smem4) / float(HEAD_DIM) + eps);
  float normed = v * rstd * w;
  s_normed[tid] = normed;
  __syncthreads();

  __nv_bfloat16* kdst = k_cache + (size_t)row * cache_row_stride + head * HEAD_DIM;
  if (tid < HALF) {
    float partner = s_normed[tid + HALF];
    float c = __bfloat162float(cos_r[tid]);
    float sn = __bfloat162float(sin_r[tid]);
    kdst[tid] = __float2bfloat16(normed * c - partner * sn);
  } else {
    float partner = s_normed[tid - HALF];
    int hi = tid - HALF;
    float c = __bfloat162float(cos_r[hi]);
    float sn = __bfloat162float(sin_r[hi]);
    kdst[tid] = __float2bfloat16(normed * c + partner * sn);
  }
  // V is copied (no norm, no RoPE).
  v_cache[(size_t)row * cache_row_stride + head * HEAD_DIM + tid] = v_row[tid];
}

// ── v3 (HEAD_DIM=128 only): WARP-per-row with VECTORIZED contiguous-4 access.
// Lane t owns the CONTIGUOUS 4 elements {4t,4t+1,4t+2,4t+3} via one 8-byte (uint2)
// load — so the warp issues ONE 256-byte coalesced transaction instead of v1/v2's
// 64-256B fragments (this is the lever: q_norm_rope is memory-throughput-bound at
// ~44% BW, NOT sync-bound). The RMS sum is a warp-shuffle; the RoPE partner of an
// element i is i±64, owned by lane t^16, fetched with one __shfl_xor(.,16) per
// element — no smem, no __syncthreads. cos/sin index = 4*(t&15)+j; the rotate sign
// is - for the low half (t<16) and + for the high half. NOT bit-identical (fp32
// ssq grouping ~1 ULP). blockDim.x = 32 (1 warp = 1 row). ──
__device__ __forceinline__ void load4_bf16(const __nv_bfloat16* p,
                                            float& a, float& b, float& c, float& d) {
  uint2 raw = *reinterpret_cast<const uint2*>(p);
  // Convert 2-at-once (one cvt per pair) instead of 4 scalar cvts.
  float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
  float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
  a = lo.x; b = lo.y; c = hi.x; d = hi.y;
}
__device__ __forceinline__ void store4_bf16(__nv_bfloat16* p,
                                             float a, float b, float c, float d) {
  __nv_bfloat162 lo = __float22bfloat162_rn(make_float2(a, b));
  __nv_bfloat162 hi = __float22bfloat162_rn(make_float2(c, d));
  uint2 raw;
  raw.x = *reinterpret_cast<const uint32_t*>(&lo);
  raw.y = *reinterpret_cast<const uint32_t*>(&hi);
  *reinterpret_cast<uint2*>(p) = raw;
}

__device__ __forceinline__ void norm_rope_warp4(
    const __nv_bfloat16* row_in, const __nv_bfloat16* norm_w,
    const __nv_bfloat16* cos_r, const __nv_bfloat16* sin_r,
    __nv_bfloat16* dst, int lane, float eps) {
  float v0, v1, v2, v3;
  load4_bf16(row_in + 4 * lane, v0, v1, v2, v3);
  float sq = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, off);
  float rstd = rsqrtf(sq / float(HEAD_DIM) + eps);
  float w0, w1, w2, w3;
  load4_bf16(norm_w + 4 * lane, w0, w1, w2, w3);
  float n0 = v0 * rstd * w0, n1 = v1 * rstd * w1, n2 = v2 * rstd * w2, n3 = v3 * rstd * w3;
  float pn0 = __shfl_xor_sync(0xffffffff, n0, 16);
  float pn1 = __shfl_xor_sync(0xffffffff, n1, 16);
  float pn2 = __shfl_xor_sync(0xffffffff, n2, 16);
  float pn3 = __shfl_xor_sync(0xffffffff, n3, 16);
  int ci = 4 * (lane & 15);
  float c0, c1, c2, c3, s0, s1, s2, s3;
  load4_bf16(cos_r + ci, c0, c1, c2, c3);
  load4_bf16(sin_r + ci, s0, s1, s2, s3);
  float sgn = (lane < 16) ? -1.f : 1.f;
  store4_bf16(dst + 4 * lane,
              n0 * c0 + sgn * pn0 * s0, n1 * c1 + sgn * pn1 * s1,
              n2 * c2 + sgn * pn2 * s2, n3 * c3 + sgn * pn3 * s3);
}

// ── v3 = ONE BLOCK PER ROW (grid=(S,), block=256=8 warps); each warp loops over
// its heads (head = warp, += n_warps) via the warp-per-row vectorized helper.
// This cuts the block count from n_heads*S (16384) to S (512) — matching the fast
// row-grid kernels — killing the per-tiny-block overhead that left the old
// one-block-per-(head,row) kernel at 2.5× its 4.1µs memory-copy floor. (Earlier
// single-head warp variants — sync-elimination, vectorized transactions — gave 0;
// the bottleneck was block COUNT, not sync/transaction/compute.) ──
__global__ void q_norm_rope_qstage_prefill_v3_kernel(
    const __nv_bfloat16* __restrict__ q_pre, const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ cos_v, const __nv_bfloat16* __restrict__ sin_v,
    __nv_bfloat16* __restrict__ q_buf,
    int n_q, int S, int in_row_stride, int out_row_stride, float eps) {
  int row = blockIdx.x;
  if (row >= S) return;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n_warps = blockDim.x >> 5;
  const __nv_bfloat16* cos_r = cos_v + (size_t)row * HALF;
  const __nv_bfloat16* sin_r = sin_v + (size_t)row * HALF;
  for (int head = warp; head < n_q; head += n_warps) {
    norm_rope_warp4(q_pre + (size_t)row * in_row_stride + head * HEAD_DIM, q_norm_w,
                    cos_r, sin_r,
                    q_buf + (size_t)row * out_row_stride + head * HEAD_DIM, lane, eps);
  }
}

__global__ void k_norm_rope_kvwrite_prefill_v3_kernel(
    const __nv_bfloat16* __restrict__ k_pre, const __nv_bfloat16* __restrict__ v_pre,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const __nv_bfloat16* __restrict__ cos_v, const __nv_bfloat16* __restrict__ sin_v,
    __nv_bfloat16* __restrict__ k_cache, __nv_bfloat16* __restrict__ v_cache,
    int n_kv, int S, int in_row_stride, int cache_row_stride, float eps) {
  int row = blockIdx.x;
  if (row >= S) return;
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int n_warps = blockDim.x >> 5;
  const __nv_bfloat16* cos_r = cos_v + (size_t)row * HALF;
  const __nv_bfloat16* sin_r = sin_v + (size_t)row * HALF;
  for (int head = warp; head < n_kv; head += n_warps) {
    norm_rope_warp4(k_pre + (size_t)row * in_row_stride + head * HEAD_DIM, k_norm_w,
                    cos_r, sin_r,
                    k_cache + (size_t)row * cache_row_stride + head * HEAD_DIM, lane, eps);
    const __nv_bfloat16* v_row = v_pre + (size_t)row * in_row_stride + head * HEAD_DIM;
    __nv_bfloat16* vdst = v_cache + (size_t)row * cache_row_stride + head * HEAD_DIM;
    *reinterpret_cast<uint2*>(vdst + 4 * lane) =
        *reinterpret_cast<const uint2*>(v_row + 4 * lane);
  }
}

}  // namespace

int qwen3_q_norm_rope_qstage_prefill_bf16(
    const void* q_pre, const void* q_norm_w, const void* cos, const void* sin,
    void* q_buf_dst, int n_q_heads, int S, int in_row_stride,
    int out_row_stride, float eps, cudaStream_t stream) {
  if (!q_pre || !q_norm_w || !cos || !sin || !q_buf_dst) return 1;
  if (n_q_heads <= 0 || S <= 0) return 2;
  q_norm_rope_qstage_prefill_kernel<<<dim3(n_q_heads, S), dim3(THREADS), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_pre),
      reinterpret_cast<const __nv_bfloat16*>(q_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(q_buf_dst),
      n_q_heads, S, in_row_stride, out_row_stride, eps);
  return 0;
}

int qwen3_k_norm_rope_kvwrite_prefill_bf16(
    const void* k_pre, const void* v_pre, const void* k_norm_w,
    const void* cos, const void* sin, void* k_cache_dst, void* v_cache_dst,
    int n_kv_heads, int S, int in_row_stride, int cache_row_stride,
    float eps, cudaStream_t stream) {
  if (!k_pre || !v_pre || !k_norm_w || !cos || !sin
      || !k_cache_dst || !v_cache_dst) return 1;
  if (n_kv_heads <= 0 || S <= 0) return 2;
  k_norm_rope_kvwrite_prefill_kernel<<<dim3(n_kv_heads, S), dim3(THREADS), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_pre),
      reinterpret_cast<const __nv_bfloat16*>(v_pre),
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(k_cache_dst),
      reinterpret_cast<__nv_bfloat16*>(v_cache_dst),
      n_kv_heads, S, in_row_stride, cache_row_stride, eps);
  return 0;
}

// v3 launches: 32 threads (1 warp) per (head, row), vectorized contiguous-4.
int qwen3_q_norm_rope_qstage_prefill_v3_bf16(
    const void* q_pre, const void* q_norm_w, const void* cos, const void* sin,
    void* q_buf_dst, int n_q_heads, int S, int in_row_stride,
    int out_row_stride, float eps, cudaStream_t stream) {
  if (!q_pre || !q_norm_w || !cos || !sin || !q_buf_dst) return 1;
  if (n_q_heads <= 0 || S <= 0) return 2;
  q_norm_rope_qstage_prefill_v3_kernel<<<dim3(S), dim3(256), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_pre),
      reinterpret_cast<const __nv_bfloat16*>(q_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(q_buf_dst),
      n_q_heads, S, in_row_stride, out_row_stride, eps);
  return 0;
}

int qwen3_k_norm_rope_kvwrite_prefill_v3_bf16(
    const void* k_pre, const void* v_pre, const void* k_norm_w,
    const void* cos, const void* sin, void* k_cache_dst, void* v_cache_dst,
    int n_kv_heads, int S, int in_row_stride, int cache_row_stride,
    float eps, cudaStream_t stream) {
  if (!k_pre || !v_pre || !k_norm_w || !cos || !sin
      || !k_cache_dst || !v_cache_dst) return 1;
  if (n_kv_heads <= 0 || S <= 0) return 2;
  // n_kv (8) < warps(8); use n_kv warps (256 threads still fine, extra warps idle).
  k_norm_rope_kvwrite_prefill_v3_kernel<<<dim3(S), dim3(256), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_pre),
      reinterpret_cast<const __nv_bfloat16*>(v_pre),
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(k_cache_dst),
      reinterpret_cast<__nv_bfloat16*>(v_cache_dst),
      n_kv_heads, S, in_row_stride, cache_row_stride, eps);
  return 0;
}

int qwen3_k_norm_rope_kvwrite_devpos_bf16(
    const void* k_pre, const void* v_pre, const void* k_norm_w,
    const void* cos, const void* sin, void* k_cache_base, void* v_cache_base,
    const void* cur_pos, int row_elems, int n_kv_heads, float eps,
    cudaStream_t stream) {
  if (!k_pre || !v_pre || !k_norm_w || !cos || !sin
      || !k_cache_base || !v_cache_base || !cur_pos) return 1;
  if (n_kv_heads <= 0) return 2;
  k_norm_rope_kvwrite_devpos_kernel<<<dim3(n_kv_heads), dim3(THREADS), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_pre),
      reinterpret_cast<const __nv_bfloat16*>(v_pre),
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(k_cache_base),
      reinterpret_cast<__nv_bfloat16*>(v_cache_base),
      reinterpret_cast<const int*>(cur_pos),
      row_elems, n_kv_heads, eps);
  return 0;
}

int qwen3_q_norm_rope_qstage_bf16(
    const void* q_pre,
    const void* q_norm_w,
    const void* cos,
    const void* sin,
    void*       q_buf_dst,
    int         n_q_heads,
    float       eps,
    cudaStream_t stream) {
  if (!q_pre || !q_norm_w || !cos || !sin || !q_buf_dst) return 1;
  if (n_q_heads <= 0) return 2;
  dim3 grid(n_q_heads);
  dim3 block(THREADS);
  q_norm_rope_qstage_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q_pre),
      reinterpret_cast<const __nv_bfloat16*>(q_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(q_buf_dst),
      n_q_heads, eps);
  return 0;
}

int qwen3_k_norm_rope_kvwrite_bf16(
    const void* k_pre,
    const void* v_pre,
    const void* k_norm_w,
    const void* cos,
    const void* sin,
    void*       k_cache_dst,
    void*       v_cache_dst,
    int         n_kv_heads,
    float       eps,
    cudaStream_t stream) {
  if (!k_pre || !v_pre || !k_norm_w || !cos || !sin
      || !k_cache_dst || !v_cache_dst) return 1;
  if (n_kv_heads <= 0) return 2;
  dim3 grid(n_kv_heads);
  dim3 block(THREADS);
  k_norm_rope_kvwrite_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_pre),
      reinterpret_cast<const __nv_bfloat16*>(v_pre),
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(k_cache_dst),
      reinterpret_cast<__nv_bfloat16*>(v_cache_dst),
      n_kv_heads, eps);
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
