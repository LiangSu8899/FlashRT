// SPDX-License-Identifier: Apache-2.0
//
// Gated DeltaNet recurrent (single-token decode) kernel.
//
// Block layout: one block per (b, h) where h indexes ``num_v_heads``.
// Within a block, threadIdx.x = t in [0, head_v_dim) owns column t of
// the state matrix state[b, h, :, t] (head_k_dim elements).
//
// Per-thread state column lives in registers (head_k_dim fp32 = 128
// regs/thread on Qwen3.6). Q/K/V are loaded into shared memory once
// per block, then broadcast across threads.

#include "gated_deltanet_qwen36.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

namespace {

constexpr int kHD = 128;   // Qwen3.6 head_k_dim == head_v_dim
constexpr float kEps = 1e-6f;

template <int HD>
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
  // Warp reduce.
  for (int off = 16; off > 0; off >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, off);
  }
  // Cross-warp reduce via smem (4 warps for HD=128).
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = val;
  __syncthreads();

  if (warp == 0) {
    val = (lane < (HD / 32)) ? smem[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
      val += __shfl_xor_sync(0xffffffff, val, off);
    }
    if (lane == 0) smem[0] = val;
  }
  __syncthreads();
  return smem[0];
}

template <int HD>
__global__ void gated_deltanet_recurrent_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __nv_bfloat16* __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ out_,
    int num_v_heads,
    bool use_qk_l2norm)
{
  static_assert(HD == 128, "HD must be 128 for Qwen3.6 (single instantiation)");
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int t = threadIdx.x;
  if (t >= HD) return;

  // Smem layout: qs[HD], ks[HD], scratch[8] (warp-reduce buffer).
  __shared__ float smem[2 * HD + 32];
  float* qs = smem;
  float* ks = smem + HD;
  float* scratch = smem + 2 * HD;

  // Load Q and K to smem (each thread loads its element).
  const size_t qkv_off = ((size_t)b * num_v_heads + h) * HD + t;
  qs[t] = static_cast<float>(q_in[qkv_off]);
  ks[t] = static_cast<float>(k_in[qkv_off]);
  __syncthreads();

  // L2 norm Q and K (in-place in smem).
  if (use_qk_l2norm) {
    float q_sq = qs[t] * qs[t];
    float k_sq = ks[t] * ks[t];
    q_sq = block_reduce_sum<HD>(q_sq, scratch);
    k_sq = block_reduce_sum<HD>(k_sq, scratch);
    const float q_inv = rsqrtf(q_sq + kEps);
    const float k_inv = rsqrtf(k_sq + kEps);
    qs[t] *= q_inv;
    ks[t] *= k_inv;
    __syncthreads();
  }

  // Scale Q by 1 / sqrt(HD).
  qs[t] *= rsqrtf(static_cast<float>(HD));
  __syncthreads();

  // exp(g_t) and beta_t (broadcast scalars).
  const float g_t =
      __expf(static_cast<float>(g_in[b * num_v_heads + h]));
  const float beta_t =
      static_cast<float>(beta_in[b * num_v_heads + h]);

  // Each thread holds column t of state[b, h, :, :] in registers.
  float col[HD];
  const size_t state_h_off =
      (((size_t)b * num_v_heads + h)) * HD * HD;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] =
        static_cast<float>(state[state_h_off + (size_t)i * HD + t]) * g_t;
  }

  // kv_mem[t] = sum_i col[i] * ks[i]
  float kv_mem = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    kv_mem = fmaf(col[i], ks[i], kv_mem);
  }

  // delta[t] = (V[t] - kv_mem) * beta
  const float v_t =
      static_cast<float>(v_in[(size_t)b * num_v_heads * HD + h * HD + t]);
  const float delta = (v_t - kv_mem) * beta_t;

  // state[i, t] += k[i] * delta
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] = fmaf(ks[i], delta, col[i]);
  }

  // Write back state column.
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state[state_h_off + (size_t)i * HD + t] = __float2bfloat16(col[i]);
  }

  // out[t] = sum_i col[i] * qs[i]
  float out_t = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    out_t = fmaf(col[i], qs[i], out_t);
  }
  out_[(size_t)b * num_v_heads * HD + h * HD + t] =
      __float2bfloat16(out_t);
}

}  // namespace

void gated_deltanet_recurrent_qwen36_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    void*       state,
    void*       out,
    int B, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (head_k_dim != kHD || head_v_dim != kHD) {
    // Could template more dims; for Qwen3.6 only HD=128 is needed.
    return;  // silently no-op; caller checks output is unchanged
  }

  dim3 grid(num_v_heads, B);
  dim3 block(kHD);
  gated_deltanet_recurrent_kernel<kHD><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<__nv_bfloat16*>(state),
      reinterpret_cast<__nv_bfloat16*>(out),
      num_v_heads, use_qk_l2norm);
}

// In/out-state variant: reads col from state_in, writes updated col
// to state_out (separate buffer). Eliminates the standalone
// .copy_(state_out, state) launch in the K-iter verify loop by
// chaining state_in[k+1] := state_out[k] across iterations. Bit-
// identical to (existing kernel + .copy_) under same inputs because
// the math is unchanged; only the writeback target differs.
namespace {

template <int HD>
__global__ void gated_deltanet_recurrent_inout_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    const __nv_bfloat16* __restrict__ g_in,
    const __nv_bfloat16* __restrict__ beta_in,
    const __nv_bfloat16* __restrict__ state_in,
    __nv_bfloat16* __restrict__ state_out,
    __nv_bfloat16* __restrict__ out_,
    int num_v_heads,
    bool use_qk_l2norm)
{
  static_assert(HD == 128, "HD must be 128 for Qwen3.6");
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int t = threadIdx.x;
  if (t >= HD) return;

  __shared__ float smem[2 * HD + 32];
  float* qs = smem;
  float* ks = smem + HD;
  float* scratch = smem + 2 * HD;

  const size_t qkv_off = ((size_t)b * num_v_heads + h) * HD + t;
  qs[t] = static_cast<float>(q_in[qkv_off]);
  ks[t] = static_cast<float>(k_in[qkv_off]);
  __syncthreads();

  if (use_qk_l2norm) {
    float q_sq = qs[t] * qs[t];
    float k_sq = ks[t] * ks[t];
    q_sq = block_reduce_sum<HD>(q_sq, scratch);
    k_sq = block_reduce_sum<HD>(k_sq, scratch);
    const float q_inv = rsqrtf(q_sq + kEps);
    const float k_inv = rsqrtf(k_sq + kEps);
    qs[t] *= q_inv;
    ks[t] *= k_inv;
    __syncthreads();
  }

  qs[t] *= rsqrtf(static_cast<float>(HD));
  __syncthreads();

  const float g_t =
      __expf(static_cast<float>(g_in[b * num_v_heads + h]));
  const float beta_t =
      static_cast<float>(beta_in[b * num_v_heads + h]);

  float col[HD];
  const size_t state_h_off =
      (((size_t)b * num_v_heads + h)) * HD * HD;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] =
        static_cast<float>(state_in[state_h_off + (size_t)i * HD + t]) * g_t;
  }

  float kv_mem = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    kv_mem = fmaf(col[i], ks[i], kv_mem);
  }

  const float v_t =
      static_cast<float>(v_in[(size_t)b * num_v_heads * HD + h * HD + t]);
  const float delta = (v_t - kv_mem) * beta_t;

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    col[i] = fmaf(ks[i], delta, col[i]);
  }

  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    state_out[state_h_off + (size_t)i * HD + t] = __float2bfloat16(col[i]);
  }

  float out_t = 0.0f;
  #pragma unroll 16
  for (int i = 0; i < HD; ++i) {
    out_t = fmaf(col[i], qs[i], out_t);
  }
  out_[(size_t)b * num_v_heads * HD + h * HD + t] =
      __float2bfloat16(out_t);
}

}  // namespace

void gated_deltanet_recurrent_inout_qwen36_bf16(
    const void* q,
    const void* k,
    const void* v,
    const void* g,
    const void* beta,
    const void* state_in,
    void*       state_out,
    void*       out,
    int B, int num_v_heads, int head_k_dim, int head_v_dim,
    bool use_qk_l2norm,
    cudaStream_t stream)
{
  if (head_k_dim != kHD || head_v_dim != kHD) {
    return;
  }
  dim3 grid(num_v_heads, B);
  dim3 block(kHD);
  gated_deltanet_recurrent_inout_kernel<kHD><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q),
      reinterpret_cast<const __nv_bfloat16*>(k),
      reinterpret_cast<const __nv_bfloat16*>(v),
      reinterpret_cast<const __nv_bfloat16*>(g),
      reinterpret_cast<const __nv_bfloat16*>(beta),
      reinterpret_cast<const __nv_bfloat16*>(state_in),
      reinterpret_cast<__nv_bfloat16*>(state_out),
      reinterpret_cast<__nv_bfloat16*>(out),
      num_v_heads, use_qk_l2norm);
}

}  // namespace kernels
}  // namespace flash_rt
