// SPDX-License-Identifier: Apache-2.0
//
// See attn_qkv_norm_rope_write_sm80.cuh. SM80-family facilities only.
//
// A block owns one head of one row. Query heads normalize, rotate and publish
// their query and, when the projection carries one, their gate; key heads do
// the same for the key and carry the value across unchanged. Both read the
// same row of the fused projection, which is the reason they share a launch:
// separately this is a split, two norms, two rotations and two cache writes,
// and each of those is a pass over a few kilobytes.

#include "kernels/attn_qkv_norm_rope_write_sm80.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {
namespace {

constexpr int kThreads = 128;
constexpr int kMaxHeadDim = 1024;

__device__ __forceinline__ float block_sum(float value, float* scratch) {
  for (int offset = 16; offset; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (!lane) scratch[warp] = value;
  __syncthreads();
  value = (threadIdx.x < (kThreads >> 5)) ? scratch[threadIdx.x] : 0.0f;
  for (int offset = (kThreads >> 6); offset; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  if (!threadIdx.x) scratch[0] = value;
  __syncthreads();
  return scratch[0];
}

// Normalize one head, rotate the part of it that carries position, and leave
// the result in ``staged`` in shared memory.
__device__ void norm_and_rotate(
    const __nv_bfloat16* __restrict__ source,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ cos_row,
    const __nv_bfloat16* __restrict__ sin_row,
    float* __restrict__ staged, float* __restrict__ scratch,
    int head_dim, int rope_dim, float eps) {
  float squares = 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += kThreads) {
    const float value = __bfloat162float(source[i]);
    staged[i] = value;
    squares += value * value;
  }
  const float inverse =
      rsqrtf(block_sum(squares, scratch) / head_dim + eps);
  for (int i = threadIdx.x; i < head_dim; i += kThreads) {
    staged[i] = staged[i] * inverse * __bfloat162float(weight[i]);
  }
  __syncthreads();

  // The second half of the rotation reads the first, so both halves are read
  // before either is written.
  const int half = rope_dim >> 1;
  for (int i = threadIdx.x; i < half; i += kThreads) {
    const float c = __bfloat162float(cos_row[i]);
    const float s = __bfloat162float(sin_row[i]);
    const float low = staged[i];
    const float high = staged[i + half];
    staged[i] = low * c - high * s;
    staged[i + half] = high * c + low * s;
  }
  __syncthreads();
}

__global__ void qkv_norm_rope_write_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const __nv_bfloat16* __restrict__ cos, const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out, __nv_bfloat16* __restrict__ gate_out,
    __nv_bfloat16* __restrict__ k_cache, __nv_bfloat16* __restrict__ v_cache,
    int pos, const int* __restrict__ pos_device,
    int q_heads, int kv_heads, int head_dim, int rope_dim,
    bool has_gate, float eps) {
  extern __shared__ float shared[];
  float* staged = shared;
  float* scratch = shared + head_dim;

  const int head = blockIdx.x;
  const int row = blockIdx.y;
  const int position = (pos_device ? *pos_device : pos) + row;

  const int query_width = q_heads * head_dim * (has_gate ? 2 : 1);
  const int key_width = kv_heads * head_dim;
  const __nv_bfloat16* qkv_row =
      qkv + static_cast<size_t>(row) * (query_width + 2 * key_width);
  const __nv_bfloat16* cos_row =
      cos + static_cast<size_t>(position) * (rope_dim >> 1);
  const __nv_bfloat16* sin_row =
      sin + static_cast<size_t>(position) * (rope_dim >> 1);

  if (head < q_heads) {
    const int stride = has_gate ? 2 * head_dim : head_dim;
    norm_and_rotate(qkv_row + head * stride, q_norm_w, cos_row, sin_row,
                    staged, scratch, head_dim, rope_dim, eps);
    __nv_bfloat16* destination =
        q_out + (static_cast<size_t>(row) * q_heads + head) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += kThreads) {
      destination[i] = __float2bfloat16(staged[i]);
    }
    if (has_gate && gate_out) {
      const __nv_bfloat16* source = qkv_row + head * stride + head_dim;
      __nv_bfloat16* gate =
          gate_out + (static_cast<size_t>(row) * q_heads + head) * head_dim;
      for (int i = threadIdx.x; i < head_dim; i += kThreads) {
        gate[i] = source[i];
      }
    }
    return;
  }

  const int key_head = head - q_heads;
  norm_and_rotate(qkv_row + query_width + key_head * head_dim, k_norm_w,
                  cos_row, sin_row, staged, scratch, head_dim, rope_dim, eps);
  const size_t slot =
      (static_cast<size_t>(position) * kv_heads + key_head) * head_dim;
  const __nv_bfloat16* value =
      qkv_row + query_width + key_width + key_head * head_dim;
  for (int i = threadIdx.x; i < head_dim; i += kThreads) {
    k_cache[slot + i] = __float2bfloat16(staged[i]);
    v_cache[slot + i] = value[i];
  }
}

}  // namespace

int attn_qkv_norm_rope_write_bf16(
    const void* qkv,
    const void* q_norm_w,
    const void* k_norm_w,
    const void* cos, const void* sin,
    void* q_out, void* gate_out,
    void* k_cache, void* v_cache,
    int S, int pos, const int* pos_device,
    int q_heads, int kv_heads, int head_dim, int rope_dim,
    bool has_gate, float eps,
    cudaStream_t stream) {
  if (S <= 0) return 0;
  if (q_heads <= 0 || kv_heads <= 0) return -1;
  if (head_dim <= 0 || head_dim > kMaxHeadDim) return -1;
  if (rope_dim < 0 || rope_dim > head_dim || (rope_dim & 1)) return -1;

  const dim3 grid(q_heads + kv_heads, S);
  const size_t shared = (head_dim + 32) * sizeof(float);
  qkv_norm_rope_write_kernel<<<grid, kThreads, shared, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(qkv),
      reinterpret_cast<const __nv_bfloat16*>(q_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
      reinterpret_cast<const __nv_bfloat16*>(cos),
      reinterpret_cast<const __nv_bfloat16*>(sin),
      reinterpret_cast<__nv_bfloat16*>(q_out),
      reinterpret_cast<__nv_bfloat16*>(gate_out),
      reinterpret_cast<__nv_bfloat16*>(k_cache),
      reinterpret_cast<__nv_bfloat16*>(v_cache),
      pos, pos_device, q_heads, kv_heads, head_dim, rope_dim, has_gate, eps);
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
