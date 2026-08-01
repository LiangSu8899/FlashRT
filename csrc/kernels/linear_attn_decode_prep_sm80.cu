// SPDX-License-Identifier: Apache-2.0
//
// See linear_attn_decode_prep_sm80.cuh. SM80-family facilities only.
//
// A block owns one (row, value head): it copies that head's slice of q, k and
// v out of the convolved stream and, on one thread, produces the head's decay
// and update rate. Query and key are read from the key head this value head
// belongs to, which is the whole of the broadcast -- no tensor is expanded,
// the reads just repeat.

#include "kernels/linear_attn_decode_prep_sm80.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {
namespace {

constexpr int kThreads = 128;
// Above this the exponential in softplus overflows well before the result
// differs from its argument, which is the threshold the reference uses.
constexpr float kSoftplusLinear = 20.0f;

__global__ void split_broadcast_gate_kernel(
    const __nv_bfloat16* __restrict__ conv_out,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ neg_exp_a_log,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ g,
    __nv_bfloat16* __restrict__ beta,
    int k_heads, int v_heads, int head_k, int head_v,
    int a_stride, int b_stride, int repeat) {
  const int head = blockIdx.x;
  const int row = blockIdx.y;
  const int key_head = head / repeat;

  const int key_width = k_heads * head_k;
  const size_t source = static_cast<size_t>(row) *
                        (2 * key_width + v_heads * head_v);
  const __nv_bfloat16* q_src = conv_out + source + key_head * head_k;
  const __nv_bfloat16* k_src = q_src + key_width;
  const __nv_bfloat16* v_src =
      conv_out + source + 2 * key_width + head * head_v;

  const size_t key_dst = (static_cast<size_t>(row) * v_heads + head) * head_k;
  const size_t value_dst = (static_cast<size_t>(row) * v_heads + head) * head_v;
  for (int i = threadIdx.x; i < head_k; i += kThreads) {
    q[key_dst + i] = q_src[i];
    k[key_dst + i] = k_src[i];
  }
  for (int i = threadIdx.x; i < head_v; i += kThreads) {
    v[value_dst + i] = v_src[i];
  }

  if (threadIdx.x != 0) return;
  const float decay =
      __bfloat162float(a[static_cast<size_t>(row) * a_stride + head]) +
      dt_bias[head];
  const float softplus =
      decay > kSoftplusLinear ? decay : log1pf(__expf(decay));
  const float rate =
      __bfloat162float(b[static_cast<size_t>(row) * b_stride + head]);
  const size_t gate_dst = static_cast<size_t>(row) * v_heads + head;
  g[gate_dst] = __float2bfloat16(neg_exp_a_log[head] * softplus);
  beta[gate_dst] = __float2bfloat16(1.0f / (1.0f + __expf(-rate)));
}

}  // namespace

int linear_attn_split_broadcast_gate_bf16(
    const void* conv_out,
    const void* a,
    const void* b,
    const float* neg_exp_a_log,
    const float* dt_bias,
    void* q, void* k, void* v,
    void* g, void* beta,
    int S, int k_heads, int v_heads, int head_k, int head_v,
    int a_stride, int b_stride,
    cudaStream_t stream) {
  if (S <= 0) return 0;
  if (k_heads <= 0 || v_heads <= 0 || v_heads % k_heads) return -1;

  const dim3 grid(v_heads, S);
  split_broadcast_gate_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(conv_out),
      reinterpret_cast<const __nv_bfloat16*>(a),
      reinterpret_cast<const __nv_bfloat16*>(b),
      neg_exp_a_log, dt_bias,
      reinterpret_cast<__nv_bfloat16*>(q),
      reinterpret_cast<__nv_bfloat16*>(k),
      reinterpret_cast<__nv_bfloat16*>(v),
      reinterpret_cast<__nv_bfloat16*>(g),
      reinterpret_cast<__nv_bfloat16*>(beta),
      k_heads, v_heads, head_k, head_v, a_stride, b_stride,
      v_heads / k_heads);
  return 0;
}

}  // namespace kernels
}  // namespace flash_rt
