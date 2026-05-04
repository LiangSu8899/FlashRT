// SPDX-License-Identifier: Apache-2.0
//
// Fused RMSNorm + weight + silu(gate). See header for spec.

#include "rms_norm_gated_silu_qwen36.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

namespace {

template <int DIM>
__global__ void rms_norm_gated_silu_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int M, float eps)
{
  static_assert(DIM == 128, "DIM=128 specialization for Qwen3.6");
  const int m = blockIdx.x;
  const int t = threadIdx.x;
  if (m >= M || t >= DIM) return;

  const size_t row_off = (size_t)m * DIM + t;
  const float xv = static_cast<float>(x[row_off]);
  const float gv = static_cast<float>(gate[row_off]);

  // Block-reduce sum-of-squares (4 warps for DIM=128).
  float sq = xv * xv;
  for (int off = 16; off > 0; off >>= 1) {
    sq += __shfl_xor_sync(0xffffffff, sq, off);
  }
  __shared__ float warp_sq[4];
  __shared__ float reduced;
  const int lane = t & 31;
  const int warp = t >> 5;
  if (lane == 0) warp_sq[warp] = sq;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < 4) ? warp_sq[lane] : 0.0f;
    v += __shfl_xor_sync(0xffffffff, v, 1);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    if (lane == 0) reduced = v;
  }
  __syncthreads();

  const float rms_inv = rsqrtf(reduced / static_cast<float>(DIM) + eps);

  // Weighted norm: cast to bf16 then back (HF's quirky dtype path).
  const float wv = static_cast<float>(weight[t]);
  const __nv_bfloat16 norm_bf = __float2bfloat16(xv * rms_inv);
  const __nv_bfloat16 weighted_bf =
      __float2bfloat16(wv * static_cast<float>(norm_bf));

  // silu(gate) computed in fp32, multiply with weighted (re-cast fp32),
  // cast back to bf16 for output.
  const float silu_g = gv / (1.0f + __expf(-gv));
  const float out_f =
      static_cast<float>(weighted_bf) * silu_g;
  out[row_off] = __float2bfloat16(out_f);
}

}  // namespace

void rms_norm_gated_silu_qwen36_bf16(
    const void* x,
    const void* gate,
    const void* weight,
    void* out,
    int M, int dim, float eps,
    cudaStream_t stream)
{
  if (dim != 128) {
    return;  // Qwen3.6 only; caller fall back if dim differs
  }
  dim3 grid(M);
  dim3 block(dim);
  rms_norm_gated_silu_kernel<128><<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<const __nv_bfloat16*>(gate),
      reinterpret_cast<const __nv_bfloat16*>(weight),
      reinterpret_cast<__nv_bfloat16*>(out),
      M, eps);
}

}  // namespace kernels
}  // namespace flash_rt
