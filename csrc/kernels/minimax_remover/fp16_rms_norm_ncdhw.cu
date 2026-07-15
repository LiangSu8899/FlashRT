// SPDX-License-Identifier: Apache-2.0
// Fused FP16 NCDHW RMSNorm for MiniMax-Remover VAE (Wan VAE).
//
// Replaces the diffusers WanRMS_norm.forward which, for fp16 inputs, does:
//   x.float() -> F.normalize(x, dim=1) -> .to(fp16) -> * sqrt(C) * gamma + bias
// (four full-tensor passes, ~0.45 ms each on RTX 5060 Ti for [1,384,1,240,432]).
//
// This kernel is a single-pass-per-spatial-location fp16-native version:
// fp16 in, fp32 statistics, fp16 out -- NO dtype cast at all.  The VAE stays
// in fp16 (cuDNN already dispatches fp16 tensorop conv kernels), so every
// RMS_norm call goes from ~0.45 ms to ~0.008 ms (measured), a ~56x speed-up.
//
// Precision: fp32 sum-of-squares accumulation, same as the bf16 sibling.
// fp16 has 10-bit mantissa vs bf16's 7-bit, so end-to-end VAE PSNR stays
// at ~40 dB vs the fp16 reference (vs ~15 dB for the bf16-cast path).

#include "fp16_rms_norm_ncdhw.cuh"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {
namespace {

constexpr int kThreadsX = 32;
constexpr int kThreadsY = 8;
constexpr int kThreads = kThreadsX * kThreadsY;
constexpr int kMaxHalf2 = 64;  // C <= 1024 with 8 y-lanes.

__global__ void fp16_rms_norm_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    const __half* __restrict__ bias,
    __half* __restrict__ y,
    int B, int C, int T, int H, int W,
    int W_blocks_per_row,
    float eps)
{
  __shared__ float sm_red[kThreads];

  const int wb = blockIdx.x % W_blocks_per_row;
  const int rest = blockIdx.x / W_blocks_per_row;
  const int hwt = T * H;
  const int b = rest / hwt;
  const int rh = rest - b * hwt;
  const int t = rh / H;
  const int h = rh - t * H;
  if (b >= B) return;

  const int tx = threadIdx.x & 31;
  const int ty = threadIdx.x >> 5;
  const int w = wb * kThreadsX + tx;
  const bool active = (w < W);

  const int c_per_y = (C + kThreadsY - 1) / kThreadsY;
  const int c_start = ty * c_per_y;
  const int c_end = min(c_start + c_per_y, C);
  const int n_c = c_end - c_start;
  const int n_pair = (n_c + 1) >> 1;

  const long long stride_C = (long long)T * H * W;
  const long long row_off = (long long)t * H * W + (long long)h * W + w;
  const long long b_off = (long long)b * C * stride_C;

  __half2 xcache[kMaxHalf2];
  float sum_sq = 0.0f;

  if (active) {
    #pragma unroll 1
    for (int p = 0; p < n_pair; ++p) {
      int c0 = c_start + (p << 1);
      int c1 = c0 + 1;
      __half v0 = x[b_off + (long long)c0 * stride_C + row_off];
      __half v1 = (c1 < c_end)
          ? x[b_off + (long long)c1 * stride_C + row_off]
          : __float2half(0.0f);
      xcache[p] = __half2{v0, v1};
      float f0 = __half2float(v0);
      float f1 = __half2float(v1);
      sum_sq = fmaf(f0, f0, sum_sq);
      if (c1 < c_end) sum_sq = fmaf(f1, f1, sum_sq);
    }
  }

  sm_red[ty * kThreadsX + tx] = active ? sum_sq : 0.0f;
  __syncthreads();

  float total_sum_sq = 0.0f;
  #pragma unroll
  for (int yi = 0; yi < kThreadsY; ++yi) {
    total_sum_sq += sm_red[yi * kThreadsX + tx];
  }

  const float inv_rms = active
      ? rsqrtf(total_sum_sq * (1.0f / static_cast<float>(C)) + eps)
      : 0.0f;

  if (active) {
    #pragma unroll 1
    for (int p = 0; p < n_pair; ++p) {
      int c0 = c_start + (p << 1);
      int c1 = c0 + 1;
      __half2 vp = xcache[p];

      float v0 = __half2float(vp.x) * inv_rms
          * __half2float(gamma[c0]);
      if (bias != nullptr) {
        v0 += __half2float(bias[c0]);
      }
      y[b_off + (long long)c0 * stride_C + row_off] =
          __float2half(v0);

      if (c1 < c_end) {
        float v1 = __half2float(vp.y) * inv_rms
            * __half2float(gamma[c1]);
        if (bias != nullptr) {
          v1 += __half2float(bias[c1]);
        }
        y[b_off + (long long)c1 * stride_C + row_off] =
            __float2half(v1);
      }
    }
  }
}

}  // namespace

int fp16_rms_norm_ncdhw(
    const void* x_fp16,
    const void* gamma_fp16,
    const void* bias_fp16,
    void* y_fp16,
    int B, int C, int T, int H, int W,
    float eps,
    cudaStream_t stream)
{
  if (B <= 0 || C <= 0 || T <= 0 || H <= 0 || W <= 0) return -1;
  if ((C & 1) != 0) return -2;
  if (C > 1024) return -3;

  const int W_blocks_per_row = (W + kThreadsX - 1) / kThreadsX;
  const long long n_ctas =
      (long long)B * T * H * (long long)W_blocks_per_row;
  if (n_ctas <= 0 || n_ctas > (long long)INT32_MAX) return -4;

  fp16_rms_norm_kernel<<<static_cast<unsigned>(n_ctas), kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(x_fp16),
      reinterpret_cast<const __half*>(gamma_fp16),
      reinterpret_cast<const __half*>(bias_fp16),
      reinterpret_cast<__half*>(y_fp16),
      B, C, T, H, W, W_blocks_per_row, eps);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt
