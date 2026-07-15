// ================================================================
// flash_rt — MiniMax-Remover FP8 per-tensor activation quantize.
//
// 2-pass fused launcher (no host sync):
//   Pass 1: grid-stride amax reduction → atomicMax into amax_buf.
//   Pass 2: read amax_buf on device, compute scale, grid-stride
//           quantize fp16 → fp8 e4m3.  Writes scale to scale_out.
// ================================================================

#include "fp16_quant_fp8_per_tensor.cuh"
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

namespace {

constexpr int Q_THREADS = 256;
constexpr int Q_BLOCKS = 256;  // cap grid for amax pass

__global__ void amax_fp16_kernel(
    const __half* __restrict__ x, int n, float* amax_out)
{
  int tid = threadIdx.x;
  int idx = blockIdx.x * Q_THREADS + tid;
  int stride = Q_THREADS * gridDim.x;
  float local = 0.0f;
  for (int i = idx; i < n; i += stride) {
    local = fmaxf(local, fabsf(__half2float(x[i])));
  }
  // Warp reduce
  for (int d = 16; d > 0; d >>= 1)
    local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, d));
  // Block reduce via shared memory
  __shared__ float smem[8];  // Q_THREADS / 32 = 8 warps
  int warp_id = tid / 32;
  int lane = tid % 32;
  if (lane == 0) smem[warp_id] = local;
  __syncthreads();
  if (warp_id == 0) {
    local = (lane < 8) ? smem[lane] : 0.0f;
    for (int d = 4; d > 0; d >>= 1)
      local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, d));
    if (lane == 0) {
      // atomicMax on reinterpreted int works for non-negative floats
      // (IEEE 754 preserves ordering for sign bit = 0).
      atomicMax(reinterpret_cast<int*>(amax_out), __float_as_int(local));
    }
  }
}

__global__ void quantize_fp16_to_fp8_kernel(
    const __half* __restrict__ x,
    __nv_fp8_e4m3* __restrict__ y,
    int n, const float* amax_in, float* scale_out)
{
  __shared__ float s_inv_scale;
  if (threadIdx.x == 0) {
    float amax = fmaxf(*amax_in, 0.0f);
    float scale = amax * (1.0f / 448.0f);
    if (scale < 1e-6f) scale = 1e-6f;
    if (scale_out != nullptr) *scale_out = scale;
    s_inv_scale = 1.0f / scale;
  }
  __syncthreads();
  float inv_scale = s_inv_scale;

  int idx = blockIdx.x * Q_THREADS + threadIdx.x;
  int stride = Q_THREADS * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    float val = __half2float(x[i]) * inv_scale;
    val = fminf(fmaxf(val, -448.0f), 448.0f);
    y[i] = __nv_fp8_e4m3(val);
  }
}

}  // anonymous namespace

int fp16_quant_fp8_per_tensor(
    const void* x_fp16, void* y_fp8,
    void* scale_out, void* amax_buf,
    int n, cudaStream_t stream)
{
  if (n <= 0) return -1;
  cudaError_t e;
  e = cudaMemsetAsync(amax_buf, 0, sizeof(float), stream);
  if (e != cudaSuccess) return -2;

  int blocks_amax = (n + Q_THREADS - 1) / Q_THREADS;
  if (blocks_amax > Q_BLOCKS) blocks_amax = Q_BLOCKS;

  amax_fp16_kernel<<<blocks_amax, Q_THREADS, 0, stream>>>(
      reinterpret_cast<const __half*>(x_fp16), n,
      reinterpret_cast<float*>(amax_buf));

  int blocks_q = (n + Q_THREADS - 1) / Q_THREADS;
  quantize_fp16_to_fp8_kernel<<<blocks_q, Q_THREADS, 0, stream>>>(
      reinterpret_cast<const __half*>(x_fp16),
      reinterpret_cast<__nv_fp8_e4m3*>(y_fp8), n,
      reinterpret_cast<const float*>(amax_buf),
      reinterpret_cast<float*>(scale_out));

  e = cudaGetLastError();
  if (e != cudaSuccess) return -3;
  return 0;
}

// ── Standalone amax (for multi-tensor shared-scale quantization) ──
int amax_fp16(
    const void* x_fp16, void* amax_buf,
    int n, cudaStream_t stream)
{
  if (n <= 0) return -1;
  int blocks = (n + Q_THREADS - 1) / Q_THREADS;
  if (blocks > Q_BLOCKS) blocks = Q_BLOCKS;
  amax_fp16_kernel<<<blocks, Q_THREADS, 0, stream>>>(
      reinterpret_cast<const __half*>(x_fp16), n,
      reinterpret_cast<float*>(amax_buf));
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -2;
}

// ── Standalone quantize (reads pre-computed amax from device) ──
int quantize_fp16_fp8_with_amax(
    const void* x_fp16, void* y_fp8,
    const void* amax_buf, void* scale_out,
    int n, cudaStream_t stream)
{
  if (n <= 0) return -1;
  int blocks = (n + Q_THREADS - 1) / Q_THREADS;
  quantize_fp16_to_fp8_kernel<<<blocks, Q_THREADS, 0, stream>>>(
      reinterpret_cast<const __half*>(x_fp16),
      reinterpret_cast<__nv_fp8_e4m3*>(y_fp8), n,
      reinterpret_cast<const float*>(amax_buf),
      reinterpret_cast<float*>(scale_out));
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -2;
}

// ── Dual quantize: two buffers, one shared amax, one launch ──
__global__ void quantize_fp16_to_fp8_dual_kernel(
    const __half* __restrict__ x1,
    __nv_fp8_e4m3* __restrict__ y1, int n1,
    const __half* __restrict__ x2,
    __nv_fp8_e4m3* __restrict__ y2, int n2,
    const float* amax_in, float* scale_out)
{
  __shared__ float s_inv_scale;
  if (threadIdx.x == 0) {
    float amax = fmaxf(*amax_in, 0.0f);
    float scale = amax * (1.0f / 448.0f);
    if (scale < 1e-6f) scale = 1e-6f;
    if (scale_out != nullptr) *scale_out = scale;
    s_inv_scale = 1.0f / scale;
  }
  __syncthreads();
  float inv_scale = s_inv_scale;

  // Grid-stride over both buffers using a unified index space
  int total = n1 + n2;
  int idx = blockIdx.x * Q_THREADS + threadIdx.x;
  int stride = Q_THREADS * gridDim.x;
  for (int i = idx; i < total; i += stride) {
    if (i < n1) {
      float val = __half2float(x1[i]) * inv_scale;
      val = fminf(fmaxf(val, -448.0f), 448.0f);
      y1[i] = __nv_fp8_e4m3(val);
    } else {
      int j = i - n1;
      float val = __half2float(x2[j]) * inv_scale;
      val = fminf(fmaxf(val, -448.0f), 448.0f);
      y2[j] = __nv_fp8_e4m3(val);
    }
  }
}

int quantize_fp16_fp8_with_amax_dual(
    const void* x1_fp16, void* y1_fp8, int n1,
    const void* x2_fp16, void* y2_fp8, int n2,
    const void* amax_buf, void* scale_out,
    cudaStream_t stream)
{
  if (n1 <= 0 && n2 <= 0) return -1;
  int total = n1 + n2;
  int blocks = (total + Q_THREADS - 1) / Q_THREADS;
  quantize_fp16_to_fp8_dual_kernel<<<blocks, Q_THREADS, 0, stream>>>(
      reinterpret_cast<const __half*>(x1_fp16),
      reinterpret_cast<__nv_fp8_e4m3*>(y1_fp8), n1,
      reinterpret_cast<const __half*>(x2_fp16),
      reinterpret_cast<__nv_fp8_e4m3*>(y2_fp8), n2,
      reinterpret_cast<const float*>(amax_buf),
      reinterpret_cast<float*>(scale_out));
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -2;
}

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt
