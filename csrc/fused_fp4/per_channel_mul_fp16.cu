// ============================================================================
//  Per-input-channel multiply (AWQ activation inverse-scale).
//
//  Applies x[i, k] *= inv_s[k]  in-place (or to out buffer).
//
//  Used after residual_add_rms_norm_noweight_fp16 / gate_silu_mul_merged_fp16
//  to undo the offline weight pre-scaling in AWQ-style calibration-aware FP4
//  quantization. Inverse scale is computed offline from activation statistics;
//  weight has been pre-multiplied by s[k], activation gets 1/s[k].
//
//  Result: x_scaled fed into quantize_fp4_dynamic_sfa_fp16.
//
//  Additive, zero modification to existing kernels.
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace flash_rt {
namespace fused_fp4 {

__global__ void kernel_per_channel_mul_fp16(
    __half* __restrict__ x,
    const __half* __restrict__ inv_s,
    int S, int D) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (row >= S || col >= D) return;
  __half s = inv_s[col];
  x[row * D + col] = __hmul(x[row * D + col], s);
}

void per_channel_mul_fp16(
    void* x, const void* inv_s, int S, int D, cudaStream_t stream) {
  const int threads = 256;
  dim3 grid((D + threads - 1) / threads, S);
  dim3 block(threads);
  kernel_per_channel_mul_fp16<<<grid, block, 0, stream>>>(
      reinterpret_cast<__half*>(x),
      reinterpret_cast<const __half*>(inv_s),
      S, D);
}

}  // namespace fused_fp4
}  // namespace flash_rt

extern "C" int flash_rt_per_channel_mul_fp16(
    uintptr_t x, uintptr_t inv_s, int S, int D, uintptr_t stream) {
  flash_rt::fused_fp4::per_channel_mul_fp16(
      reinterpret_cast<void*>(x),
      reinterpret_cast<const void*>(inv_s),
      S, D, reinterpret_cast<cudaStream_t>(stream));
  return 0;
}
