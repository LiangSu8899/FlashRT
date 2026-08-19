// SPDX-License-Identifier: Apache-2.0
//
// Fused (1+w)-form RMSNorm + NVFP4 quantize producer. Computes the
// host norm exactly — y = x * rsqrt(mean(x^2) + eps) * (1 + w), all in
// fp32, one bf16 round at the write — and quantizes the same values
// with the production per-16-block amax/6 scale selection, e2m1
// rounding table and 128x64 SFA layout. Emits BOTH the normed bf16 row
// (for hosts that still read it) and the packed FP4 + SFA the bound
// projections consume, off one read of x. Additive.
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// x: (N, D) bf16 row-major; w: (D) bf16 (the residual-form weight, the
// kernel applies 1+w). normed: (N, D) bf16. packed: (N, D/2) bytes.
// sfa: ((N+127)/128)*((D+63)/64)*512 bytes. D%16==0, D<=8192.
int rms_norm_quantize_fp4_sfa_bf16(
    const void* x_bf16, const void* w_bf16, float eps, void* normed_bf16,
    void* dst_packed, void* dst_sfa, int N, int D, cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
