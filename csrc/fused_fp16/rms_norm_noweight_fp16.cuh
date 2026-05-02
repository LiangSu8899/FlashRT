// ============================================================================
//  FlashVLA — fp16-output RMSNorm kernels (additive, for FP4 frontend path).
//
//  Design intent:
//    The existing fp8-output RMSNorm kernels (norm.cu) fuse RMSNorm with
//    quantize-to-fp8-with-descale. For FP4 layers we want the RMSNorm result
//    in fp16 so the downstream quantize_fp4_dynamic_fp16 kernel can ingest it
//    without a lossy fp8 intermediate.
//
//    These kernels are a STRICT SUBSET of the fp8 versions: identical RMS
//    computation, but the output is fp16 * (rsqrt scale) — no descale division,
//    no fp8 cast, no saturation clamp. When the output is subsequently cast
//    to fp8 with the same descale factor, results must be bit-exact with the
//    existing fp8-out kernels (verified in test Phase 4.3 Step B).
//
//  Lives in a separate .so (flash_rt_fp4). Does NOT touch norm.cu.
// ============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace flash_rt {
namespace fused_fp16 {

// fp16 [R,C] → fp16 [R,C], output = x * rsqrt(mean(x^2) + eps).
// No descale, no quant. Drop-in replacement for the normalization step of
// rms_norm_fp8_noweight_fp16 minus the fp8 cast.
void rms_norm_noweight_fp16(const __half* x, __half* out,
                             int seq_len, int dim,
                             cudaStream_t stream);

// Residual + RMSNorm variant (mirrors residual_add_rms_norm_fp8_noweight_fp16
// minus the fp8 cast). Updates `residual` in place with residual += x, then
// writes normalized fp16 result to `out`.
void residual_add_rms_norm_noweight_fp16(__half* residual, const __half* x,
                                          __half* out,
                                          int seq_len, int dim,
                                          cudaStream_t stream);

}  // namespace fused_fp16
}  // namespace flash_rt
