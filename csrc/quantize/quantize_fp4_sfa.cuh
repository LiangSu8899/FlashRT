// ============================================================================
//  FlashRT — fused (FP4 quantize + CUTLASS SFA/SFB tile-interleave) kernel.
//
//  Equivalent to:
//      quantize_fp4_dynamic_fp16(src, packed, linear_scales, N, D)
//      reshape_linear_scales_to_sfa(linear_scales, sfa, N, D, is_sfb)
//  in a SINGLE kernel launch. Scale byte is written directly to the CUTLASS
//  tile-interleaved offset — linear_scales intermediate buffer is gone.
//
//  Additive: does NOT modify quantize_fp4_dynamic.* or reshape_scales_sfa.*.
//  Both remain callable for existing paths.
// ============================================================================
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// fp16 [N, D] → packed [N, D/2] (e2m1) + SFA/SFB tile-interleaved UE4M3 scales.
//   is_sfb = false  → SFA layout (use for A = activation, shape [M=N, K=D])
//   is_sfb = true   → SFB layout (use for B = weight,     shape [N=N, K=D])
// Returns 0 on success.
int quantize_fp4_dynamic_sfa_fp16(
    const void* src_fp16,
    void* dst_packed,
    void* dst_sfa,
    int N, int D, bool is_sfb,
    cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
