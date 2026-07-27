// Fused LayerNorm (gamma/beta) + NVFP4 quantize + CUTLASS SFA write.
// Bit-identical to a fp16 LayerNorm followed by
// quantize_fp4_dynamic_sfa_fp16. Returns nonzero without launching on
// unaligned buffers or dim % 16 != 0.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace flash_rt {
namespace fused_fp4 {

int layer_norm_fp4_sfa_fp16(
    const __half* x, const __half* gamma, const __half* beta,
    void* packed, void* sfa,
    int seq_len, int dim, float eps, cudaStream_t stream);

}  // namespace fused_fp4
}  // namespace flash_rt
