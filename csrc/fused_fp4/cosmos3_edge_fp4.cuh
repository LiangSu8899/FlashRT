// Cosmos3-Edge model-specific fused NVFP4 quant kernels (additive).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace fused_fp4 {

void cosmos3_edge_res_rms_fp4_sfa_bf16(
    __nv_bfloat16* residual, const __nv_bfloat16* x, const __nv_bfloat16* weight,
    uint8_t* packed, uint8_t* sfa,
    int seq_len, int dim, float eps, cudaStream_t stream);

void cosmos3_edge_relu2_fp4_sfa_fp16(
    const __half* x, uint8_t* packed, uint8_t* sfa,
    int seq_len, int dim, cudaStream_t stream);

}  // namespace fused_fp4
}  // namespace flash_rt
