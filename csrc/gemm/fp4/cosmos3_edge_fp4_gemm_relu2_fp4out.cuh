#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// Cosmos3-Edge up projection with ReLU(x)^2 and NVFP4 output fused into the
// CUTLASS epilogue. The output is directly consumable by the down projection.
int cosmos3_edge_fp4_gemm_relu2_fp4out(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void* D_packed, void* D_SFD,
    int M, int N, int K,
    cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
