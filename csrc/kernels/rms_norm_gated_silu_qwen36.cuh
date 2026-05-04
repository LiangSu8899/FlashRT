// SPDX-License-Identifier: Apache-2.0
//
// Fused RMSNorm + weight + silu(gate) for Qwen3.6 linear-attention output.
// Replaces Qwen3_5RMSNormGated.forward (the slow Python path used when
// FusedRMSNormGated is not installed via causal_conv1d).
//
// Math (matching HF's exact dtype routing):
//   norm     = x * rsqrt(mean(x^2) + eps)        [fp32]
//   weighted = (weight * norm).to(bf16)          [bf16 mul]
//   silu_g   = gate / (1 + exp(-gate))           [fp32]
//   out      = (weighted.to(fp32) * silu_g).to(bf16)
//
// Each block handles one row (M); 128 threads cover dim (Qwen3.6
// head_v_dim = 128). Block-reduce sum-of-squares in shmem + warp shuffle.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// (M, dim) bf16 in / (M, dim) bf16 out. weight (dim,) bf16.
// Caller-provided output buffer; M and dim unrestricted (kernel
// templated for dim == 128 in the impl, can be extended).
void rms_norm_gated_silu_qwen36_bf16(
    const void* x,
    const void* gate,
    const void* weight,
    void*       out,
    int M, int dim, float eps,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
