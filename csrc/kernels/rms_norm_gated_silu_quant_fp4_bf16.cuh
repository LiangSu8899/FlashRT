// SPDX-License-Identifier: Apache-2.0
//
// Fused RMSNorm + weight + silu(gate) that also emits its consumer's
// NVFP4 input. The gated norm's output has exactly one consumer — the
// output projection — whose first act is to quantize it, and the
// quantizer's 16-element blocks tile a head's 128 lanes exactly, so
// the whole quantization fits inside the block that just produced the
// row. The norm arithmetic is transcribed unchanged from the packaged
// kernel; the quantize stage is the production path verbatim.
// Additive.
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// x, gate: (M, 128) bf16. weight: (128) bf16. out: (M, 128) bf16 —
// the normed rows, still written for hosts that read them. packed:
// (1, M*128/2) bytes; sfa: the 128x64-atom block for (1, M*128).
// dim must be 128. Returns 0 on success.
int rms_norm_gated_silu_quant_fp4_bf16(
    const void* x, const void* gate, const void* weight, void* out,
    void* packed, void* sfa, int M, int dim, float eps,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
