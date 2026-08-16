// SPDX-License-Identifier: Apache-2.0
//
// Fused SwiGLU activation + NVFP4 quantize producer: silu(gate)·up
// computed in fp32, rounded through bf16 (mirroring the elementwise
// producer it replaces), then quantized with the exact per-16-block
// amax/6 scale selection, e2m1 rounding table and 128x64 SFA layout of
// the production quantize kernel — bit-exact against that chain by
// construction. Additive.
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace fp4 {

// merged_bf16: (N, 2H) row-major, halves ordered [gate | up]; H%16==0.
// dst_packed: (N, H/2) bytes. dst_sfa: the 128x64-atom SFA block for
// (N, H), ((N+127)/128)*((H+63)/64)*512 bytes. Returns 0 on success.
int silu_mul_quantize_fp4_sfa_bf16(
    const void* merged_bf16, void* dst_packed, void* dst_sfa,
    int N, int H, cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
