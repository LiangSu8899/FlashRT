// SPDX-License-Identifier: Apache-2.0
//
// Chunk-parallel causal conv1d update (K=4) with per-thread step
// batching and GQA split outputs. The packaged chunk-parallel kernel
// computes one (token, channel) per thread, so every input element is
// re-read K times from DRAM; batching STEPS consecutive tokens per
// thread rolls the taps through registers ((STEPS+K-1)/STEPS read
// amplification instead of K). Tap order and fp32 fma chain match the
// packaged kernel exactly — bit-exact outputs. Additive.
#pragma once
#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// x: (S, 10240) bf16 (the fixed 2048/2048/6144 q/k/v channel family,
// conv K=4). w: (10240, 4) bf16. state: (1, 10240, 3) bf16 (the last
// K-1 raw inputs). bias: (10240) bf16 or null. q16/k16: (S, 2048),
// v48: (S, 6144) bf16, silu applied. Returns 0 on success.
int causal_conv1d_update_steps_gqa_bf16(
    const void* x, const void* w, const void* bias, const void* state,
    void* q16, void* k16, void* v48, int S, bool apply_silu,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt
