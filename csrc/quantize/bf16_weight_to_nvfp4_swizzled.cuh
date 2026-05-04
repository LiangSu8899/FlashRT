// SPDX-License-Identifier: Apache-2.0
//
// Single-shot BF16 weight tensor -> NVFP4 (swizzled SF + per-tensor
// global_scale) conversion. Sibling of fp8_block128_to_nvfp4_swizzled_bf16
// for tensors that arrive as BF16 (no FP8 block scales).
//
// Use case: the prithivMLmods Qwen3.6 NVFP4 checkpoint quantizes
// 70% of weights (full-attn + MLP) but leaves the linear-attention
// in_proj_qkv / in_proj_z / out_proj projections as BF16. Those
// 100+60+60 MB per layer × 48 lin layers = 9.6 GB extra BF16 weight
// reads per forward, which is ~2× the FP8 path's BW for the same
// projections. Quantizing them to NVFP4 at load time eliminates the
// gap (NVFP4 = 25+15+15 MB packed per layer).
//
// Two-launch design (same as fp8_block128_to_nvfp4_swizzled_bf16):
//   Pass 1: row-block atomicMax over |w_fp32|. Inputs are BF16 so
//           we promote to fp32 for amax (no precision floor).
//   Finalize: out_global_scale = global_amax / 2688.
//   Pass 2: per-NVFP4-block (16 elements) compute SF in UE4M3 with
//           proper division by global_scale, pack to e2m1, write
//           swizzled SF. Identical math to the FP8-input sibling.
//
// All add-only.

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace flash_rt {
namespace quantize {

void bf16_weight_to_nvfp4_swizzled(
    const __nv_bfloat16* w_bf16,
    uint8_t* nvfp4_packed,
    uint8_t* nvfp4_sf_swizzled,
    float* scratch_global_amax,
    float* out_global_scale,
    int N, int K,
    cudaStream_t stream);

}  // namespace quantize
}  // namespace flash_rt
