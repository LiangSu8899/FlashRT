// FlashRT — MiniMax-Remover WanVAE NVFP4 conv3d fprop, sm_120a.
//
// Purpose-built NVFP4 (W4A4) implicit-GEMM conv3d for the WanVAE.
// Adapted from motus_fp4_conv3d_v19sfb with three WanVAE-specific optimizations:
//
//   1. **fp16 output** (not bf16): WanVAE is fp16-native; eliminates the
//      bf16→fp16 conversion that cost ~0.3 ms/layer in the Python wrapper.
//
//   2. **NDHWC output** (not NCDHW): matches the channels-last 3D pipeline;
//      eliminates the NCDHW→channels-last format conversion. The downstream
//      conv receives data in its preferred layout with zero-copy.
//
//   3. **fp16 bias** (not bf16): matches WanCausalConv3d.bias dtype.
//
// The compute path (MMA, im2col, cp.async pipeline, SF loading) is identical
// to the proven motus kernel. Only the epilogue changes.
//
// Tile: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, 8 warps, cp.async 2-stage.
// MMA: mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
//
// Constraints: Ci % 64 == 0, T_cache == 2, kernel = 3×3×3, pad = 1.
// (WanVAE Ci=192/384 qualify; Ci=96 stays on FP8.)
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

extern "C" int nvfp4_conv3d_ndhwc_fp16out(
    const void*  cache_x_fp4,   // [B,T_cache,H,W,Ci/2] uint8 packed FP4
    const void*  new_x_fp4,     // [B,T_new,H,W,Ci/2]  uint8 packed FP4
    const void*  w_fp4,         // [Co,3,3,3,Ci/2]     uint8 packed FP4
    const void*  cache_sfa,     // [B,T_cache,H,W,Ci/16] uint8 UE4M3
    const void*  new_sfa,       // [B,T_new,H,W,Ci/16]  uint8 UE4M3
    const void*  w_sfb,         // [Co,3,3,3,Ci/16]     uint8 UE4M3
    void*        y_fp16,        // [B,T_new,H,W,Co] __half NDHWC output
    const void*  bias_fp16,     // [Co] __half, or nullptr
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream);

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt
