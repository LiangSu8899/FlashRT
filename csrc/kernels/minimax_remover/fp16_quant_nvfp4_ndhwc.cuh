// FlashRT — MiniMax-Remover WanVAE NVFP4 fused quantization kernels.
//
// Two kernels:
//   1. fp16_rms_silu_quant_nvfp4_ndhwc:
//        Fused RMS_norm + SiLU + NVFP4 quantization.
//        fp16 NCDHW [B,C,T,H,W] → FP4 packed [B,T,H,W,C/2] + UE4M3 SF [B,T,H,W,C/16]
//        Eliminates 3 separate passes (norm, silu, quant) into one kernel.
//
//   2. fp16_quant_nvfp4_ndhwc:
//        Plain NVFP4 quantization (no norm/silu).
//        fp16 NCDHW [B,C,T,H,W] → FP4 packed + UE4M3 SF (NDHWC layout).
//        Used for causal-conv cache quantization.
//
// Output format matches motus_fp4_conv3d_v19sfb kernel input requirements:
//   - FP4 data:  [B,T,H,W, C/2]  uint8  (2 e2m1 values packed per byte)
//   - SF data:   [B,T,H,W, C/16] uint8  (UE4M3 block scale, 1 per 16 elements)
//
// Thread layout: kThreadsX=32 (one per W position), kThreadsY=6
//   (chosen so C/6 is always a multiple of 16 for WanVAE channels 96/192/384:
//    96/6=16, 192/6=32, 384/6=64 — all clean SF-block multiples).
//
// Precision: RMS statistics accumulated in fp32 (bit-exact with WanRMS_norm).
// SiLU computed in fp32 (NOT rounded to bf16 like the motus variant — preserves
// fp16's 10-bit mantissa through the activation). FP4 quantization uses
// per-16-element UE4M3 block scales (same as NVFP4 hardware format).
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

// Fused: fp16 NCDHW → RMS_norm(+bias) → SiLU → NVFP4 quant (NDHWC out).
// Returns 0 on success, negative on error.
extern "C" int fp16_rms_silu_quant_nvfp4_ndhwc(
    const void*  x_fp16,         // [B,C,T,H,W] __half
    const void*  gamma_fp16,     // [C] __half
    const void*  bias_fp16,      // [C] __half, or nullptr
    void*        y_fp4,          // [B,T,H,W, C/2] uint8 packed
    void*        y_sf,           // [B,T,H,W, C/16] uint8 UE4M3
    int B, int C, int T, int H, int W,
    float eps, cudaStream_t stream);

// Plain: fp16 NCDHW → NVFP4 quant (NDHWC out). No norm/silu.
extern "C" int fp16_quant_nvfp4_ndhwc(
    const void*  x_fp16,         // [B,C,T,H,W] __half NCDHW contiguous
    void*        y_fp4,          // [B,T,H,W, C/2] uint8 packed
    void*        y_sf,           // [B,T,H,W, C/16] uint8 UE4M3
    int B, int C, int T, int H, int W,
    cudaStream_t stream);

// Channels-last variant: fp16 channels-last 3D → NVFP4 quant (NDHWC out).
// Eliminates the contiguous() copy when input is already channels-last.
extern "C" int fp16_quant_nvfp4_cl_ndhwc(
    const void*  x_fp16,         // [B,C,T,H,W] __half channels-last 3D
    void*        y_fp4,          // [B,T,H,W, C/2] uint8 packed
    void*        y_sf,           // [B,T,H,W, C/16] uint8 UE4M3
    int B, int C, int T, int H, int W,
    cudaStream_t stream);

// Fused RMS+SiLU+NVFP4 quant, channels-last input.
extern "C" int fp16_rms_silu_quant_nvfp4_cl_ndhwc(
    const void*  x_fp16,         // [B,C,T,H,W] __half channels-last 3D
    const void*  gamma_fp16,     // [C] __half
    const void*  bias_fp16,      // [C] __half, or nullptr
    void*        y_fp4,          // [B,T,H,W, C/2] uint8 packed
    void*        y_sf,           // [B,T,H,W, C/16] uint8 UE4M3
    int B, int C, int T, int H, int W,
    float eps, cudaStream_t stream);

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt
