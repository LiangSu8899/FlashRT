// ============================================================================
//  FlashVLA — fp16 → NVFP4 (e2m1 + UE4M3 block scale) dynamic quantization.
//
//  Per-row, per-16-element-block. Used at runtime to quantize activation
//  tensors before FP4 GEMM. Weights go through the offline `Quant4` transform
//  in torch_weights.py which mirrors the same math.
//
//  Output layout (linear, row-major — NOT CUTLASS SFA/SFB tile-interleaved):
//    fp4_packed : uint8 [N, D/2]   — 2 e2m1 elements per byte
//                                    low nibble  = element 2*i
//                                    high nibble = element 2*i + 1
//    block_scales : fp8_e4m3 [N, D/16]  — UE4M3 scale per 16-element block
//                                         (positive value stored in signed e4m3)
//
//  If the CUTLASS mainloop wants tile-interleaved scale layout (SFA),
//  call `reshape_block_scales_to_sfa()` after this kernel at integration time.
//  Keeping the quantize kernel layout-agnostic makes it easier to unit-test
//  against a pytorch reference.
// ============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace fp4 {

// Launch parameters automatic: one thread per 16-element block per row.
// Preconditions: D % 16 == 0, N * D fits in int, tensors on device.
int quantize_fp4_dynamic_fp16(
    const void* src_fp16,          // [N, D] fp16 device ptr
    void* dst_fp4_packed,          // [N, D/2] uint8 device ptr
    void* dst_block_scales,        // [N, D/16] fp8_e4m3 device ptr (positive)
    int N, int D,
    cudaStream_t stream);

// Round-trip dequantize on device: used for unit tests / debugging.
// Reverses the quantize step into fp16 (lossy, matches pytorch fake_nvfp4).
int dequantize_fp4_to_fp16(
    const void* src_fp4_packed,    // [N, D/2]
    const void* src_block_scales,  // [N, D/16] fp8_e4m3
    void* dst_fp16,                // [N, D]
    int N, int D,
    cudaStream_t stream);

}  // namespace fp4
}  // namespace flash_rt
