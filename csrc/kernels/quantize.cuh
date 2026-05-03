// ================================================================
// FlashRT — Quantization kernel declarations
// FP8 dynamic/static quantize, NVFP4 block-scaled (SM120+)
// FP8 functions support: __half (FP16), __nv_bfloat16 (BF16) input
// ================================================================
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ── BF16 (original signatures, backward compatible) ──

// FP8 quantize with host sync (NOT CUDA Graph compatible)
float quantize_fp8(const __nv_bfloat16* input, __nv_fp8_e4m3* output,
                   float* d_scale, int n, cudaStream_t stream = 0);

// FP8 quantize with pre-computed static scale (CUDA Graph compatible)
void quantize_fp8_static(const __nv_bfloat16* input, __nv_fp8_e4m3* output,
                         const float* d_scale, int n, cudaStream_t stream = 0);

// FP8 quantize device-only: scale computed on device (CUDA Graph compatible)
void quantize_fp8_device(const __nv_bfloat16* input, __nv_fp8_e4m3* output,
                         float* d_scale, int n, cudaStream_t stream = 0);

// ── FP16 variants ──

float quantize_fp8_fp16(const __half* input, __nv_fp8_e4m3* output,
                        float* d_scale, int n, cudaStream_t stream = 0);

void quantize_fp8_static_fp16(const __half* input, __nv_fp8_e4m3* output,
                               const float* d_scale, int n, cudaStream_t stream = 0);

void quantize_fp8_device_fp16(const __half* input, __nv_fp8_e4m3* output,
                               float* d_scale, int n, cudaStream_t stream = 0);

// ── L2 weight prefetch ──

void prefetch_l2(const void* data, size_t num_bytes, cudaStream_t stream = 0);

// ── NVFP4 (BF16-only, SM120+) ──

#ifdef ENABLE_NVFP4
void quantize_bf16_to_nvfp4(const __nv_bfloat16* input, uint8_t* fp4_data,
                              uint8_t* scale_factors, int rows, int cols,
                              cudaStream_t stream = 0);

void quantize_bf16_to_nvfp4_swizzled(const __nv_bfloat16* input, uint8_t* fp4_data,
                                       uint8_t* scale_factors, int rows, int cols,
                                       cudaStream_t stream = 0);

void quantize_bf16_to_mxfp8(const __nv_bfloat16* input, __nv_fp8_e4m3* fp8_data,
                              uint8_t* scale_factors, int rows, int cols,
                              cudaStream_t stream = 0);

int get_mxfp8_sf_size(int rows, int cols);

void quantize_bf16_to_mxfp4_cutlass(const __nv_bfloat16* input, uint8_t* fp4_data,
                                      uint8_t* scale_factors, int N, int K,
                                      cudaStream_t stream = 0);

int get_mxfp4_sf_size(int N, int K);
#endif
