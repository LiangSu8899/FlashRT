#pragma once

#include <cstddef>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

void flashrt_native_silu_inplace_bf16(
    __nv_bfloat16* values, int elements, cudaStream_t stream = nullptr);

cudaError_t flashrt_native_silu_inplace_fp16_precise(
    __half* values, std::size_t elements, cudaStream_t stream = nullptr);

void flashrt_native_fill_negative_infinity_f32(
    float* values, std::size_t count, cudaStream_t stream = nullptr);

void flashrt_native_generate_rope_table_f16(
    __half* output, int start_position, int positions,
    int frequencies, float theta, cudaStream_t stream = nullptr);

void flashrt_native_quantize_fp8_device_precise(
    const __nv_bfloat16* input, __nv_fp8_e4m3* output,
    float* device_scale, int elements, cudaStream_t stream = nullptr);

void flashrt_native_quantize_fp8_device_f16_precise(
    const __half* input, __nv_fp8_e4m3* output,
    float* device_scale, int elements, cudaStream_t stream = nullptr);

void flashrt_native_quantize_fp8_weight_bf16(
    const __nv_bfloat16* input, __nv_fp8_e4m3* output,
    float* device_scale, int elements, cudaStream_t stream = nullptr);

void flashrt_native_scale_bf16_weight(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    float scale, int elements, cudaStream_t stream = nullptr);

void flashrt_native_quantize_fp8_weight_f16(
    const __half* input, __nv_fp8_e4m3* output,
    float* device_scale, int rows, int columns, bool transpose,
    cudaStream_t stream = nullptr);

void flashrt_native_quantize_fp8_weight_f16_pair(
    const __half* first, const __half* second,
    __nv_fp8_e4m3* output, float* device_scale,
    int rows, int columns, bool transpose,
    cudaStream_t stream = nullptr);

void flashrt_native_layer_norm_fp8_f16(
    const __half* input, __nv_fp8_e4m3* output,
    const __half* gamma, const __half* beta,
    int rows, int columns, float epsilon,
    cudaStream_t stream = nullptr);

void flashrt_native_attention_qkv_fp16_seqused(
    cublasHandle_t handle,
    const __half* query, const __half* key, const __half* value,
    __half* logits, __half* output,
    int query_rows, int key_rows, int heads, int head_dimension,
    const int* seqused_key, float attention_scale,
    cudaStream_t stream = nullptr);

// Existing common operation used by the native frontend. This private
// declaration avoids expanding the installed common header surface.
void add_bias_bf16(__nv_bfloat16* values,
                   const __nv_bfloat16* bias,
                   int rows,
                   int columns,
                   cudaStream_t stream = nullptr);

void silu_inplace_fp16(
    __half* values, int elements, cudaStream_t stream = nullptr);
