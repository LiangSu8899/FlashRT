// Cosmos3-Reasoner single-query GQA decode attention (device-side length).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::kernels {

void cosmos3_reasoner_decode_attn_bf16(
    const __nv_bfloat16* q,        // [num_heads, 128]
    const __nv_bfloat16* k_cache,  // [max_seq, num_kv_heads, 128]
    const __nv_bfloat16* v_cache,  // [max_seq, num_kv_heads, 128]
    const int* len_ptr,            // device int32: valid prefix rows
    __nv_bfloat16* out,            // [num_heads, 128]
    float* part_acc,               // scratch [num_heads, 20, 128] f32
    float* part_ml,                // scratch [num_heads, 20, 2] f32
    int num_heads,
    int num_kv_heads,
    float scale,
    cudaStream_t stream);

void cosmos3_reasoner_rope_kv_bf16(
    const __nv_bfloat16* q_in,
    const __nv_bfloat16* k_in,
    const __nv_bfloat16* v_in,
    const __nv_bfloat16* cos_table,   // [max_seq, 128]
    const __nv_bfloat16* sin_table,
    const long long* pos_ptr,         // device scalar
    const long long* slot_ptr,        // device scalar
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_cache,           // [max_seq, num_kv_heads, 128]
    __nv_bfloat16* v_cache,
    int num_heads,
    int num_kv_heads,
    cudaStream_t stream);

void cosmos3_reasoner_decode_attn_fp8kv_bf16(
    const __nv_bfloat16* q,
    const __nv_fp8_e4m3* k_cache,
    const __nv_fp8_e4m3* v_cache,
    const int* len_ptr,
    __nv_bfloat16* out,
    float* part_acc,
    float* part_ml,
    int num_heads,
    int num_kv_heads,
    float scale,
    cudaStream_t stream);

void cosmos3_reasoner_rope_kv_fp8_bf16(
    const __nv_bfloat16* q_in,
    const __nv_bfloat16* k_in,
    const __nv_bfloat16* v_in,
    const __nv_bfloat16* cos_table,
    const __nv_bfloat16* sin_table,
    const long long* pos_ptr,
    const long long* slot_ptr,
    __nv_bfloat16* q_out,
    __nv_fp8_e4m3* k_cache,
    __nv_fp8_e4m3* v_cache,
    int num_heads,
    int num_kv_heads,
    cudaStream_t stream);

}  // namespace flash_rt::kernels
