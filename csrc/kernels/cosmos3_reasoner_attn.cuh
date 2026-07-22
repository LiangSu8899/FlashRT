// Cosmos3-Reasoner single-query GQA decode attention (device-side length).
#pragma once

#include <cuda_bf16.h>
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

}  // namespace flash_rt::kernels
