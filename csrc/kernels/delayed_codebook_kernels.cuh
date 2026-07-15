#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::kernels {

void delayed_codebook_argmax_embed_bf16(
    const __nv_bfloat16* logits,
    const __nv_bfloat16* codebook,
    int64_t* codes_out,
    __nv_bfloat16* embed_out,
    int num_codebooks,
    int codebook_vocab,
    int hidden,
    int delay,
    int boc,
    cudaStream_t stream);

void delayed_codebook_sample_embed_bf16(
    const __nv_bfloat16* logits,
    const __nv_bfloat16* codebook,
    int64_t* codes_out,
    __nv_bfloat16* embed_out,
    int num_codebooks,
    int codebook_vocab,
    int hidden,
    int delay,
    int boc,
    float temperature,
    uint64_t seed,
    uint64_t step,
    cudaStream_t stream);

}  // namespace flash_rt::kernels
