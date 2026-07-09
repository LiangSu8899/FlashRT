#ifndef FLASHRT_ATTENTION_FA2_WRAPPER_H
#define FLASHRT_ATTENTION_FA2_WRAPPER_H

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

void fvk_attention_fa2_fwd_fp16(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream);

void fvk_attention_fa2_fwd_bf16(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream);

void fvk_attention_fa2_fwd_bf16_seqused(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr, const void* seqused_k_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream);

void fvk_attention_fa2_fwd_bf16_seqused_splitkv(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr, const void* seqused_k_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream);

void fvk_attention_fa2_fwd_bf16_causal(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* o_ptr, void* softmax_lse_ptr,
    void* softmax_lse_accum_ptr, void* o_accum_ptr,
    int batch, int seqlen_q, int seqlen_k,
    int num_heads_q, int num_heads_kv, int head_dim,
    int q_batch_stride, int q_row_stride, int q_head_stride,
    int k_batch_stride, int k_row_stride, int k_head_stride,
    int v_batch_stride, int v_row_stride, int v_head_stride,
    int o_batch_stride, int o_row_stride, int o_head_stride,
    float softmax_scale, int num_sms, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // FLASHRT_ATTENTION_FA2_WRAPPER_H
