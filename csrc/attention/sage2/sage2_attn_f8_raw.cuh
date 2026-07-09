#pragma once

#include <cuda_runtime.h>

// Additive fp8-QK attention entry points (qwen3 prefill).
//
// These adapt the in-tree SageAttention-2 SM120 core to fp8 (e4m3) QK^T while
// keeping the PV path in fp16 (the PV matmul dtype is NOT the bottleneck at the
// qwen3 prefill shape — int8/fp8/fp16 PV measured equal — so V stays high
// precision and needs no transpose/per-channel quant). The int8 sage2 kernels
// (motus) and the FA2 bf16 path (qwen3 fallback) are left fully intact.
//
//   q_fp8 / k_fp8 : e4m3 bytes, NHD contiguous [B, S, H*, 128]
//   v_fp16        : fp16, NHD contiguous [B, S, Hkv, 128]
//   q_scale       : per-warp  (WARP_Q=32) [B, Hq,  ceil(Lq/128)*4]
//   k_scale       : per-warp  (WARP_K=64) [B, Hkv, ceil(Lk/64)]
//   out_bf16      : [B, S, Hq, 128]
//   causal masking is always on (prefill).

namespace flash_rt::attention::sage2 {

int qk_f8_sv_f16_bf16_gqa_nhd_d128_causal(
    const void* q_fp8,
    const void* k_fp8,
    const void* v_fp16,
    void* out_bf16,
    const void* q_scale,
    const void* k_scale,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_q_heads,
    int num_kv_heads,
    float softmax_scale,
    bool per_token,
    cudaStream_t stream);

}  // namespace flash_rt::attention::sage2
