/*
 * FlashRT additive fp8-QK / fp16-PV attention wrapper (qwen3 prefill).
 *
 * Source basis: SageAttention qk_int_sv_f16 core, Apache-2.0. This file only
 * ADDS a new fp8-QK entry point; the int8 sage2 kernels and FA2 are untouched.
 *
 * Shape contract: NHD contiguous Q/K/V/O, GQA (num_kv_groups = Hq/Hkv), causal,
 * head_dim = 128, BF16 output. Q/K are e4m3 (passed as int8 bytes), V is fp16.
 */

#include "sage2_attn_f8_raw.cuh"

#include <algorithm>
#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <mutex>

#include "qattn/qk_f8_sv_f16_core.cuh"

namespace flash_rt::attention::sage2 {
namespace {

constexpr int kHeadDim = 128;
constexpr int kCtaQ = 128;  // 128 beat 64 in-graph (4 warps/CTA hide cold-L2
constexpr int kCtaK = 64;   // latency better than the extra CTAs of CTA_Q=64).
constexpr int kWarpQ = 32;
constexpr int kWarpK = 64;

inline int div_up_int(int x, int y) { return (x + y - 1) / y; }

}  // namespace

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
    cudaStream_t stream) {
  if (!q_fp8 || !k_fp8 || !v_fp16 || !out_bf16 || !q_scale || !k_scale) {
    return -1;
  }
  if (batch <= 0 || seqlen_q <= 0 || seqlen_k <= 0 ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0) {
    return -2;
  }

  const int num_kv_groups = num_q_heads / num_kv_heads;

  const uint32_t stride_bz_q = static_cast<uint32_t>(seqlen_q * num_q_heads * kHeadDim);
  const uint32_t stride_seq_q = static_cast<uint32_t>(num_q_heads * kHeadDim);
  const uint32_t stride_h_q = static_cast<uint32_t>(kHeadDim);

  const uint32_t stride_bz_k = static_cast<uint32_t>(seqlen_k * num_kv_heads * kHeadDim);
  const uint32_t stride_seq_k = static_cast<uint32_t>(num_kv_heads * kHeadDim);
  const uint32_t stride_h_k = static_cast<uint32_t>(kHeadDim);

  // V: NHD fp16 [B, S, Hkv, D].
  const uint32_t stride_bz_v = static_cast<uint32_t>(seqlen_k * num_kv_heads * kHeadDim);
  const uint32_t stride_seq_v = static_cast<uint32_t>(num_kv_heads * kHeadDim);
  const uint32_t stride_h_v = static_cast<uint32_t>(kHeadDim);

  const uint32_t stride_bz_o = static_cast<uint32_t>(seqlen_q * num_q_heads * kHeadDim);
  const uint32_t stride_seq_o = static_cast<uint32_t>(num_q_heads * kHeadDim);
  const uint32_t stride_h_o = static_cast<uint32_t>(kHeadDim);

  using Kernel = decltype(&qk_f8_sv_f16_attn_kernel<
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim,
      DataType::kInt8,                 // sizing only (8-bit, k=32) — correct for fp8
      QuantGranularity::kPerWarp,
      QuantGranularity::kPerWarp,
      float,                           // DTypeSVAccum
      false,                           // use_inst_buffer
      nv_bfloat16,                     // DTypeOut
      ComputeUnit::kTensorCore,        // denominator accum
      MaskMode::kCausal,
      false,                           // return_lse
      false,                           // fuse_v_mean
      false>);                         // PER_TOKEN

  // PER_TOKEN=true uses per-(token,head) q_scale[B,Hq,Lq] / k_scale[B,Hkv,Lk]
  // and applies the dequant per score entry; false uses the per-warp scalars.
  Kernel kernel = per_token
      ? qk_f8_sv_f16_attn_kernel<
            kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim, DataType::kInt8,
            QuantGranularity::kPerWarp, QuantGranularity::kPerWarp,
            float, false, nv_bfloat16, ComputeUnit::kTensorCore,
            MaskMode::kCausal, false, false, true>
      : qk_f8_sv_f16_attn_kernel<
            kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim, DataType::kInt8,
            QuantGranularity::kPerWarp, QuantGranularity::kPerWarp,
            float, false, nv_bfloat16, ComputeUnit::kTensorCore,
            MaskMode::kCausal, false, false, false>;

  const size_t smem_qkv =
      static_cast<size_t>(kCtaQ * kHeadDim * sizeof(int8_t) +
                          kCtaK * kHeadDim * sizeof(int8_t) +
                          kCtaK * kHeadDim * sizeof(half));
  const size_t smem_o = static_cast<size_t>(kCtaQ * kHeadDim * sizeof(half));
  const size_t smem_max = std::max(smem_qkv, smem_o);
  static std::once_flag attr_once;
  std::call_once(attr_once, [&]() {
    // set for both PER_TOKEN instantiations (smem 40KB < 48KB default, but
    // explicit keeps parity with the int8 wrappers and is future-proof).
    cudaFuncSetAttribute(
        qk_f8_sv_f16_attn_kernel<
            kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim, DataType::kInt8,
            QuantGranularity::kPerWarp, QuantGranularity::kPerWarp,
            float, false, nv_bfloat16, ComputeUnit::kTensorCore,
            MaskMode::kCausal, false, false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_max));
    cudaFuncSetAttribute(
        qk_f8_sv_f16_attn_kernel<
            kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim, DataType::kInt8,
            QuantGranularity::kPerWarp, QuantGranularity::kPerWarp,
            float, false, nv_bfloat16, ComputeUnit::kTensorCore,
            MaskMode::kCausal, false, false, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_max));
  });

  dim3 grid(div_up_int(seqlen_q, kCtaQ), num_q_heads, batch);
  dim3 block(32, (kCtaQ / kWarpQ) * (kCtaK / kWarpK));

  kernel<<<grid, block, smem_max, stream>>>(
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(q_fp8)),
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(k_fp8)),
      const_cast<half*>(reinterpret_cast<const half*>(v_fp16)),
      reinterpret_cast<nv_bfloat16*>(out_bf16),
      nullptr,
      const_cast<float*>(reinterpret_cast<const float*>(q_scale)),
      const_cast<float*>(reinterpret_cast<const float*>(k_scale)),
      nullptr,
      static_cast<uint32_t>(seqlen_q),
      static_cast<uint32_t>(seqlen_k),
      static_cast<uint32_t>(num_kv_groups),
      stride_bz_q, stride_seq_q, stride_h_q,
      stride_bz_k, stride_seq_k, stride_h_k,
      stride_bz_v, stride_seq_v, stride_h_v,
      stride_bz_o, stride_seq_o, stride_h_o,
      softmax_scale);

  return static_cast<int>(cudaGetLastError());
}

}  // namespace flash_rt::attention::sage2
