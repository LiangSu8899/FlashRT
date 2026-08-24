// FlashRT NVFP4 kernels for Jetson AGX Thor (SM110).
//
// C-style entry points implemented in the fr_*.cu translation units, which
// are compiled separately for sm_110a with CUTLASS. This header must stay
// free of CUTLASS and ggml includes so it can be consumed from the regular
// ggml-cuda translation units.
//
// Wire format (NVFP4):
//   packed:  uint8 [rows, K/2], adjacent-pair nibbles (elem 2i low, 2i+1 high)
//   scales:  e4m3 (positive/ue4m3), one per 16 elements along K, stored in
//            the CUTLASS Sm1xx block-scaled atom layout for the GEMM shape
//
// ggml's block_nvfp4 scale bytes carry standard e4m3 semantics (its dequant
// table doubles the e2m1 values and its ue4m3 decode halves the scale, which
// cancel), so they pass through to the GEMM unmodified with alpha = 1.0.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ggml_cuda_flashrt {

static inline int64_t round_up_i64(int64_t x, int64_t m) { return (x + m - 1) / m * m; }

// Scale-factor buffer size in bytes for a [rows, K] operand: the Sm1xx atom
// layout tiles rows in chunks of 128 and K/16 scale columns in chunks of 4.
static inline int64_t sf_bytes(int64_t rows, int64_t K) {
    return round_up_i64(rows, 128) * round_up_i64(K / 16, 4);
}

static inline int64_t packed_bytes(int64_t rows, int64_t K) {
    return rows * (K / 2);
}

// Block-scaled NVFP4 x NVFP4 GEMM, D = alpha * (A x B), fp32 output.
//   A_packed [M, K/2] row-major, B_packed [N, K/2] row-major (used as
//   column-major [K, N]), D fp32 [M, N] row-major.
// Requires K % 64 == 0, N % 16 == 0, D 16-byte aligned.
// widen selects a wide-N tile (use for N >= 8192).
// Returns 0 on success.
int gemm_f32out(const void * A_packed, const void * SFA,
                const void * B_packed, const void * SFB,
                float * D, int M, int N, int K,
                float alpha, bool widen, cudaStream_t stream);

// Quantize fp32 activations [M, K] row-major (contiguous) to NVFP4 packed
// [M, K/2] plus SFA scales in the atom layout for problem (M, x, K).
// Requires K % 16 == 0. Returns 0 on success.
int quantize_act_f32(const float * src, void * dst_packed, void * dst_sfa,
                     int M, int K, cudaStream_t stream);

// Repack a ggml GGML_TYPE_NVFP4 weight tensor [N rows, K] into the GEMM's
// B-side wire format: packed [N, K/2] with adjacent-pair nibbles plus SFB
// scales (ggml half-scale bytes, unmodified) in the atom layout.
// Requires K % 64 == 0. Returns 0 on success.
int repack_weight(const void * ggml_blocks, void * dst_packed, void * dst_sf,
                  int N, int K, cudaStream_t stream);

// Pairwise-interleave two ggml NVFP4 weight tensors (gate, up; each
// [n_ff rows, K]) into one B operand for the fused GeGLU GEMM: output row 2j
// is gate row j, row 2j+1 is up row j. dst_packed holds 2*n_ff rows; dst_sf
// is sized sf_bytes(2*n_ff, K).
int repack_weight_pair_interleaved(const void * gate_blocks, const void * up_blocks,
                                   void * dst_packed, void * dst_sf,
                                   int n_ff, int K, cudaStream_t stream);

// Rows-padded repack for the SigLIP FFN Up weight: rows >= N_src are zeros.
int repack_weight_rows_padded(const void * ggml_blocks, void * dst_packed, void * dst_sf,
                              int N_src, int N_pad, int K, cudaStream_t stream);

// Quantize an fp16 weight [N, K_src] to NVFP4 wire format, K zero-padded to K_pad.
int quantize_weight_f16_padded(const void * w_f16, void * dst_packed, void * dst_sf,
                               int N, int K_src, int K_pad, cudaStream_t stream);

// The SigLIP FFN GEMM pair and gemm_bias_f32out are declared in
// flashrt-public's cutlass_fp4_gemm_siglip_ffn_f32out_sm100.cuh
// (namespace flash_rt::fp4).

// Group-padded rows repack: n_groups groups widened group_in -> group_out
// rows, pad rows zero (used to widen per-head projections for FA head sizes).
int repack_weight_rows_grouppad(const void * ggml_blocks, void * dst_packed, void * dst_sf,
                                int group_in, int group_out, int n_groups, int K, cudaStream_t stream);

// Row-concat repack of three NVFP4 weights (shared K) for the fused QKV GEMM.
int repack_weight_concat3(const void * b0, int N0, const void * b1, int N1,
                          const void * b2, int N2,
                          void * dst_packed, void * dst_sf,
                          int K, cudaStream_t stream);

// Fused QKV post: RoPE+f16-store K, f16-store V (into the persistent KV
// suffix), RoPE+scale Q (f32 out) from the fused GEMM's [M, Nk+Nv+Nq] rows.
int qkv_post(const float * qkv_cat, float * q_out, void * k_out_f16, void * v_out_f16,
             const int32_t * pos, const float * freq_factors,
             int M, int Nk, int Nv, int Nq, int head_dim, int n_dims,
             float freq_scale, float ext_factor, float attn_factor,
             float corr_low, float corr_high, float theta_scale, float q_scale,
             cudaStream_t stream);

// Fused adaLN modulate: out[m,c] = norm(x[m])[c] * (1 + scale[c]) + shift[c],
// norm = rms-normalize when with_rms else identity. x/out are [M, C]
// contiguous fp32; scale/shift are [C] vectors.
int ada_rms_mod(const float * x, const float * scale, const float * shift,
                float * out, int M, int C, float eps, bool with_rms,
                cudaStream_t stream);

// Fused gated residual: out[m,c] = residual[m,c] + branch[m,c] * gate[c].
int gated_residual(const float * residual, const float * branch, const float * gate,
                   float * out, int M, int C, cudaStream_t stream);

// Quant-emitting variants: additionally write the result quantized to
// NVFP4 packed + SFA (atom layout for an [M, C] activation operand).
int ada_rms_mod_quant(const float * x, const float * scale, const float * shift,
                      float * out, void * dst_packed, void * dst_sfa,
                      int M, int C, float eps, bool with_rms, cudaStream_t stream);
int layer_norm_affine_quant(const float * x, const float * w, const float * b,
                            float * out, void * dst_packed, void * dst_sfa,
                            int M, int C, float eps, cudaStream_t stream);

// Fused LayerNorm + affine: out[m,c] = normalize(x[m])[c] * w[c] + b[c].
int layer_norm_affine(const float * x, const float * w, const float * b,
                      float * out, int M, int C, float eps, cudaStream_t stream);

// out[i] = a[i] + b[i] for n fp32 elements.
int vec_add_f32(const float * a, const float * b, float * out, int n, cudaStream_t stream);

} // namespace ggml_cuda_flashrt
