// ggml-facing interface of the FlashRT NVFP4 path (Thor SM110).
// Included from ggml-cuda.cu under #ifdef GGML_CUDA_FLASHRT.
#pragma once

// Host dependency: ggml-cuda's internal common header. The consuming build
// must put ggml/src/ggml-cuda on the include path (the ggml adapter is
// compiled inside the host's build tree, like the vllm/sglang adapters run
// inside their host's runtime).
#include "common.cuh"

// Called at the start of every backend graph evaluation; invalidates the
// per-evaluation quantized-activation cache.
void ggml_cuda_flashrt_begin_eval();

// True when this mul_mat should be routed to the FlashRT block-scaled NVFP4
// GEMM: NVFP4 weights, fp32 contiguous activations/dst, no batch dims,
// shapes within kernel alignment. cc must already be checked by the caller.
bool ggml_cuda_flashrt_should_use(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// dst = src1 @ src0 via activation quantize + NVFP4 x NVFP4 tcgen05 GEMM.
// Weights are repacked into the CUTLASS wire format on first use and cached
// for the process lifetime.
void ggml_cuda_flashrt_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// True when the 4-node FFN subgraph {mul_mat gate, mul_mat up, GEGLU,
// mul_mat down} can run as one fused GeGLU GEMM (interleaved gate/up
// weights, FP4 intermediate) followed by the down GEMM.
bool ggml_cuda_flashrt_should_fuse_geglu(const ggml_tensor * gate_mm, const ggml_tensor * up_mm, const ggml_tensor * glu, const ggml_tensor * down_mm);

// Execute that fused FFN; writes the down mul_mat's dst.
void ggml_cuda_flashrt_geglu_ffn(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_mm, const ggml_tensor * up_mm, const ggml_tensor * glu, ggml_tensor * down_mm);

// pi0.5 adaLN modulate window: {rms_norm?, mul_mat mod, add bias, view,
// repeat, mul, add, view, repeat, add}. rms is null for the variant whose
// normalized input arrives from a previous graph split. Outputs written by
// the fused execution: the bias add (consumed by the gate view later) and
// the final add.
bool ggml_cuda_flashrt_should_fuse_ada(const ggml_tensor * rms, const ggml_tensor * mm, const ggml_tensor * bias_add,
                                       const ggml_tensor * view_scale, const ggml_tensor * repeat_scale,
                                       const ggml_tensor * mul, const ggml_tensor * add1,
                                       const ggml_tensor * view_shift, const ggml_tensor * repeat_shift,
                                       const ggml_tensor * add2);
void ggml_cuda_flashrt_ada_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mm,
                                ggml_tensor * bias_add, const ggml_tensor * view_scale, const ggml_tensor * mul,
                                const ggml_tensor * view_shift, ggml_tensor * add2);

// LayerNorm + affine window: {NORM, MUL weight, ADD bias} -> one kernel.
bool ggml_cuda_flashrt_should_fuse_ada_cached(
        const ggml_tensor * rms, const ggml_tensor * view_col,
        const ggml_tensor * view_scale, const ggml_tensor * repeat_scale,
        const ggml_tensor * mul, const ggml_tensor * add1,
        const ggml_tensor * view_shift, const ggml_tensor * repeat_shift,
        const ggml_tensor * add2);
void ggml_cuda_flashrt_ada_norm_cached(ggml_backend_cuda_context & ctx, const ggml_tensor * rms,
                                       const ggml_tensor * view_scale, const ggml_tensor * view_shift,
                                       ggml_tensor * add2);

bool ggml_cuda_flashrt_should_fuse_ln(const ggml_tensor * norm, const ggml_tensor * mul, const ggml_tensor * add);
void ggml_cuda_flashrt_ln_affine(ggml_backend_cuda_context & ctx, const ggml_tensor * norm, const ggml_tensor * mul, ggml_tensor * add);

// SigLIP FFN window: {mul_mat up (NVFP4), add bias, GELU, cont, mul_mat
// down (f16), add bias, cont, add residual} -> fused FP4 Up GEMM (gelu
// epilogue, FP4 hidden) + Down GEMM (bias + residual epilogue).
bool ggml_cuda_flashrt_should_fuse_siglip_ffn(const ggml_tensor * up_mm, const ggml_tensor * bias1, const ggml_tensor * gelu,
                                              const ggml_tensor * cont1, const ggml_tensor * dn_mm, const ggml_tensor * bias2,
                                              const ggml_tensor * cont2, const ggml_tensor * res_add);
void ggml_cuda_flashrt_siglip_ffn(ggml_backend_cuda_context & ctx, const ggml_tensor * up_mm, const ggml_tensor * bias1,
                                  const ggml_tensor * dn_mm, const ggml_tensor * bias2,
                                  const ggml_tensor * cont2, ggml_tensor * res_add);

// pi0.5 AE fused QKV window: {mm k, reshape, rope, view, cpy, mm v,
// reshape, view, cpy, mm q, reshape, rope, scale} -> one fused GEMM over
// row-concatenated [k|v|q] weights + one post kernel (rope/scale/f16 KV
// suffix stores).
bool ggml_cuda_flashrt_should_fuse_qkv(const ggml_tensor * k_mm, const ggml_tensor * k_rope, const ggml_tensor * k_cpy,
                                       const ggml_tensor * v_mm, const ggml_tensor * v_cpy,
                                       const ggml_tensor * q_mm, const ggml_tensor * q_rope, const ggml_tensor * q_scale);
void ggml_cuda_flashrt_qkv(ggml_backend_cuda_context & ctx,
                           const ggml_tensor * k_mm, const ggml_tensor * k_rope, const ggml_tensor * k_cpy,
                           const ggml_tensor * v_mm, const ggml_tensor * v_cpy,
                           const ggml_tensor * q_mm, const ggml_tensor * q_rope, ggml_tensor * q_scale);

// pi0.5 gated residual window: {view gate, repeat, mul, add}.
// Vision QKV pad window: {mul_mat, add bias, reshape}x3 + {pad}x3 -> three
// padded-weight GEMMs (bias in epilogue) writing the pad buffers directly.
bool ggml_cuda_flashrt_should_fuse_vis_qkv_pad(
        const ggml_tensor * mm_q, const ggml_tensor * add_q, const ggml_tensor * resh_q,
        const ggml_tensor * mm_k, const ggml_tensor * add_k, const ggml_tensor * resh_k,
        const ggml_tensor * mm_v, const ggml_tensor * add_v, const ggml_tensor * resh_v,
        const ggml_tensor * pad_q, const ggml_tensor * pad_k, const ggml_tensor * pad_v);
void ggml_cuda_flashrt_vis_qkv_pad(ggml_backend_cuda_context & ctx,
        const ggml_tensor * mm_q, const ggml_tensor * add_q, ggml_tensor * pad_q,
        const ggml_tensor * mm_k, const ggml_tensor * add_k, ggml_tensor * pad_k,
        const ggml_tensor * mm_v, const ggml_tensor * add_v, ggml_tensor * pad_v);

// GEMM + optional bias + residual add fused into one epilogue.
bool ggml_cuda_flashrt_should_fuse_mm_res(const ggml_tensor * mm, const ggml_tensor * bias_add,
                                          const ggml_tensor * res_add);
bool ggml_cuda_flashrt_mm_res(ggml_backend_cuda_context & ctx, const ggml_tensor * mm,
                              const ggml_tensor * bias_add, ggml_tensor * res_add);

bool ggml_cuda_flashrt_should_fuse_gated_res(const ggml_tensor * view, const ggml_tensor * repeat,
                                             const ggml_tensor * mul, const ggml_tensor * add);
void ggml_cuda_flashrt_gated_residual(ggml_backend_cuda_context & ctx, const ggml_tensor * view,
                                      const ggml_tensor * mul, ggml_tensor * add);
