// ================================================================
// FlashRT — Elementwise kernel declarations
// Residual add, gate multiply, bias residual
// Supports: __half (FP16), __nv_bfloat16 (BF16)
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ── BF16 (original signatures, backward compatible) ──

void gate_mul_residual(__nv_bfloat16* residual, const __nv_bfloat16* x,
                       const __nv_bfloat16* gate, int n,
                       cudaStream_t stream = 0);

void bias_residual(__nv_bfloat16* residual, const __nv_bfloat16* x,
                   const __nv_bfloat16* bias, int seq_len, int dim,
                   cudaStream_t stream = 0);

void residual_add(__nv_bfloat16* residual, const __nv_bfloat16* x, int n,
                  cudaStream_t stream = 0);

// ── FP16 variants ──

void gate_mul_residual_fp16(__half* residual, const __half* x,
                            const __half* gate, int n,
                            cudaStream_t stream = 0);

void bias_residual_fp16(__half* residual, const __half* x,
                        const __half* bias, int seq_len, int dim,
                        cudaStream_t stream = 0);

void residual_add_fp16(__half* residual, const __half* x, int n,
                       cudaStream_t stream = 0);

// ── Classifier-Free Guidance combine ──
// In-place: noise[i] += v_uncond[i] + beta * (v_cond[i] - v_uncond[i])
//         = noise[i] + (1 - beta) * v_uncond[i] + beta * v_cond[i]
// Used by Pi05CFGPipeline per denoising step (arXiv:2511.14759 App. E).

void cfg_combine_into_residual(__nv_bfloat16* residual,
                               const __nv_bfloat16* v_cond,
                               const __nv_bfloat16* v_uncond,
                               float beta, int n,
                               cudaStream_t stream = 0);

void cfg_combine_into_residual_fp16(__half* residual,
                                    const __half* v_cond,
                                    const __half* v_uncond,
                                    float beta, int n,
                                    cudaStream_t stream = 0);
