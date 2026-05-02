// ================================================================
// FlashRT — Attention dispatch layer
//
// Unified interface for attention backends:
//   - FlashAttention-2/3 (SM89/SM120, pip install)
//   - CUTLASS FMHA (SM100/SM110, patched example 77)
//
// Backend selection is automatic based on SM version,
// or can be overridden at init time.
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

enum class FMHABackend {
    AUTO,           // Auto-detect based on SM version
    FLASH_ATTN,     // FlashAttention-2/3 (Python, pip install)
    CUTLASS_FMHA,   // CUTLASS example 77 (SM100/SM110, .so)
    CUBLAS_BATCH,   // cuBLAS batched matmul fallback
};

// ── Standard FMHA interface ──
// Q: (seq, num_heads, head_dim)  contiguous
// K: (seq_kv, num_heads, head_dim)  contiguous
// V: (seq_kv, num_heads, head_dim)  contiguous
// O: (seq, num_heads, head_dim)  contiguous
int fmha_forward(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    int seq_q, int seq_kv,
    int num_heads, int head_dim,
    float scale,
    cudaStream_t stream = 0);

// ── Strided FMHA interface (Thor optimization) ──
// Q/K/V read from interleaved QKV buffer with custom strides.
// Eliminates deinterleave kernel for SigLIP (saves ~0.5-1ms).
//
// qkv_buf: (seq, 3 * num_heads * head_dim) contiguous
//   Q starts at offset 0,    stride between tokens = 3 * num_heads * head_dim
//   K starts at offset D,    stride between tokens = 3 * num_heads * head_dim
//   V starts at offset 2*D,  stride between tokens = 3 * num_heads * head_dim
int fmha_strided_forward(
    const __nv_bfloat16* qkv_buf,  // interleaved QKV
    __nv_bfloat16* O,              // contiguous output
    int seq, int num_heads, int head_dim,
    float scale,
    cudaStream_t stream = 0);

// ── FMHA library management ──
// Load external FMHA .so (for CUTLASS FMHA on SM100/110)
int load_fmha_library(const char* path);
int load_fmha_strided_library(const char* path);

// Full strided FMHA: separate Q/K/V pointers + batch + strides
// Used by SigLIP multi-view (batch=NV, stride=3*D)
int fmha_strided_full(
    const void* Q, const void* K, const void* V, void* O,
    int batch, int seq_q, int seq_kv,
    int nheads_q, int nheads_kv, int head_dim,
    int stride_q, int stride_kv,
    cudaStream_t stream = 0);

// Check available backends
bool has_flash_attn();       // Python FlashAttention available?
bool has_cutlass_fmha();     // CUTLASS FMHA .so loaded?
FMHABackend get_active_backend();
