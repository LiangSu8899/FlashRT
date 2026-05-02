// ================================================================
// FlashRT — Attention dispatch implementation
//
// Manages FMHA backends (FlashAttention / CUTLASS FMHA / cuBLAS)
// with dynamic library loading for CUTLASS FMHA on SM100/110.
// ================================================================

#include "fmha_dispatch.h"
#include <cstdio>
#include <cublasLt.h>

// ── Cross-platform dynamic library loading ──
// Linux: dlopen/dlsym/dlclose (POSIX).
// Windows: LoadLibraryA/GetProcAddress/FreeLibrary (Win32).
// The wrapper keeps the rest of the file dlopen-style and lets the
// SM100/SM110 CUTLASS FMHA path stay available on Windows builds —
// callers just have to pass a .dll path instead of a .so.
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  using DynLibHandle = HMODULE;
  static DynLibHandle dyn_open(const char* path)                  { return LoadLibraryA(path); }
  static void*        dyn_sym(DynLibHandle h, const char* name)   { return reinterpret_cast<void*>(GetProcAddress(h, name)); }
  static void         dyn_close(DynLibHandle h)                   { if (h) FreeLibrary(h); }
  static const char*  dyn_error()                                 { return "LoadLibrary/GetProcAddress failed (Win32 GetLastError available via system error code)"; }
#else
  #include <dlfcn.h>
  using DynLibHandle = void*;
  static DynLibHandle dyn_open(const char* path)                  { return dlopen(path, RTLD_LAZY); }
  static void*        dyn_sym(DynLibHandle h, const char* name)   { return dlsym(h, name); }
  static void         dyn_close(DynLibHandle h)                   { if (h) dlclose(h); }
  static const char*  dyn_error()                                 { return dlerror(); }
#endif

// ── Function pointer types for dlopen'd FMHA ──
typedef int (*fmha_fn_t)(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, int, int, int, int, int, cudaStream_t);

// Strided FMHA: Q/K/V from interleaved buffer with custom stride
// Signature matches libfmha_fp16_strided.so: fmha_fp16_strided(
//   Q, K, V, O, batch, seq_q, seq_kv, nheads_q, nheads_kv, head_dim, stride_q, stride_kv, stream)
typedef int (*fmha_strided_fn_t)(
    const void*, const void*, const void*, void*,
    int, int, int, int, int, int, int, int, void*);

// ── Global state ──
static DynLibHandle g_fmha_lib = nullptr;
static DynLibHandle g_fmha_strided_lib = nullptr;
static fmha_fn_t g_fmha_fn = nullptr;
static fmha_strided_fn_t g_fmha_strided_fn = nullptr;
static FMHABackend g_active_backend = FMHABackend::AUTO;

// ── Library loading ──

int load_fmha_library(const char* path) {
    g_fmha_lib = dyn_open(path);
    if (!g_fmha_lib) {
        fprintf(stderr, "[FlashRT] Failed to load FMHA library: %s\n  %s\n",
                path, dyn_error());
        return -1;
    }
    g_fmha_fn = (fmha_fn_t)dyn_sym(g_fmha_lib, "fmha_fp16_attn");
    if (!g_fmha_fn) {
        fprintf(stderr, "[FlashRT] Symbol 'fmha_fp16_attn' not found in %s\n", path);
        return -1;
    }
    g_active_backend = FMHABackend::CUTLASS_FMHA;
    return 0;
}

int load_fmha_strided_library(const char* path) {
    g_fmha_strided_lib = dyn_open(path);
    if (!g_fmha_strided_lib) {
        fprintf(stderr, "[FlashRT] Failed to load strided FMHA library: %s\n  %s\n",
                path, dyn_error());
        return -1;
    }
    g_fmha_strided_fn = (fmha_strided_fn_t)dyn_sym(g_fmha_strided_lib, "fmha_fp16_strided");
    if (!g_fmha_strided_fn) {
        fprintf(stderr, "[FlashRT] Symbol 'fmha_fp16_strided' not found in %s\n", path);
        return -1;
    }
    return 0;
}

// ── Standard FMHA dispatch ──

int fmha_forward(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    int seq_q, int seq_kv,
    int num_heads, int head_dim,
    float scale,
    cudaStream_t stream) {

    if (g_fmha_fn) {
        // CUTLASS FMHA path (SM100/110)
        return g_fmha_fn(Q, K, V, O, seq_q, seq_kv, num_heads, head_dim,
                         0 /* batch=0 means infer */, stream);
    }

    // FlashAttention path: must be called from Python side
    // (flash_attn_func is a Python function, not C-callable)
    fprintf(stderr, "[FlashRT] No FMHA backend loaded. "
                    "Call load_fmha_library() or use Python FlashAttention.\n");
    return -1;
}

// ── Strided FMHA dispatch ──

int fmha_strided_forward(
    const __nv_bfloat16* qkv_buf,
    __nv_bfloat16* O,
    int seq, int num_heads, int head_dim,
    float scale,
    cudaStream_t stream) {
    // Convenience wrapper: single-batch, Q/K/V interleaved in qkv_buf
    if (g_fmha_strided_fn) {
        int D = num_heads * head_dim;
        int D3 = 3 * D;
        const void* Q = qkv_buf;
        const void* K = (const __nv_bfloat16*)qkv_buf + D;
        const void* V = (const __nv_bfloat16*)qkv_buf + 2 * D;
        return g_fmha_strided_fn(Q, K, V, O,
                                  1, seq, seq, num_heads, num_heads, head_dim,
                                  D3, D3, (void*)stream);
    }
    fprintf(stderr, "[FlashRT] Strided FMHA not loaded.\n");
    return -1;
}

// Full strided FMHA: separate Q/K/V pointers + batch + strides
// Matches libfmha_fp16_strided.so signature exactly (used by SigLIP multi-view)
int fmha_strided_full(
    const void* Q, const void* K, const void* V, void* O,
    int batch, int seq_q, int seq_kv,
    int nheads_q, int nheads_kv, int head_dim,
    int stride_q, int stride_kv,
    cudaStream_t stream) {
    if (g_fmha_strided_fn) {
        return g_fmha_strided_fn(Q, K, V, O,
                                  batch, seq_q, seq_kv, nheads_q, nheads_kv, head_dim,
                                  stride_q, stride_kv, (void*)stream);
    }
    fprintf(stderr, "[FlashRT] Strided FMHA not loaded.\n");
    return -1;
}

// ── Backend queries ──

bool has_flash_attn() {
    // Check at Python level; C++ always returns false
    return false;
}

bool has_cutlass_fmha() {
    return g_fmha_fn != nullptr;
}

FMHABackend get_active_backend() {
    return g_active_backend;
}
