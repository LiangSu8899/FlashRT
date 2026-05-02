// ================================================================
// flash_wm_kernels — Python bindings for BAGEL BF16 kernels
// ================================================================

#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <stdexcept>

namespace py = pybind11;

// ── FvkContext: import from FlashRT for cuBLAS handle ──
// Minimal reproduction — only need cublas_handle for CUDA Graph compatibility
struct WmContext {
    cublasHandle_t cublas_handle;
    WmContext() : cublas_handle(nullptr) { cublasCreate(&cublas_handle); }
    ~WmContext() { if (cublas_handle) { cublasDestroy(cublas_handle); cublas_handle = nullptr; } }
    WmContext(const WmContext&) = delete;
    WmContext& operator=(const WmContext&) = delete;
};

// ── Extern declarations ──

extern void attention_mha_bf16(
    cublasHandle_t handle,
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* logits, __nv_bfloat16* out,
    int S_q, int S_kv, int NH, int HD,
    float attn_scale, cudaStream_t stream);

extern void rope_rotate_half_bf16(
    __nv_bfloat16* x, const __nv_bfloat16* cos_table, const __nv_bfloat16* sin_table,
    int S, int NH, int HD, cudaStream_t stream);

extern void silu_mul_split_fp8_bf16(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_fp8_e4m3* out, int n, const float* d_scale, cudaStream_t stream);

extern void silu_mul_bf16(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* out, int n, cudaStream_t stream);

extern void tiny_bf16_matmul_m2(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int N, int K, cudaStream_t stream);

extern void silu_mul_merged_fp8_bf16(
    const __nv_bfloat16* gate_up, __nv_fp8_e4m3* out,
    int seq, int ffn, const float* d_scale, cudaStream_t stream);

extern void gpu_fill_neginf_bf16(__nv_bfloat16* dst, int n, cudaStream_t stream);

extern void gpu_add_bias_bf16(__nv_bfloat16* data, const __nv_bfloat16* bias,
                              int rows, int cols, cudaStream_t stream);

extern void bf16_text_gather(const __nv_bfloat16* src, __nv_bfloat16* dst,
                              int B, int Sq, int N, cudaStream_t stream);
extern void bf16_text_scatter(__nv_bfloat16* dst, const __nv_bfloat16* src,
                               int B, int Sq, int N, cudaStream_t stream);

extern void qk_rmsnorm_rope_fused_bf16(
    __nv_bfloat16* qk, const __nv_bfloat16* w,
    const __nv_bfloat16* cos_t, const __nv_bfloat16* sin_t,
    int rows, int NH, int HD, float eps, cudaStream_t stream);

extern void bf16_gemm_nn(
    cublasHandle_t handle,
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K, cudaStream_t stream);

extern void gpu_repeat_interleave_heads_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int S, int NH_src, int HD, int repeat, cudaStream_t stream);

#include <cuda_fp16.h>
extern void cast_bf16_to_fp16(
    const __nv_bfloat16* src, __half* dst, int n, cudaStream_t stream);
extern void cast_fp16_to_bf16(
    const __half* src, __nv_bfloat16* dst, int n, cudaStream_t stream);
extern void residual_add_rms_norm_bf16_to_fp16(
    __nv_bfloat16* x, const __nv_bfloat16* r, const __nv_bfloat16* w,
    __half* out, int rows, int D, float eps, cudaStream_t stream);
extern void silu_mul_fp16(
    const __half* gate, const __half* up, __half* out, int n, cudaStream_t stream);
extern void silu_mul_clamp_fp16(
    const __half* gate, const __half* up, __half* out, int n,
    float max_abs, cudaStream_t stream);
extern void residual_add_fp16_to_bf16(
    __nv_bfloat16* acc, const __half* delta, int n, cudaStream_t stream);

// bf16-out FP4 GEMM (mirror of upstream cutlass_fp4_gemm_variant but emits bf16).
extern "C" int cutlass_fp4_gemm_bf16out_variant(int idx,
    void const* A, void const* SFA, void const* B, void const* SFB,
    void* D_bf16, int M, int N, int K, float alpha, float beta,
    cudaStream_t stream);

// BAGEL BF16 residual+rms+mul+FP4+SFA fused (ROI A).
extern "C" void bagel_res_rms_mul_fp4_sfa_bf16(
    __nv_bfloat16* x, const __nv_bfloat16* r, const __nv_bfloat16* w,
    uint8_t* packed, uint8_t* sfa,
    int rows, int D, float eps, cudaStream_t stream);

// BAGEL true-SiLU fused gate_silu_mul + fp4 + SFA (Class 1c).
extern "C" void bagel_silu_mul_fp4_sfa_v2_fp16(
    const __half* merged, uint8_t* packed, uint8_t* sfa,
    int seq_len, int half_dim, cudaStream_t stream);

// FP8 SqGemm with per-col BF16 bias epilogue, BF16 output (flash_wm only).
extern "C" int cutlass_fp8_sq_bias_bf16out(
    void const* A, void const* B, void const* bias, void* D,
    int M, int N, int K, float alpha, cudaStream_t stream);

// CUTLASS Blackwell FMHA, FP8 E4M3 in / BF16 out (cross-attn, GQA capable).
extern "C" int cutlass_fp8_fmha_strided(
    const void* Q, const void* K, const void* V, void* O,
    int B, int SQ, int SK, int NQ, int NKV, int HD,
    int q_seq_stride, int k_seq_stride,
    float scale_q, float scale_k, float scale_v, float inv_scale_o,
    cudaStream_t stream);
extern "C" int cutlass_fp8_fmha_prepare(int max_SQ, int max_SK, int max_B,
                                         int max_NQ, int max_HD);

// ── Python module ──

PYBIND11_MODULE(flash_wm_kernels, m) {
    m.doc() = "flash_wm_kernels: BF16 kernel variants for BAGEL world model";

    // ── WmContext (holds cuBLAS handle for CUDA Graph compatibility) ──
    py::class_<WmContext>(m, "WmContext")
        .def(py::init<>());

    m.def("attention_mha_bf16",
        [](WmContext& ctx, uintptr_t Q, uintptr_t K, uintptr_t V,
           uintptr_t logits, uintptr_t out,
           int S_q, int S_kv, int NH, int HD,
           float attn_scale, uintptr_t stream) {
            attention_mha_bf16(
                ctx.cublas_handle,
                reinterpret_cast<const __nv_bfloat16*>(Q),
                reinterpret_cast<const __nv_bfloat16*>(K),
                reinterpret_cast<const __nv_bfloat16*>(V),
                reinterpret_cast<__nv_bfloat16*>(logits),
                reinterpret_cast<__nv_bfloat16*>(out),
                S_q, S_kv, NH, HD, attn_scale,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "BF16 multi-head attention via cuBLAS batched GEMM",
        py::arg("ctx"), py::arg("Q"), py::arg("K"), py::arg("V"),
        py::arg("logits"), py::arg("out"),
        py::arg("S_q"), py::arg("S_kv"), py::arg("NH"), py::arg("HD"),
        py::arg("attn_scale") = 1.0f, py::arg("stream") = 0);

    m.def("rope_rotate_half_bf16",
        [](uintptr_t x, uintptr_t cos_table, uintptr_t sin_table,
           int S, int NH, int HD, uintptr_t stream) {
            rope_rotate_half_bf16(
                reinterpret_cast<__nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(cos_table),
                reinterpret_cast<const __nv_bfloat16*>(sin_table),
                S, NH, HD,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "BF16 Qwen-style RoPE (rotate_half, in-place)",
        py::arg("x"), py::arg("cos_table"), py::arg("sin_table"),
        py::arg("S"), py::arg("NH"), py::arg("HD"),
        py::arg("stream") = 0);

    m.def("silu_mul_split_fp8_bf16",
        [](uintptr_t gate, uintptr_t up, uintptr_t out,
           int n, uintptr_t d_scale, uintptr_t stream) {
            silu_mul_split_fp8_bf16(
                reinterpret_cast<const __nv_bfloat16*>(gate),
                reinterpret_cast<const __nv_bfloat16*>(up),
                reinterpret_cast<__nv_fp8_e4m3*>(out),
                n, reinterpret_cast<const float*>(d_scale),
                reinterpret_cast<cudaStream_t>(stream));
        },
        "SiLU(gate) * up -> FP8 (BF16 input)",
        py::arg("gate"), py::arg("up"), py::arg("out"),
        py::arg("n"), py::arg("d_scale"), py::arg("stream") = 0);

    m.def("silu_mul_bf16",
        [](uintptr_t gate, uintptr_t up, uintptr_t out,
           int n, uintptr_t stream) {
            silu_mul_bf16(
                reinterpret_cast<const __nv_bfloat16*>(gate),
                reinterpret_cast<const __nv_bfloat16*>(up),
                reinterpret_cast<__nv_bfloat16*>(out), n,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "SiLU(gate) * up -> BF16 (BF16 input, BF16 output, no FP8 quantize)",
        py::arg("gate"), py::arg("up"), py::arg("out"),
        py::arg("n"), py::arg("stream") = 0);

    m.def("tiny_bf16_matmul_m2",
        [](uintptr_t A, uintptr_t B, uintptr_t C,
           int N, int K, uintptr_t stream) {
            tiny_bf16_matmul_m2(
                reinterpret_cast<const __nv_bfloat16*>(A),
                reinterpret_cast<const __nv_bfloat16*>(B),
                reinterpret_cast<__nv_bfloat16*>(C),
                N, K,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "BF16 tiny matmul: C[2,N] = A[2,K] * B[N,K]^T (hand-rolled, for MoT text path)",
        py::arg("A"), py::arg("B"), py::arg("C"),
        py::arg("N"), py::arg("K"), py::arg("stream") = 0);

    m.def("silu_mul_merged_fp8_bf16",
        [](uintptr_t gate_up, uintptr_t out,
           int seq, int ffn, uintptr_t d_scale, uintptr_t stream) {
            silu_mul_merged_fp8_bf16(
                reinterpret_cast<const __nv_bfloat16*>(gate_up),
                reinterpret_cast<__nv_fp8_e4m3*>(out),
                seq, ffn,
                reinterpret_cast<const float*>(d_scale),
                reinterpret_cast<cudaStream_t>(stream));
        },
        "SiLU(gate) * up -> FP8 (packed gate+up buffer [Sq, 2*FFN] input)",
        py::arg("gate_up"), py::arg("out"),
        py::arg("seq"), py::arg("ffn"), py::arg("d_scale"),
        py::arg("stream") = 0);

    m.def("gpu_fill_neginf_bf16",
        [](uintptr_t dst, int n, uintptr_t stream) {
            gpu_fill_neginf_bf16(
                reinterpret_cast<__nv_bfloat16*>(dst), n,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "Fill BF16 buffer with -inf",
        py::arg("dst"), py::arg("n"), py::arg("stream") = 0);

    m.def("gpu_repeat_interleave_heads_bf16",
        [](uintptr_t src, uintptr_t dst,
           int S, int NH_src, int HD, int repeat, uintptr_t stream) {
            gpu_repeat_interleave_heads_bf16(
                reinterpret_cast<const __nv_bfloat16*>(src),
                reinterpret_cast<__nv_bfloat16*>(dst),
                S, NH_src, HD, repeat,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "Repeat KV heads for GQA (BF16)",
        py::arg("src"), py::arg("dst"),
        py::arg("S"), py::arg("NH_src"), py::arg("HD"), py::arg("repeat"),
        py::arg("stream") = 0);

    m.def("gpu_add_bias_bf16",
        [](uintptr_t data, uintptr_t bias, int rows, int cols, uintptr_t stream) {
            gpu_add_bias_bf16(
                reinterpret_cast<__nv_bfloat16*>(data),
                reinterpret_cast<const __nv_bfloat16*>(bias),
                rows, cols,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "Add bias (broadcast over rows, BF16, in-place)",
        py::arg("data"), py::arg("bias"), py::arg("rows"), py::arg("cols"),
        py::arg("stream") = 0);

    m.def("bf16_gemm_nn",
        [](WmContext& ctx, uintptr_t A, uintptr_t B, uintptr_t C,
           int M, int N, int K, uintptr_t stream) {
            bf16_gemm_nn(ctx.cublas_handle,
                reinterpret_cast<const __nv_bfloat16*>(A),
                reinterpret_cast<const __nv_bfloat16*>(B),
                reinterpret_cast<__nv_bfloat16*>(C),
                M, N, K,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "BF16 GEMM: C[M,N] = A[M,K] * B[N,K]^T (cuBLAS)",
        py::arg("ctx"), py::arg("A"), py::arg("B"), py::arg("C"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("stream") = 0);

    // ── B5 FP4 FFN bridge kernels ───────────────────────────────
    m.def("cast_bf16_to_fp16",
        [](uintptr_t src, uintptr_t dst, int n, uintptr_t stream) {
            cast_bf16_to_fp16(
                reinterpret_cast<const __nv_bfloat16*>(src),
                reinterpret_cast<__half*>(dst), n,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "BF16 → FP16 cast. NEVER call on raw residual (|x| can exceed 65504); "
        "safe on post-rms-norm activations.",
        py::arg("src"), py::arg("dst"), py::arg("n"), py::arg("stream") = 0);

    m.def("cast_fp16_to_bf16",
        [](uintptr_t src, uintptr_t dst, int n, uintptr_t stream) {
            cast_fp16_to_bf16(
                reinterpret_cast<const __half*>(src),
                reinterpret_cast<__nv_bfloat16*>(dst), n,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "FP16 → BF16 cast. Used to fold FP4 down-GEMM fp16 output back into "
        "the bf16 b_down buffer for text-row overlay + parent's residual flow.",
        py::arg("src"), py::arg("dst"), py::arg("n"), py::arg("stream") = 0);

    m.def("residual_add_rms_norm_bf16_to_fp16",
        [](uintptr_t x, uintptr_t r, uintptr_t w, uintptr_t out,
           int rows, int D, float eps, uintptr_t stream) {
            residual_add_rms_norm_bf16_to_fp16(
                reinterpret_cast<__nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(r),
                reinterpret_cast<const __nv_bfloat16*>(w),
                reinterpret_cast<__half*>(out), rows, D, eps,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "Fused BF16 residual_add + weighted rms_norm + cast to FP16. "
        "x ← x + r (in-place bf16); out = fp16(norm(x) * w). Replaces 3 "
        "separate kernels (residual_add + rms_norm + cast_bf16_to_fp16).",
        py::arg("x"), py::arg("r"), py::arg("w"), py::arg("out"),
        py::arg("rows"), py::arg("D"), py::arg("eps") = 1e-6f,
        py::arg("stream") = 0);

    m.def("silu_mul_fp16",
        [](uintptr_t gate, uintptr_t up, uintptr_t out, int n, uintptr_t stream) {
            silu_mul_fp16(
                reinterpret_cast<const __half*>(gate),
                reinterpret_cast<const __half*>(up),
                reinterpret_cast<__half*>(out), n,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "SiLU(gate) * up → FP16 (FP16 in / FP16 out). Mirror of silu_mul_bf16.",
        py::arg("gate"), py::arg("up"), py::arg("out"),
        py::arg("n"), py::arg("stream") = 0);

    m.def("silu_mul_clamp_fp16",
        [](uintptr_t gate, uintptr_t up, uintptr_t out, int n,
           float max_abs, uintptr_t stream) {
            silu_mul_clamp_fp16(
                reinterpret_cast<const __half*>(gate),
                reinterpret_cast<const __half*>(up),
                reinterpret_cast<__half*>(out), n, max_abs,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "SiLU(gate) * up saturated to |x| <= max_abs (Path C FP4 Down "
        "accumulator safety). Same I/O as silu_mul_fp16.",
        py::arg("gate"), py::arg("up"), py::arg("out"), py::arg("n"),
        py::arg("max_abs"), py::arg("stream") = 0);

    m.def("cutlass_fp8_fmha_prepare",
        &cutlass_fp8_fmha_prepare,
        "Pre-allocate workspace + LSE buffers for cutlass_fp8_fmha_strided. "
        "MUST be called ONCE before any CUDA Graph capture that includes "
        "the FMHA call (cudaMalloc is not graph-captureable).",
        py::arg("max_SQ"), py::arg("max_SK"),
        py::arg("max_B") = 1, py::arg("max_NQ") = 28, py::arg("max_HD") = 128);

    m.def("cutlass_fp8_fmha_strided",
        [](uintptr_t Q, uintptr_t K, uintptr_t V, uintptr_t O,
           int B, int SQ, int SK, int NQ, int NKV, int HD,
           int q_seq_stride, int k_seq_stride,
           float scale_q, float scale_k, float scale_v, float inv_scale_o,
           uintptr_t stream) {
            return cutlass_fp8_fmha_strided(
                reinterpret_cast<const void*>(Q),
                reinterpret_cast<const void*>(K),
                reinterpret_cast<const void*>(V),
                reinterpret_cast<void*>(O),
                B, SQ, SK, NQ, NKV, HD,
                q_seq_stride, k_seq_stride,
                scale_q, scale_k, scale_v, inv_scale_o,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "CUTLASS Blackwell FMHA (SM100/110), FP8 E4M3 input / BF16 output. "
        "Supports GQA (NQ must be divisible by NKV). q_seq_stride/k_seq_stride "
        "let Q and K/V live in interleaved buffers. Scales map FP8 back to "
        "original magnitudes (bf16 ~= fp8 * scale_*).",
        py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
        py::arg("B"), py::arg("SQ"), py::arg("SK"),
        py::arg("NQ"), py::arg("NKV"), py::arg("HD"),
        py::arg("q_seq_stride"), py::arg("k_seq_stride"),
        py::arg("scale_q") = 1.0f, py::arg("scale_k") = 1.0f,
        py::arg("scale_v") = 1.0f, py::arg("inv_scale_o") = 1.0f,
        py::arg("stream") = 0);

    m.def("cutlass_fp4_gemm_bf16out_variant",
        [](int idx, uintptr_t A, uintptr_t SFA, uintptr_t B, uintptr_t SFB,
           uintptr_t D_bf16, int M, int N, int K, float alpha, float beta,
           uintptr_t stream) {
            return cutlass_fp4_gemm_bf16out_variant(
                idx,
                reinterpret_cast<void const*>(A),
                reinterpret_cast<void const*>(SFA),
                reinterpret_cast<void const*>(B),
                reinterpret_cast<void const*>(SFB),
                reinterpret_cast<void*>(D_bf16),
                M, N, K, alpha, beta,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "NVFP4 GEMM with BF16 output (same interface as upstream variant, "
        "but D is bf16 not fp16). Currently implemented for idx={6,8}.",
        py::arg("idx"), py::arg("A"), py::arg("SFA"),
        py::arg("B"), py::arg("SFB"), py::arg("D_bf16"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("alpha"), py::arg("beta"), py::arg("stream") = 0);

    m.def("qk_rmsnorm_rope_fused_bf16",
        [](uintptr_t qk, uintptr_t w, uintptr_t cos_t, uintptr_t sin_t,
           int rows, int NH, int HD, float eps, uintptr_t stream) {
            qk_rmsnorm_rope_fused_bf16(
                reinterpret_cast<__nv_bfloat16*>(qk),
                reinterpret_cast<const __nv_bfloat16*>(w),
                reinterpret_cast<const __nv_bfloat16*>(cos_t),
                reinterpret_cast<const __nv_bfloat16*>(sin_t),
                rows, NH, HD, eps,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "Class D: fused per-head rms_norm(qk × w) + RoPE rotate_half (BF16, "
        "in-place). One kernel per (row, head) replaces rms_norm + rope "
        "(2 launches → 1). qk is [rows, NH, HD]; w is [HD]; cos_t/sin_t are "
        "[seq_rows, HD] with first HD/2 columns populated.",
        py::arg("qk"), py::arg("w"),
        py::arg("cos_t"), py::arg("sin_t"),
        py::arg("rows"), py::arg("NH"), py::arg("HD"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("bf16_text_gather",
        [](uintptr_t src, uintptr_t dst, int B, int Sq, int N, uintptr_t stream) {
            bf16_text_gather(
                reinterpret_cast<const __nv_bfloat16*>(src),
                reinterpret_cast<__nv_bfloat16*>(dst),
                B, Sq, N,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "Class 1d: gather the 2*B text-bracket rows (first+last of each "
        "batch) from src [B*Sq, N] into dst [2*B, N] in one kernel launch, "
        "replacing 2*B separate gpu_copy calls.",
        py::arg("src"), py::arg("dst"),
        py::arg("B"), py::arg("Sq"), py::arg("N"),
        py::arg("stream") = 0);

    m.def("bf16_text_scatter",
        [](uintptr_t dst, uintptr_t src, int B, int Sq, int N, uintptr_t stream) {
            bf16_text_scatter(
                reinterpret_cast<__nv_bfloat16*>(dst),
                reinterpret_cast<const __nv_bfloat16*>(src),
                B, Sq, N,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "Class 1d: scatter 2*B text-bracket rows from src [2*B, N] back to "
        "dst [B*Sq, N] (first+last positions of each batch), one kernel "
        "launch replacing 2*B gpu_copy calls.",
        py::arg("dst"), py::arg("src"),
        py::arg("B"), py::arg("Sq"), py::arg("N"),
        py::arg("stream") = 0);

    m.def("bagel_res_rms_mul_fp4_sfa_bf16",
        [](uintptr_t x, uintptr_t r, uintptr_t w,
           uintptr_t packed, uintptr_t sfa,
           int rows, int D, float eps, uintptr_t stream) {
            bagel_res_rms_mul_fp4_sfa_bf16(
                reinterpret_cast<__nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(r),
                reinterpret_cast<const __nv_bfloat16*>(w),
                reinterpret_cast<uint8_t*>(packed),
                reinterpret_cast<uint8_t*>(sfa),
                rows, D, eps,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "Fused BF16 residual+rms_norm(×w)+FP4+SFA (ROI A). In-place: "
        "x ← x+r (bf16). Output: packed FP4 [rows, D/2] + SFA. "
        "Eliminates fp16 intermediate buffer; keeps bf16 residual for "
        "L5/L9 overflow safety. w is ln_baked (ln2_w × inv_s_gu AWQ).",
        py::arg("x"), py::arg("r"), py::arg("w"),
        py::arg("packed"), py::arg("sfa"),
        py::arg("rows"), py::arg("D"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("bagel_silu_mul_fp4_sfa_v2_fp16",
        [](uintptr_t merged, uintptr_t packed, uintptr_t sfa,
           int seq_len, int half_dim, uintptr_t stream) {
            bagel_silu_mul_fp4_sfa_v2_fp16(
                reinterpret_cast<const __half*>(merged),
                reinterpret_cast<uint8_t*>(packed),
                reinterpret_cast<uint8_t*>(sfa),
                seq_len, half_dim,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "True SiLU fused gate_silu_mul + FP4 + SFA (BAGEL-specific, vs "
        "upstream's mis-named GELU-tanh kernel). merged: fp16 [S, 2H] with "
        "gate in [:, :H) and up in [:, H:2H). Output: packed fp4 + SFA.",
        py::arg("merged"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("half_dim"), py::arg("stream") = 0);

    m.def("cutlass_fp8_sq_bias_bf16out",
        [](uintptr_t A, uintptr_t B, uintptr_t bias, uintptr_t D,
           int M, int N, int K, float alpha, uintptr_t stream) {
            return cutlass_fp8_sq_bias_bf16out(
                reinterpret_cast<void const*>(A),
                reinterpret_cast<void const*>(B),
                reinterpret_cast<void const*>(bias),
                reinterpret_cast<void*>(D),
                M, N, K, alpha,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "FP8 SqGemm + per-col BF16 bias epilogue, BF16 output. "
        "Fuses GEMM + gpu_add_bias_bf16 into one kernel. "
        "D[M,N] = alpha * (A[M,K] @ B[N,K]^T) + bias[N].",
        py::arg("A"), py::arg("B"), py::arg("bias"), py::arg("D"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("alpha"), py::arg("stream") = 0);

    m.def("residual_add_fp16_to_bf16",
        [](uintptr_t acc, uintptr_t delta, int n, uintptr_t stream) {
            residual_add_fp16_to_bf16(
                reinterpret_cast<__nv_bfloat16*>(acc),
                reinterpret_cast<const __half*>(delta), n,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "BF16 residual += FP16 delta (fp32 accumulate internally)",
        py::arg("acc"), py::arg("delta"), py::arg("n"), py::arg("stream") = 0);
}
