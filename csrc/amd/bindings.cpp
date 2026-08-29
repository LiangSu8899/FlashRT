// ================================================================
// FlashRT AMD — pybind11 bindings (module: flash_rt_amd_kernels)
//
// Same ABI contract as csrc/bindings.cpp: every entry takes
// uintptr_t device pointers + a uintptr_t stream, never tensors.
// Entries keep the CUDA-side names and signatures so the pipeline
// layer is portable text across platforms.
//
// This is the AMD-only module. The CUDA module (flash_rt_kernels)
// is untouched; the two never build together.
// ================================================================

#include <pybind11/pybind11.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <cstdint>
#include <string>

#include "context.h"
#include "gemm/hipblaslt_runner.h"

namespace py = pybind11;

// ── Pointer helpers (mirror csrc/bindings.cpp) ──
static void* to_ptr(uintptr_t addr) { return reinterpret_cast<void*>(addr); }
template<typename T> static T* typed_ptr(uintptr_t addr) { return reinterpret_cast<T*>(addr); }
static hipStream_t to_stream(uintptr_t s) { return reinterpret_cast<hipStream_t>(s); }

// ── Kernel declarations (defined in kernels/*.hip) ──
void rms_norm(const __hip_bfloat16* x, const __hip_bfloat16* weight,
              __hip_bfloat16* out, int seq_len, int dim, float eps,
              hipStream_t stream);
void rms_norm_fp16(const __half* x, const __half* weight,
                   __half* out, int seq_len, int dim, float eps,
                   hipStream_t stream);
void rms_norm_inplace(const __hip_bfloat16* weight,
                      __hip_bfloat16* x, int seq_len, int dim, float eps,
                      hipStream_t stream);
void layer_norm(const __hip_bfloat16* x, const __hip_bfloat16* weight,
                const __hip_bfloat16* bias, __hip_bfloat16* out,
                int seq_len, int dim, float eps, hipStream_t stream);
void ada_rms_norm_style(const __hip_bfloat16* x, const __hip_bfloat16* weight,
                        const __hip_bfloat16* style,
                        __hip_bfloat16* out, __hip_bfloat16* gate_out,
                        int seq_len, int dim, float eps, hipStream_t stream);
void bias_residual_layer_norm_bf16(
    __hip_bfloat16* residual, const __hip_bfloat16* x,
    const __hip_bfloat16* bias_pre,
    const __hip_bfloat16* ln_weight, const __hip_bfloat16* ln_bias,
    __hip_bfloat16* out, int seq_len, int dim, float eps,
    hipStream_t stream);
void avg_pool_vision_tokens(const __hip_bfloat16* x, __hip_bfloat16* out,
                            int nv, int H, int W, int dim, int pool_factor,
                            hipStream_t stream);
void gate_geglu(const __hip_bfloat16* gate, const __hip_bfloat16* up,
                __hip_bfloat16* out, int n, hipStream_t stream);
void gelu_inplace(__hip_bfloat16* x, int n, hipStream_t stream);
void bias_gelu_inplace_bf16_strict(__hip_bfloat16* x,
                                   const __hip_bfloat16* bias,
                                   int M, int N, hipStream_t stream);
void gate_geglu_merged(const __hip_bfloat16* merged, __hip_bfloat16* out,
                       int seq, int half_dim, hipStream_t stream);
void gate_mul_residual(__hip_bfloat16* residual, const __hip_bfloat16* x,
                       const __hip_bfloat16* gate, int n, hipStream_t stream);
void bias_residual(__hip_bfloat16* residual, const __hip_bfloat16* x,
                   const __hip_bfloat16* bias, int seq_len, int dim,
                   hipStream_t stream);
void residual_add(__hip_bfloat16* residual, const __hip_bfloat16* x, int n,
                  hipStream_t stream);
void add_bias_bf16(__hip_bfloat16* x, const __hip_bfloat16* b,
                   int S, int D, hipStream_t stream);
void qkv_split(const __hip_bfloat16* qkv,
               __hip_bfloat16* Q, __hip_bfloat16* K, __hip_bfloat16* V,
               int seq, int q_dim, int k_dim, int v_dim, hipStream_t stream);
void qkv_split_rope(const __hip_bfloat16* qkv,
                    const __hip_bfloat16* rope_weights,
                    __hip_bfloat16* Q, __hip_bfloat16* K, __hip_bfloat16* V,
                    int seq, int q_dim, int k_dim, int v_dim, int head_dim,
                    hipStream_t stream);
void patch_im2col(const __half* input, __half* output, int nv,
                  hipStream_t stream);
void patch_embed_bias_pos(__half* output, const __half* bias,
                          const __half* pos_emb,
                          int S, int D, int S_per_view, hipStream_t stream);
void rms_norm_fp8(const __hip_bfloat16* x, const __hip_bfloat16* weight,
                  __hip_fp8_e4m3* out, int seq_len, int dim, float eps,
                  const float* d_scale, hipStream_t stream);
void ada_rms_norm_style_fp8(const __hip_bfloat16* x, const __hip_bfloat16* weight,
                            const __hip_bfloat16* style,
                            __hip_fp8_e4m3* out, __hip_bfloat16* gate_out,
                            int seq_len, int dim, float eps,
                            const float* d_scale, hipStream_t stream);
void residual_add_rms_norm_fp8(__hip_bfloat16* residual, const __hip_bfloat16* x,
                               const __hip_bfloat16* weight, __hip_fp8_e4m3* out,
                               int seq_len, int dim, float eps,
                               const float* d_scale, hipStream_t stream);
void gate_geglu_merged_fp8(const __hip_bfloat16* merged, __hip_fp8_e4m3* out,
                           int seq, int half_dim,
                           const float* d_scale, hipStream_t stream);
void gate_residual_ada_norm_fp8(__hip_bfloat16* residual, const __hip_bfloat16* x,
                                const __hip_bfloat16* gate, const __hip_bfloat16* weight,
                                const __hip_bfloat16* style,
                                __hip_fp8_e4m3* out, __hip_bfloat16* gate_out,
                                int seq_len, int dim, float eps,
                                const float* d_scale, hipStream_t stream);
void quantize_fp8_static(const __hip_bfloat16* input, __hip_fp8_e4m3* output,
                         const float* d_scale, int n, hipStream_t stream);
void quantize_fp8_device(const __hip_bfloat16* input, __hip_fp8_e4m3* output,
                         float* d_scale, int n, hipStream_t stream);
void fp8_accumulate_scale_max(const float* src_scale, float* dst_scale,
                              hipStream_t stream);
void qkv_split_rope_devpos(const __hip_bfloat16* qkv,
                           const __hip_bfloat16* rope_weights,
                           __hip_bfloat16* Q, __hip_bfloat16* K, __hip_bfloat16* V,
                           const int* devpos,
                           int seq, int q_dim, int k_dim, int v_dim,
                           int head_dim, hipStream_t stream);

PYBIND11_MODULE(flash_rt_amd_kernels, m) {
    m.doc() = "FlashRT AMD (ROCm/HIP) kernels — raw-pointer ABI";

    m.def("build_info", []() {
        py::dict info;
        info["platform"] = "hip";
#ifdef FLASHRT_AMD_GPU_ARCH
        info["gpu_arch"] = FLASHRT_AMD_GPU_ARCH;
#endif
        int rt_version = 0;
        (void)hipRuntimeGetVersion(&rt_version);
        info["hip_runtime_version"] = rt_version;
        return info;
    });

    m.def("device_arch", []() {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) return std::string("none");
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return std::string("unknown");
        return std::string(prop.gcnArchName);
    });

    // ── Norm ──
    m.def("rms_norm", [](uintptr_t x, uintptr_t weight, uintptr_t out,
                         int seq_len, int dim, float eps, uintptr_t stream) {
        rms_norm(typed_ptr<__hip_bfloat16>(x), typed_ptr<__hip_bfloat16>(weight),
                 typed_ptr<__hip_bfloat16>(out), seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("rms_norm_fp16", [](uintptr_t x, uintptr_t weight, uintptr_t out,
                              int seq_len, int dim, float eps, uintptr_t stream) {
        rms_norm_fp16(typed_ptr<__half>(x), typed_ptr<__half>(weight),
                      typed_ptr<__half>(out), seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("rms_norm_inplace", [](uintptr_t weight, uintptr_t x,
                                 int seq_len, int dim, float eps, uintptr_t stream) {
        rms_norm_inplace(typed_ptr<__hip_bfloat16>(weight), typed_ptr<__hip_bfloat16>(x),
                         seq_len, dim, eps, to_stream(stream));
    }, py::arg("weight"), py::arg("x"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("layer_norm", [](uintptr_t x, uintptr_t weight, uintptr_t bias,
                            uintptr_t out, int seq_len, int dim, float eps, uintptr_t stream) {
        layer_norm(typed_ptr<__hip_bfloat16>(x), typed_ptr<__hip_bfloat16>(weight),
                   typed_ptr<__hip_bfloat16>(bias), typed_ptr<__hip_bfloat16>(out),
                   seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("ada_rms_norm_style", [](uintptr_t x, uintptr_t weight, uintptr_t style,
                                    uintptr_t out, uintptr_t gate_out,
                                    int seq_len, int dim, float eps, uintptr_t stream) {
        ada_rms_norm_style(typed_ptr<__hip_bfloat16>(x), typed_ptr<__hip_bfloat16>(weight),
                           typed_ptr<__hip_bfloat16>(style),
                           typed_ptr<__hip_bfloat16>(out), typed_ptr<__hip_bfloat16>(gate_out),
                           seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("style"),
       py::arg("out"), py::arg("gate_out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    // Fused Norm → FP8
    m.def("rms_norm_fp8", [](uintptr_t x, uintptr_t weight, uintptr_t out,
                              int seq_len, int dim, float eps,
                              uintptr_t d_scale, uintptr_t stream) {
        rms_norm_fp8(typed_ptr<__hip_bfloat16>(x), typed_ptr<__hip_bfloat16>(weight),
                     typed_ptr<__hip_fp8_e4m3>(out), seq_len, dim, eps,
                     reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    m.def("ada_rms_norm_style_fp8", [](uintptr_t x, uintptr_t weight, uintptr_t style,
                                        uintptr_t out, uintptr_t gate_out,
                                        int seq_len, int dim, float eps,
                                        uintptr_t d_scale, uintptr_t stream) {
        ada_rms_norm_style_fp8(typed_ptr<__hip_bfloat16>(x), typed_ptr<__hip_bfloat16>(weight),
                               typed_ptr<__hip_bfloat16>(style),
                               typed_ptr<__hip_fp8_e4m3>(out), typed_ptr<__hip_bfloat16>(gate_out),
                               seq_len, dim, eps,
                               reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("style"),
       py::arg("out"), py::arg("gate_out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    m.def("residual_add_rms_norm_fp8", [](uintptr_t residual, uintptr_t x,
                                           uintptr_t weight, uintptr_t out,
                                           int seq_len, int dim, float eps,
                                           uintptr_t d_scale, uintptr_t stream) {
        residual_add_rms_norm_fp8(typed_ptr<__hip_bfloat16>(residual),
                                   typed_ptr<__hip_bfloat16>(x),
                                   typed_ptr<__hip_bfloat16>(weight),
                                   typed_ptr<__hip_fp8_e4m3>(out),
                                   seq_len, dim, eps,
                                   reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    m.def("bias_residual_layer_norm_bf16", [](uintptr_t residual, uintptr_t x,
                                                uintptr_t bias_pre,
                                                uintptr_t ln_weight, uintptr_t ln_bias,
                                                uintptr_t out,
                                                int seq_len, int dim, float eps,
                                                uintptr_t stream) {
        bias_residual_layer_norm_bf16(
            typed_ptr<__hip_bfloat16>(residual), typed_ptr<__hip_bfloat16>(x),
            typed_ptr<__hip_bfloat16>(bias_pre),
            typed_ptr<__hip_bfloat16>(ln_weight),
            typed_ptr<__hip_bfloat16>(ln_bias),
            typed_ptr<__hip_bfloat16>(out), seq_len, dim, eps,
            to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("bias_pre"),
       py::arg("ln_weight"), py::arg("ln_bias"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("stream") = 0);

    m.def("avg_pool_vision_tokens", [](uintptr_t x, uintptr_t out,
                                        int nv, int H, int W, int dim,
                                        int pool_factor, uintptr_t stream) {
        avg_pool_vision_tokens(
            typed_ptr<__hip_bfloat16>(x), typed_ptr<__hip_bfloat16>(out),
            nv, H, W, dim, pool_factor, to_stream(stream));
    }, py::arg("x"), py::arg("out"), py::arg("nv"), py::arg("H"), py::arg("W"),
       py::arg("dim"), py::arg("pool_factor"), py::arg("stream") = 0);

    // ── Activation — GEGLU (tanh-approx GELU(gate) * up), not SiLU ──
    m.def("gate_geglu", [](uintptr_t gate, uintptr_t up, uintptr_t out, int n, uintptr_t stream) {
        gate_geglu(typed_ptr<__hip_bfloat16>(gate), typed_ptr<__hip_bfloat16>(up),
                      typed_ptr<__hip_bfloat16>(out), n, to_stream(stream));
    }, py::arg("gate"), py::arg("up"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);

    m.def("gelu_inplace", [](uintptr_t x, int n, uintptr_t stream) {
        gelu_inplace(typed_ptr<__hip_bfloat16>(x), n, to_stream(stream));
    }, py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    m.def("bias_gelu_bf16_strict", [](uintptr_t x, uintptr_t bias,
                                       int seq_len, int dim, uintptr_t stream) {
        bias_gelu_inplace_bf16_strict(typed_ptr<__hip_bfloat16>(x),
                                      typed_ptr<__hip_bfloat16>(bias),
                                      seq_len, dim, to_stream(stream));
    }, py::arg("x"), py::arg("bias"), py::arg("seq_len"), py::arg("dim"),
       py::arg("stream") = 0);

    m.def("gate_geglu_merged", [](uintptr_t merged, uintptr_t out,
                                   int seq, int half_dim, uintptr_t stream) {
        gate_geglu_merged(typed_ptr<__hip_bfloat16>(merged),
                              typed_ptr<__hip_bfloat16>(out), seq, half_dim, to_stream(stream));
    }, py::arg("merged"), py::arg("out"), py::arg("seq"), py::arg("half_dim"), py::arg("stream") = 0);

    m.def("gate_geglu_merged_fp8", [](uintptr_t merged, uintptr_t out,
                                       int seq, int half_dim,
                                       uintptr_t d_scale, uintptr_t stream) {
        gate_geglu_merged_fp8(typed_ptr<__hip_bfloat16>(merged),
                                  typed_ptr<__hip_fp8_e4m3>(out), seq, half_dim,
                                  reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("merged"), py::arg("out"), py::arg("seq"), py::arg("half_dim"),
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    // ── Elementwise ──
    m.def("gate_mul_residual", [](uintptr_t residual, uintptr_t x, uintptr_t gate, int n, uintptr_t stream) {
        gate_mul_residual(typed_ptr<__hip_bfloat16>(residual),
                          typed_ptr<__hip_bfloat16>(x), typed_ptr<__hip_bfloat16>(gate), n, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("gate"), py::arg("n"), py::arg("stream") = 0);

    m.def("bias_residual", [](uintptr_t residual, uintptr_t x, uintptr_t bias,
                               int seq_len, int dim, uintptr_t stream) {
        bias_residual(typed_ptr<__hip_bfloat16>(residual),
                      typed_ptr<__hip_bfloat16>(x), typed_ptr<__hip_bfloat16>(bias),
                      seq_len, dim, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("bias"),
       py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0);

    m.def("residual_add", [](uintptr_t residual, uintptr_t x, int n, uintptr_t stream) {
        residual_add(typed_ptr<__hip_bfloat16>(residual),
                     typed_ptr<__hip_bfloat16>(x), n, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    m.def("add_bias_bf16",
          [](uintptr_t x, uintptr_t b, int S, int D, uintptr_t stream) {
        add_bias_bf16(typed_ptr<__hip_bfloat16>(x),
                       typed_ptr<__hip_bfloat16>(b),
                       S, D, to_stream(stream));
    }, py::arg("x"), py::arg("b"), py::arg("S"), py::arg("D"),
       py::arg("stream") = 0);

    // ── QKV split ──
    m.def("qkv_split", [](uintptr_t qkv, uintptr_t Q, uintptr_t K, uintptr_t V,
                           int seq, int q_dim, int k_dim, int v_dim, uintptr_t stream) {
        qkv_split(typed_ptr<__hip_bfloat16>(qkv),
                   typed_ptr<__hip_bfloat16>(Q), typed_ptr<__hip_bfloat16>(K),
                   typed_ptr<__hip_bfloat16>(V), seq, q_dim, k_dim, v_dim, to_stream(stream));
    }, py::arg("qkv"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("seq"), py::arg("q_dim"), py::arg("k_dim"), py::arg("v_dim"), py::arg("stream") = 0);

    m.def("qkv_split_rope", [](uintptr_t qkv, uintptr_t rope_weights,
                                 uintptr_t Q, uintptr_t K, uintptr_t V,
                                 int seq, int q_dim, int k_dim, int v_dim,
                                 int head_dim, uintptr_t stream) {
        qkv_split_rope(typed_ptr<__hip_bfloat16>(qkv), typed_ptr<__hip_bfloat16>(rope_weights),
                        typed_ptr<__hip_bfloat16>(Q), typed_ptr<__hip_bfloat16>(K),
                        typed_ptr<__hip_bfloat16>(V),
                        seq, q_dim, k_dim, v_dim, head_dim, to_stream(stream));
    }, py::arg("qkv"), py::arg("rope_weights"),
       py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("seq"), py::arg("q_dim"), py::arg("k_dim"), py::arg("v_dim"),
       py::arg("head_dim"), py::arg("stream") = 0);

    // QKV split + RoPE with a runtime device K/V-cache row offset (devpos).
    // K/V are cache BASE pointers; row written = devpos[0] + token index.
    m.def("qkv_split_rope_devpos", [](uintptr_t qkv, uintptr_t rope_weights,
                                       uintptr_t Q, uintptr_t K, uintptr_t V,
                                       uintptr_t devpos,
                                       int seq, int q_dim, int k_dim, int v_dim,
                                       int head_dim, uintptr_t stream) {
        qkv_split_rope_devpos(typed_ptr<__hip_bfloat16>(qkv),
                              typed_ptr<__hip_bfloat16>(rope_weights),
                              typed_ptr<__hip_bfloat16>(Q), typed_ptr<__hip_bfloat16>(K),
                              typed_ptr<__hip_bfloat16>(V),
                              typed_ptr<int>(devpos),
                              seq, q_dim, k_dim, v_dim, head_dim, to_stream(stream));
    }, py::arg("qkv"), py::arg("rope_weights"),
       py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("devpos"),
       py::arg("seq"), py::arg("q_dim"), py::arg("k_dim"), py::arg("v_dim"),
       py::arg("head_dim"), py::arg("stream") = 0);

    // ── Fusion ──
    m.def("gate_residual_ada_norm_fp8", [](uintptr_t residual, uintptr_t x,
                                            uintptr_t gate, uintptr_t weight,
                                            uintptr_t style,
                                            uintptr_t out, uintptr_t gate_out,
                                            int seq_len, int dim, float eps,
                                            uintptr_t d_scale, uintptr_t stream) {
        gate_residual_ada_norm_fp8(typed_ptr<__hip_bfloat16>(residual),
                                    typed_ptr<__hip_bfloat16>(x),
                                    typed_ptr<__hip_bfloat16>(gate),
                                    typed_ptr<__hip_bfloat16>(weight),
                                    typed_ptr<__hip_bfloat16>(style),
                                    typed_ptr<__hip_fp8_e4m3>(out),
                                    typed_ptr<__hip_bfloat16>(gate_out),
                                    seq_len, dim, eps,
                                    reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("gate"), py::arg("weight"),
       py::arg("style"), py::arg("out"), py::arg("gate_out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    // ── Quantize ──
    m.def("quantize_fp8_static", [](uintptr_t input, uintptr_t output,
                                     uintptr_t d_scale, int n, uintptr_t stream) {
        quantize_fp8_static(typed_ptr<__hip_bfloat16>(input),
                            typed_ptr<__hip_fp8_e4m3>(output),
                            reinterpret_cast<const float*>(d_scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scale"), py::arg("n"), py::arg("stream") = 0);

    m.def("quantize_fp8_device", [](uintptr_t input, uintptr_t output,
                                     uintptr_t d_scale, int n, uintptr_t stream) {
        quantize_fp8_device(typed_ptr<__hip_bfloat16>(input),
                            typed_ptr<__hip_fp8_e4m3>(output),
                            reinterpret_cast<float*>(d_scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scale"), py::arg("n"), py::arg("stream") = 0);

    m.def("fp8_accumulate_scale_max", [](uintptr_t src_scale,
                                         uintptr_t dst_scale,
                                         uintptr_t stream) {
        fp8_accumulate_scale_max(reinterpret_cast<const float*>(src_scale),
                                 reinterpret_cast<float*>(dst_scale),
                                 to_stream(stream));
    }, py::arg("src_scale"), py::arg("dst_scale"), py::arg("stream") = 0);

    // ── GPU memory ops (HIP Graph compatible, explicit stream) ──
    m.def("gpu_copy", [](uintptr_t dst, uintptr_t src, int nbytes, uintptr_t stream) {
        extern void gpu_copy_async(void*, const void*, size_t, hipStream_t);
        gpu_copy_async(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src),
                        nbytes, to_stream(stream));
    }, py::arg("dst"), py::arg("src"), py::arg("nbytes"), py::arg("stream") = 0);

    // ── Patch embedding (FP16, matching the CUDA surface) ──
    m.def("patch_im2col", [](uintptr_t input, uintptr_t output, int nv, uintptr_t stream) {
        patch_im2col(typed_ptr<__half>(input),
                     typed_ptr<__half>(output), nv, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("nv"), py::arg("stream") = 0);

    m.def("patch_embed_bias_pos", [](uintptr_t output, uintptr_t bias, uintptr_t pos_emb,
                                      int S, int D, int S_per_view, uintptr_t stream) {
        patch_embed_bias_pos(typed_ptr<__half>(output),
                             typed_ptr<__half>(bias),
                             typed_ptr<__half>(pos_emb),
                             S, D, S_per_view, to_stream(stream));
    }, py::arg("output"), py::arg("bias"), py::arg("pos_emb"),
       py::arg("S"), py::arg("D"), py::arg("S_per_view"), py::arg("stream") = 0);

    // ── GEMM: FvkContext + hipBLASLt GemmRunner ──
#include "gemm/bindings_gemm.inc"
}
