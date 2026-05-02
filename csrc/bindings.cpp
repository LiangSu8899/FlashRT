// ================================================================
// FlashVLA — pybind11 bindings
// Exposes GemmRunner + all CUDA kernels to Python
// ================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include "context.h"
#include "gemm/gemm_runner.h"
#include "kernels/kernels.h"
#include "attention/fmha_dispatch.h"

namespace py = pybind11;

static void* to_ptr(uintptr_t addr) { return reinterpret_cast<void*>(addr); }
template<typename T> static T* typed_ptr(uintptr_t addr) { return reinterpret_cast<T*>(addr); }
static cudaStream_t to_stream(uintptr_t s) { return reinterpret_cast<cudaStream_t>(s); }

#ifdef ENABLE_NVFP4
extern "C" int run_w4a8_gemm(void*, void*, void*, void*, void*, int, int, int, cudaStream_t);
extern "C" float launch_w4a8_gemm(void*, void*, void*, void*, void*, void*, int, int, int, float, float, int, int);
#endif

// ENABLE_FA2 moved to a separate pybind module (flash_rt_fa2.so —
// csrc/fa2_bindings.cpp). This keeps the main flash_rt_kernels.so
// small and its build fast by isolating FA2's heavy CUTLASS 3.x
// template codegen.

#ifdef ENABLE_SM100_CUTLASS
extern "C" int cutlass_fp8_sq(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_t1(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_wide(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_plain(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_gelu(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_sq_f32out(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_wide_f32out(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_sq_bf16out(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_wide_bf16out(void*, void*, void*, int, int, int, float, float, cudaStream_t);
extern "C" int cutlass_fp8_t1_bf16out(void*, void*, void*, int, int, int, float, float, cudaStream_t);
#endif

PYBIND11_MODULE(flash_rt_kernels, m) {
    m.doc() = "FlashVLA C++/CUDA inference kernels";

    // ── FvkContext: per-instance cuBLAS handle ──
    py::class_<FvkContext>(m, "FvkContext")
        .def(py::init<>())
        .def_property_readonly("handle_ptr", [](const FvkContext& ctx) {
            return reinterpret_cast<uintptr_t>(ctx.cublas_handle);
        });

    // ── GemmRunner ──
    py::class_<GemmRunner>(m, "GemmRunner")
        .def(py::init<>())
        .def("bf16_gemm", [](GemmRunner& self,
                              uintptr_t A, uintptr_t B, uintptr_t D,
                              int M, int N, int K,
                              float alpha, float beta,
                              int warmup, int iters) {
            return self.bf16_gemm(to_ptr(A), to_ptr(B), to_ptr(D),
                                  M, N, K, alpha, beta, warmup, iters);
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
           py::arg("warmup") = 3, py::arg("iters") = 100)
        .def("fp8_gemm", [](GemmRunner& self,
                             uintptr_t A, uintptr_t B, uintptr_t D,
                             int M, int N, int K,
                             float scale_a, float scale_b,
                             int warmup, int iters) {
            return self.fp8_gemm(to_ptr(A), to_ptr(B), to_ptr(D),
                                 M, N, K, scale_a, scale_b, warmup, iters);
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("scale_a") = 1.0f, py::arg("scale_b") = 1.0f,
           py::arg("warmup") = 3, py::arg("iters") = 100)
#ifdef ENABLE_NVFP4
        .def("fp4_gemm", [](GemmRunner& self,
                             uintptr_t A, uintptr_t SFA,
                             uintptr_t B, uintptr_t SFB,
                             uintptr_t D,
                             int M, int N, int K,
                             int warmup, int iters) {
            return self.fp4_gemm(to_ptr(A), to_ptr(SFA),
                                 to_ptr(B), to_ptr(SFB),
                                 to_ptr(D), M, N, K, warmup, iters);
        }, py::arg("A"), py::arg("SFA"),
           py::arg("B"), py::arg("SFB"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("warmup") = 3, py::arg("iters") = 100)
#endif
        // Inference methods (stream-based, CUDA Graph compatible)
        .def("fp16_nn", [](GemmRunner& self,
                            uintptr_t A, uintptr_t B, uintptr_t D,
                            int M, int N, int K, uintptr_t stream) {
            self.fp16_nn(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("bf16_nn", [](GemmRunner& self,
                            uintptr_t A, uintptr_t B, uintptr_t D,
                            int M, int N, int K, uintptr_t stream) {
            self.bf16_nn(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("bf16_nn_res", [](GemmRunner& self,
                                uintptr_t A, uintptr_t B, uintptr_t D,
                                int M, int N, int K, uintptr_t stream) {
            self.bf16_nn_res(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("bf16_nn_bias", [](GemmRunner& self,
                                 uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                 int M, int N, int K, uintptr_t stream) {
            self.bf16_nn_bias(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                               M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("bf16_nn_bias_gelu", [](GemmRunner& self,
                                      uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                      int M, int N, int K, uintptr_t stream) {
            self.bf16_nn_bias_gelu(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                                    M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("bf16_nn_bias_res", [](GemmRunner& self,
                                     uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                     int M, int N, int K, uintptr_t stream) {
            self.bf16_nn_bias_res(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                                   M, N, K, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("fp8_run_dev", [](GemmRunner& self,
                                uintptr_t A, uintptr_t B, uintptr_t D,
                                int M, int N, int K,
                                uintptr_t d_scale_a, uintptr_t d_scale_b,
                                uintptr_t stream) {
            self.fp8_run_dev(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
                              reinterpret_cast<float*>(d_scale_a),
                              reinterpret_cast<float*>(d_scale_b), to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("d_scale_a"), py::arg("d_scale_b"), py::arg("stream") = 0)
        .def("fp8_nn_dev", [](GemmRunner& self,
                               uintptr_t A, uintptr_t B, uintptr_t D,
                               int M, int N, int K,
                               uintptr_t d_scale_a, uintptr_t d_scale_b,
                               uintptr_t stream) {
            self.fp8_nn_dev(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
                             reinterpret_cast<float*>(d_scale_a),
                             reinterpret_cast<float*>(d_scale_b), to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("d_scale_a"), py::arg("d_scale_b"), py::arg("stream") = 0)
        // FP8 with device descale → FP16 (GemmRunner handle, matching pi05)
        .def("fp8_descale_fp16", [](GemmRunner& self,
                                     uintptr_t A, uintptr_t B, uintptr_t D,
                                     int M, int N, int K,
                                     uintptr_t act_descale, uintptr_t w_descale, uintptr_t stream) {
            self.fp8_descale_fp16(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
                                   reinterpret_cast<float*>(act_descale),
                                   reinterpret_cast<float*>(w_descale), to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("act_descale"), py::arg("w_descale"), py::arg("stream") = 0)
        // FP8 GEMM with epilogues (matches pi05 cublaslt_fp8.cuh)
        .def("fp8_nn_bias", [](GemmRunner& self,
                                uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                int M, int N, int K, float alpha, uintptr_t stream) {
            self.fp8_nn_bias(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                              M, N, K, alpha, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha") = 1.0f, py::arg("stream") = 0)
        .def("fp8_nn_bias_res", [](GemmRunner& self,
                                    uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                    int M, int N, int K, float alpha, uintptr_t stream) {
            self.fp8_nn_bias_res(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                                  M, N, K, alpha, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha") = 1.0f, py::arg("stream") = 0)
        .def("fp8_nn_gelu_bias", [](GemmRunner& self,
                                     uintptr_t A, uintptr_t B, uintptr_t D, uintptr_t bias,
                                     int M, int N, int K, float alpha, uintptr_t stream) {
            self.fp8_nn_gelu_bias(to_ptr(A), to_ptr(B), to_ptr(D), to_ptr(bias),
                                   M, N, K, alpha, to_stream(stream));
        }, py::arg("A"), py::arg("B"), py::arg("D"), py::arg("bias"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha") = 1.0f, py::arg("stream") = 0)
        // Autotune
        .def("autotune_bf16_nn", [](GemmRunner& self,
                                     uintptr_t A, uintptr_t B, uintptr_t D,
                                     int M, int N, int K, int num_algos) {
            self.autotune_bf16_nn(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, num_algos);
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("num_algos") = 16)
        .def("autotune_fp8_nn_dev", [](GemmRunner& self,
                                        uintptr_t A, uintptr_t B, uintptr_t D,
                                        int M, int N, int K,
                                        uintptr_t d_scale_a, uintptr_t d_scale_b,
                                        int num_algos) {
            self.autotune_fp8_nn_dev(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K,
                                      reinterpret_cast<float*>(d_scale_a),
                                      reinterpret_cast<float*>(d_scale_b), num_algos);
        }, py::arg("A"), py::arg("B"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"),
           py::arg("d_scale_a"), py::arg("d_scale_b"), py::arg("num_algos") = 16)
#ifdef ENABLE_NVFP4
        .def("fp4_nn_dev", [](GemmRunner& self,
                               uintptr_t A_fp4, uintptr_t SFA,
                               uintptr_t B_fp4, uintptr_t SFB,
                               uintptr_t D,
                               int M, int N, int K, uintptr_t stream) {
            self.fp4_nn_dev(to_ptr(A_fp4), to_ptr(SFA),
                             to_ptr(B_fp4), to_ptr(SFB),
                             to_ptr(D), M, N, K, to_stream(stream));
        }, py::arg("A_fp4"), py::arg("SFA"),
           py::arg("B_fp4"), py::arg("SFB"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0)
        .def("autotune_fp4_nn_dev", [](GemmRunner& self,
                                        uintptr_t A_fp4, uintptr_t SFA,
                                        uintptr_t B_fp4, uintptr_t SFB,
                                        uintptr_t D,
                                        int M, int N, int K, int num_algos) {
            self.autotune_fp4_nn_dev(to_ptr(A_fp4), to_ptr(SFA),
                                      to_ptr(B_fp4), to_ptr(SFB),
                                      to_ptr(D), M, N, K, num_algos);
        }, py::arg("A_fp4"), py::arg("SFA"),
           py::arg("B_fp4"), py::arg("SFB"), py::arg("D"),
           py::arg("M"), py::arg("N"), py::arg("K"), py::arg("num_algos") = 16)
#endif
    ;

    // ── Kernel functions ──
    // Norm
    m.def("rms_norm", [](uintptr_t x, uintptr_t weight, uintptr_t out,
                          int seq_len, int dim, float eps, uintptr_t stream) {
        rms_norm(typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(weight),
                 typed_ptr<__nv_bfloat16>(out), seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("rms_norm_inplace", [](uintptr_t weight, uintptr_t x,
                                  int seq_len, int dim, float eps, uintptr_t stream) {
        rms_norm_inplace(typed_ptr<__nv_bfloat16>(weight),
                         typed_ptr<__nv_bfloat16>(x), seq_len, dim, eps, to_stream(stream));
    }, py::arg("weight"), py::arg("x"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("layer_norm", [](uintptr_t x, uintptr_t weight, uintptr_t bias,
                            uintptr_t out, int seq_len, int dim, float eps, uintptr_t stream) {
        layer_norm(typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(weight),
                   typed_ptr<__nv_bfloat16>(bias), typed_ptr<__nv_bfloat16>(out),
                   seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("ada_rms_norm_style", [](uintptr_t x, uintptr_t weight, uintptr_t style,
                                    uintptr_t out, uintptr_t gate_out,
                                    int seq_len, int dim, float eps, uintptr_t stream) {
        ada_rms_norm_style(typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(weight),
                           typed_ptr<__nv_bfloat16>(style),
                           typed_ptr<__nv_bfloat16>(out), typed_ptr<__nv_bfloat16>(gate_out),
                           seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("style"),
       py::arg("out"), py::arg("gate_out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    // Fused Norm → FP8
    m.def("rms_norm_fp8", [](uintptr_t x, uintptr_t weight, uintptr_t out,
                              int seq_len, int dim, float eps,
                              uintptr_t d_scale, uintptr_t stream) {
        rms_norm_fp8(typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(weight),
                     typed_ptr<__nv_fp8_e4m3>(out), seq_len, dim, eps,
                     reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    m.def("ada_rms_norm_style_fp8", [](uintptr_t x, uintptr_t weight, uintptr_t style,
                                        uintptr_t out, uintptr_t gate_out,
                                        int seq_len, int dim, float eps,
                                        uintptr_t d_scale, uintptr_t stream) {
        ada_rms_norm_style_fp8(typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(weight),
                               typed_ptr<__nv_bfloat16>(style),
                               typed_ptr<__nv_fp8_e4m3>(out), typed_ptr<__nv_bfloat16>(gate_out),
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
        residual_add_rms_norm_fp8(typed_ptr<__nv_bfloat16>(residual),
                                   typed_ptr<__nv_bfloat16>(x),
                                   typed_ptr<__nv_bfloat16>(weight),
                                   typed_ptr<__nv_fp8_e4m3>(out),
                                   seq_len, dim, eps,
                                   reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    m.def("residual_add_rms_norm", [](uintptr_t residual, uintptr_t x,
                                       uintptr_t weight, uintptr_t out,
                                       int seq_len, int dim, float eps, uintptr_t stream) {
        residual_add_rms_norm(typed_ptr<__nv_bfloat16>(residual),
                               typed_ptr<__nv_bfloat16>(x),
                               typed_ptr<__nv_bfloat16>(weight),
                               typed_ptr<__nv_bfloat16>(out),
                               seq_len, dim, eps, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    // Activation — GEGLU (tanh-approx GELU(gate) * up), not SiLU.
    m.def("gate_geglu", [](uintptr_t gate, uintptr_t up, uintptr_t out, int n, uintptr_t stream) {
        gate_silu_mul(typed_ptr<__nv_bfloat16>(gate), typed_ptr<__nv_bfloat16>(up),
                      typed_ptr<__nv_bfloat16>(out), n, to_stream(stream));
    }, py::arg("gate"), py::arg("up"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);

    m.def("gate_geglu_fp16", [](uintptr_t gate, uintptr_t up, uintptr_t out, int n, uintptr_t stream) {
        gate_silu_mul_fp16(typed_ptr<__half>(gate), typed_ptr<__half>(up),
                           typed_ptr<__half>(out), n, to_stream(stream));
    }, py::arg("gate"), py::arg("up"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);

    m.def("gelu_inplace", [](uintptr_t x, int n, uintptr_t stream) {
        gelu_inplace(typed_ptr<__nv_bfloat16>(x), n, to_stream(stream));
    }, py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    m.def("gate_geglu_merged", [](uintptr_t merged, uintptr_t out,
                                   int seq, int half_dim, uintptr_t stream) {
        gate_silu_mul_merged(typed_ptr<__nv_bfloat16>(merged),
                              typed_ptr<__nv_bfloat16>(out), seq, half_dim, to_stream(stream));
    }, py::arg("merged"), py::arg("out"), py::arg("seq"), py::arg("half_dim"), py::arg("stream") = 0);

    m.def("gate_geglu_merged_fp8", [](uintptr_t merged, uintptr_t out,
                                       int seq, int half_dim,
                                       uintptr_t d_scale, uintptr_t stream) {
        gate_silu_mul_merged_fp8(typed_ptr<__nv_bfloat16>(merged),
                                  typed_ptr<__nv_fp8_e4m3>(out), seq, half_dim,
                                  reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("merged"), py::arg("out"), py::arg("seq"), py::arg("half_dim"),
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    // RoPE
    m.def("rope_apply", [](uintptr_t rope_weights, uintptr_t Q, uintptr_t K,
                            int seq_len, int num_heads, int head_dim, uintptr_t stream) {
        rope_apply(typed_ptr<__nv_bfloat16>(rope_weights),
                   typed_ptr<__nv_bfloat16>(Q), typed_ptr<__nv_bfloat16>(K),
                   seq_len, num_heads, head_dim, to_stream(stream));
    }, py::arg("rope_weights"), py::arg("Q"), py::arg("K"),
       py::arg("seq_len"), py::arg("num_heads"), py::arg("head_dim"), py::arg("stream") = 0);

    m.def("qkv_split", [](uintptr_t qkv, uintptr_t Q, uintptr_t K, uintptr_t V,
                           int seq, int q_dim, int k_dim, int v_dim, uintptr_t stream) {
        qkv_split(typed_ptr<__nv_bfloat16>(qkv),
                   typed_ptr<__nv_bfloat16>(Q), typed_ptr<__nv_bfloat16>(K),
                   typed_ptr<__nv_bfloat16>(V), seq, q_dim, k_dim, v_dim, to_stream(stream));
    }, py::arg("qkv"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("seq"), py::arg("q_dim"), py::arg("k_dim"), py::arg("v_dim"), py::arg("stream") = 0);

    m.def("qkv_split_fp16", [](uintptr_t qkv, uintptr_t Q, uintptr_t K, uintptr_t V,
                                int seq, int q_dim, int k_dim, int v_dim, uintptr_t stream) {
        qkv_split_fp16(typed_ptr<__half>(qkv),
                        typed_ptr<__half>(Q), typed_ptr<__half>(K),
                        typed_ptr<__half>(V), seq, q_dim, k_dim, v_dim, to_stream(stream));
    }, py::arg("qkv"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("seq"), py::arg("q_dim"), py::arg("k_dim"), py::arg("v_dim"), py::arg("stream") = 0);

    m.def("qkv_split_rope", [](uintptr_t qkv, uintptr_t rope_weights,
                                 uintptr_t Q, uintptr_t K, uintptr_t V,
                                 int seq, int q_dim, int k_dim, int v_dim,
                                 int head_dim, uintptr_t stream) {
        qkv_split_rope(typed_ptr<__nv_bfloat16>(qkv), typed_ptr<__nv_bfloat16>(rope_weights),
                        typed_ptr<__nv_bfloat16>(Q), typed_ptr<__nv_bfloat16>(K),
                        typed_ptr<__nv_bfloat16>(V),
                        seq, q_dim, k_dim, v_dim, head_dim, to_stream(stream));
    }, py::arg("qkv"), py::arg("rope_weights"),
       py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("seq"), py::arg("q_dim"), py::arg("k_dim"), py::arg("v_dim"),
       py::arg("head_dim"), py::arg("stream") = 0);

    // Elementwise
    m.def("gate_mul_residual", [](uintptr_t residual, uintptr_t x, uintptr_t gate, int n, uintptr_t stream) {
        gate_mul_residual(typed_ptr<__nv_bfloat16>(residual),
                          typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(gate), n, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("gate"), py::arg("n"), py::arg("stream") = 0);

    m.def("bias_residual", [](uintptr_t residual, uintptr_t x, uintptr_t bias,
                               int seq_len, int dim, uintptr_t stream) {
        bias_residual(typed_ptr<__nv_bfloat16>(residual),
                      typed_ptr<__nv_bfloat16>(x), typed_ptr<__nv_bfloat16>(bias),
                      seq_len, dim, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("bias"),
       py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0);

    m.def("residual_add", [](uintptr_t residual, uintptr_t x, int n, uintptr_t stream) {
        residual_add(typed_ptr<__nv_bfloat16>(residual),
                     typed_ptr<__nv_bfloat16>(x), n, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    m.def("cfg_combine_into_residual",
          [](uintptr_t residual, uintptr_t v_cond, uintptr_t v_uncond,
             float beta, int n, uintptr_t stream) {
        cfg_combine_into_residual(typed_ptr<__nv_bfloat16>(residual),
                                  typed_ptr<__nv_bfloat16>(v_cond),
                                  typed_ptr<__nv_bfloat16>(v_uncond),
                                  beta, n, to_stream(stream));
    }, py::arg("residual"), py::arg("v_cond"), py::arg("v_uncond"),
       py::arg("beta"), py::arg("n"), py::arg("stream") = 0);

    // Fusion
    m.def("gate_residual_ada_norm_fp8", [](uintptr_t residual, uintptr_t x,
                                            uintptr_t gate, uintptr_t weight,
                                            uintptr_t style,
                                            uintptr_t out, uintptr_t gate_out,
                                            int seq_len, int dim, float eps,
                                            uintptr_t d_scale, uintptr_t stream) {
        gate_residual_ada_norm_fp8(typed_ptr<__nv_bfloat16>(residual),
                                    typed_ptr<__nv_bfloat16>(x),
                                    typed_ptr<__nv_bfloat16>(gate),
                                    typed_ptr<__nv_bfloat16>(weight),
                                    typed_ptr<__nv_bfloat16>(style),
                                    typed_ptr<__nv_fp8_e4m3>(out),
                                    typed_ptr<__nv_bfloat16>(gate_out),
                                    seq_len, dim, eps,
                                    reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("gate"), py::arg("weight"),
       py::arg("style"), py::arg("out"), py::arg("gate_out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    // Quantize
    m.def("quantize_fp8", [](uintptr_t input, uintptr_t output,
                              uintptr_t d_scale, int n, uintptr_t stream) {
        return quantize_fp8(typed_ptr<__nv_bfloat16>(input),
                            typed_ptr<__nv_fp8_e4m3>(output),
                            reinterpret_cast<float*>(d_scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scale"), py::arg("n"), py::arg("stream") = 0);

    m.def("quantize_fp8_static", [](uintptr_t input, uintptr_t output,
                                     uintptr_t d_scale, int n, uintptr_t stream) {
        quantize_fp8_static(typed_ptr<__nv_bfloat16>(input),
                            typed_ptr<__nv_fp8_e4m3>(output),
                            reinterpret_cast<const float*>(d_scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scale"), py::arg("n"), py::arg("stream") = 0);

    m.def("quantize_fp8_device", [](uintptr_t input, uintptr_t output,
                                     uintptr_t d_scale, int n, uintptr_t stream) {
        quantize_fp8_device(typed_ptr<__nv_bfloat16>(input),
                            typed_ptr<__nv_fp8_e4m3>(output),
                            reinterpret_cast<float*>(d_scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scale"), py::arg("n"), py::arg("stream") = 0);

    // FP16 device-only FP8 quantize (GPU absmax + scale + quantize, CUDA Graph compatible)
    m.def("quantize_fp8_device_fp16", [](uintptr_t input, uintptr_t output,
                                          uintptr_t d_scale, int n, uintptr_t stream) {
        quantize_fp8_device_fp16(reinterpret_cast<const __half*>(input),
                                  typed_ptr<__nv_fp8_e4m3>(output),
                                  reinterpret_cast<float*>(d_scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scale"), py::arg("n"), py::arg("stream") = 0);

#ifdef ENABLE_NVFP4
    m.def("quantize_bf16_to_nvfp4", [](uintptr_t input, uintptr_t fp4_data,
                                         uintptr_t scale_factors, int rows, int cols,
                                         uintptr_t stream) {
        quantize_bf16_to_nvfp4(typed_ptr<__nv_bfloat16>(input),
                                reinterpret_cast<uint8_t*>(fp4_data),
                                reinterpret_cast<uint8_t*>(scale_factors),
                                rows, cols, to_stream(stream));
    }, py::arg("input"), py::arg("fp4_data"), py::arg("scale_factors"),
       py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);

    m.def("quantize_bf16_to_nvfp4_swizzled", [](uintptr_t input, uintptr_t fp4_data,
                                                  uintptr_t scale_factors, int rows, int cols,
                                                  uintptr_t stream) {
        quantize_bf16_to_nvfp4_swizzled(typed_ptr<__nv_bfloat16>(input),
                                         reinterpret_cast<uint8_t*>(fp4_data),
                                         reinterpret_cast<uint8_t*>(scale_factors),
                                         rows, cols, to_stream(stream));
    }, py::arg("input"), py::arg("fp4_data"), py::arg("scale_factors"),
       py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);
#endif

    // Patch embedding
    m.def("patch_im2col", [](uintptr_t input, uintptr_t output, int nv, uintptr_t stream) {
        patch_im2col(reinterpret_cast<const half*>(input),
                     reinterpret_cast<half*>(output), nv, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("nv"), py::arg("stream") = 0);

    m.def("patch_embed_bias_pos", [](uintptr_t output, uintptr_t bias, uintptr_t pos_emb,
                                      int S, int D, int S_per_view, uintptr_t stream) {
        patch_embed_bias_pos(reinterpret_cast<half*>(output),
                             reinterpret_cast<const half*>(bias),
                             reinterpret_cast<const half*>(pos_emb),
                             S, D, S_per_view, to_stream(stream));
    }, py::arg("output"), py::arg("bias"), py::arg("pos_emb"),
       py::arg("S"), py::arg("D"), py::arg("S_per_view"), py::arg("stream") = 0);

    // ── FP16 variants (Thor SM110 path) ──
    // All use uintptr_t for pointers, same as BF16 versions.

    // Norm FP16
    m.def("rms_norm_fp16", [](uintptr_t x, uintptr_t weight, uintptr_t out,
                               int seq_len, int dim, float eps, uintptr_t stream) {
        rms_norm_fp16(reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(weight),
                       reinterpret_cast<__half*>(out), seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("layer_norm_fp16", [](uintptr_t x, uintptr_t weight, uintptr_t bias,
                                 uintptr_t out, int seq_len, int dim, float eps, uintptr_t stream) {
        layer_norm_fp16(reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(weight),
                         reinterpret_cast<const __half*>(bias), reinterpret_cast<__half*>(out),
                         seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("layer_norm_fp8", [](uintptr_t x, uintptr_t out, uintptr_t gamma, uintptr_t beta,
                                int seq_len, int dim, float eps, uintptr_t stream) {
        layer_norm_fp8(reinterpret_cast<const __half*>(x), reinterpret_cast<__nv_fp8_e4m3*>(out),
                        reinterpret_cast<const __half*>(gamma), reinterpret_cast<const __half*>(beta),
                        seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("out"), py::arg("gamma"), py::arg("beta"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    // QKV Split + RoPE + KV Cache (FP16, matches pi05 qkv_split_rope_kvcache_k)
    m.def("qkv_split_rope_kvcache_fp16", [](uintptr_t qkv, uintptr_t rope,
                                              uintptr_t Q, uintptr_t Kc, uintptr_t Vc,
                                              int S, int Q_dim, int K_dim, int HD, int qkv_stride,
                                              int kc_offset, int kc_stride, uintptr_t stream) {
        qkv_split_rope_kvcache_fp16(reinterpret_cast<const __half*>(qkv),
                                     reinterpret_cast<const __half*>(rope),
                                     reinterpret_cast<__half*>(Q),
                                     reinterpret_cast<__half*>(Kc),
                                     reinterpret_cast<__half*>(Vc),
                                     S, Q_dim, K_dim, HD, qkv_stride,
                                     kc_offset, kc_stride, to_stream(stream));
    }, py::arg("qkv"), py::arg("rope"), py::arg("Q"), py::arg("Kc"), py::arg("Vc"),
       py::arg("S"), py::arg("Q_dim"), py::arg("K_dim"), py::arg("HD"), py::arg("qkv_stride"),
       py::arg("kc_offset"), py::arg("kc_stride"), py::arg("stream") = 0);

    // Elementwise FP16
    m.def("bias_residual_fp16", [](uintptr_t residual, uintptr_t x, uintptr_t bias,
                                    int seq_len, int dim, uintptr_t stream) {
        bias_residual_fp16(reinterpret_cast<__half*>(residual), reinterpret_cast<const __half*>(x),
                            reinterpret_cast<const __half*>(bias), seq_len, dim, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("bias"),
       py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0);

    m.def("residual_add_fp16", [](uintptr_t residual, uintptr_t x, int n, uintptr_t stream) {
        residual_add_fp16(reinterpret_cast<__half*>(residual), reinterpret_cast<const __half*>(x),
                           n, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    m.def("cfg_combine_into_residual_fp16",
          [](uintptr_t residual, uintptr_t v_cond, uintptr_t v_uncond,
             float beta, int n, uintptr_t stream) {
        cfg_combine_into_residual_fp16(reinterpret_cast<__half*>(residual),
                                       reinterpret_cast<const __half*>(v_cond),
                                       reinterpret_cast<const __half*>(v_uncond),
                                       beta, n, to_stream(stream));
    }, py::arg("residual"), py::arg("v_cond"), py::arg("v_uncond"),
       py::arg("beta"), py::arg("n"), py::arg("stream") = 0);

    // Activation FP16
    m.def("gelu_inplace_fp16", [](uintptr_t x, int n, uintptr_t stream) {
        gelu_inplace_fp16(reinterpret_cast<__half*>(x), n, to_stream(stream));
    }, py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    m.def("gate_geglu_merged_fp16", [](uintptr_t merged, uintptr_t out,
                                        int seq, int half_dim, uintptr_t stream) {
        gate_silu_mul_merged_fp16(reinterpret_cast<const __half*>(merged),
                                   reinterpret_cast<__half*>(out), seq, half_dim, to_stream(stream));
    }, py::arg("merged"), py::arg("out"), py::arg("seq"), py::arg("half_dim"), py::arg("stream") = 0);

    // Merged GEGLU (tanh-approx GELU) → FP8 (FP16 input, matches pi05 FFN quant path)
    m.def("gate_geglu_merged_fp8_fp16", [](uintptr_t merged, uintptr_t out,
                                            int seq, int half_dim,
                                            uintptr_t d_scale, uintptr_t stream) {
        gate_silu_mul_merged_fp8_fp16(reinterpret_cast<const __half*>(merged),
                                       typed_ptr<__nv_fp8_e4m3>(out), seq, half_dim,
                                       reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("merged"), py::arg("out"), py::arg("seq"), py::arg("half_dim"),
       py::arg("d_scale"), py::arg("stream") = 0);

    // Norm FP16 → FP8 (fused, with scale)
    m.def("rms_norm_fp8_fp16", [](uintptr_t x, uintptr_t weight, uintptr_t out,
                                    int seq_len, int dim, float eps,
                                    uintptr_t d_scale, uintptr_t stream) {
        rms_norm_fp8_fp16(reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(weight),
                           typed_ptr<__nv_fp8_e4m3>(out), seq_len, dim, eps,
                           reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    // RMSNorm → FP8 without weight (FP16, verbatim production rms_norm_fp8_static_k)
    m.def("rms_norm_fp8_noweight_fp16", [](uintptr_t x, uintptr_t out,
                                            int seq_len, int dim,
                                            uintptr_t d_scale, uintptr_t stream) {
        rms_norm_fp8_noweight_fp16(reinterpret_cast<const __half*>(x),
                                    typed_ptr<__nv_fp8_e4m3>(out), seq_len, dim,
                                    reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("x"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"),
       py::arg("d_scale"), py::arg("stream") = 0);

    // Residual + RMSNorm → FP8 without weight (FP16, matches production res_rms_fp8_static_k)
    m.def("residual_add_rms_norm_fp8_noweight_fp16", [](uintptr_t residual, uintptr_t x,
                                                          uintptr_t out,
                                                          int seq_len, int dim,
                                                          uintptr_t d_scale, uintptr_t stream) {
        residual_add_rms_norm_fp8_noweight_fp16(reinterpret_cast<__half*>(residual),
                                                  reinterpret_cast<const __half*>(x),
                                                  typed_ptr<__nv_fp8_e4m3>(out), seq_len, dim,
                                                  reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"),
       py::arg("d_scale"), py::arg("stream") = 0);

    // Residual + RMSNorm → FP8 (FP16)
    m.def("residual_add_rms_norm_fp8_fp16", [](uintptr_t residual, uintptr_t x,
                                                 uintptr_t weight, uintptr_t out,
                                                 int seq_len, int dim, float eps,
                                                 uintptr_t d_scale, uintptr_t stream) {
        residual_add_rms_norm_fp8_fp16(reinterpret_cast<__half*>(residual),
                                        reinterpret_cast<const __half*>(x),
                                        reinterpret_cast<const __half*>(weight),
                                        typed_ptr<__nv_fp8_e4m3>(out), seq_len, dim, eps,
                                        reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f,
       py::arg("d_scale") = 0, py::arg("stream") = 0);

    // Split SiLU → FP8 (FP16, separate gate/up)
    m.def("silu_mul_split_fp8_fp16", [](uintptr_t gate, uintptr_t up, uintptr_t out,
                                         int n, uintptr_t d_scale, uintptr_t stream) {
        silu_mul_split_fp8_fp16(reinterpret_cast<const __half*>(gate),
                                 reinterpret_cast<const __half*>(up),
                                 typed_ptr<__nv_fp8_e4m3>(out), n,
                                 reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("gate"), py::arg("up"), py::arg("out"),
       py::arg("n"), py::arg("d_scale"), py::arg("stream") = 0);

    // ── Production-exact kernels (no weight, no scale) ──

    // RMSNorm → FP8 (no weight, no d_scale). Matches pi05 fused_rms_fp8.
    m.def("plain_rms_fp8_fp16", [](uintptr_t x, uintptr_t out,
                                     int seq_len, int dim, uintptr_t stream) {
        plain_rms_fp8_fp16(reinterpret_cast<const __half*>(x),
                            typed_ptr<__nv_fp8_e4m3>(out), seq_len, dim, to_stream(stream));
    }, py::arg("x"), py::arg("out"), py::arg("seq_len"), py::arg("dim"),
       py::arg("stream") = 0);

    // Residual + RMSNorm → FP8 (no weight, no d_scale). Matches pi05 res_rms_fp8_k.
    m.def("plain_res_rms_fp8_fp16", [](uintptr_t residual, uintptr_t x,
                                         uintptr_t out, int seq_len, int dim,
                                         uintptr_t stream) {
        plain_res_rms_fp8_fp16(reinterpret_cast<__half*>(residual),
                                reinterpret_cast<const __half*>(x),
                                typed_ptr<__nv_fp8_e4m3>(out), seq_len, dim, to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0);

    // Cast FP16 → FP8 (no scale). Matches pi05 cast_fp16_fp8_k.
    m.def("cast_fp16_fp8", [](uintptr_t input, uintptr_t output, int n, uintptr_t stream) {
        cast_fp16_fp8(reinterpret_cast<const __half*>(input),
                       typed_ptr<__nv_fp8_e4m3>(output), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("n"), py::arg("stream") = 0);

    // Quantize FP16→FP8
    m.def("quantize_fp8_static_fp16", [](uintptr_t input, uintptr_t output,
                                          uintptr_t d_scale, int n, uintptr_t stream) {
        quantize_fp8_static_fp16(reinterpret_cast<const __half*>(input),
                                  typed_ptr<__nv_fp8_e4m3>(output),
                                  reinterpret_cast<const float*>(d_scale), n, to_stream(stream));
    }, py::arg("input"), py::arg("output"), py::arg("d_scale"), py::arg("n"), py::arg("stream") = 0);

    // ── Decoder fused kernels (FP16, matching pi05 ae_forward_static) ──
    m.def("fused_adarms_fp8_static_fp16", [](uintptr_t x, uintptr_t style,
            uintptr_t out, uintptr_t gate_out, int S, int D, uintptr_t descale, uintptr_t stream) {
        fused_adarms_fp8_static_fp16(reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(style),
            typed_ptr<__nv_fp8_e4m3>(out), reinterpret_cast<__half*>(gate_out),
            S, D, reinterpret_cast<const float*>(descale), to_stream(stream));
    }, py::arg("x"), py::arg("style"), py::arg("out"), py::arg("gate_out"),
       py::arg("S"), py::arg("D"), py::arg("descale"), py::arg("stream") = 0);

    m.def("gate_res_adarms_fp8_static_fp16", [](uintptr_t gemm_out, uintptr_t prev_gate,
            uintptr_t residual, uintptr_t style, uintptr_t fp8_out, uintptr_t gate_out,
            int S, int D, uintptr_t descale, uintptr_t stream) {
        gate_res_adarms_fp8_static_fp16(reinterpret_cast<const __half*>(gemm_out),
            reinterpret_cast<const __half*>(prev_gate), reinterpret_cast<__half*>(residual),
            reinterpret_cast<const __half*>(style), typed_ptr<__nv_fp8_e4m3>(fp8_out),
            reinterpret_cast<__half*>(gate_out), S, D, reinterpret_cast<const float*>(descale), to_stream(stream));
    }, py::arg("gemm_out"), py::arg("prev_gate"), py::arg("residual"), py::arg("style"),
       py::arg("fp8_out"), py::arg("gate_out"), py::arg("S"), py::arg("D"), py::arg("descale"), py::arg("stream") = 0);

    m.def("geglu_fp8_static_fp16", [](uintptr_t merged, uintptr_t out, int S, int H,
            uintptr_t descale, uintptr_t stream) {
        geglu_fp8_static_fp16(reinterpret_cast<const __half*>(merged), typed_ptr<__nv_fp8_e4m3>(out),
            S, H, reinterpret_cast<const float*>(descale), to_stream(stream));
    }, py::arg("merged"), py::arg("out"), py::arg("S"), py::arg("H"), py::arg("descale"), py::arg("stream") = 0);

    m.def("gate_res_fp16", [](uintptr_t gemm_out, uintptr_t gate, uintptr_t residual, int n, uintptr_t stream) {
        gate_res_fp16(reinterpret_cast<const __half*>(gemm_out), reinterpret_cast<const __half*>(gate),
            reinterpret_cast<__half*>(residual), n, to_stream(stream));
    }, py::arg("gemm_out"), py::arg("gate"), py::arg("residual"), py::arg("n"), py::arg("stream") = 0);

    m.def("adarms_fp16", [](uintptr_t x, uintptr_t style, uintptr_t out, uintptr_t gate_out,
            int S, int D, uintptr_t stream) {
        adarms_fp16(reinterpret_cast<const __half*>(x), reinterpret_cast<const __half*>(style),
            reinterpret_cast<__half*>(out), reinterpret_cast<__half*>(gate_out), S, D, to_stream(stream));
    }, py::arg("x"), py::arg("style"), py::arg("out"), py::arg("gate_out"),
       py::arg("S"), py::arg("D"), py::arg("stream") = 0);

    // Simple bias add (pi05 bias_k)
    m.def("add_bias_fp16", [](uintptr_t x, uintptr_t b, int S, int D, uintptr_t stream) {
        add_bias_fp16(reinterpret_cast<__half*>(x), reinterpret_cast<const __half*>(b),
                       S, D, to_stream(stream));
    }, py::arg("x"), py::arg("b"), py::arg("S"), py::arg("D"), py::arg("stream") = 0);

    // cuBLAS NN GEMM: C = A @ B + beta * C (pi05 gmm)
    // Requires FvkContext for cuBLAS handle.
    m.def("gmm_fp16", [](FvkContext& ctx, uintptr_t A, uintptr_t B, uintptr_t C,
                           int M, int N, int K, float beta, uintptr_t stream) {
        gmm_fp16(ctx.cublas_handle,
                  reinterpret_cast<const __half*>(A), reinterpret_cast<const __half*>(B),
                  reinterpret_cast<__half*>(C), M, N, K, beta, to_stream(stream));
    }, py::arg("ctx"), py::arg("A"), py::arg("B"), py::arg("C"),
       py::arg("M"), py::arg("N"), py::arg("K"), py::arg("beta") = 0.0f, py::arg("stream") = 0);

    // FP8 GEMM with device descale → FP16 output (pi05 gmm_fp8_kn_descale)
    m.def("fp8_gemm_descale_fp16", [](uintptr_t A, uintptr_t B, uintptr_t C,
            int M, int N, int K, uintptr_t act_descale, uintptr_t w_descale, uintptr_t stream) {
        fp8_gemm_descale_fp16(to_ptr(A), to_ptr(B), to_ptr(C), M, N, K,
            reinterpret_cast<const float*>(act_descale), reinterpret_cast<const float*>(w_descale),
            to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("C"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("act_descale"), py::arg("w_descale"), py::arg("stream") = 0);

    // FP8 GEMM with device descale → FP32 output (for models with activations > FP16 range)
    m.def("fp8_gemm_descale_f32out", [](uintptr_t A, uintptr_t B, uintptr_t C,
            int M, int N, int K, uintptr_t act_descale, uintptr_t w_descale, uintptr_t stream) {
        fp8_gemm_descale_f32out(to_ptr(A), to_ptr(B), to_ptr(C), M, N, K,
            reinterpret_cast<const float*>(act_descale), reinterpret_cast<const float*>(w_descale),
            to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("C"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("act_descale"), py::arg("w_descale"), py::arg("stream") = 0);

    // FP8 GEMM with device descale → BF16 output (Pi0-FAST decode_step path)
    m.def("fp8_gemm_descale_bf16out", [](uintptr_t A, uintptr_t B, uintptr_t C,
            int M, int N, int K, uintptr_t act_descale, uintptr_t w_descale, uintptr_t stream) {
        fp8_gemm_descale_bf16out(to_ptr(A), to_ptr(B), to_ptr(C), M, N, K,
            reinterpret_cast<const float*>(act_descale), reinterpret_cast<const float*>(w_descale),
            to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("C"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("act_descale"), py::arg("w_descale"), py::arg("stream") = 0);

    // cuBLAS decomposed attention (GQA, matching pi05 engine)
    // Requires FvkContext for cuBLAS handle.
    m.def("attention_qkv_fp16", [](FvkContext& ctx, uintptr_t Q, uintptr_t K, uintptr_t V,
                                    uintptr_t logits, uintptr_t out,
                                    int S, int S_kv, int NH, int HD,
                                    float attn_scale, uintptr_t stream) {
        attention_qkv_fp16(ctx.cublas_handle,
                            reinterpret_cast<const __half*>(Q),
                            reinterpret_cast<const __half*>(K),
                            reinterpret_cast<const __half*>(V),
                            reinterpret_cast<__half*>(logits),
                            reinterpret_cast<__half*>(out),
                            S, S_kv, NH, HD, attn_scale, to_stream(stream));
    }, py::arg("ctx"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("logits"), py::arg("out"),
       py::arg("S"), py::arg("S_kv"), py::arg("NH"), py::arg("HD"),
       py::arg("attn_scale") = 1.0f, py::arg("stream") = 0);

    // Padded attention: supports odd S_kv (pads logits lda to even).
    // logits buffer must have room for S*NH * (S_kv + S_kv%2) elements.
    m.def("attention_qkv_fp16_padded", [](FvkContext& ctx, uintptr_t Q, uintptr_t K, uintptr_t V,
                                    uintptr_t logits, uintptr_t out,
                                    int S, int S_kv, int NH, int HD,
                                    float attn_scale, uintptr_t stream) {
        attention_qkv_fp16_padded(ctx.cublas_handle,
                            reinterpret_cast<const __half*>(Q),
                            reinterpret_cast<const __half*>(K),
                            reinterpret_cast<const __half*>(V),
                            reinterpret_cast<__half*>(logits),
                            reinterpret_cast<__half*>(out),
                            S, S_kv, NH, HD, attn_scale, to_stream(stream));
    }, py::arg("ctx"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("logits"), py::arg("out"),
       py::arg("S"), py::arg("S_kv"), py::arg("NH"), py::arg("HD"),
       py::arg("attn_scale") = 1.0f, py::arg("stream") = 0);

    // State-masked attention: single call with AR mask for Pi0 state token.
    m.def("attention_qkv_fp16_state_masked", [](FvkContext& ctx, uintptr_t Q, uintptr_t K, uintptr_t V,
                                    uintptr_t logits, uintptr_t out,
                                    int S, int S_kv, int NH, int HD,
                                    int state_nk, float attn_scale, uintptr_t stream) {
        attention_qkv_fp16_state_masked(ctx.cublas_handle,
                            reinterpret_cast<const __half*>(Q),
                            reinterpret_cast<const __half*>(K),
                            reinterpret_cast<const __half*>(V),
                            reinterpret_cast<__half*>(logits),
                            reinterpret_cast<__half*>(out),
                            S, S_kv, NH, HD, state_nk, attn_scale, to_stream(stream));
    }, py::arg("ctx"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("logits"), py::arg("out"),
       py::arg("S"), py::arg("S_kv"), py::arg("NH"), py::arg("HD"),
       py::arg("state_nk"), py::arg("attn_scale") = 1.0f, py::arg("stream") = 0);

    m.def("softmax_fp16", [](uintptr_t data, int rows, int cols, uintptr_t stream) {
        softmax_fp16(reinterpret_cast<__half*>(data), rows, cols, to_stream(stream));
    }, py::arg("data"), py::arg("rows"), py::arg("cols"), py::arg("stream") = 0);

    // ── CUTLASS FP8 GEMMs (SM100/SM110, pi05-equivalent tile configs) ──
#ifdef ENABLE_SM100_CUTLASS
    m.def("cutlass_fp8_sq", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                 int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_sq(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp8_t1", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                 int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_t1(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp8_wide", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                   int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_wide(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp8_plain", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                    int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_plain(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp8_gelu", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                   int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_gelu(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    // FP32 output variants — for models with activations exceeding FP16 range
    m.def("cutlass_fp8_sq_f32out", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                       int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_sq_f32out(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp8_wide_f32out", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                         int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_wide_f32out(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    // BF16 output variants — for models trained in BF16 with large activations
    m.def("cutlass_fp8_sq_bf16out", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                        int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_sq_bf16out(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp8_wide_bf16out", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                          int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_wide_bf16out(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("cutlass_fp8_t1_bf16out", [](uintptr_t A, uintptr_t B, uintptr_t D,
                                        int M, int N, int K, float alpha, float beta, uintptr_t stream) {
        return cutlass_fp8_t1_bf16out(to_ptr(A), to_ptr(B), to_ptr(D), M, N, K, alpha, beta, to_stream(stream));
    }, py::arg("A"), py::arg("B"), py::arg("D"),
       py::arg("M"), py::arg("N"), py::arg("K"),
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("stream") = 0);

    m.def("has_cutlass_sm100", []() { return true; });
#else
    m.def("has_cutlass_sm100", []() { return false; });
#endif

    // BF16 noweight norm kernels (always available, not SM100-gated)
    m.def("rms_norm_fp8_noweight_bf16", [](uintptr_t x, uintptr_t out,
            int seq_len, int dim, uintptr_t d_scale, uintptr_t stream) {
        rms_norm_fp8_noweight_bf16(reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<__nv_fp8_e4m3*>(out), seq_len, dim,
            reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("x"), py::arg("out"), py::arg("seq_len"), py::arg("dim"),
       py::arg("d_scale"), py::arg("stream") = 0);

    m.def("residual_add_rms_norm_fp8_noweight_bf16", [](uintptr_t residual, uintptr_t x,
            uintptr_t out, int seq_len, int dim, uintptr_t d_scale, uintptr_t stream) {
        residual_add_rms_norm_fp8_noweight_bf16(reinterpret_cast<__nv_bfloat16*>(residual),
            reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_fp8_e4m3*>(out),
            seq_len, dim, reinterpret_cast<const float*>(d_scale), to_stream(stream));
    }, py::arg("residual"), py::arg("x"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("d_scale"), py::arg("stream") = 0);

    // Hardware info
    m.def("get_sm_version", []() {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        return prop.major * 10 + prop.minor;
    });

#ifdef ENABLE_NVFP4
    m.def("has_nvfp4", []() { return true; });
#else
    m.def("has_nvfp4", []() { return false; });
#endif

    // ── Attention dispatch ──
    m.def("load_fmha_library", [](const std::string& path) {
        return load_fmha_library(path.c_str());
    }, py::arg("path"));

    m.def("load_fmha_strided_library", [](const std::string& path) {
        return load_fmha_strided_library(path.c_str());
    }, py::arg("path"));

    m.def("has_cutlass_fmha", &has_cutlass_fmha);

    m.def("fmha_forward", [](uintptr_t Q, uintptr_t K, uintptr_t V, uintptr_t O,
                              int seq_q, int seq_kv, int num_heads, int head_dim,
                              float scale, uintptr_t stream) {
        return fmha_forward(typed_ptr<__nv_bfloat16>(Q), typed_ptr<__nv_bfloat16>(K),
                            typed_ptr<__nv_bfloat16>(V), typed_ptr<__nv_bfloat16>(O),
                            seq_q, seq_kv, num_heads, head_dim, scale, to_stream(stream));
    }, py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
       py::arg("seq_q"), py::arg("seq_kv"), py::arg("num_heads"), py::arg("head_dim"),
       py::arg("scale") = 1.0f, py::arg("stream") = 0);

    m.def("fmha_strided_forward", [](uintptr_t qkv_buf, uintptr_t O,
                                      int seq, int num_heads, int head_dim,
                                      float scale, uintptr_t stream) {
        return fmha_strided_forward(typed_ptr<__nv_bfloat16>(qkv_buf),
                                     typed_ptr<__nv_bfloat16>(O),
                                     seq, num_heads, head_dim, scale, to_stream(stream));
    }, py::arg("qkv_buf"), py::arg("O"),
       py::arg("seq"), py::arg("num_heads"), py::arg("head_dim"),
       py::arg("scale") = 1.0f, py::arg("stream") = 0);

    // Full strided FMHA: Q/K/V separate pointers + batch + strides
    // Used by SigLIP multi-view: batch=NV, seq=256, stride=3*D
    m.def("fmha_strided_full", [](uintptr_t Q, uintptr_t K, uintptr_t V, uintptr_t O,
                                   int batch, int seq_q, int seq_kv,
                                   int nheads_q, int nheads_kv, int head_dim,
                                   int stride_q, int stride_kv, uintptr_t stream) {
        return fmha_strided_full(to_ptr(Q), to_ptr(K), to_ptr(V), to_ptr(O),
                                  batch, seq_q, seq_kv, nheads_q, nheads_kv, head_dim,
                                  stride_q, stride_kv, to_stream(stream));
    }, py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("O"),
       py::arg("batch"), py::arg("seq_q"), py::arg("seq_kv"),
       py::arg("nheads_q"), py::arg("nheads_kv"), py::arg("head_dim"),
       py::arg("stride_q"), py::arg("stride_kv"), py::arg("stream") = 0);

    // ── DiT kernels (GROOT N1.6) ──

    // LayerNorm without affine parameters (elementwise_affine=False)
    m.def("layer_norm_no_affine_fp16", [](uintptr_t x, uintptr_t out,
                                           int seq_len, int dim, float eps, uintptr_t stream) {
        extern void layer_norm_no_affine_fp16(const __half*, __half*, int, int, float, cudaStream_t);
        layer_norm_no_affine_fp16(reinterpret_cast<const __half*>(x),
                                   reinterpret_cast<__half*>(out),
                                   seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-5f, py::arg("stream") = 0);

    // Fused AdaLayerNorm: LN(x, no_affine) * (1 + scale) + shift
    m.def("ada_layer_norm_fp16", [](uintptr_t x, uintptr_t scale, uintptr_t shift,
                                     uintptr_t out, int seq_len, int dim, float eps, uintptr_t stream) {
        extern void ada_layer_norm_fp16(const __half*, const __half*, const __half*,
                                         __half*, int, int, float, cudaStream_t);
        ada_layer_norm_fp16(reinterpret_cast<const __half*>(x),
                             reinterpret_cast<const __half*>(scale),
                             reinterpret_cast<const __half*>(shift),
                             reinterpret_cast<__half*>(out),
                             seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("scale"), py::arg("shift"),
       py::arg("out"), py::arg("seq_len"), py::arg("dim"),
       py::arg("eps") = 1e-5f, py::arg("stream") = 0);

    // ── GPU memory ops (CUDA Graph compatible, explicit stream) ──
    m.def("gpu_copy", [](uintptr_t dst, uintptr_t src, int nbytes, uintptr_t stream) {
        extern void gpu_copy_async(void*, const void*, size_t, cudaStream_t);
        gpu_copy_async(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src),
                        nbytes, to_stream(stream));
    }, py::arg("dst"), py::arg("src"), py::arg("nbytes"), py::arg("stream") = 0);

    m.def("gpu_fill_neginf_fp16", [](uintptr_t dst, int n, uintptr_t stream) {
        extern void gpu_fill_neginf_fp16(__half*, int, cudaStream_t);
        gpu_fill_neginf_fp16(reinterpret_cast<__half*>(dst), n, to_stream(stream));
    }, py::arg("dst"), py::arg("n"), py::arg("stream") = 0);

    m.def("gpu_strided_copy_fp16", [](uintptr_t src, uintptr_t dst,
                                       int rows, int dst_cols, int src_stride, int col_offset,
                                       uintptr_t stream) {
        extern void gpu_strided_copy_fp16(const __half*, __half*, int, int, int, int, cudaStream_t);
        gpu_strided_copy_fp16(reinterpret_cast<const __half*>(src), reinterpret_cast<__half*>(dst),
                               rows, dst_cols, src_stride, col_offset, to_stream(stream));
    }, py::arg("src"), py::arg("dst"),
       py::arg("rows"), py::arg("dst_cols"), py::arg("src_stride"), py::arg("col_offset"),
       py::arg("stream") = 0);

    m.def("gpu_cast_fp32_to_fp16", [](uintptr_t src, uintptr_t dst, int n, uintptr_t stream) {
        extern void gpu_cast_fp32_to_fp16(const float*, __half*, int, cudaStream_t);
        gpu_cast_fp32_to_fp16(reinterpret_cast<const float*>(src),
                               reinterpret_cast<__half*>(dst), n, to_stream(stream));
    }, py::arg("src"), py::arg("dst"), py::arg("n"), py::arg("stream") = 0);

    m.def("gpu_euler_step", [](uintptr_t actions, uintptr_t velocity,
                                int T, int action_dim, float dt, int vel_elem_offset,
                                uintptr_t stream) {
        extern void gpu_euler_step(float*, const __half*, int, int, float, int, cudaStream_t);
        gpu_euler_step(reinterpret_cast<float*>(actions),
                        reinterpret_cast<const __half*>(velocity),
                        T, action_dim, dt, vel_elem_offset, to_stream(stream));
    }, py::arg("actions"), py::arg("velocity"),
       py::arg("T"), py::arg("action_dim"), py::arg("dt"), py::arg("vel_elem_offset"),
       py::arg("stream") = 0);

    // SiLU in-place FP16 (for DiT action encoder)
    m.def("silu_inplace_fp16", [](uintptr_t x, int n, uintptr_t stream) {
        extern void silu_inplace_fp16(__half*, int, cudaStream_t);
        silu_inplace_fp16(reinterpret_cast<__half*>(x), n, to_stream(stream));
    }, py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    // Fused add + SiLU in-place: a = silu(a + b). Used by Pi0 action_time_mlp.
    m.def("fused_add_silu_fp16", [](uintptr_t a, uintptr_t b, int n, uintptr_t stream) {
        extern void fused_add_silu_fp16(__half*, const __half*, int, cudaStream_t);
        fused_add_silu_fp16(reinterpret_cast<__half*>(a),
                            reinterpret_cast<const __half*>(b), n, to_stream(stream));
    }, py::arg("a"), py::arg("b"), py::arg("n"), py::arg("stream") = 0);

    m.def("fused_add_silu_bf16", [](uintptr_t a, uintptr_t b, int n, uintptr_t stream) {
        extern void fused_add_silu_bf16(__nv_bfloat16*, const __nv_bfloat16*, int, cudaStream_t);
        fused_add_silu_bf16(reinterpret_cast<__nv_bfloat16*>(a),
                            reinterpret_cast<const __nv_bfloat16*>(b), n, to_stream(stream));
    }, py::arg("a"), py::arg("b"), py::arg("n"), py::arg("stream") = 0);

    // ReLU in-place FP16 (for DiT action decoder)
    m.def("relu_inplace_fp16", [](uintptr_t x, int n, uintptr_t stream) {
        extern void relu_inplace_fp16(__half*, int, cudaStream_t);
        relu_inplace_fp16(reinterpret_cast<__half*>(x), n, to_stream(stream));
    }, py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    // GQA KV repeat interleave (for Qwen3 8→16 heads)
    m.def("gpu_repeat_interleave_heads", [](uintptr_t src, uintptr_t dst,
                                             int S, int NH_src, int HD, int repeat, uintptr_t stream) {
        extern void gpu_repeat_interleave_heads(const __half*, __half*, int, int, int, int, cudaStream_t);
        gpu_repeat_interleave_heads(reinterpret_cast<const __half*>(src),
                                     reinterpret_cast<__half*>(dst),
                                     S, NH_src, HD, repeat, to_stream(stream));
    }, py::arg("src"), py::arg("dst"),
       py::arg("S"), py::arg("NH_src"), py::arg("HD"), py::arg("repeat"),
       py::arg("stream") = 0);

    // Qwen3 RoPE (rotate_half style, in-place)
    m.def("rope_rotate_half_fp16", [](uintptr_t x, uintptr_t cos_table, uintptr_t sin_table,
                                       int S, int NH, int HD, uintptr_t stream) {
        extern void rope_rotate_half_fp16(__half*, const __half*, const __half*, int, int, int, cudaStream_t);
        rope_rotate_half_fp16(reinterpret_cast<__half*>(x),
                               reinterpret_cast<const __half*>(cos_table),
                               reinterpret_cast<const __half*>(sin_table),
                               S, NH, HD, to_stream(stream));
    }, py::arg("x"), py::arg("cos_table"), py::arg("sin_table"),
       py::arg("S"), py::arg("NH"), py::arg("HD"), py::arg("stream") = 0);

    // MHA batched cuBLAS attention (for DiT — per-head independent attention)
    m.def("attention_mha_fp16", [](FvkContext& ctx, uintptr_t Q, uintptr_t K, uintptr_t V,
                                    uintptr_t logits, uintptr_t out,
                                    int S_q, int S_kv, int NH, int HD,
                                    float attn_scale, uintptr_t stream) {
        extern void attention_mha_fp16(cublasHandle_t, const __half*, const __half*, const __half*,
                                        __half*, __half*, int, int, int, int, float, cudaStream_t);
        attention_mha_fp16(ctx.cublas_handle,
                            reinterpret_cast<const __half*>(Q),
                            reinterpret_cast<const __half*>(K),
                            reinterpret_cast<const __half*>(V),
                            reinterpret_cast<__half*>(logits),
                            reinterpret_cast<__half*>(out),
                            S_q, S_kv, NH, HD, attn_scale, to_stream(stream));
    }, py::arg("ctx"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("logits"), py::arg("out"),
       py::arg("S_q"), py::arg("S_kv"), py::arg("NH"), py::arg("HD"),
       py::arg("attn_scale") = 1.0f, py::arg("stream") = 0);

    // Causal MHA — N1.7 Qwen3-VL LLM (is_causal=True per HF). Same layout
    // and cuBLAS path as attention_mha_fp16; differs only in the softmax
    // step (strict upper-triangular mask via softmax_causal_fp16).
    m.def("attention_mha_causal_fp16",
          [](FvkContext& ctx, uintptr_t Q, uintptr_t K, uintptr_t V,
             uintptr_t logits, uintptr_t out,
             int S_q, int S_kv, int NH, int HD,
             float attn_scale, uintptr_t stream) {
        extern void attention_mha_causal_fp16(
            cublasHandle_t, const __half*, const __half*, const __half*,
            __half*, __half*, int, int, int, int, float, cudaStream_t);
        attention_mha_causal_fp16(ctx.cublas_handle,
                                   reinterpret_cast<const __half*>(Q),
                                   reinterpret_cast<const __half*>(K),
                                   reinterpret_cast<const __half*>(V),
                                   reinterpret_cast<__half*>(logits),
                                   reinterpret_cast<__half*>(out),
                                   S_q, S_kv, NH, HD, attn_scale,
                                   to_stream(stream));
    }, py::arg("ctx"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("logits"), py::arg("out"),
       py::arg("S_q"), py::arg("S_kv"), py::arg("NH"), py::arg("HD"),
       py::arg("attn_scale") = 1.0f, py::arg("stream") = 0);

    // FA2 bindings (fvk.attention_fa2_fwd_fp16/bf16) moved to a separate
    // pybind module flash_rt_fa2.so — see csrc/fa2_bindings.cpp. This
    // keeps flash_rt_kernels.so small and fast to rebuild.

    // ── DiT bf16 helpers (Phase 5a-2) ────────────────────────────────
    m.def("layer_norm_no_affine_bf16",
          [](uintptr_t x, uintptr_t out, int seq_len, int dim, float eps,
             uintptr_t stream) {
        extern void layer_norm_no_affine_bf16(
            const __nv_bfloat16*, __nv_bfloat16*, int, int, float, cudaStream_t);
        layer_norm_no_affine_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<__nv_bfloat16*>(out),
            seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("out"), py::arg("seq_len"), py::arg("dim"),
       py::arg("eps") = 1e-5f, py::arg("stream") = 0);

    m.def("ada_layer_norm_bf16",
          [](uintptr_t x, uintptr_t scale, uintptr_t shift, uintptr_t out,
             int seq_len, int dim, float eps, uintptr_t stream) {
        extern void ada_layer_norm_bf16(
            const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
            __nv_bfloat16*, int, int, float, cudaStream_t);
        ada_layer_norm_bf16(
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(scale),
            reinterpret_cast<const __nv_bfloat16*>(shift),
            reinterpret_cast<__nv_bfloat16*>(out),
            seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("scale"), py::arg("shift"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"),
       py::arg("eps") = 1e-5f, py::arg("stream") = 0);

    m.def("add_bias_bf16",
          [](uintptr_t x, uintptr_t b, int S, int D, uintptr_t stream) {
        extern void add_bias_bf16(
            __nv_bfloat16*, const __nv_bfloat16*, int, int, cudaStream_t);
        add_bias_bf16(reinterpret_cast<__nv_bfloat16*>(x),
                       reinterpret_cast<const __nv_bfloat16*>(b),
                       S, D, to_stream(stream));
    }, py::arg("x"), py::arg("b"), py::arg("S"), py::arg("D"),
       py::arg("stream") = 0);

    m.def("cast_fp16_to_bf16",
          [](uintptr_t in, uintptr_t out, int n, uintptr_t stream) {
        extern void cast_fp16_to_bf16(
            const __half*, __nv_bfloat16*, int, cudaStream_t);
        cast_fp16_to_bf16(reinterpret_cast<const __half*>(in),
                           reinterpret_cast<__nv_bfloat16*>(out),
                           n, to_stream(stream));
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);

    m.def("cast_bf16_to_fp16",
          [](uintptr_t in, uintptr_t out, int n, uintptr_t stream) {
        extern void cast_bf16_to_fp16(
            const __nv_bfloat16*, __half*, int, cudaStream_t);
        cast_bf16_to_fp16(reinterpret_cast<const __nv_bfloat16*>(in),
                           reinterpret_cast<__half*>(out),
                           n, to_stream(stream));
    }, py::arg("in"), py::arg("out"), py::arg("n"), py::arg("stream") = 0);

    // ── DiT bf16 attention path (Phase 5a-2) ─────────────────────────
    m.def("softmax_bf16", [](uintptr_t data, int rows, int cols,
                              uintptr_t stream) {
        extern void softmax_bf16(__nv_bfloat16*, int, int, cudaStream_t);
        softmax_bf16(reinterpret_cast<__nv_bfloat16*>(data),
                      rows, cols, to_stream(stream));
    }, py::arg("data"), py::arg("rows"), py::arg("cols"),
       py::arg("stream") = 0);

    m.def("gpu_fill_neginf_bf16", [](uintptr_t x, int n, uintptr_t stream) {
        extern void gpu_fill_neginf_bf16(__nv_bfloat16*, int, cudaStream_t);
        gpu_fill_neginf_bf16(reinterpret_cast<__nv_bfloat16*>(x),
                              n, to_stream(stream));
    }, py::arg("x"), py::arg("n"), py::arg("stream") = 0);

    m.def("attention_mha_bf16",
          [](FvkContext& ctx, uintptr_t Q, uintptr_t K, uintptr_t V,
             uintptr_t logits, uintptr_t out,
             int S_q, int S_kv, int NH, int HD,
             float attn_scale, int logits_kv_stride, uintptr_t stream) {
        extern void attention_mha_bf16(
            cublasHandle_t, const __nv_bfloat16*, const __nv_bfloat16*,
            const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*,
            int, int, int, int, float, int, cudaStream_t);
        attention_mha_bf16(ctx.cublas_handle,
                            reinterpret_cast<const __nv_bfloat16*>(Q),
                            reinterpret_cast<const __nv_bfloat16*>(K),
                            reinterpret_cast<const __nv_bfloat16*>(V),
                            reinterpret_cast<__nv_bfloat16*>(logits),
                            reinterpret_cast<__nv_bfloat16*>(out),
                            S_q, S_kv, NH, HD, attn_scale,
                            logits_kv_stride, to_stream(stream));
    }, py::arg("ctx"), py::arg("Q"), py::arg("K"), py::arg("V"),
       py::arg("logits"), py::arg("out"),
       py::arg("S_q"), py::arg("S_kv"), py::arg("NH"), py::arg("HD"),
       py::arg("attn_scale") = 1.0f, py::arg("logits_kv_stride") = 0,
       py::arg("stream") = 0);
}
