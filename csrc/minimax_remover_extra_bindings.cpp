// ================================================================
// flash_rt_minimax_remover — standalone pybind module for
// MiniMax-Remover VAE-specific fused fp16 kernels.
//
// Kept separate from flash_rt_kernels so they can be added/rebuilt
// independently without touching the main bindings.  Build with:
//   cmake -DFLASHRT_ENABLE_MINIMAX_REMOVER=ON -DGPU_ARCH=120 ...
//
// Kernels: fp16_rms_norm_ncdhw, fp16_rms_silu_ncdhw
// ================================================================
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "kernels/minimax_remover/fp16_rms_norm_ncdhw.cuh"
#include "kernels/minimax_remover/fp16_rms_silu_ncdhw.cuh"
#include "kernels/minimax_remover/fp16_rms_norm_ndhwc.cuh"
#include "kernels/minimax_remover/fp8_conv3d_mm_ndhwc_fp16out.cuh"
#include "kernels/minimax_remover/fp16_quant_fp8_per_tensor.cuh"
#include "kernels/minimax_remover/fp16_rms_silu_fp8_ndhwc.cuh"
#include "kernels/minimax_remover/fp16_bias_gelu_quant_fp8.cuh"
#include "kernels/minimax_remover/fp16_bias_gate_residual.cuh"
#include "kernels/minimax_remover/fp16_ada_layernorm_quant_fp8.cuh"
#include "kernels/minimax_remover/fp16_rmsnorm_rope.cuh"
#include "kernels/minimax_remover/fp16_rmsnorm_rope_quant_int8.cuh"
#include "kernels/minimax_remover/fp16_quant_nvfp4_ndhwc.cuh"
#include "kernels/minimax_remover/nvfp4_conv3d_ndhwc_fp16out.cuh"

namespace py = pybind11;

static inline void* to_ptr(uintptr_t p) { return reinterpret_cast<void*>(p); }
static inline cudaStream_t to_stream(uintptr_t s) {
    return reinterpret_cast<cudaStream_t>(s);
}

PYBIND11_MODULE(flash_rt_minimax_remover, m) {
    m.doc() = "MiniMax-Remover VAE fused fp16 kernels";

    m.def("fp16_rms_norm_ncdhw",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp16, int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::fp16_rms_norm_ncdhw(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp16), B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp16"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused FP16 NCDHW RMSNorm (fp16 in/out, fp32 stats, no cast). "
        "Replaces WanRMS_norm.forward (4 full-tensor fp32 passes).");

    m.def("fp16_rms_silu_ncdhw",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp16, int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::fp16_rms_silu_ncdhw(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp16), B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp16"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused FP16 NCDHW RMSNorm + SiLU (fp16 in/out, fp32 stats+act, "
        "no cast). Replaces norm->silu two-pass in WanResidualBlock.");

    // ── Channels-last (NDHWC) norm kernels ──
    m.def("fp16_rms_norm_ndhwc",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp16, int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::fp16_rms_norm_ndhwc(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp16), B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp16"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused FP16 channels-last (NDHWC) RMSNorm. C values contiguous, "
        "eliminates nchw<->nhwc format conversion for cuDNN conv3d.");

    m.def("fp16_rms_silu_ndhwc",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp16, int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::fp16_rms_silu_ndhwc(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp16), B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp16"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused FP16 channels-last (NDHWC) RMSNorm + SiLU.");

    // ── FP8 implicit-GEMM conv3d (3×3×3 causal, NDHWC, fp16 output) ──
    m.def("fp8_conv3d_mm_ndhwc_fp16out",
        [](uintptr_t cache_x_fp8, uintptr_t new_x_fp8,
           uintptr_t w_fp8, uintptr_t y_fp16,
           uintptr_t bias_fp16, uintptr_t alpha_vec,
           int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
           uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp8_conv3d_mm_ndhwc_fp16out(
                to_ptr(cache_x_fp8), to_ptr(new_x_fp8),
                to_ptr(w_fp8), to_ptr(y_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                alpha_vec ? to_ptr(alpha_vec) : nullptr,
                N, T_cache, T_new, H, W, Ci, Co,
                to_stream(stream));
        },
        py::arg("cache_x_fp8"), py::arg("new_x_fp8"),
        py::arg("w_fp8"), py::arg("y_fp16"),
        py::arg("bias_fp16"), py::arg("alpha_vec"),
        py::arg("N"), py::arg("T_cache"), py::arg("T_new"),
        py::arg("H"), py::arg("W"), py::arg("Ci"), py::arg("Co"),
        py::arg("stream") = 0,
        "FP8 e4m3 implicit-GEMM conv3d fprop (3x3x3 causal, NDHWC, "
        "fp16 output). Per-channel alpha vector [Co] float and fp16 "
        "bias [Co]. No im2col materialization.");

    // ── Fused fp16→fp8 per-tensor quantize (2-pass, no host sync) ──
    m.def("fp16_quant_fp8_per_tensor",
        [](uintptr_t x_fp16, uintptr_t y_fp8,
           uintptr_t scale_out, uintptr_t amax_buf,
           int n, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp16_quant_fp8_per_tensor(
                to_ptr(x_fp16), to_ptr(y_fp8),
                to_ptr(scale_out), to_ptr(amax_buf),
                n, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("y_fp8"),
        py::arg("scale_out"), py::arg("amax_buf"),
        py::arg("n"), py::arg("stream") = 0,
        "Fused per-tensor fp16→fp8 e4m3 quantize (amax + scale on "
        "device, no host sync). Writes float scale to scale_out.");

    m.def("amax_fp16",
        [](uintptr_t x_fp16, uintptr_t amax_buf,
           int n, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::amax_fp16(
                to_ptr(x_fp16), to_ptr(amax_buf), n, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("amax_buf"),
        py::arg("n"), py::arg("stream") = 0,
        "Grid-stride amax reduction into amax_buf via atomicMax. "
        "Caller must zero amax_buf before first call. Multiple calls "
        "accumulate (for multi-tensor shared-scale quantization).");

    m.def("quantize_fp16_fp8_with_amax",
        [](uintptr_t x_fp16, uintptr_t y_fp8,
           uintptr_t amax_buf, uintptr_t scale_out,
           int n, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                quantize_fp16_fp8_with_amax(
                to_ptr(x_fp16), to_ptr(y_fp8),
                to_ptr(amax_buf), to_ptr(scale_out),
                n, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("y_fp8"),
        py::arg("amax_buf"), py::arg("scale_out"),
        py::arg("n"), py::arg("stream") = 0,
        "Quantize fp16→fp8 using pre-computed amax in amax_buf. "
        "Writes float scale to scale_out.");

    // ── Dual quantize: two buffers, one shared amax, one launch ──
    m.def("quantize_fp16_fp8_with_amax_dual",
        [](uintptr_t x1_fp16, uintptr_t y1_fp8, int n1,
           uintptr_t x2_fp16, uintptr_t y2_fp8, int n2,
           uintptr_t amax_buf, uintptr_t scale_out,
           uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                quantize_fp16_fp8_with_amax_dual(
                to_ptr(x1_fp16), to_ptr(y1_fp8), n1,
                to_ptr(x2_fp16), to_ptr(y2_fp8), n2,
                to_ptr(amax_buf),
                scale_out ? to_ptr(scale_out) : nullptr,
                to_stream(stream));
        },
        py::arg("x1_fp16"), py::arg("y1_fp8"), py::arg("n1"),
        py::arg("x2_fp16"), py::arg("y2_fp8"), py::arg("n2"),
        py::arg("amax_buf"), py::arg("scale_out") = 0,
        py::arg("stream") = 0,
        "Dual quantize: two fp16 buffers → fp8 with shared amax in "
        "one kernel launch. Saves one launch vs two separate calls.");

    // ── Fused norm+silu+amax / norm+silu+quant_fp8 (NDHWC) ──
    m.def("fp16_rms_silu_amax_ndhwc",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp16, uintptr_t amax_buf,
           int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp16_rms_silu_amax_ndhwc(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp16), to_ptr(amax_buf),
                B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp16"), py::arg("amax_buf"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused FP16 NDHWC RMSNorm+SiLU+amax. Writes fp16 output and "
        "accumulates |output| into amax_buf via atomicMax (caller must "
        "zero amax_buf before first call). Saves one full read of y "
        "vs separate norm+silu then amax.");

    m.def("fp16_rms_silu_quant_fp8_ndhwc",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp8, uintptr_t amax_buf, uintptr_t scale_out,
           int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp16_rms_silu_quant_fp8_ndhwc(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp8), to_ptr(amax_buf),
                scale_out ? to_ptr(scale_out) : nullptr,
                B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp8"), py::arg("amax_buf"), py::arg("scale_out") = 0,
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused FP16 NDHWC RMSNorm+SiLU → FP8 e4m3 quantize. Reads "
        "pre-computed amax from device. Does NOT write fp16 output — "
        "eliminates the fp16 intermediate between norm and conv.");

    m.def("fp16_rms_silu_amax_quant_fp8_ndhwc",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp8, uintptr_t scale_out, uintptr_t amax_buf,
           int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp16_rms_silu_amax_quant_fp8_ndhwc(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp8), to_ptr(scale_out), to_ptr(amax_buf),
                B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp8"), py::arg("scale_out"), py::arg("amax_buf"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "2-pass fused norm+silu+amax+quant → FP8. Pass 1 computes amax "
        "(no write); pass 2 re-reads x and quantizes. Produces ONLY fp8 "
        "output + scale, no fp16 intermediate.");

    m.def("fp16_rms_silu_amax_quant_fp8_ndhwc_nozero",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp8, uintptr_t scale_out, uintptr_t amax_buf,
           int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp16_rms_silu_amax_quant_fp8_ndhwc_nozero(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp8), to_ptr(scale_out), to_ptr(amax_buf),
                B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp8"), py::arg("scale_out"), py::arg("amax_buf"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Same as fp16_rms_silu_amax_quant_fp8_ndhwc but does NOT zero "
        "amax_buf before pass 1. For running-max mode: caller seeds "
        "amax_buf with historical max; pass 1 accumulates current output; "
        "pass 2 quantizes with max(historical, current).");

    // ── Fused FFN epilogue: bias + gelu + quant → fp8 (transformer) ──
    m.def("bias_gelu_quant_fp16_fp8",
        [](uintptr_t gemm_out, uintptr_t bias, uintptr_t out,
           uintptr_t d_scale, int M, int N, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                bias_gelu_quant_fp16_fp8(
                to_ptr(gemm_out), to_ptr(bias), to_ptr(out),
                reinterpret_cast<const float*>(to_ptr(d_scale)),
                M, N, to_stream(stream));
        },
        py::arg("gemm_out"), py::arg("bias"), py::arg("out"),
        py::arg("d_scale"), py::arg("M"), py::arg("N"),
        py::arg("stream") = 0,
        "Fused FFN epilogue: fp16 GEMM-out + bias → tanh-gelu → fp8 e4m3. "
        "Replaces add_bias_fp16 + gelu_inplace_fp16 + quantize_fp8 (3 "
        "kernels → 1). Output is the pre-quantised input of the next FP8 "
        "Linear, which skips its own activation quantise.");

    m.def("bias_quant_fp16_fp8",
        [](uintptr_t gemm_out, uintptr_t bias, uintptr_t out,
           uintptr_t d_scale, int M, int N, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                bias_quant_fp16_fp8(
                to_ptr(gemm_out), to_ptr(bias), to_ptr(out),
                reinterpret_cast<const float*>(to_ptr(d_scale)),
                M, N, to_stream(stream));
        },
        py::arg("gemm_out"), py::arg("bias"), py::arg("out"),
        py::arg("d_scale"), py::arg("M"), py::arg("N"),
        py::arg("stream") = 0,
        "Fused: fp16 GEMM-out + bias → fp8 e4m3 (identity activation). "
        "For Linear→Linear chains with no activation in between.");

    // ── Fused bias + gate·residual (eliminates the mid-block bias-add RMW) ──
    m.def("fp16_bias_gate_residual_bcast",
        [](uintptr_t out_fp16, uintptr_t bias_fp16, uintptr_t gate_fp16,
           uintptr_t residual_fp16, int M, int D, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp16_bias_gate_residual_bcast(
                to_ptr(out_fp16), to_ptr(bias_fp16), to_ptr(gate_fp16),
                to_ptr(residual_fp16), M, D, to_stream(stream));
        },
        py::arg("out_fp16"), py::arg("bias_fp16"),
        py::arg("gate_fp16"), py::arg("residual_fp16"),
        py::arg("M"), py::arg("D"),
        py::arg("stream") = 0,
        "Fused: residual[m,d] += (out[m,d] + bias[d]) * gate[d]. "
        "Replaces add_bias_fp16 + gate_mul_residual_bcast (2 kernels → 1) "
        "for the O-proj and FFN-down slots — cuts one full [M,D] fp16 "
        "read-modify-write per call.  D must be a multiple of 8.");

    m.def("fp16_add_bias_vec8",
        [](uintptr_t x_fp16, uintptr_t bias_fp16,
           int M, int D, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::fp16_add_bias_vec8(
                to_ptr(x_fp16), to_ptr(bias_fp16), M, D, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("bias_fp16"),
        py::arg("M"), py::arg("D"),
        py::arg("stream") = 0,
        "Vectorised (fp16x8) in-place add_bias: x[m,d] += bias[d]. "
        "Replaces the scalar decoder_fused add_bias_fp16 kernel for the "
        "Q/K/V projections; ~8× fewer memory transactions.  D must be "
        "a multiple of 8.");

    m.def("fp16_ada_layernorm_quant_fp8",
        [](uintptr_t x_fp16, uintptr_t scale_fp32, uintptr_t shift_fp32,
           uintptr_t act_scale_fp32, uintptr_t out_fp8,
           int S, int D, float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp16_ada_layernorm_quant_fp8(
                to_ptr(x_fp16), to_ptr(scale_fp32), to_ptr(shift_fp32),
                to_ptr(act_scale_fp32), to_ptr(out_fp8),
                S, D, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("scale_fp32"), py::arg("shift_fp32"),
        py::arg("act_scale_fp32"), py::arg("out_fp8"),
        py::arg("S"), py::arg("D"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused fp32-stat LayerNorm + adaLN modulation + per-tensor fp8 "
        "e4m3 quantise.  Reads x[S,D] fp16, applies "
        "y = (x-mean)/std * (1+scale[d]) + shift[d], then divides by the "
        "target Linear's static act_scale and casts to fp8_e4m3fn.  "
        "Replaces the 3-kernel ada_layernorm_fp16_io + quantize_fp8 + "
        "gemm-descale sequence with a single pass — eliminates the "
        "intermediate fp16 read of the LayerNorm output.  D must be a "
        "multiple of 8.");

    m.def("fp16_rmsnorm_rope_bshd",
        [](uintptr_t x_fp16, uintptr_t weight_fp16,
           uintptr_t cos_fp32, uintptr_t sin_fp32,
           int B, int S, int H, int Dd,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::fp16_rmsnorm_rope_bshd(
                to_ptr(x_fp16), to_ptr(weight_fp16),
                to_ptr(cos_fp32), to_ptr(sin_fp32),
                B, S, H, Dd, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("weight_fp16"),
        py::arg("cos_fp32"), py::arg("sin_fp32"),
        py::arg("B"), py::arg("S"), py::arg("H"), py::arg("Dd"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused per-token RMSNorm (fp32 stats, fp16 affine) + interleaved "
        "RoPE on the native [B,S,H,Dd] fp16 layout.  RMS reduction is "
        "across the full D = H*Dd (matches qk_norm='rms_norm_across_heads'). "
        "Replaces the 2-kernel rms_norm_fp32stat + rope_apply_bshd Triton "
        "sequence with one pass — eliminates one full fp16 read+write of "
        "the Q/K tensor.  Dd must be a multiple of 8.");

    // ── Fused RMSNorm + RoPE + int8 quantize (Q, per-warp) ──────────
    m.def("fp16_rmsnorm_rope_quant_int8_q",
        [](uintptr_t x_fp16, uintptr_t weight_fp16,
           uintptr_t bias_fp16,
           uintptr_t cos_fp32, uintptr_t sin_fp32,
           uintptr_t out_int8, uintptr_t scale_fp32,
           int B, int S, int H, int Dd,
           float eps, float sm_scale,
           uintptr_t rstd_buf, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp16_rmsnorm_rope_quant_int8_q(
                to_ptr(x_fp16), to_ptr(weight_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(cos_fp32), to_ptr(sin_fp32),
                to_ptr(out_int8), to_ptr(scale_fp32),
                B, S, H, Dd, eps, sm_scale,
                rstd_buf ? to_ptr(rstd_buf) : nullptr,
                to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("weight_fp16"),
        py::arg("bias_fp16") = 0,
        py::arg("cos_fp32"), py::arg("sin_fp32"),
        py::arg("out_int8"), py::arg("scale_fp32"),
        py::arg("B"), py::arg("S"), py::arg("H"), py::arg("Dd"),
        py::arg("eps") = 1e-6f, py::arg("sm_scale") = 1.0f,
        py::arg("rstd_buf") = 0, py::arg("stream") = 0,
        "Fused RMSNorm + RoPE + per-warp int8 quantization for Q. "
        "Eliminates the fp16 intermediate between norm+rope and quantize. "
        "bias_fp16 (Q-proj bias) is added pre-norm (fused: replaces the "
        "separate add_bias kernel); pass 0 to skip. "
        "rstd_buf: caller-owned [B*S] fp32 scratch (reused across calls to "
        "avoid hot-path allocation); pass 0 for a transient allocation. "
        "Output: int8 [B*S, H*Dd], scale [B, H, ceil(S/32)]. "
        "sm_scale is folded into quantization (softmax scale pre-multiply).");

    // ── Fused RMSNorm + RoPE + int8 quantize (K, per-block + smooth_k) ─
    m.def("fp16_rmsnorm_rope_quant_int8_k",
        [](uintptr_t x_fp16, uintptr_t weight_fp16,
           uintptr_t bias_fp16,
           uintptr_t cos_fp32, uintptr_t sin_fp32,
           uintptr_t km_fp16,
           uintptr_t out_int8, uintptr_t scale_fp32,
           int B, int S, int H, int Dd,
           float eps, float sm_scale,
           uintptr_t rstd_buf, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp16_rmsnorm_rope_quant_int8_k(
                to_ptr(x_fp16), to_ptr(weight_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(cos_fp32), to_ptr(sin_fp32),
                km_fp16 ? to_ptr(km_fp16) : nullptr,
                to_ptr(out_int8), to_ptr(scale_fp32),
                B, S, H, Dd, eps, sm_scale,
                rstd_buf ? to_ptr(rstd_buf) : nullptr,
                to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("weight_fp16"),
        py::arg("bias_fp16") = 0,
        py::arg("cos_fp32"), py::arg("sin_fp32"),
        py::arg("km_fp16"),
        py::arg("out_int8"), py::arg("scale_fp32"),
        py::arg("B"), py::arg("S"), py::arg("H"), py::arg("Dd"),
        py::arg("eps") = 1e-6f, py::arg("sm_scale") = 1.0f,
        py::arg("rstd_buf") = 0, py::arg("stream") = 0,
        "Fused RMSNorm + RoPE + per-block int8 quantization for K with "
        "smooth_k (subtract key mean). Eliminates fp16 intermediate. "
        "bias_fp16 (K-proj bias) added pre-norm (fused); pass 0 to skip. "
        "rstd_buf: caller-owned [B*S] fp32 scratch (reused across calls); "
        "pass 0 for a transient allocation. "
        "Output: int8 [B*S, H*Dd], scale [B, H, ceil(S/64)]. "
        "km_fp16 can be 0 (nullptr) to skip smooth_k.");

    // ── NVFP4 fused quantization kernels (WanVAE FP4 path) ──

    m.def("fp16_rms_silu_quant_nvfp4_ndhwc",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp4, uintptr_t y_sf,
           int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp16_rms_silu_quant_nvfp4_ndhwc(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp4), to_ptr(y_sf),
                B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp4"), py::arg("y_sf"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused fp16 NCDHW → RMS_norm + SiLU + NVFP4 quant (NDHWC out). "
        "Output: fp4 [B,T,H,W,C/2] uint8, sf [B,T,H,W,C/16] uint8 UE4M3. "
        "Requires C%96==0 (WanVAE channels 96/192/384). "
        "Eliminates 3 separate passes into one kernel.");

    m.def("fp16_quant_nvfp4_ndhwc",
        [](uintptr_t x_fp16, uintptr_t y_fp4, uintptr_t y_sf,
           int B, int C, int T, int H, int W, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::fp16_quant_nvfp4_ndhwc(
                to_ptr(x_fp16), to_ptr(y_fp4), to_ptr(y_sf),
                B, C, T, H, W, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("y_fp4"), py::arg("y_sf"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("stream") = 0,
        "Plain fp16 NCDHW → NVFP4 quant (NDHWC out), no norm/silu. "
        "Used for causal-conv cache quantization.");

    m.def("fp16_quant_nvfp4_cl_ndhwc",
        [](uintptr_t x_fp16, uintptr_t y_fp4, uintptr_t y_sf,
           int B, int C, int T, int H, int W, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::fp16_quant_nvfp4_cl_ndhwc(
                to_ptr(x_fp16), to_ptr(y_fp4), to_ptr(y_sf),
                B, C, T, H, W, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("y_fp4"), py::arg("y_sf"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("stream") = 0,
        "fp16 channels-last 3D → NVFP4 quant (NDHWC out). "
        "Eliminates contiguous() copy for channels-last inputs.");

    m.def("fp16_rms_silu_quant_nvfp4_cl_ndhwc",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp4, uintptr_t y_sf,
           int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp16_rms_silu_quant_nvfp4_cl_ndhwc(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp4), to_ptr(y_sf),
                B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp4"), py::arg("y_sf"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused fp16 channels-last → RMS_norm + SiLU + NVFP4 quant (NDHWC out).");

    // ── WanVAE NVFP4 conv3d (purpose-built, fp16 NDHWC output) ──

    m.def("nvfp4_conv3d_ndhwc_fp16out",
        [](uintptr_t cache_x_fp4, uintptr_t new_x_fp4, uintptr_t w_fp4,
           uintptr_t cache_sfa, uintptr_t new_sfa, uintptr_t w_sfb,
           uintptr_t y_fp16, uintptr_t bias_fp16,
           int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
           float alpha, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                nvfp4_conv3d_ndhwc_fp16out(
                to_ptr(cache_x_fp4), to_ptr(new_x_fp4), to_ptr(w_fp4),
                to_ptr(cache_sfa), to_ptr(new_sfa), to_ptr(w_sfb),
                to_ptr(y_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                N, T_cache, T_new, H, W, Ci, Co, alpha, to_stream(stream));
        },
        py::arg("cache_x_fp4"), py::arg("new_x_fp4"), py::arg("w_fp4"),
        py::arg("cache_sfa"), py::arg("new_sfa"), py::arg("w_sfb"),
        py::arg("y_fp16"), py::arg("bias_fp16"),
        py::arg("N"), py::arg("T_cache"), py::arg("T_new"),
        py::arg("H"), py::arg("W"), py::arg("Ci"), py::arg("Co"),
        py::arg("alpha") = 1.0f, py::arg("stream") = 0,
        "WanVAE NVFP4 W4A4 conv3d (3x3x3 causal, fp16 NDHWC output). "
        "Purpose-built: eliminates bf16→fp16 + NCDHW→NDHWC conversions. "
        "Requires Ci%64==0, T_cache==2.");
}
