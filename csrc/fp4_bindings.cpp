// ============================================================================
//  FlashRT — pybind module for NVFP4 kernels.
//
//  Built as a SEPARATE .so from flash_rt_kernels.so (which stays untouched).
//  Python-side usage:
//
//      import flash_rt.flash_rt_kernels as fvk        # unchanged
//      import flash_rt.flash_rt_fp4    as fvk_fp4     # new, additive
//
//  All pointer args are passed as int (ctypes.c_void_p.value) to mirror the
//  existing fvk convention; everything is host/device pointer pass-through.
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gemm/fp4/cutlass_fp4_gemm.cuh"
#include "gemm/fp4/cutlass_fp4_gemm_fp4out.cuh"
#include "quantize/quantize_fp4_dynamic.cuh"
#include "quantize/quantize_fp4_sfa.cuh"
#include "quantize/reshape_scales_sfa.cuh"
#include "fused_fp16/rms_norm_noweight_fp16.cuh"
#include "fused_fp4/norm_silu_fp4_sfa.cuh"
#include "fused_fp4/silu_mul_two_fp4_to_fp4.cuh"

extern "C" int flash_rt_per_channel_mul_fp16(
    uintptr_t x, uintptr_t inv_s, int S, int D, uintptr_t stream);

namespace py = pybind11;

PYBIND11_MODULE(flash_rt_fp4, m) {
  m.doc() = "FlashRT — NVFP4 (Thor SM110) add-on kernels";

  // ── GEMM ──
  m.def("cutlass_fp4_sq_fp16",
        [](uintptr_t A, uintptr_t SFA,
           uintptr_t B, uintptr_t SFB,
           uintptr_t D, int M, int N, int K,
           float alpha, float beta, uintptr_t stream) -> int {
          return flash_rt::fp4::cutlass_fp4_sq_fp16(
              reinterpret_cast<void const*>(A),
              reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B),
              reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void*>(D),
              M, N, K, alpha, beta,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("A"), py::arg("SFA"),
        py::arg("B"), py::arg("SFB"),
        py::arg("D"), py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
        py::arg("stream") = 0,
        R"pbdoc(
NVFP4 block-scaled GEMM:  D[M,N] (fp16) = A[M,K] (fp4) @ B[N,K]^T (fp4)

A and B are NVFP4 (e2m1) packed as 2 elements per byte, with per-16-element
UE4M3 block scales (SFA, SFB). All pointers are device-resident; int-typed
(e.g. t.data_ptr()). Returns 0 on success, nonzero on error.

NOTE: SFA/SFB must be in the CUTLASS Sm1xxBlkScaledConfig tile-interleaved
layout, NOT the linear [N, D/16] layout produced by quantize_fp4_dynamic_fp16.
Phase 4 will add the layout conversion helper.
)pbdoc");

  // ── Dynamic quantize ──
  m.def("quantize_fp4_dynamic_fp16",
        [](uintptr_t src, uintptr_t packed, uintptr_t scales,
           int N, int D, uintptr_t stream) -> int {
          return flash_rt::fp4::quantize_fp4_dynamic_fp16(
              reinterpret_cast<void const*>(src),
              reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(scales),
              N, D, reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("src"), py::arg("packed"), py::arg("scales"),
        py::arg("N"), py::arg("D"), py::arg("stream") = 0,
        R"pbdoc(
fp16 [N, D] → NVFP4 packed [N, D/2] uint8 + UE4M3 scales [N, D/16].
Linear (row-major) scale layout. For CUTLASS GEMM consumption, additional
tile-interleave conversion is required.
)pbdoc");

  m.def("dequantize_fp4_to_fp16",
        [](uintptr_t packed, uintptr_t scales, uintptr_t dst,
           int N, int D, uintptr_t stream) -> int {
          return flash_rt::fp4::dequantize_fp4_to_fp16(
              reinterpret_cast<void const*>(packed),
              reinterpret_cast<void const*>(scales),
              reinterpret_cast<void*>(dst),
              N, D, reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("packed"), py::arg("scales"), py::arg("dst"),
        py::arg("N"), py::arg("D"), py::arg("stream") = 0,
        "Inverse of quantize_fp4_dynamic_fp16. Used for unit tests.");

  m.def("quantize_fp4_dynamic_sfa_fp16",
        [](uintptr_t src, uintptr_t packed, uintptr_t sfa,
           int N, int D, bool is_sfb, uintptr_t stream) -> int {
          return flash_rt::fp4::quantize_fp4_dynamic_sfa_fp16(
              reinterpret_cast<void const*>(src),
              reinterpret_cast<void*>(packed),
              reinterpret_cast<void*>(sfa),
              N, D, is_sfb,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("src"), py::arg("packed"), py::arg("sfa"),
        py::arg("N"), py::arg("D"), py::arg("is_sfb"), py::arg("stream") = 0,
        R"pbdoc(
Fused: fp16 [N, D] → NVFP4 packed [N, D/2] + CUTLASS tile-interleaved SFA/SFB.
Bit-exact equivalent of quantize_fp4_dynamic_fp16 followed by
reshape_linear_scales_to_sfa, in a single kernel launch.
)pbdoc");

  m.def("reshape_linear_scales_to_sfa",
        [](uintptr_t src, uintptr_t dst, int rows, int D, bool is_sfb,
           uintptr_t stream) -> int {
          return flash_rt::fp4::reshape_linear_scales_to_sfa(
              reinterpret_cast<void const*>(src),
              reinterpret_cast<void*>(dst),
              rows, D, is_sfb,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("src"), py::arg("dst"), py::arg("rows"), py::arg("D"),
        py::arg("is_sfb"), py::arg("stream") = 0,
        "Permute linear [rows, D/16] fp8 scales into CUTLASS SFA (is_sfb=False) "
        "or SFB (is_sfb=True) tile-interleaved layout.");

  m.def("sfa_size_bytes",
        &flash_rt::fp4::sfa_size_bytes,
        py::arg("rows"), py::arg("D"), py::arg("is_sfb"),
        "Byte size of the CUTLASS SFA (or SFB) buffer for the given problem.");

  // Tuning variants
  m.def("cutlass_fp4_gemm_variant",
        [](int idx, uintptr_t A, uintptr_t SFA, uintptr_t B, uintptr_t SFB,
           uintptr_t D, int M, int N, int K, float alpha, float beta,
           uintptr_t stream) -> int {
          return flash_rt::fp4::cutlass_fp4_gemm_variant(
              idx, reinterpret_cast<void const*>(A), reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B), reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void*>(D), M, N, K, alpha, beta,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("idx"), py::arg("A"), py::arg("SFA"),
        py::arg("B"), py::arg("SFB"), py::arg("D"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
        py::arg("stream") = 0,
        "Call one of the NVFP4 GEMM variants by index. Used for tile/schedule tuning.");

  m.def("cutlass_fp4_gemm_variant_name", &flash_rt::fp4::cutlass_fp4_gemm_variant_name,
        py::arg("idx"), "Human-readable name of variant at index.");
  m.def("cutlass_fp4_gemm_num_variants", &flash_rt::fp4::cutlass_fp4_gemm_num_variants,
        "Count of available GEMM variants.");

  m.def("has_nvfp4", &flash_rt::fp4::has_nvfp4_sm110,
        "True iff this .so was built with CUTLASS SM100 support (NVFP4 usable).");

  // ── fp16-output fused norm kernels (additive, for FP4 frontend path) ──
  m.def("rms_norm_noweight_fp16",
        [](uintptr_t x, uintptr_t out, int seq_len, int dim,
           uintptr_t stream) {
          flash_rt::fused_fp16::rms_norm_noweight_fp16(
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<__half*>(out),
              seq_len, dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("out"), py::arg("seq_len"), py::arg("dim"),
        py::arg("stream") = 0,
        "fp16 [S,D] → fp16 [S,D]. RMSNorm without weight, no descale. "
        "Bit-exact with rms_norm_fp8_noweight_fp16 when subsequently quantized "
        "to fp8 with the same descale factor.");

  m.def("residual_add_rms_norm_noweight_fp16",
        [](uintptr_t residual, uintptr_t x, uintptr_t out,
           int seq_len, int dim, uintptr_t stream) {
          flash_rt::fused_fp16::residual_add_rms_norm_noweight_fp16(
              reinterpret_cast<__half*>(residual),
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<__half*>(out),
              seq_len, dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("out"),
        py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0,
        "Residual += x (in place) then RMSNorm to fp16 (no descale). "
        "Bit-exact with residual_add_rms_norm_fp8_noweight_fp16 modulo fp8 cast.");

  // ── Fused FP4 pre-GEMM kernels (F2/F3/F4) ──
  m.def("rms_norm_fp4_sfa_fp16",
        [](uintptr_t x, uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, uintptr_t stream) {
          flash_rt::fused_fp4::rms_norm_fp4_sfa_fp16(
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("x"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0,
        "F2: fused rms_norm + fp4_quant + SFA write.");

  m.def("residual_add_rms_norm_fp4_sfa_v2_fp16",
        [](uintptr_t residual, uintptr_t x,
           uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, uintptr_t stream) {
          flash_rt::fused_fp4::residual_add_rms_norm_fp4_sfa_v2_fp16(
              reinterpret_cast<__half*>(residual),
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0,
        "F3 v2 (register-only, 1 thread = 1 NVFP4 block): fused res+rms+fp4+SFA.");

  m.def("residual_add_rms_norm_fp4_sfa_fp16",
        [](uintptr_t residual, uintptr_t x,
           uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, uintptr_t stream) {
          flash_rt::fused_fp4::residual_add_rms_norm_fp4_sfa_fp16(
              reinterpret_cast<__half*>(residual),
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0,
        "F3: fused residual+rms_norm + fp4_quant + SFA write.");

  // GEGLU (tanh-approx GELU(gate) * up) fused FP4 kernels.
  m.def("gate_geglu_fp4_sfa_fp16",
        [](uintptr_t merged, uintptr_t packed, uintptr_t sfa,
           int seq_len, int half_dim, uintptr_t stream) {
          flash_rt::fused_fp4::gate_silu_mul_fp4_sfa_fp16(
              reinterpret_cast<const __half*>(merged),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, half_dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("merged"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("half_dim"), py::arg("stream") = 0,
        "F4 v1 (smem-staged): fused GEGLU + fp4_quant + SFA write.");

  m.def("gate_geglu_fp4_sfa_v2_fp16",
        [](uintptr_t merged, uintptr_t packed, uintptr_t sfa,
           int seq_len, int half_dim, uintptr_t stream) {
          flash_rt::fused_fp4::gate_silu_mul_fp4_sfa_v2_fp16(
              reinterpret_cast<const __half*>(merged),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, half_dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("merged"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("half_dim"), py::arg("stream") = 0,
        "F4 v2 (register-only, no smem): same semantics as v1, faster at H=8192.");

  // ── AWQ fused: F3 + per-channel-mul ──
  m.def("residual_add_rms_norm_mul_fp4_sfa_fp16",
        [](uintptr_t residual, uintptr_t x, uintptr_t inv_s,
           uintptr_t packed, uintptr_t sfa,
           int seq_len, int dim, uintptr_t stream) {
          flash_rt::fused_fp4::residual_add_rms_norm_mul_fp4_sfa_fp16(
              reinterpret_cast<__half*>(residual),
              reinterpret_cast<const __half*>(x),
              reinterpret_cast<const __half*>(inv_s),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("residual"), py::arg("x"), py::arg("inv_s"),
        py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("dim"), py::arg("stream") = 0,
        "F3 + AWQ: fused res+rms+inv_s_mul+fp4_quant+SFA (1 launch).");

  // ── AWQ fused: F4 v2 + per-channel-mul ──
  m.def("gate_geglu_mul_fp4_sfa_v2_fp16",
        [](uintptr_t merged, uintptr_t inv_s,
           uintptr_t packed, uintptr_t sfa,
           int seq_len, int half_dim, uintptr_t stream) {
          flash_rt::fused_fp4::gate_silu_mul_mul_fp4_sfa_v2_fp16(
              reinterpret_cast<const __half*>(merged),
              reinterpret_cast<const __half*>(inv_s),
              reinterpret_cast<uint8_t*>(packed),
              reinterpret_cast<uint8_t*>(sfa),
              seq_len, half_dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("merged"), py::arg("inv_s"), py::arg("packed"), py::arg("sfa"),
        py::arg("seq_len"), py::arg("half_dim"), py::arg("stream") = 0,
        "F4 v2 + AWQ: fused GEGLU+inv_s_mul+fp4_quant+SFA (1 launch).");

  // ── AWQ per-channel inverse scale multiply ──
  m.def("per_channel_mul_fp16",
        [](uintptr_t x, uintptr_t inv_s, int S, int D, uintptr_t stream) {
          flash_rt_per_channel_mul_fp16(x, inv_s, S, D, stream);
        },
        py::arg("x"), py::arg("inv_s"), py::arg("S"), py::arg("D"),
        py::arg("stream") = 0,
        "x[i,k] *= inv_s[k] in-place. Per-input-channel activation scaling "
        "for AWQ-style FP4 inference (paired with offline weight pre-scaling).");

  // ── P1: NVFP4 GEMM with FP4 packed output (LinCombBlockScaleFactor epilogue) ──
  m.def("cutlass_fp4_gemm_fp4out",
        [](uintptr_t A_packed, uintptr_t SFA,
           uintptr_t B_packed, uintptr_t SFB,
           uintptr_t D_packed, uintptr_t D_SFD,
           int M, int N, int K, uintptr_t stream) -> int {
          return flash_rt::fp4::cutlass_fp4_gemm_fp4out(
              reinterpret_cast<void const*>(A_packed),
              reinterpret_cast<void const*>(SFA),
              reinterpret_cast<void const*>(B_packed),
              reinterpret_cast<void const*>(SFB),
              reinterpret_cast<void*>(D_packed),
              reinterpret_cast<void*>(D_SFD),
              M, N, K,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("A_packed"), py::arg("SFA"),
        py::arg("B_packed"), py::arg("SFB"),
        py::arg("D_packed"), py::arg("D_SFD"),
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("stream") = 0,
        R"pbdoc(
P1 NVFP4 GEMM:  D[M,N/2] (fp4) + D_SFD = LinCombBlockScaleFactor(A @ B^T).
Drop-in for cutlass_fp4_sq_fp16 when downstream consumes FP4 + SFA directly.
)pbdoc");

  // ── P1 + AWQ: geglu_two_mul_fp4_to_fp4 — GEGLU combiner with Down inv_s mul ──
  m.def("geglu_two_mul_fp4_to_fp4",
        [](uintptr_t gate_packed, uintptr_t gate_sfa,
           uintptr_t up_packed,   uintptr_t up_sfa,
           uintptr_t inv_s,
           uintptr_t out_packed,  uintptr_t out_sfa,
           int seq_len, int half_dim, uintptr_t stream) {
          flash_rt::fused_fp4::silu_mul_two_mul_fp4_to_fp4(
              reinterpret_cast<const uint8_t*>(gate_packed),
              reinterpret_cast<const uint8_t*>(gate_sfa),
              reinterpret_cast<const uint8_t*>(up_packed),
              reinterpret_cast<const uint8_t*>(up_sfa),
              reinterpret_cast<const __half*>(inv_s),
              reinterpret_cast<uint8_t*>(out_packed),
              reinterpret_cast<uint8_t*>(out_sfa),
              seq_len, half_dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("gate_packed"), py::arg("gate_sfa"),
        py::arg("up_packed"),   py::arg("up_sfa"),
        py::arg("inv_s"),
        py::arg("out_packed"),  py::arg("out_sfa"),
        py::arg("seq_len"), py::arg("half_dim"), py::arg("stream") = 0,
        "P1 + AWQ-Down: GEGLU + per-input-channel inv_s mul → FP4 + SFA.");

  // ── P1: geglu_two_fp4_to_fp4 — combiner kernel for split-GU FFN path ──
  m.def("geglu_two_fp4_to_fp4",
        [](uintptr_t gate_packed, uintptr_t gate_sfa,
           uintptr_t up_packed,   uintptr_t up_sfa,
           uintptr_t out_packed,  uintptr_t out_sfa,
           int seq_len, int half_dim, uintptr_t stream) {
          flash_rt::fused_fp4::silu_mul_two_fp4_to_fp4(
              reinterpret_cast<const uint8_t*>(gate_packed),
              reinterpret_cast<const uint8_t*>(gate_sfa),
              reinterpret_cast<const uint8_t*>(up_packed),
              reinterpret_cast<const uint8_t*>(up_sfa),
              reinterpret_cast<uint8_t*>(out_packed),
              reinterpret_cast<uint8_t*>(out_sfa),
              seq_len, half_dim,
              reinterpret_cast<cudaStream_t>(stream));
        },
        py::arg("gate_packed"), py::arg("gate_sfa"),
        py::arg("up_packed"),   py::arg("up_sfa"),
        py::arg("out_packed"),  py::arg("out_sfa"),
        py::arg("seq_len"), py::arg("half_dim"), py::arg("stream") = 0,
        "P1: GEGLU over two FP4 inputs → FP4 + SFA.");

  m.attr("__version__") = "0.1.0-dev";
  m.attr("layout_note") = "scales are linear [N, D/16]; Phase 4 adds tile-interleave conversion";
}
