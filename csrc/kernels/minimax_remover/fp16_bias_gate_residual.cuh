// ================================================================
// flash_rt_minimax_remover — fused bias + gate·residual kernel.
//
// The MiniMax-Remover transformer block runs the following pattern
// after every attention output (O-proj) and every FFN down proj:
//
//     out_fp16 = FP8_GEMM(...)                      // no bias yet
//     add_bias_fp16(out, bias, S, D)                // RMW on out
//     gate_mul_residual_bcast(residual, out, gate)  // residual += out*gate
//
// The middle step is a full fp16 read-modify-write pass over out that
// happens JUST BEFORE the residual add reads out again — cache-hostile
// (2 * S * D fp16 traffic per pass, ~100 MB per call at S=32256/D=1536).
//
// This kernel folds them into one:
//     residual[i] += (out[i] + bias[i % D]) * gate[i % D]
// which drops the intermediate RMW on `out` entirely (one fewer full
// pass over ~50M fp16 elements per call).  Node-level profiling put
// `add_bias_fp16` at 280 ms / 1812 calls; the O-proj + FFN-down slots
// account for 720 of those calls — this kernel eliminates them.
//
// Layout / semantics:
//   out       : [M, D]  fp16, row-major (read-only)
//   bias      : [D]     fp16 (broadcast along M)
//   gate      : [D]     fp16 (broadcast along M)
//   residual  : [M, D]  fp16 (accumulated in place)
//
// Uses fp16x8 vector loads/stores (uint4) so each thread processes 8
// elements, cutting global memory transactions 8× versus the scalar
// `add_bias_fp16` kernel.  D is assumed to be a multiple of 8 (true
// for all MiniMax-Remover Linears; D ∈ {1536, 8960}).
// ================================================================
#ifndef FLASHRT_KERNELS_MINIMAX_REMOVER_FP16_BIAS_GATE_RES_CUH
#define FLASHRT_KERNELS_MINIMAX_REMOVER_FP16_BIAS_GATE_RES_CUH

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

// Fused: residual[m,d] += (out[m,d] + bias[d]) * gate[d]
//
// All buffers are fp16.  bias and gate are broadcast vectors of length D.
// Returns 0 on success, negative on invalid args.
int fp16_bias_gate_residual_bcast(
    const void* out_fp16,
    const void* bias_fp16,
    const void* gate_fp16,
    void* residual_fp16,
    int M, int D,
    cudaStream_t stream);

// Vectorised replacement for the generic scalar add_bias_fp16:
//   x[m,d] = x[m,d] + bias[d]     (broadcast bias, in-place)
// Uses fp16x8 (uint4) accesses; D must be a multiple of 8.
int fp16_add_bias_vec8(
    void* x_fp16,
    const void* bias_fp16,
    int M, int D,
    cudaStream_t stream);

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt

#endif  // FLASHRT_KERNELS_MINIMAX_REMOVER_FP16_BIAS_GATE_RES_CUH
