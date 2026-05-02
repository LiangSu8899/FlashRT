// ============================================================================
//  flash_wm — CUTLASS Blackwell FMHA forward, FP8 E4M3 in / BF16 out.
//
//  Reuses flash_rt's mainloop collective (which has kPRescale precision
//  protection for FP8 softmax denorm) + CUTLASS example 77 kernel/epilogue.
//  Only new instantiation: Element=float_e4m3_t, ElementOut=bfloat16_t.
//
//  Scope: additive. Lives entirely under flash_wm/csrc/. Pulls headers from:
//    - CUTLASS examples/77_blackwell_fmha (device/, kernel/, most of collective/)
//    - flash_rt/csrc/attention/collective/ (mainloop_tma_warpspecialized.hpp
//        — has FP8 kPRescale) via CMake include path
//
//  Target shape for BAGEL Pi0.7 denoise:
//    Q [1, 786, 28, 128]  (subgoal queries), BF16 input, quantized to FP8 per tensor
//    K [1, 7984, 4, 128]  (prefilled obs+text+subgoal KV, BF16 input, FP8 per tensor)
//    V same shape as K
//    O [1, 786, 28, 128]  BF16 output (resid stream)
//    GQA 28:4 (group size 7), HD=128, BF16 residual stream.
//
//  Scale convention: scale_q, scale_k, scale_v are such that
//     bf16_value ≈ fp8_value * scale_*
//  i.e. quantize by  fp8 = bf16 / scale;  dequantize by  bf16 = fp8 * scale.
//  The FMHA epilogue absorbs scale_q*scale_k into softmax and scale_v into
//  the PV output.
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"

// ── SM110 (Thor) compatibility shim ─────────────────────────────────
// CUTLASS 4.x's example-77 FMHA kernel only gates on SM100A_ENABLED /
// SM100F_ENABLED. On Thor we compile for sm_110a, so CUTLASS sets
// SM110A_ENABLED instead and the FMHA kernel's arch check fails at
// launch with "Arch conditional MMA instruction used without targeting
// appropriate compute capability. Aborting." Thor's tcgen05 tensor
// cores are forward-compatible with SM100's MMA encoding (same Blackwell
// family), so we force-enable SM100A when building for SM110A.
#include "cutlass/arch/config.h"
#if defined(CUTLASS_ARCH_MMA_SM110A_ENABLED) && !defined(CUTLASS_ARCH_MMA_SM100A_ENABLED)
  #define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
#endif
#if defined(CUTLASS_ARCH_MMA_SM110F_ENABLED) && !defined(CUTLASS_ARCH_MMA_SM100F_ENABLED)
  #define CUTLASS_ARCH_MMA_SM100F_ENABLED 1
#endif

// CMake include order: flash_rt/csrc/attention/ FIRST, then CUTLASS
// example77. This guarantees flash_rt's modified mainloop (kPRescale FP8
// softmax denorm protection) wins over example77's stock mainloop for
// "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp".
#include "device/fmha.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"   // flash_rt version
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_load_tma_warpspecialized.hpp"
#include "collective/fmha_fusion.hpp"

namespace flash_wm {
namespace fmha {

using namespace cute;

using Element      = cutlass::float_e4m3_t;       // FP8 E4M3 for Q/K/V
using ElementAccQK = float;
using ElementAccPV = float;
using ElementOut   = cutlass::bfloat16_t;         // BAGEL residual stream

// Tile shape: 256 Q tokens × 128 KV tokens × 128 HD. Matches FP16 variant;
// FP8 halves smem so stage counts are bumped internally by the mainloop.
using TileShape = Shape<_256, _128, _128>;

// Contiguous [seq, nheads, hd]:
//   stride_seq = NH * HD   (int)
//   stride_hd  = 1         (_1)
//   stride per (head_group, head, batch)
using StrideQ   = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK   = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;
using StrideV   = StrideK;
using StrideO   = StrideQ;
using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;
using ProblemShape = cute::tuple<int, int, int,
                                  cute::tuple<cute::tuple<int, int>, int>>;

using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
    Element, ElementAccQK, ElementAccPV, TileShape,
    StrideQ, StrideK, StrideV, cutlass::fmha::collective::NoMask>;
using Epilogue = cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
    ElementOut, ElementAccPV, typename Mainloop::TileShapePV, StrideO, StrideLSE>;
using Kernel = cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
    ProblemShape, Mainloop, Epilogue,
    cutlass::fmha::kernel::IndividualTileScheduler>;
using FmhaOp = cutlass::fmha::device::FMHA<Kernel>;

// Persistent workspace + LSE buffer (reused across calls — same pattern as
// upstream fmha_fp16_strided.cu). For CUDA Graph capture we need these
// allocated BEFORE capture begins — cudaMalloc is not captureable. Callers
// should invoke cutlass_fp8_fmha_prepare(max_SQ, max_SK, max_B, max_NQ, HD)
// once before graph capture.
static void* g_ws = nullptr;  static size_t g_ws_sz = 0;
static float* g_lse = nullptr; static size_t g_lse_sz = 0;
static int g_sm_count = 0;  // cached at first prepare call

// Pre-allocate workspace + LSE to the given upper bound + cache SM count.
// Must be called from a non-capture context before any graph capture that
// includes cutlass_fp8_fmha_strided. Safe to call repeatedly; grows buffers
// only if requested size exceeds current capacity.
extern "C" int cutlass_fp8_fmha_prepare(
    int max_SQ, int max_SK, int max_B, int max_NQ, int max_HD) {
    int D = cutlass::round_up(max_HD, 8);
    int SQ_r = ((max_SQ + 127) / 128) * 128;
    size_t lsz = (size_t)max_B * max_NQ * SQ_r * sizeof(float);
    if (lsz > g_lse_sz) {
        if (g_lse) cudaFree(g_lse);
        if (cudaMalloc(&g_lse, lsz) != cudaSuccess) return -10;
        g_lse_sz = lsz;
    }
    if (g_sm_count == 0) {
        cudaDeviceGetAttribute(&g_sm_count, cudaDevAttrMultiProcessorCount, 0);
    }
    // Probe workspace size with a plausible args and pre-alloc. Using max
    // shape so later smaller invocations fit without re-malloc.
    // Workspace for SM100 FMHA is typically small (~64KB of counters);
    // reserve 4MB to be safe across all shape variants.
    size_t wsz = 4 * 1024 * 1024;
    if (wsz > g_ws_sz) {
        if (g_ws) cudaFree(g_ws);
        if (cudaMalloc(&g_ws, wsz) != cudaSuccess) return -11;
        g_ws_sz = wsz;
    }
    return 0;
}

// Public entry: FP8 FMHA forward, strided Q/K/V, BF16 output.
//
//   Q    : FP8 e4m3, [B, SQ, NQ, HD], stride q_seq_stride between seqlen
//   K, V : FP8 e4m3, [B, SK, NKV, HD], stride k_seq_stride between seqlen
//   O    : BF16,     [B, SQ, NQ, HD]   contiguous
//
//   scale_q, scale_k, scale_v, inv_scale_o : per-tensor scales
//     bf16(q) ≈ fp8(q) * scale_q (same for k, v)
//     output_bf16 = FP8_FMHA_output * scale_v / scale_o  (scale_o defaults 1)
//
// Returns 0 on success, negative on failure.
extern "C" int cutlass_fp8_fmha_strided(
    const void* Q, const void* K, const void* V, void* O,
    int B, int SQ, int SK, int NQ, int NKV, int HD,
    int q_seq_stride,   // stride between Q tokens (contiguous: NH*HD, or 3*NH*HD if interleaved)
    int k_seq_stride,   // stride between K/V tokens (same layout)
    float scale_q, float scale_k, float scale_v, float inv_scale_o,
    cudaStream_t stream)
{
    int H_Q = NQ / NKV, H_K = NKV, H = H_Q * H_K;
    int D = cutlass::round_up(HD, 8);
    auto ps = cute::make_tuple(SQ, SK, D,
                                cute::make_tuple(cute::make_tuple(H_Q, H_K), B));

    StrideQ sQ = make_stride(q_seq_stride, _1{},
                              make_stride(make_stride(D, H_Q * D),
                                          q_seq_stride * SQ));
    StrideO sO = make_stride(H * D, _1{},
                              make_stride(make_stride(D, H_Q * D),
                                          H * D * SQ));
    StrideK sK = make_stride(k_seq_stride, _1{},
                              make_stride(make_stride(_0{}, D),
                                          k_seq_stride * SK));

    int SQ_r = ((SQ + 127) / 128) * 128;
    StrideLSE sL = make_stride(_1{},
                                make_stride(make_stride(SQ_r, SQ_r * H_Q),
                                            SQ_r * H));

    // GRAPH CAPTURE SAFETY: we must NOT call cudaMalloc / cudaDeviceGetAttribute
    // in the hot path. Caller must invoke cutlass_fp8_fmha_prepare() first to
    // pre-allocate g_ws/g_lse and cache g_sm_count.
    size_t lsz = (size_t)B * H * SQ_r * sizeof(float);
    if (lsz > g_lse_sz) {
        printf("[FP8 FMHA] LSE buffer too small (%zu > %zu); call cutlass_fp8_fmha_prepare first\n",
               lsz, g_lse_sz);
        return -10;
    }
    if (g_sm_count == 0) {
        printf("[FP8 FMHA] g_sm_count==0; call cutlass_fp8_fmha_prepare first\n");
        return -12;
    }

    typename FmhaOp::Arguments args{
        ps,
        { { (Element const*)Q, sQ,
            (Element const*)K, sK,
            (Element const*)V, sK },
          /* scale_softmax = */ 0.0f,   // let mainloop default to 1/sqrt(HD)
          scale_q, scale_k, scale_v, inv_scale_o },
        { (ElementOut*)O, sO, g_lse, sL },
        { 0, g_sm_count }
    };

    FmhaOp op;
    auto st = op.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        printf("[FP8 FMHA] can_implement FAILED (%d) SQ=%d SK=%d NQ=%d NKV=%d HD=%d qstride=%d kstride=%d\n",
               (int)st, SQ, SK, NQ, NKV, HD, q_seq_stride, k_seq_stride);
        return -1;
    }
    size_t wsz = FmhaOp::get_workspace_size(args);
    if (wsz > g_ws_sz) {
        printf("[FP8 FMHA] workspace too small (%zu > %zu); grow prepare() reserve\n",
               wsz, g_ws_sz);
        return -11;
    }
    if (op.initialize(args, g_ws, stream) != cutlass::Status::kSuccess) return -2;
    return (op.run(stream) == cutlass::Status::kSuccess) ? 0 : -3;
}

}  // namespace fmha
}  // namespace flash_wm
