// ============================================================================
//  flash_wm — CUTLASS FP8 SqGemm with per-column BIAS epilogue, BF16 output.
//
//  Class-1a of the Pi0.7 FP4 optimization stack. Fuses the post-GEMM
//  `gpu_add_bias_bf16` (168 tiny launches per fwd) into the SqGemm epilogue,
//  eliminating the launch bubble.
//
//  Mathematically equivalent to:
//     D = alpha * (A @ B^T) + bias[j]
//  where A: FP8 E4M3 RowMajor [M, K], B: FP8 E4M3 ColumnMajor [N, K],
//  bias: BF16 [N] (per-col), D: BF16 RowMajor [M, N], alpha: FP32 scalar.
//
//  Additive only — upstream flash_rt/csrc/gemm/cutlass_sm100.cu is untouched.
//  Self-contained kernel lives in flash_wm/csrc/, uses CUTLASS 3.x collective
//  builder with LinCombPerColBias fusion.
// ============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

namespace flash_wm {
namespace fp8_bias {

using namespace cute;

using ElementA   = cutlass::float_e4m3_t;
using LayoutA    = cutlass::layout::RowMajor;
using ElementB   = cutlass::float_e4m3_t;
using LayoutB    = cutlass::layout::ColumnMajor;
using ElementC   = cutlass::bfloat16_t;
using ElementD   = cutlass::bfloat16_t;
using LayoutC    = cutlass::layout::RowMajor;
using LayoutD    = cutlass::layout::RowMajor;
using ElementBias    = cutlass::bfloat16_t;
using ElementAcc     = float;
using ElementCompute = float;

static constexpr int AlignA    = 16;                                  // 128b / 8b = 16
static constexpr int AlignB    = 16;
static constexpr int AlignC    = 128 / cutlass::sizeof_bits<ElementC>::value;     // 8
static constexpr int AlignD    = AlignC;
static constexpr int AlignBias = 128 / cutlass::sizeof_bits<ElementBias>::value;  // 8

using ArchTag   = cutlass::arch::Sm100;
using OpClass   = cutlass::arch::OpClassTensorOp;

// Mirror upstream sm100_sq_bf16out tile/cluster exactly so we pick up the same
// tactic Myelin/prod have been validating against.
using TileShape    = Shape<_256, _256, _128>;
using ClusterShape = Shape<_2, _2, _1>;

// D = alpha * acc + per-col bias.  (No Source C path, just bias.)
using Fusion = cutlass::epilogue::fusion::LinCombPerColBias<
    ElementD, ElementCompute, ElementBias,
    ElementC, ElementCompute,
    AlignBias, cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementCompute,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    Fusion
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop, CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

static int run(void const* A, void const* B, void const* bias,
               void* D, int M, int N, int K, float alpha,
               cudaStream_t stream) {
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        { reinterpret_cast<ElementA const*>(A), stride_A,
          reinterpret_cast<ElementB const*>(B), stride_B },
        { {},  // epilogue.thread (fusion args filled below)
          reinterpret_cast<ElementC const*>(D), stride_C,  // C unused (beta=0)
          reinterpret_cast<ElementD*>(D), stride_D }
    };

    auto& fusion_args = args.epilogue.thread;
    fusion_args.alpha = alpha;
    fusion_args.beta  = 0.0f;
    fusion_args.bias_ptr = reinterpret_cast<ElementBias const*>(bias);

    Gemm gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
        fprintf(stderr, "[fp8_bias] can_implement failed: M=%d N=%d K=%d code=%d\n",
                M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x10000;
    }

    size_t ws_sz = Gemm::get_workspace_size(args);
    static void*  ws_ptr  = nullptr;
    static size_t ws_cap  = 0;
    if (ws_sz > ws_cap) {
        if (ws_ptr) cudaFree(ws_ptr);
        if (cudaMalloc(&ws_ptr, ws_sz) != cudaSuccess) { ws_ptr = nullptr; ws_cap = 0; return -1; }
        ws_cap = ws_sz;
    }

    st = gemm.initialize(args, ws_ptr, stream);
    if (st != cutlass::Status::kSuccess) {
        fprintf(stderr, "[fp8_bias] init failed: M=%d N=%d K=%d code=%d\n",
                M, N, K, static_cast<int>(st));
        return static_cast<int>(st) | 0x20000;
    }

    st = gemm.run(stream);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

}  // namespace fp8_bias
}  // namespace flash_wm

// Public C entry: FP8 SqGemm with per-col BF16 bias epilogue, BF16 output.
//   A:    FP8 E4M3, [M, K] RowMajor
//   B:    FP8 E4M3, [N, K] ColumnMajor (i.e. [K, N] RowMajor in storage)
//   bias: BF16, [N]
//   D:    BF16, [M, N] RowMajor
//   alpha: FP32 scalar (act_scale product). beta is always 0.
extern "C" int cutlass_fp8_sq_bias_bf16out(
    void const* A, void const* B, void const* bias, void* D,
    int M, int N, int K, float alpha, cudaStream_t stream) {
    return flash_wm::fp8_bias::run(A, B, bias, D, M, N, K, alpha, stream);
}
