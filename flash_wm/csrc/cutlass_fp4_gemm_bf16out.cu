// ============================================================================
//  flash_wm — NVFP4 GEMM with BF16 output (Thor SM100/SM110).
//
//  Direct mirror of the upstream flash_rt FP4 GEMM variant template, with
//  ElementC/ElementD swapped from cutlass::half_t to cutlass::bfloat16_t.
//  Needed because BAGEL's residual stream is BF16 and the FP4 Down GEMM's
//  fp16 output accumulator overflows 65504 at some layers/timesteps (L5 @
//  t=0.2, L9 @ t=0.4 — see flash_wm/tests/debug_l9_t04.py).
//
//  Scope: additive. Doesn't touch upstream .so or existing kernels in
//  flash_wm. Only the Down GEMM path is rewired to use this; Gate/Up stay
//  fp16-out since their output magnitudes are <283 (fp16-safe).
//
//  Based on CUTLASS example 72a (Blackwell NVFP4 × NVFP4 → bfloat16).
// ============================================================================

#include <cuda_runtime.h>
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

namespace flash_wm {
namespace fp4 {

using namespace cute;

// Parametric variant — same template as upstream, ElementC/D = bfloat16_t.
template <class MmaTile, class Cluster>
struct VariantBf16 {
  using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  // ── Only difference vs upstream: bf16 output instead of fp16 ──
  using ElementD   = cutlass::bfloat16_t;
  using ElementC   = cutlass::bfloat16_t;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ArchTag            = cutlass::arch::Sm100;
  using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      MmaTile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      MmaTile, Cluster,
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
  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  static int run(void const* A, void const* SFA, void const* B, void const* SFB,
                 void* D, int M, int N, int K, float alpha, float beta,
                 cudaStream_t stream) {
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    using EA = typename ElementA::DataType;
    using SA = typename ElementA::ScaleFactorType;
    using EB = typename ElementB::DataType;
    using SB = typename ElementB::ScaleFactorType;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        { reinterpret_cast<EA const*>(A), stride_A,
          reinterpret_cast<EB const*>(B), stride_B,
          reinterpret_cast<SA const*>(SFA), layout_SFA,
          reinterpret_cast<SB const*>(SFB), layout_SFB },
        { {alpha, beta},
          reinterpret_cast<ElementC*>(D), stride_C,
          reinterpret_cast<ElementD*>(D), stride_D }
    };
    Gemm gemm;
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x10000;
    size_t ws_sz = Gemm::get_workspace_size(args);
    void* ws = nullptr;
    if (ws_sz > 0 && cudaMalloc(&ws, ws_sz) != cudaSuccess) return -1;
    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) { if (ws) cudaFree(ws); return static_cast<int>(st) | 0x20000; }
    st = gemm.run(stream);
    if (ws) cudaFree(ws);
    return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
  }
};

// ── Only the variants we need for BAGEL's FFN Down path (V8 wide-NK).
//    Add more later if we want bf16-out Gate/Up (V6 wide-N).
using V6_bf16 = VariantBf16<Shape<_128,_256,_128>, Shape<_1,_1,_1>>;  // Gate/Up
using V8_bf16 = VariantBf16<Shape<_128,_256,_256>, Shape<_1,_1,_1>>;  // Down

// Public entry: dispatch by variant index (matches upstream indexing where
// possible — only V6 and V8 implemented for now).
extern "C" int cutlass_fp4_gemm_bf16out_variant(int idx,
    void const* A, void const* SFA, void const* B, void const* SFB,
    void* D_bf16, int M, int N, int K, float alpha, float beta,
    cudaStream_t stream) {
  switch (idx) {
    case 6: return V6_bf16::run(A, SFA, B, SFB, D_bf16, M, N, K, alpha, beta, stream);
    case 8: return V8_bf16::run(A, SFA, B, SFB, D_bf16, M, N, K, alpha, beta, stream);
    default: return -99;  // unimplemented variant
  }
}

}  // namespace fp4
}  // namespace flash_wm
