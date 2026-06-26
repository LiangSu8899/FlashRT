// SPDX-License-Identifier: Apache-2.0
//
// NormFoldBuilder — assembles the forked NVFP4 sm120 blockscaled CollectiveMma
// (sm120_normfold_mma_tma.hpp) without re-deriving the ~30 intermediate CuTe
// types the stock CUTLASS CollectiveBuilder computes.
//
// Strategy (minimal plumbing, maximally maintainable):
//   1. Instantiate the *stock* sm120 blockscaled CollectiveBuilder for the same
//      (TileShape, ClusterShape, schedule) the production GEMM uses. Its
//      CollectiveOp re-exposes every type we need: TiledMma, the SF / smem
//      layout atoms, the gmem/smem copy pairs, the stride pairs, and the
//      resolved DispatchPolicy (which carries the auto-computed PipelineStages,
//      SchedulerPipelineStageCount and the BlockScaled KernelSchedule).
//   2. Re-assemble CollectiveMma with the *MainloopSm120NormFold* dispatch tag
//      (distinct type → selects our forked specialization, no ODR conflict),
//      passing the stock-computed types through verbatim.
//
// At identity (no A-path edits) this MUST produce a kernel bit-identical to the
// production fp4 GEMM — that is milestone M-FULL-3a-i, the proof the fork is
// instantiable before any norm-fold transform is introduced.

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "sm120_normfold_mma_tma.hpp"  // forked CollectiveMma + MainloopSm120NormFold

namespace flash_rt {
namespace gemm {
namespace normfold {

// TileShape_MNK / ClusterShape_MNK are static CuTe shapes (e.g. Shape<_128,_128,_256>).
// StageCountType matches whatever the production GEMM passes (StageCountAutoCarveout<...>).
template <class TileShape_MNK, class ClusterShape_MNK, class StageCountType>
struct NormFoldBuilder {
  using ElementA           = cutlass::float_e2m1_t;
  using ElementB           = cutlass::float_e2m1_t;
  using ElementAccumulator = float;
  using ElementSF          = cutlass::float_ue4m3_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  static constexpr int AlignmentA = 16 * 8 / cutlass::sizeof_bits<ElementA>::value;  // 32
  static constexpr int AlignmentB = 16 * 8 / cutlass::sizeof_bits<ElementB>::value;  // 32

  // (1) Stock builder — the vendor-tuned config the production GEMM uses.
  using StockOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementPairA, LayoutA, AlignmentA,
      ElementPairB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      StageCountType,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  // The stock DispatchPolicy (MainloopSm120TmaWarpSpecializedBlockScaled) carries
  // the auto-resolved pipeline depth + the BlockScaled kernel schedule.
  using StockDP = typename StockOp::DispatchPolicy;

  // (2) Swap *only* the dispatch tag → selects the forked specialization.
  using DispatchPolicy = cutlass::gemm::collective::MainloopSm120NormFold<
      StockDP::Stages,
      StockDP::SchedulerPipelineStageCount,
      typename StockDP::ClusterShape,
      typename StockDP::Schedule>;

  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      cute::tuple<ElementA, ElementSF>,
      typename StockOp::StridePairA,
      cute::tuple<ElementB, ElementSF>,
      typename StockOp::StridePairB,
      typename StockOp::TiledMma,
      typename StockOp::GmemTiledCopyPairA,
      typename StockOp::SmemLayoutAtomsA,
      typename StockOp::SmemCopyAtomsA,
      cute::identity,
      typename StockOp::GmemTiledCopyPairB,
      typename StockOp::SmemLayoutAtomsB,
      typename StockOp::SmemCopyAtomsB,
      cute::identity>;
};

}  // namespace normfold
}  // namespace gemm
}  // namespace flash_rt
