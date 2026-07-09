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

#include "sm120_normfold_mma_tma.hpp"        // forked CollectiveMma + MainloopSm120NormFold
#include "sm120_normfold_mma_tma_bf16a.hpp"  // bf16-A CollectiveMma + MainloopSm120NormFoldBf16A
#include "sm120_normfold_mma_tma_pq.hpp"     // producer-quant CollectiveMma + MainloopSm120NormFoldPQ

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

// ── bf16-A variant: the norm-fold collective that loads A as bf16 and quantizes
// to NVFP4 in the consumer registers. Delegates ALL CuTe type derivation to the
// stock fp4 builder EXCEPT (a) the dispatch tag (MainloopSm120NormFoldBf16A) and
// (b) the A smem layout/copy atoms, which must be the standard 16-bit path (the
// fp4 rr selector is <=8-bit only). B / SFB / TiledMma / SFA-layout-atom / stride
// pairs / dispatch stages are reused verbatim.
template <class TileShape_MNK, class ClusterShape_MNK, class StageCountType>
struct NormFoldBuilderBf16A {
  using ElementA           = cutlass::float_e2m1_t;     // MMA operand type
  using ElementALoad       = cutlass::bfloat16_t;       // gmem/smem load type
  using ElementB           = cutlass::float_e2m1_t;
  using ElementAccumulator = float;
  using ElementSF          = cutlass::float_ue4m3_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  static constexpr int AlignmentA = 16 * 8 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 16 * 8 / cutlass::sizeof_bits<ElementB>::value;

  using StockOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementPairA, LayoutA, AlignmentA,
      ElementPairB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      StageCountType,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;
  using StockDP = typename StockOp::DispatchPolicy;

  using DispatchPolicy = cutlass::gemm::collective::MainloopSm120NormFoldBf16A<
      StockDP::Stages,
      StockDP::SchedulerPipelineStageCount,
      typename StockDP::ClusterShape,
      typename StockDP::Schedule>;

  // bf16 A smem: standard 16-bit K-major swizzle atom (TMA-compatible). The stock
  // SFA-layout atom (get<1>) is kept for the (computed) tCrSFA fragment shape.
  using SmemLayoutAtomA_bf16 = decltype(cute::UMMA::Layout_K_SW128_Atom<ElementALoad>());
  using SmemLayoutAtomsA = decltype(cute::make_tuple(
      SmemLayoutAtomA_bf16{},
      cute::get<1>(typename StockOp::SmemLayoutAtomsA{})));
  // A smem->rmem copy: a 16-bit ldmatrix (the consumer reads bf16 via partition_A;
  // this atom satisfies the collective's non-void static_assert).
  using SmemCopyAtomA_bf16 = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementALoad>;
  using SmemCopyAtomsA = decltype(cute::make_tuple(
      SmemCopyAtomA_bf16{},
      cute::get<1>(typename StockOp::SmemCopyAtomsA{})));

  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      cute::tuple<ElementA, ElementSF>,
      typename StockOp::StridePairA,
      cute::tuple<ElementB, ElementSF>,
      typename StockOp::StridePairB,
      typename StockOp::TiledMma,
      typename StockOp::GmemTiledCopyPairA,
      SmemLayoutAtomsA,
      SmemCopyAtomsA,
      cute::identity,
      typename StockOp::GmemTiledCopyPairB,
      typename StockOp::SmemLayoutAtomsB,
      typename StockOp::SmemCopyAtomsB,
      cute::identity>;
};

// ── producer-quant (PQ) variant: smem_A is FP4 (consumer reads it stock = roofline),
// the producer TMAs bf16 A into a separate staging buffer, and the consumer quantizes
// it (natural layout) into smem_A + SFA. Uses the IDENTITY fp4 atoms verbatim (the
// consumer is unchanged) — only the dispatch tag swaps to MainloopSm120NormFoldPQ.
template <class TileShape_MNK, class ClusterShape_MNK, class StageCountType>
struct NormFoldBuilderPQ {
  using ElementA           = cutlass::float_e2m1_t;
  using ElementB           = cutlass::float_e2m1_t;
  using ElementAccumulator = float;
  using ElementSF          = cutlass::float_ue4m3_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  static constexpr int AlignmentA = 16 * 8 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 16 * 8 / cutlass::sizeof_bits<ElementB>::value;

  using StockOp = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementPairA, LayoutA, AlignmentA,
      ElementPairB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape_MNK, ClusterShape_MNK,
      StageCountType,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;
  using StockDP = typename StockOp::DispatchPolicy;

  using DispatchPolicy = cutlass::gemm::collective::MainloopSm120NormFoldPQ<
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
