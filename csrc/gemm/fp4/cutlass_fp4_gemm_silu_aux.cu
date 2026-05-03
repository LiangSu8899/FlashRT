// ============================================================================
//  FlashRT P1 — NVFP4 GEMM with fused silu(aux_gate) * acc → fp4 + SFA out.
//
//  STATUS (2026-04-15, Day 1 of P1): COMPILES, RUNTIME FAILS.
//    Custom FusionOp `LinCombSiLuAuxMulBlockScaleFactor` + FusionCallbacks
//    specialization for Sm100TmaWarpSpecialized + IsAuxInSupported=true.
//    CallbacksBuilder injects SmemLayoutAtomAux + CopyOpS2RAux from
//    Sm100AuxLoadDescriptor → my specialization matches.
//
//    Build: SUCCESS (461KB .o, no warnings).
//    Runtime at M=968 N=8192 K=2048: "Invalid __global__ read of size 4
//    bytes — Access to 0x0 is out of bounds" inside the kernel during
//    epilogue execution. The compute-sanitizer trace confirms my
//    FusionCallbacks specialization IS matched, so the failure is inside
//    the EVT data flow. Aux pointer appears to be null at use-site
//    despite being passed correctly via Arguments::operator() conversion.
//
//    Sm100 + Aux load via FusionOp::IsAuxInSupported has ZERO working
//    examples in CUTLASS 4.3.1 (verified by grep). The plumbing exists
//    (Sm100AuxLoadDescriptor + CallbacksBuilder enable_if branch) but no
//    in-tree consumer demonstrates it working. Likely a latent CUTLASS
//    issue or undocumented additional init step required.
//
//    DECISION: pivot to a 2-GEMM split design that uses only the proven
//    LinCombBlockScaleFactor / LinCombEltActBlockScaleFactor paths
//    (production-ready in CUTLASS examples 72b/75/79b/92). See
//    docs/v2/fp4_p1_pivot_plan.md for the new approach. This file kept
//    as the EVT exploration artifact for future re-attempts when
//    CUTLASS adds Sm100+Aux examples.
//
//  Original goal (still valid for the 2-GEMM design, just split across
//  3 kernels instead of fused into 1):
//
//  Eliminates the standalone F4 v2 / F4 v2+mul kernel (143 μs/layer) by
//  absorbing silu_mul + fp4 quant + SFA write into the GEMM epilogue. Used
//  as the second leg of the split-GU FFN path:
//
//      gate_acc = X @ Wg^T          (separate normal NVFP4 GEMM, fp16 out)
//      out_fp4  = NVFP4 GEMM(X, Wu) with epilogue:
//                   acc_fp32 = X @ Wu^T  (FP4 mma)
//                   gate_fp16  = aux_load[m, n]
//                   v_fp32     = silu(gate_fp16) * acc_fp32
//                   block-scale (SFVecSize=16, UE4M3 SF)
//                   pack to e2m1, write packed[m, n/2] + SFA tile-interleaved
//
//  EVT shape:
//      Sm90EVT<                                                     // ROOT
//          Sm100BlockScaleFactorRowStore<...>,                      // pack fp4 + SFA
//          Sm90EVT<                                                 // silu(aux) * acc
//              Sm90Compute<multiplies, ...>,
//              Sm90EVT< Sm90Compute<SiLu, ...>, Sm90AuxLoad<0, ...> >,
//              Sm90AccFetch
//          >
//      >
//
//  Sm90AuxLoad<Stages=0, ...> uses the direct global→register specialisation
//  (no smem, no TMA descriptor). This keeps the kernel buildable without
//  driving Cluster-shape-specific TMA descriptor wiring.
//
//  Additive: does NOT modify existing cutlass_fp4_gemm.cu / variants.
// ============================================================================
#include "gemm/fp4/cutlass_fp4_gemm_silu_aux.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// New fusion op: D = blockscale(silu(aux_gate) * acc)
//                aux_gate is fp16 [M, N] RowMajor
//                Output: D fp4 packed + per-block UE4M3 SF (RowMajor)
//
/////////////////////////////////////////////////////////////////////////////////////////////////
template<
  int SFVecSize_,
  class ElementOutput_,                                  // e.g. cutlass::float_e2m1_t
  class ElementCompute_,                                 // e.g. float
  class ElementBlockScaleFactor_,                        // e.g. cutlass::float_ue4m3_t
  class ElementAux_   = cutlass::half_t,                 // gate fp16
  class GmemLayoutTagAux_ = cutlass::layout::RowMajor,
  int   AlignmentAux_     = 8,
  class GmemLayoutTagScalefactor_ = cutlass::layout::RowMajor,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct LinCombSiLuAuxMulBlockScaleFactor : FusionOperation {
  using ElementOutput            = ElementOutput_;
  using ElementCompute           = ElementCompute_;
  using ElementSource            = ElementOutput_;
  using ElementScalar            = ElementCompute_;
  using ElementAux               = ElementAux_;
  using ElementBlockScaleFactor  = ElementBlockScaleFactor_;
  using GmemLayoutTagAux         = GmemLayoutTagAux_;
  using GmemLayoutTagScalefactor = GmemLayoutTagScalefactor_;

  static constexpr int SFVecSize     = SFVecSize_;
  static constexpr int AlignmentAux  = AlignmentAux_;
  static constexpr FloatRoundStyle RoundStyle = RoundStyle_;
  static constexpr bool IsSourceSupported     = false;     // no C tensor
  static constexpr bool IsAuxOutSupported     = false;
  static constexpr bool IsAuxInSupported      = true;      // <-- gates Sm100 CallbacksBuilder
  static constexpr bool IsBlockScaleSupported = true;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// EVT alias for the Sm100 RowMajor SF case
//
/////////////////////////////////////////////////////////////////////////////////////////////////
template<
  int Stages,
  int SFVecSize,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  class ElementAux,
  class StrideAuxMNL,
  class SmemLayoutAtomAux,
  class CopyOpS2RAux,
  int   AlignmentAux,
  FloatRoundStyle RoundStyle
>
using Sm100SiLuAuxMulRowBlockScaleFactor =
  Sm90EVT<
    Sm100BlockScaleFactorRowStore<
        SFVecSize, EpilogueTile, ElementOutput, ElementCompute, ElementBlockScaleFactor, RoundStyle>,
    Sm90EVT<
      Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
      Sm90EVT<
        Sm90Compute<cutlass::epilogue::thread::SiLu, ElementCompute, ElementCompute, RoundStyle>,
        Sm90AuxLoad<
            Stages,
            EpilogueTile,
            ElementAux,
            StrideAuxMNL,
            SmemLayoutAtomAux,
            CopyOpS2RAux,
            AlignmentAux,
            /*EnableNullptr=*/false>
      >,
      Sm90AccFetch
    >
  >;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// FusionCallbacks specialisation: maps our LinCombSiLuAuxMulBlockScaleFactor
// to the EVT impl above.
//
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int  SFVecSize,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  class ElementAux,
  class GmemLayoutTagAux,
  int  AlignmentAux,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class SmemLayoutAtomAux,        // <-- injected by CallbacksBuilder (Sm100AuxLoadDescriptor)
  class CopyOpS2RAux              // <-- injected by CallbacksBuilder
>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::LinCombSiLuAuxMulBlockScaleFactor<
        SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor,
        ElementAux, GmemLayoutTagAux, AlignmentAux,
        cutlass::layout::RowMajor /*ScaleFactor layout*/, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile,
    SmemLayoutAtomAux,
    CopyOpS2RAux
> : Sm100SiLuAuxMulRowBlockScaleFactor<
        StagesC, SFVecSize, EpilogueTile,
        typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type,
        ElementCompute, ElementBlockScaleFactor,
        ElementAux,
        cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>,
        SmemLayoutAtomAux, CopyOpS2RAux,
        AlignmentAux, RoundStyle>
{
  using Impl = Sm100SiLuAuxMulRowBlockScaleFactor<
      StagesC, SFVecSize, EpilogueTile,
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type,
      ElementCompute, ElementBlockScaleFactor,
      ElementAux,
      cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>,
      SmemLayoutAtomAux, CopyOpS2RAux,
      AlignmentAux, RoundStyle>;
  using Operation = fusion::LinCombSiLuAuxMulBlockScaleFactor<
      SFVecSize, ElementOutput, ElementCompute, ElementBlockScaleFactor,
      ElementAux, GmemLayoutTagAux, AlignmentAux,
      cutlass::layout::RowMajor, RoundStyle>;

  using StrideAux = cutlass::gemm::TagToStrideC_t<GmemLayoutTagAux>;

  struct Arguments {
    // Block-scaled FP4 output config
    ElementBlockScaleFactor* block_scale_factor_ptr = nullptr;
    using StrideNormConst = Stride<_0, _0, int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {_0{}, _0{}, 0};

    // Aux gate input
    ElementAux const* aux_ptr  = nullptr;
    ElementAux        aux_null = ElementAux(0);
    StrideAux         dAux     = {};

    operator typename Impl::Arguments() const {
      return
        // Sm90EVT root (BlockScaleFactor RowStore over child subtree)
        {
          // Child subtree: silu(aux) * acc
          {
            // silu(aux)
            {
              {aux_ptr, aux_null, dAux},  // Sm90AuxLoad args
              {}                          // Sm90Compute<SiLu> args (none)
            },
            // acc
            {},                           // Sm90AccFetch args (none)
            {}                            // multiplies binary args
          },
          // BlockScaleFactor store args
          { block_scale_factor_ptr, norm_constant_ptr, dNormConst }
        };
    }
  };

  // Ctor inheritance from Impl
  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// GEMM type definition
//
/////////////////////////////////////////////////////////////////////////////////////////////////
namespace flash_rt {
namespace fp4 {
namespace silu_aux {

using namespace cute;

using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD   = cutlass::float_e2m1_t;            // FP4 packed output
using ElementC   = ElementD;                         // unused (no source)
using LayoutDTag = cutlass::layout::RowMajor;
using LayoutCTag = LayoutDTag;
constexpr int AlignmentD = 32;
constexpr int AlignmentC = AlignmentD;

using ElementSFD = cutlass::float_ue4m3_t;
using LayoutSFDTag = LayoutDTag;

using ElementAuxGate = cutlass::half_t;
using LayoutAuxTag   = cutlass::layout::RowMajor;
constexpr int AlignmentAuxGate = 8;                  // fp16 vec=8 → 16 bytes

using ElementAccumulator  = float;
using ElementCompute      = float;
using ArchTag             = cutlass::arch::Sm100;
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;

constexpr int InputSFVectorSize  = 16;
constexpr int OutputSFVectorSize = InputSFVectorSize;

// Tile / cluster ─ start with V8 shape (best for [968, 16384, 2048] in audit).
using MmaTileShape = Shape<_128, _256, _256>;
using ClusterShape = Shape<_1, _1, _1>;

using FusionOperation = cutlass::epilogue::fusion::LinCombSiLuAuxMulBlockScaleFactor<
    OutputSFVectorSize,
    ElementD,
    ElementCompute,
    ElementSFD,
    ElementAuxGate,
    LayoutAuxTag,
    AlignmentAuxGate,
    cutlass::layout::RowMajor,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperation
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop, CollectiveEpilogue, void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using StrideAux = cutlass::gemm::TagToStrideC_t<LayoutAuxTag>;

using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

}  // namespace silu_aux

int cutlass_fp4_gemm_silu_aux_fp4(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void const* aux_gate_fp16,
    void*       D_packed,
    void*       D_SFA,
    int M, int N, int K,
    cudaStream_t stream) {
  using namespace silu_aux;

  auto stride_A   = cutlass::make_cute_packed_stride(StrideA{},   {M, K, 1});
  auto stride_B   = cutlass::make_cute_packed_stride(StrideB{},   {N, K, 1});
  auto stride_C   = cutlass::make_cute_packed_stride(StrideC{},   {M, N, 1});
  auto stride_D   = cutlass::make_cute_packed_stride(StrideD{},   {M, N, 1});
  auto stride_Aux = cutlass::make_cute_packed_stride(StrideAux{}, {M, N, 1});
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

  using EA = typename ElementA::DataType;
  using SA = typename ElementA::ScaleFactorType;
  using EB = typename ElementB::DataType;
  using SB = typename ElementB::ScaleFactorType;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      { reinterpret_cast<EA const*>(A_packed), stride_A,
        reinterpret_cast<EB const*>(B_packed), stride_B,
        reinterpret_cast<SA const*>(SFA), layout_SFA,
        reinterpret_cast<SB const*>(SFB), layout_SFB },
      { /*epilogue thread args:*/
        { /*FusionCallbacks::Arguments*/
          reinterpret_cast<ElementSFD*>(D_SFA),                     // block_scale_factor_ptr
          /*norm_constant_ptr*/ nullptr,
          /*dNormConst*/ {},
          reinterpret_cast<ElementAuxGate const*>(aux_gate_fp16),   // aux_ptr
          /*aux_null*/ ElementAuxGate(0),
          stride_Aux                                                // dAux
        },
        reinterpret_cast<ElementC*>(D_packed), stride_C,
        reinterpret_cast<ElementD*>(D_packed), stride_D
      }
  };

  Gemm gemm;
  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) return static_cast<int>(st) | 0x10000;
  size_t ws_sz = Gemm::get_workspace_size(args);
  void* ws = nullptr;
  if (ws_sz > 0 && cudaMalloc(&ws, ws_sz) != cudaSuccess) return -1;
  st = gemm.initialize(args, ws, stream);
  if (st != cutlass::Status::kSuccess) {
    if (ws) cudaFree(ws);
    return static_cast<int>(st) | 0x20000;
  }
  st = gemm.run(stream);
  if (ws) cudaFree(ws);
  return (st == cutlass::Status::kSuccess) ? 0 : (static_cast<int>(st) | 0x30000);
}

}  // namespace fp4
}  // namespace flash_rt
