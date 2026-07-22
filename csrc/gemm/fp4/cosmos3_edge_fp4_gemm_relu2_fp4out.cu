#include "gemm/fp4/cosmos3_edge_fp4_gemm_relu2_fp4out.cuh"

#ifndef FLASHRT_HAVE_COSMOS3_EDGE
#error "cosmos3_edge_fp4_gemm_relu2_fp4out.cu requires FLASHRT_HAVE_COSMOS3_EDGE"
#endif

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

// CUTLASS 4.4 omits the non-grouped SM100 callback for this composition even
// though both constituent EVT nodes are supported. Keep the adapter local to
// this translation unit; no vendored header modification is required.
namespace cutlass::epilogue::fusion {

template <
    int SFVecSize, class EpilogueTile, template <class> class ActivationFn,
    class ElementOutput, class ElementCompute, class ElementBlockScaleFactor,
    class ElementSource = ElementOutput, class ElementScalar = ElementCompute,
    FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>
using Cosmos3EdgeLinCombEltActRowBlockScaleFactor = Sm90EVT<
    Sm100BlockScaleFactorRowStore<SFVecSize, EpilogueTile, ElementOutput,
                                  ElementCompute, ElementBlockScaleFactor,
                                  RoundStyle>,
    Sm90LinCombEltAct<ActivationFn, ElementCompute, ElementCompute,
                      ElementSource, ElementScalar, RoundStyle>>;

template <
    int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC,
    bool DelayTmaStore, template <class> class ActivationFn,
    class ElementOutput, class ElementCompute, class ElementBlockScaleFactor,
    int SFVecSize, class ElementSource, class ElementScalar,
    FloatRoundStyle RoundStyle, class CtaTileShapeMNK, class EpilogueTile>
struct FusionCallbacks<
    epilogue::Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize,
                                      ReuseSmemC, DelayTmaStore>,
    fusion::LinCombEltActBlockScaleFactor<
        ActivationFn, SFVecSize, ElementOutput, ElementCompute,
        ElementBlockScaleFactor, cutlass::layout::RowMajor, ElementSource,
        ElementScalar, RoundStyle>,
    CtaTileShapeMNK, EpilogueTile>
    : Cosmos3EdgeLinCombEltActRowBlockScaleFactor<
          SFVecSize, EpilogueTile, ActivationFn,
          typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type,
          ElementCompute, ElementBlockScaleFactor, ElementSource,
          ElementScalar, RoundStyle> {
  using Impl = Cosmos3EdgeLinCombEltActRowBlockScaleFactor<
      SFVecSize, EpilogueTile, ActivationFn,
      typename cutlass::detail::get_unpacked_element_type<ElementOutput>::type,
      ElementCompute, ElementBlockScaleFactor, ElementSource, ElementScalar,
      RoundStyle>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementBlockScaleFactor* block_scale_factor_ptr = nullptr;
    using StrideNormConst = cute::Stride<cute::_0, cute::_0, int64_t>;
    ElementCompute const* norm_constant_ptr = nullptr;
    StrideNormConst dNormConst = {cute::_0{}, cute::_0{}, 0};
    using StrideAlpha = cute::Stride<cute::_0, cute::_0, int64_t>;
    using StrideBeta = cute::Stride<cute::_0, cute::_0, int64_t>;
    StrideAlpha dAlpha = {cute::_0{}, cute::_0{}, 0};
    StrideBeta dBeta = {cute::_0{}, cute::_0{}, 0};
    using ActivationArguments = typename Sm90Compute<
        ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return {
          {{{{beta}, {beta_ptr}, {dBeta}}, {},
            {{{alpha}, {alpha_ptr}, {dAlpha}}, {}, {}}, {}}, activation},
          {block_scale_factor_ptr, norm_constant_ptr, dNormConst},
      };
    }
  };

  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion

namespace flash_rt {
namespace fp4 {
namespace cosmos3_edge_relu2_fp4out {

using namespace cute;

template <typename T>
struct ReLU2 {
  static constexpr bool kIsHeavy = false;

  CUTLASS_HOST_DEVICE
  T operator()(T const& value) const {
    T positive = cutlass::epilogue::thread::ReLu<T>{}(value);
    return cutlass::multiplies<T>{}(positive, positive);
  }
};

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementD = cutlass::float_e2m1_t;
using ElementC = ElementD;
using ElementSFD = cutlass::float_ue4m3_t;
using ElementAccumulator = float;
using ElementCompute = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
using LayoutC = LayoutD;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 32;
constexpr int AlignmentD = 32;
constexpr int OutputSFVectorSize = 16;

using TileShape = Shape<_128, _256, _256>;
using ClusterShape = Shape<_1, _1, _1>;
using FusionOperation = cutlass::epilogue::fusion::LinCombEltActBlockScaleFactor<
    ReLU2, OutputSFVectorSize, ElementD, ElementCompute,
    ElementSFD, LayoutD, ElementC>;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperation>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;
using Sm1xxBlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

}  // namespace cosmos3_edge_relu2_fp4out

int cosmos3_edge_fp4_gemm_relu2_fp4out(
    void const* A_packed, void const* SFA,
    void const* B_packed, void const* SFB,
    void* D_packed, void* D_SFD,
    int M, int N, int K,
    cudaStream_t stream) {
  using namespace cosmos3_edge_relu2_fp4out;
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
  auto problem = make_shape(M, N, K, 1);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem);
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem);
  using EA = typename ElementA::DataType;
  using SA = typename ElementA::ScaleFactorType;
  using EB = typename ElementB::DataType;
  using SB = typename ElementB::ScaleFactorType;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {reinterpret_cast<EA const*>(A_packed), stride_A,
       reinterpret_cast<EB const*>(B_packed), stride_B,
       reinterpret_cast<SA const*>(SFA), layout_SFA,
       reinterpret_cast<SB const*>(SFB), layout_SFB},
      {{1.0f, 0.0f}, reinterpret_cast<ElementC*>(D_packed), stride_C,
       reinterpret_cast<ElementD*>(D_packed), stride_D}};

  static float* d_norm = nullptr;
  if (!d_norm) {
    if (cudaMalloc(&d_norm, sizeof(float)) != cudaSuccess) return -1;
    float one = 1.0f;
    if (cudaMemcpyAsync(d_norm, &one, sizeof(float), cudaMemcpyHostToDevice, stream)
        != cudaSuccess) return -2;
  }
  args.epilogue.thread.block_scale_factor_ptr =
      reinterpret_cast<ElementSFD*>(D_SFD);
  args.epilogue.thread.norm_constant_ptr = d_norm;

  Gemm gemm;
  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) return static_cast<int>(status) | 0x10000;
  if (Gemm::get_workspace_size(args) != 0) return -3;
  status = gemm.initialize(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) return static_cast<int>(status) | 0x20000;
  status = gemm.run(stream);
  return status == cutlass::Status::kSuccess
      ? 0 : (static_cast<int>(status) | 0x30000);
}

}  // namespace fp4
}  // namespace flash_rt
