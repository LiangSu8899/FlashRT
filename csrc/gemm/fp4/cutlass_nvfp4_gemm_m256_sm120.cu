// SPDX-License-Identifier: Apache-2.0
//
// Large-M NVFP4 W4A4 GEMM tier for sm_120: MmaTileShape 256x128x128,
// cooperative schedule, cluster 1x1. On the M~2048 prefill shapes this
// tile wins every family over the 128x128x128 baseline (measured
// 1428/1588/1430/1360 TFLOPS on 17408x5120 / 5120x17408 / 12288x5120 /
// 16384x5120 vs 1303/1361/1311/1289; FP4 MMA roof ~2020). K traversal
// order is unchanged, so per-element accumulation matches the baseline.
// Dispatch intent: route M >= 512 here, keep the small-M tier as is.
//
// Requires CUTLASS >= 4.5 (Sm120 blockscaled collective; the Sm100-tagged
// builder path refuses initialize on sm_120 from 4.5.x). Build with
// -gencode arch=compute_120a,code=sm_120a --expt-relaxed-constexpr.
//
// Header: cutlass_nvfp4_gemm_m256_sm120.cuh.
#include "cutlass_nvfp4_gemm_m256_sm120.cuh"

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace flash_rt {
namespace gemm {
namespace {

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementD = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using EAcc = float;
using Arch = cutlass::arch::Sm120;
using Op = cutlass::arch::OpClassBlockScaledTensorOp;
using MmaTile = Shape<_256, _128, _128>;
using Cluster = Shape<_1, _1, _1>;

using CE = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, Op, MmaTile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, EAcc, EAcc,
    ElementC, cutlass::layout::RowMajor, 8,
    ElementD, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CM = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, Op,
    ElementA, cutlass::layout::RowMajor, 32,
    ElementB, cutlass::layout::ColumnMajor, 32,
    EAcc, MmaTile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CE::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GK = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                CM, CE, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>;

}  // namespace

size_t nvfp4_gemm_m256_sm120_workspace_size(int M, int N, int K) {
  using SA = typename Gemm::GemmKernel::StrideA;
  using SB = typename Gemm::GemmKernel::StrideB;
  using SD = typename Gemm::GemmKernel::StrideD;
  using Cfg = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto sa = cutlass::make_cute_packed_stride(SA{}, {M, K, 1});
  auto sb = cutlass::make_cute_packed_stride(SB{}, {N, K, 1});
  auto sd = cutlass::make_cute_packed_stride(SD{}, {M, N, 1});
  auto lsfa = Cfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto lsfb = Cfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {nullptr, sa, nullptr, sb, nullptr, lsfa, nullptr, lsfb},
      {{1.f, 0.f}, nullptr, sd, nullptr, sd}};
  return Gemm::get_workspace_size(args);
}

int nvfp4_gemm_m256_sm120_bf16(const void* A_packed, const void* SFA,
                               const void* B_packed, const void* SFB,
                               void* D_bf16, int M, int N, int K,
                               float alpha, void* workspace,
                               cudaStream_t stream) {
  using SA = typename Gemm::GemmKernel::StrideA;
  using SB = typename Gemm::GemmKernel::StrideB;
  using SD = typename Gemm::GemmKernel::StrideD;
  using Cfg = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto sa = cutlass::make_cute_packed_stride(SA{}, {M, K, 1});
  auto sb = cutlass::make_cute_packed_stride(SB{}, {N, K, 1});
  auto sd = cutlass::make_cute_packed_stride(SD{}, {M, N, 1});
  auto lsfa = Cfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto lsfb = Cfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {reinterpret_cast<const cutlass::float_e2m1_t*>(A_packed), sa,
       reinterpret_cast<const cutlass::float_e2m1_t*>(B_packed), sb,
       reinterpret_cast<const cutlass::float_ue4m3_t*>(SFA), lsfa,
       reinterpret_cast<const cutlass::float_ue4m3_t*>(SFB), lsfb},
      {{alpha, 0.f}, nullptr, sd,
       reinterpret_cast<cutlass::bfloat16_t*>(D_bf16), sd}};
  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 1;
  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess)
    return 2;
  if (gemm.run(stream) != cutlass::Status::kSuccess) return 3;
  return 0;
}

}  // namespace gemm
}  // namespace flash_rt
