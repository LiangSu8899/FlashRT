// SPDX-License-Identifier: Apache-2.0
//
// See cutlass_nvfp4_normfold_probe_sm120.cuh. Mirrors the production
// fp4_w4a16_gemm_sm120_bf16out kernel assembly (cutlass_nvfp4_w4a16_gemm_sm120.cu)
// exactly, EXCEPT the mainloop CollectiveOp comes from NormFoldBuilder (which
// selects the forked MainloopSm120NormFold CollectiveMma specialization). The
// epilogue, scheduler, tile shape, and arguments are identical, so at identity
// the numeric output is bit-identical to the production GEMM.

#include "cutlass_nvfp4_normfold_probe_sm120.cuh"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "sm120_normfold_builder.hpp"
#include "sm120_normfold_mma_tma_bf16a.hpp"

#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace flash_rt {
namespace gemm {
namespace {

using namespace cute;

using ElementA           = cutlass::float_e2m1_t;
using ElementB           = cutlass::float_e2m1_t;
using ElementC           = cutlass::bfloat16_t;
using ElementD           = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute     = float;
using ElementSF          = cutlass::float_ue4m3_t;

using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8

using ClusterShape = Shape<_1, _1, _1>;
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;

// One TileShape variant of the forked-collective GEMM. Epilogue/scheduler/args
// are the production config; only the mainloop is NormFoldBuilder::CollectiveOp.
// variant 0 = <128,128,256> (production tile, the identity anchor); variant 1 =
// <128,128,64> (the BLK_K=64 tile the bf16-A fold REQUIRES — bf16 A is 64KB/stage
// at K=256 so only 1 stage fits <100KB smem, violating Stages>=2; at K=64 it is
// 16KB/stage and fits. This variant proves the K=64 tile builds + runs correct).
template <class TileShape>
struct ProbeVariant {
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
          TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute,
          ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto
      >::CollectiveOp;

  using CollectiveMainloop = typename normfold::NormFoldBuilder<
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Per-shape workspace cache.
  struct ShapeKey {
    int M, N, K;
    bool operator==(const ShapeKey& o) const { return M == o.M && N == o.N && K == o.K; }
  };
  struct ShapeKeyHash {
    size_t operator()(const ShapeKey& k) const noexcept {
      return (static_cast<size_t>(k.M) * 1315423911u)
           ^ (static_cast<size_t>(k.N) * 2654435761u)
           ^ static_cast<size_t>(k.K);
    }
  };
  struct CachedWorkspace { void* ptr = nullptr; size_t size = 0; };

  static cutlass::Status run(
      const void* A_packed, const void* B_packed, void* D_bf16,
      int M, int N, int K, const void* SFA, const void* SFB,
      float alpha, cudaStream_t stream) {
    static std::unordered_map<ShapeKey, CachedWorkspace, ShapeKeyHash> ws_cache;
    static std::mutex ws_mu;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    auto problem_shape_MNKL = cute::make_shape(M, N, K, 1);
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape_MNKL);
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape_MNKL);

    using ArrayElementA = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementA;
    using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<ArrayElementA const*>(A_packed), stride_A,
            reinterpret_cast<ArrayElementB const*>(B_packed), stride_B,
            reinterpret_cast<ElementSF const*>(SFA), layout_SFA,
            reinterpret_cast<ElementSF const*>(SFB), layout_SFB
        },
        { {alpha, 0.0f}, nullptr, stride_C, reinterpret_cast<ElementD*>(D_bf16), stride_D }
    };

    Gemm gemm;
    size_t ws_size = Gemm::get_workspace_size(args);
    void* ws_ptr = nullptr;
    {
      std::lock_guard<std::mutex> lk(ws_mu);
      ShapeKey key{M, N, K};
      auto it = ws_cache.find(key);
      if (it != ws_cache.end() && it->second.size >= ws_size) {
        ws_ptr = it->second.ptr;
      } else {
        if (it != ws_cache.end()) { cudaFree(it->second.ptr); ws_cache.erase(it); }
        CachedWorkspace w; w.size = ws_size;
        if (ws_size > 0) cudaMalloc(&w.ptr, ws_size);
        ws_cache[key] = w; ws_ptr = w.ptr;
      }
    }

    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      std::fprintf(stderr, "[fp4_normfold_probe_sm120] can_implement FAIL M=%d N=%d K=%d (status=%d)\n",
                   M, N, K, static_cast<int>(status));
      return status;
    }
    status = gemm.initialize(args, ws_ptr, stream);
    if (status != cutlass::Status::kSuccess) {
      std::fprintf(stderr, "[fp4_normfold_probe_sm120] initialize FAIL M=%d N=%d K=%d (status=%d)\n",
                   M, N, K, static_cast<int>(status));
      return status;
    }
    return gemm.run(stream);
  }
};

using V0 = ProbeVariant<Shape<_128, _128, _256>>;  // identity anchor (production tile)
using V1 = ProbeVariant<Shape<_128, _128, _64>>;   // BLK_K=64 (bf16-A fold prereq)
using V2 = ProbeVariant<Shape<_128, _128, _128>>;  // BLK_K=128 (bf16-direct candidate)

// ── bf16-A fold variant (M-FULL-3a-ii): A is bf16, quantized in the consumer. ──
template <class TileShape>
struct ProbeVariantBf16A {
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
          TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute,
          ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto
      >::CollectiveOp;
  // bf16 A is 4x the fp4 footprint (32 KB/stage at 128x128), so force the minimum
  // 2 stages — StageCountAutoCarveout would over-provision and overflow sm120 smem.
  static constexpr int kAmaxBytes =
      static_cast<int>(cute::size<0>(TileShape{})) *
      (static_cast<int>(cute::size<2>(TileShape{})) / 16) * static_cast<int>(sizeof(float));
  using CollectiveMainloop = typename normfold::NormFoldBuilderBf16A<
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCount<2>
  >::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static cutlass::Status run(
      const void* A_bf16, const void* B_packed, void* D_bf16,
      int M, int N, int K, const void* SFB, float alpha, cudaStream_t stream) {
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    auto problem_shape_MNKL = cute::make_shape(M, N, K, 1);
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape_MNKL);
    using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<cutlass::bfloat16_t const*>(A_bf16), stride_A,
            reinterpret_cast<ArrayElementB const*>(B_packed), stride_B,
            reinterpret_cast<ElementSF const*>(SFB), layout_SFB
        },
        { {alpha, 0.0f}, nullptr, stride_C, reinterpret_cast<ElementD*>(D_bf16), stride_D }
    };
    Gemm gemm;
    static void* ws = nullptr; static size_t ws_cap = 0;
    size_t need = Gemm::get_workspace_size(args);
    if (need > ws_cap) { if (ws) cudaFree(ws); cudaMalloc(&ws, need); ws_cap = need; }
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
      std::fprintf(stderr, "[normfold_bf16a] can_implement FAIL M=%d N=%d K=%d (status=%d)\n",
                   M, N, K, static_cast<int>(st));
      return st;
    }
    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) {
      std::fprintf(stderr, "[normfold_bf16a] initialize FAIL (status=%d)\n", static_cast<int>(st));
      return st;
    }
    return gemm.run(stream);
  }
};
using VB = ProbeVariantBf16A<Shape<_128, _128, _128>>;

// ── producer-quant (PQ): smem_A is FP4 (stock consumer reads it), the producer TMAs
// bf16 A into a staging buffer, the consumer quantizes it (natural layout) → fp4 sA +
// SFA. The mainloop keeps the identity 8-field Arguments (ptr_SFA unused → nullptr). ──
template <class TileShape>
struct ProbeVariantPQ {
  // PQ K128 needs ~7KB less epilogue smem (single-buffer fp4 sA already saved 8KB) — a
  // smaller explicit EpilogueTile shrinks the epilogue smem so K128 fits the sm120 cap.
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
          TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute,
          ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto
      >::CollectiveOp;
  using CollectiveMainloop = typename normfold::NormFoldBuilderPQ<
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCount<2>
  >::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static cutlass::Status run(
      const void* A_bf16, const void* B_packed, void* D_bf16,
      int M, int N, int K, const void* SFB, float alpha, cudaStream_t stream) {
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
    auto problem_shape_MNKL = cute::make_shape(M, N, K, 1);
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape_MNKL);
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape_MNKL);
    using ArrayElementB = typename Gemm::GemmKernel::CollectiveMainloop::ArrayElementB;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {   // mainloop (identity 8-field): bf16 A, B, ptr_SFA points at SFB (a valid
            // non-null buffer — the SFA TMA descriptor is built but NEVER issued; SFA
            // is produced by the consumer's quantize_tile_natural), SFB.
            reinterpret_cast<cutlass::bfloat16_t const*>(A_bf16), stride_A,
            reinterpret_cast<ArrayElementB const*>(B_packed), stride_B,
            reinterpret_cast<ElementSF const*>(SFB), layout_SFA,
            reinterpret_cast<ElementSF const*>(SFB), layout_SFB
        },
        { {alpha, 0.0f}, nullptr, stride_C, reinterpret_cast<ElementD*>(D_bf16), stride_D }
    };
    Gemm gemm;
    static void* ws = nullptr; static size_t ws_cap = 0;
    size_t need = Gemm::get_workspace_size(args);
    if (need > ws_cap) { if (ws) cudaFree(ws); cudaMalloc(&ws, need); ws_cap = need; }
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) {
      std::fprintf(stderr, "[normfold_pq] can_implement FAIL M=%d N=%d K=%d (status=%d)\n",
                   M, N, K, static_cast<int>(st));
      return st;
    }
    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess) {
      std::fprintf(stderr, "[normfold_pq] initialize FAIL (status=%d)\n", static_cast<int>(st));
      return st;
    }
    return gemm.run(stream);
  }
};
// BLK_K=128 = BIT-EXACT. The bf16 staging is now SINGLE-buffer (PIPE=1, freed via the
// bf16_empty mbarrier after the consumer quantizes it), halving it 65536→32768 so K128
// fits the sm120 ~100KB smem cap. At K128 the BLK_K=64 SF-atom degeneracy is gone, so
// cos vs the separate quantize(A)+prod-GEMM path is ~1.0 (K128/K256 identity = 1.0).
using VPQ = ProbeVariantPQ<Shape<_128, _128, _128>>;

}  // namespace

int fp4_normfold_probe_sm120_bf16out(
    const void* A_packed, const void* B_packed, void* D_bf16,
    int M, int N, int K,
    const void* SFA, const void* SFB,
    float alpha, cudaStream_t stream)
{
  return static_cast<int>(V0::run(A_packed, B_packed, D_bf16, M, N, K, SFA, SFB, alpha, stream));
}

int fp4_normfold_probe_sm120_bf16out_v(
    int variant,
    const void* A_packed, const void* B_packed, void* D_bf16,
    int M, int N, int K,
    const void* SFA, const void* SFB,
    float alpha, cudaStream_t stream)
{
  switch (variant) {
    case 0: return static_cast<int>(V0::run(A_packed, B_packed, D_bf16, M, N, K, SFA, SFB, alpha, stream));
    case 1: return static_cast<int>(V1::run(A_packed, B_packed, D_bf16, M, N, K, SFA, SFB, alpha, stream));
    case 2: return static_cast<int>(V2::run(A_packed, B_packed, D_bf16, M, N, K, SFA, SFB, alpha, stream));
    default: return -99;
  }
}

int fp4_normfold_bf16a_probe_sm120(
    const void* A_bf16, const void* B_packed, void* D_bf16,
    int M, int N, int K, const void* SFB, float alpha, cudaStream_t stream)
{
  return static_cast<int>(VB::run(A_bf16, B_packed, D_bf16, M, N, K, SFB, alpha, stream));
}

int fp4_normfold_pq_probe_sm120(
    const void* A_bf16, const void* B_packed, void* D_bf16,
    int M, int N, int K, const void* SFB, float alpha, cudaStream_t stream)
{
  return static_cast<int>(VPQ::run(A_bf16, B_packed, D_bf16, M, N, K, SFB, alpha, stream));
}

}  // namespace gemm
}  // namespace flash_rt
