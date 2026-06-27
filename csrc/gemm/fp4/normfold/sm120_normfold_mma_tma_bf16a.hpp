/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/trace.h"
#include "cutlass/numeric_types.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace flash_rt::normfold::detail {
// ── NVFP4 quantize device helpers (verbatim math from csrc/kernels/quantize.cu)
// The in-mainloop bf16→(e2m1,UE4M3) quantize MUST reproduce these bit-for-bit so
// the fold equals the standalone quantize_bf16_to_nvfp4_swizzled numerically.
CUTLASS_DEVICE uint8_t float_to_fp4_e2m1(float v) {
  uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
  float a = fabsf(v);
  uint8_t mag;
  if      (a < 0.25f) mag = 0;
  else if (a < 0.75f) mag = 1;
  else if (a < 1.25f) mag = 2;
  else if (a < 1.75f) mag = 3;
  else if (a < 2.5f)  mag = 4;
  else if (a < 3.5f)  mag = 5;
  else if (a < 5.0f)  mag = 6;
  else                mag = 7;
  return sign | mag;
}
CUTLASS_DEVICE uint8_t float_to_ue4m3_ceil(float v) {
  if (v <= 0.0f) return 0;
  if (v > 240.0f) return 0xFE;
  uint32_t bits = __float_as_uint(v);
  int float_exp = ((bits >> 23) & 0xFF) - 127;
  uint32_t frac = bits & 0x7FFFFF;
  int ue_exp = float_exp + 7;
  if (ue_exp <= 0) {
    float scaled = v * 512.0f;
    int m = (int)ceilf(scaled);
    if (m > 7) return (1 << 3) | 0;
    if (m < 1) m = 1;
    return (uint8_t)m;
  }
  if (ue_exp >= 15) return 0xFE;
  int m = (int)(frac >> 20);
  if (frac & 0xFFFFF) m++;
  if (m >= 8) { m = 0; ue_exp++; }
  if (ue_exp >= 15) return 0xFE;
  return (uint8_t)((ue_exp << 3) | m);
}
CUTLASS_DEVICE float ue4m3_to_float(uint8_t v) {
  int e = (v >> 3) & 0xF;
  int m = v & 0x7;
  if (e == 0) return ldexpf((float)m / 8.0f, -6);
  return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}
}  // namespace flash_rt::normfold::detail

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
// NormFoldBf16A dispatch tag: distinct DispatchPolicy selecting the bf16-A
// CollectiveMma specialization (loads bf16 A, computes NVFP4 SFA + e2m1 in the
// consumer registers — folding RMSNorm into the consuming GEMM's A-load). Distinct
// from MainloopSm120NormFold (identity fork) to avoid an ODR conflict.
template <int Stages_, int SchedulerPipelineStageCount_, class ClusterShape_,
          class KernelScheduleType_>
struct MainloopSm120NormFoldBf16A {
  static constexpr int Stages = Stages_;
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  using ClusterShape = ClusterShape_;
  using KernelScheduleType = KernelScheduleType_;
  using ArchTag = cutlass::arch::Sm120;
  using Schedule = KernelScheduleType_;
  static constexpr int PipelineAsyncMmaStages = 0;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  int SchedulerPipelineStageCount,
  class ClusterShape,
  class KernelScheduleType,
  class TileShape_,
  class ElementPairA_,
  class StridePairA_,
  class ElementPairB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomsA_,
  class SmemCopyAtomsA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomsB_,
  class SmemCopyAtomsB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm120NormFoldBf16A<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>,
    TileShape_,
    ElementPairA_,
    StridePairA_,
    ElementPairB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomsA_,
    SmemCopyAtomsA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomsB_,
    SmemCopyAtomsB_,
    TransformB_> {
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm120NormFoldBf16A<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>;
  using TileShape = TileShape_;
  using ElementPairA = ElementPairA_;
  using ElementPairB = ElementPairB_;
  using StridePairA = StridePairA_;
  using StridePairB = StridePairB_;

  static_assert(cute::is_same_v<remove_cvref_t<decltype(get<1>(ElementPairA{}))>,
                                remove_cvref_t<decltype(get<1>(ElementPairB{}))>>, "SFA and SFB data types should be the same");

  using RuntimeDataTypeA = void*;
  using RuntimeDataTypeB = void*;

   // A and B matrices
  // ElementA = the MMA A operand type (e2m1); ElementALoad = the type actually
  // loaded from gmem/smem (bf16). The consumer quantizes ElementALoad -> ElementA
  // in registers (the norm-fold A-path). The B path is unchanged (fp4).
  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using ElementALoad = cutlass::bfloat16_t;
  using StrideA  = remove_cvref_t<decltype(get<0>(StridePairA{}))>;

  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using StrideB  = remove_cvref_t<decltype(get<0>(StridePairB{}))>;

  // SFA and SFB
  using ElementSF = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
  using LayoutSFA = remove_cvref_t<decltype(get<1>(StridePairA{}))>;
  using LayoutSFB = remove_cvref_t<decltype(get<1>(StridePairB{}))>;

  // Public API hands us a bf16 A pointer (the un-quantized normed activation).
  using ArrayElementA = ElementALoad;
  using ArrayElementB = ElementB;

  using TiledMma = TiledMma_;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using ElementAccumulator = typename TiledMma::ValTypeC;

  static constexpr int SFVecSize = TiledMma::Traits::SFVecSize;
  using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;

  // Gmem copies
  using GmemTiledCopyPairA = GmemTiledCopyPairA_;
  using GmemTiledCopyPairB = GmemTiledCopyPairB_;
  using GmemTiledCopyA    = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairA{}))>;
  using GmemTiledCopySFA  = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairA{}))>;
  using GmemTiledCopyB    = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairB{}))>;
  using GmemTiledCopySFB  = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairB{}))>;

  // Smem copies
  using SmemLayoutAtomsA = SmemLayoutAtomsA_;
  using SmemLayoutAtomsB = SmemLayoutAtomsB_;

  using SmemLayoutAtomA   = remove_cvref_t<decltype(get<0>(SmemLayoutAtomsA{}))>;
  using SmemLayoutAtomSFA = remove_cvref_t<decltype(get<1>(SmemLayoutAtomsA{}))>;
  using SmemLayoutAtomB   = remove_cvref_t<decltype(get<0>(SmemLayoutAtomsB{}))>;
  using SmemLayoutAtomSFB = remove_cvref_t<decltype(get<1>(SmemLayoutAtomsB{}))>;

  using SmemCopyAtomsA =  SmemCopyAtomsA_;
  using SmemCopyAtomsB =  SmemCopyAtomsB_;

  using SmemCopyAtomA   = remove_cvref_t<decltype(get<0>(SmemCopyAtomsA{}))>;
  using SmemCopyAtomSFA = remove_cvref_t<decltype(get<1>(SmemCopyAtomsA{}))>;

  using SmemCopyAtomB   = remove_cvref_t<decltype(get<0>(SmemCopyAtomsB{}))>;
  using SmemCopyAtomSFB = remove_cvref_t<decltype(get<1>(SmemCopyAtomsB{}))>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;

  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr int ThreadCount = size(TiledMma{});

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState  = typename cutlass::PipelineState<DispatchPolicy::Stages>;

  // One threads per CTA are producers (1 for operand tile)
  static constexpr int NumProducerThreadEvents = 1;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(not cute::is_void_v<SmemCopyAtomA>,
    "SM120 mainloop must specify a copy atom for A operand smem->rmem reads.");
  static_assert(not cute::is_void_v<SmemCopyAtomB>,
    "SM120 mainloop must specify a copy atom for B operand smem->rmem reads.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  // SmemLayoutAtomSFA and SmemLayoutAtomSFB are for whole CTA tiles. We add the number of pipeline stages here.
  // The number of pipeline stages is the same as the number of pipeline stages from AB Load <-> MainLoop
  using SmemLayoutSFA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
  ));

  using SmemLayoutSFB = decltype(make_layout(
    append(shape(SmemLayoutAtomSFB{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))
  ));

  static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
  static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operands from rmem for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD>, "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD>, "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  static constexpr bool IsF8F6F4 = detail::is_sm120_f8f6f4<TiledMma, ElementA, ElementB>();

  // A is loaded as bf16 (no TMA rounding concern for a 16-bit native type).
  using TmaInternalElementA = ElementALoad;

  using TmaInternalElementB = cute::conditional_t<not IsF8F6F4,
                                                  ElementB,
                              cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e2m1_t>,
                                                  cutlass::detail::float_e2m1_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e2m3_t>,
                                                cutlass::detail::float_e2m3_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementB, cutlass::float_e3m2_t>,
                                                cutlass::detail::float_e3m2_unpacksmem_t,
                                                uint_bit_t<sizeof_bits_v<ElementB>>>>>>;

  using TmaInternalElementSF = ElementSF;

  // A smem holds bf16 (the un-quantized A); B unchanged (fp4 -> uint8).
  using SmemAllocTypeA = ElementALoad;
  using SmemAllocTypeB = cute::conditional_t<IsF8F6F4, uint8_t, typename TiledMma::ValTypeB>;

  // A's TMA transaction is bf16 A bytes only — SFA is NOT loaded (computed in the
  // consumer registers from the bf16 A), so no SFA term here.
  static constexpr uint32_t TmaTransactionBytesMK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutA{})) * sizeof_bits<ElementALoad>::value));

  static constexpr uint32_t TmaTransactionBytesNK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF>) +
    cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutB{})) * sizeof_bits<ElementB>::value));

  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

  // Per-(row, 16-K-block) amax scratch for the bf16-A quantize: the consumer
  // computes block amaxes from bf16 A cooperatively here (smem broadcast avoids a
  // cross-thread shuffle), then reads them to fill tCrA(e2m1) + tCrSFA(UE4M3).
  // Sized for ONE MMA k-block (atom-K = 64 → 4 SF-blocks), reused across the
  // K_BLOCK_MAX k-blocks: BLK_M rows x 4 blocks (128x4 = 2 KB) — half the full-tile
  // buffer, which is what brings mainloop+epilogue under the sm120 99 KB cap.
  static constexpr int kAmaxRows      = size<0>(TileShape{});
  static constexpr int kAmaxBlocksPKB = 4;  // atom-K(64) / SFVecSize(16)
  using AmaxStorage = cute::array<float, kAmaxRows * kAmaxBlocksPKB>;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      alignas(1024) cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(1024) cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      // No smem_SFA: SFA is computed in the consumer (its smem layout is only used
      // to shape the tCrSFA fragment, not as storage). This reclaims ~K smem.
      alignas(16)   cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
      alignas(16)   AmaxStorage smem_amax;
    } tensors;
    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    alignas(16) PipelineStorage pipeline_storage;
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side kernel arguments. ptr_A is bf16 (un-quantized A); SFA is NOT an
  // input — it is computed in the consumer from the bf16 A.
  struct Arguments {
    ElementALoad const* ptr_A{nullptr};
    StrideA dA{};
    ElementB const* ptr_B{nullptr};
    StrideB dB{};
    ElementSF const* ptr_SFB{nullptr};
    LayoutSFB layout_SFB{};
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}));  // No programmatic multicast
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(recast_ptr<TmaInternalElementB>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        _1{}));  // No programmatic multicast

    // No TMA_SFA: SFA is computed in the consumer, not loaded.
    using TMA_SFB = decltype(make_tma_copy<uint16_t>(
        GmemTiledCopySFB{},
        make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFB{}),
        SmemLayoutSFB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        _1{}));  // No programmatic multicast

    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_SFB tma_load_sfb;
    LayoutSFB layout_SFB;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    auto ptr_A = recast_ptr<TmaInternalElementA>(args.ptr_A);
    auto ptr_B = recast_ptr<TmaInternalElementB>(args.ptr_B);

    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));

    Tensor tensor_sfb = make_tensor(args.ptr_SFB, args.layout_SFB);

    typename Params::TMA_A tma_load_a = make_tma_copy(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}); // No programmatic multicast
    typename Params::TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        _1{}); // No programmatic multicast

    typename Params::TMA_SFB tma_load_sfb = make_tma_copy<uint16_t>(
        GmemTiledCopySFB{},
        tensor_sfb,
        SmemLayoutSFB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        _1{}); // No programmatic multicast

    return {
      tma_load_a,
      tma_load_b,
      tma_load_sfb,
      args.layout_SFB,
      TmaTransactionBytes,
      TmaTransactionBytesMK,
      TmaTransactionBytesNK
    };
  }

  template<class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    constexpr int tma_alignment_bits_A = cutlass::detail::get_input_alignment_bits<ElementA, IsF8F6F4>();
    constexpr int tma_alignment_bits_B = cutlass::detail::get_input_alignment_bits<ElementB, IsF8F6F4>();

    bool implementable = true;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits_A / cutlass::sizeof_bits<ElementA>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits_B / cutlass::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_sfb.get_tma_descriptor());
  }

  // Temporary adhoc partitioning for scaling factors.
  template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE constexpr
  auto
  thrfrg_SFA(SFATensor&& sfatensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
  {
    CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

    using AtomShape_MNK  = typename Atom::Shape_MNK;
    using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;

    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<0>(permutation_mnk),
                            get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfatensor, t_tile);                 // (PermM,PermK)

    // Tile the tensor for the Atom
    auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomM,AtomK),(RestM,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{},_);           // ((ThrV,FrgV),(RestM,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<1>(thr_layout_vmnk)),
                                        make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

    return thr_tensor;
  }

  template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE constexpr
  auto
  thrfrg_SFB(SFBTensor&& sfbtensor, TiledMMA<Atom, TiledThr, TiledPerm>& mma)
  {
    CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

    using AtomShape_MNK  = typename Atom::Shape_MNK;
    using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;

    auto permutation_mnk = TiledPerm{};
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // Reorder the tensor for the TiledAtom
    auto t_tile = make_tile(get<1>(permutation_mnk),
                            get<2>(permutation_mnk));
    auto t_tensor = logical_divide(sfbtensor, t_tile);                 // (PermN,PermK)

    // Tile the tensor for the Atom
    auto a_tile = make_tile(make_layout(size<1>(AtomShape_MNK{})),
                            make_layout(size<2>(AtomShape_MNK{})));
    auto a_tensor = zipped_divide(t_tensor, a_tile);                 // ((AtomN,AtomK),(RestN,RestK))

    // Transform the Atom mode from (M,K) to (Thr,Val)
    auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{},_);           // ((ThrV,FrgV),(RestN,RestK))

    // Tile the tensor for the Thread
    auto thr_tile = make_tile(_,
                              make_tile(make_layout(size<2>(thr_layout_vmnk)),
                                        make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_tensor, thr_tile);            // ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))
    return thr_tensor;
  }

  template <class SFATensor, class ThrMma>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma)
  {
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(), thrfrg_SFA(sfatensor.layout(),thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFA =  thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition_SFA);
  }

  // Per-thread (m,k) COORDINATE tensor for the SFA fragment (norm-fold bf16-A):
  // same slicing as partition_fragment_SFA but the values are the CTA-tile (m,k)
  // coords, so the consumer knows each tCrSFA element's (row, 16-K-block). Pass an
  // identity tensor of shape (BLK_M, BLK_K).
  template <class CoordTensor, class ThrMma>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_coord_SFA(CoordTensor&& coord, ThrMma& thread_mma)
  {
    auto thr_tensor = make_tensor(static_cast<CoordTensor&&>(coord).data(), thrfrg_SFA(coord.layout(), thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
    return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  }

  template <class SFBTensor, class ThrMma>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma)
  {
    using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
    auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(), thrfrg_SFB(sfbtensor.layout(),thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
    auto partition_SFB =  thr_tensor(thr_vnk, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
    return make_fragment_like<ValTypeSF>(partition_SFB);
  }

  template<class TiledMma>
  CUTE_HOST_DEVICE constexpr
  auto
  get_layoutSFA_TV(TiledMma& mma)
  {
    // (M,K) -> (M,K)
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto atile = make_tile(_,
                          make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                make_stride(               Int<1>{} ,                Int<0>{} )),
                                    _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    // (thr_idx,val) -> (M,K)
    return thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
  }

  template<class TiledMma>
  CUTE_HOST_DEVICE constexpr
  auto
  get_layoutSFB_TV(TiledMma& mma)
  {
    // (N,K) -> (N,K)
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

    // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
    auto btile = make_tile(_,
                          make_tile(make_layout(make_shape (size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                                                make_stride(               Int<0>{} ,                Int<1>{} )),
                                    _));

    // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    // (thr_idx,val) -> (M,K)
    return thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the contract
  /// Returned tuple must contain at least two elements, with the first two elements being:
  /// gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  /// gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  /// The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M, N, K, L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = params.tma_load_a.get_tma_tensor(make_shape(M,K,L));                          // (m,k,l)
    Tensor mB_nkl = params.tma_load_b.get_tma_tensor(make_shape(N,K,L));                          // (n,k,l)
    Tensor mSFB_nkl = params.tma_load_sfb.get_tma_tensor(shape(params.layout_SFB));

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k,l)

    Tensor gSFB_nkl = local_tile(mSFB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});    // (TILE_N,TILE_K,n,k,l)

    return cute::make_tuple(gA_mkl, gB_nkl, gSFB_nkl);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TensorB,
    class TensorSFB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorSFB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {

      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE) bf16
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)
      Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});  // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA loads for A (bf16), B and SFB. SFA is NOT loaded.
      //

      auto [gA_mkl, gB_nkl, gSFB_nkl] = load_inputs;

      auto block_tma_a = params.tma_load_a.get_slice(0);
      auto block_tma_b = params.tma_load_b.get_slice(0);
      auto block_tma_sfb = params.tma_load_sfb.get_slice(0);

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

      Tensor gA =   gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
      Tensor gB =   gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)
      Tensor gSFB = gSFB_nkl(_,_,n_coord,_,l_coord);                                                   // (BLK_N,BLK_K,k)

      // Partition source and destination tensors for tma copies
      Tensor tAgA = block_tma_a.partition_S(gA);                                              // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                              // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      Tensor tBgSFB = block_tma_sfb.partition_S(gSFB);                                        // (TMA,TMA_N,TMA_K,k)
      Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);                                        // (TMA,TMA_N,TMA_K,PIPE)

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(params.tma_load_a.with(*tma_barrier), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(params.tma_load_b.with(*tma_barrier), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        copy(params.tma_load_sfb.with(*tma_barrier), tBgSFB(_,_,_,*k_tile_iter), tBsSFB(_,_,_,write_stage));

        // Advance k tile
        ++k_tile_iter;
        ++smem_pipe_write;
      }
    }
    __syncwarp();
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all
       * Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was
       * still inverted from make_producer_start_state
       */
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      [[maybe_unused]] Params const& params) {
    using namespace cute;

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    clear(accum);

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{});         // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});         // (BLK_N,BLK_K,PIPE)
    // sSFA: LAYOUT-ONLY (shapes the tCrSFA fragment via partition_fragment_SFA,
    // which make_fragment_like's fresh rmem — the smem data is never read). Reuse
    // the smem_SFB region's pointer to avoid a dedicated (dead) SFA allocation.
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});  // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    namespace nfd = flash_rt::normfold::detail;

    // Allocate MMA-operand fragments. tCrA/tCrSFA are FILLED by the in-register
    // quantize (not copied from smem). tCrB/tCrSFB use the unchanged fp4 path.
    Tensor tCrA = thread_mma.partition_fragment_A(sA(_,_,Int<0>{}));                         // (MMA,MMA_M,MMA_K) e2m1
    Tensor tCrB = thread_mma.partition_fragment_B(sB(_,_,Int<0>{}));                         // (MMA,MMA_N,MMA_K)
    Tensor tCrSFA = partition_fragment_SFA(sSFA(_,_,Int<0>{}), thread_mma);                  // (MMA,MMA_M,MMA_K) ue4m3
    Tensor tCrSFB = partition_fragment_SFB(sSFB(_,_,Int<0>{}), thread_mma);                  // (MMA,MMA_N,MMA_K)

    // ── bf16 A path: read bf16 A smem in the MMA-A fragment layout + coords ──
    // (raw copy on the swizzle partition; the SM75 ldmatrix won't vectorize the bf16
    // SW128 layout. TODO: a compatible 16-bit ldmatrix / cp path for perf.)
    Tensor tCsA_bf16 = thread_mma.partition_A(
        as_position_independent_swizzle_tensor(sA));                                    // (MMA,MMA_M,MMA_K,PIPE) bf16
    Tensor tArA = make_fragment_like<ElementALoad>(tCrA);                               // (MMA,MMA_M,MMA_K) bf16
    auto idA = make_identity_tensor(make_shape(size<0>(TileShape{}), size<2>(TileShape{})));
    Tensor cA   = thread_mma.partition_A(idA);                                          // (MMA,MMA_M,MMA_K)->(m,k)

    // B (unchanged fp4 path)
    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(as_position_independent_swizzle_tensor(sB));
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);

    auto tile_shape_mnk = tile_shape(tiled_mma);
    // SFA: build the SAME tiled-copy the production uses, but partition an IDENTITY
    // tensor to get the (m,k) coords in the copy's element order, and fill the
    // copy-view of tCrSFA. This guarantees SFA placement matches what the MMA reads
    // (my own partition_coord_SFA used a different divide → mis-placed SFA).
    auto smem_tiled_copy_SFA = make_tiled_copy_impl(SmemCopyAtomSFA{},
                                                    get_layoutSFA_TV(tiled_mma),
                                                    make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFA   = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
    Tensor cSFA              = smem_thr_copy_SFA.partition_S(
        make_identity_tensor(make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));  // (CPY,CPY_M,CPY_K)->(m,k)
    Tensor tCrSFA_cv         = smem_thr_copy_SFA.retile_D(tCrSFA);                       // (CPY,CPY_M,CPY_K)

    // SFB (unchanged fp4 path)
    auto smem_tiled_copy_SFB = make_tiled_copy_impl(SmemCopyAtomSFB{},
                                                    get_layoutSFB_TV(tiled_mma),
                                                    make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
    auto smem_thr_copy_SFB   = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
    Tensor tCsSFB            = smem_thr_copy_SFB.partition_S(as_position_independent_swizzle_tensor(sSFB));
    Tensor tCrSFB_copy_view  = smem_thr_copy_SFB.retile_D(tCrSFB);

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                                 // MMA_N
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                    // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                    // PIPE

    //
    // PIPELINED MAIN LOOP
    //
    auto K_BLOCK_MAX = size<2>(tCrA);

    int read_stage = smem_pipe_read.index();
    auto tCsB_stage   = tCsB(_,_,_,read_stage);
    auto tCsSFB_stage = tCsSFB(_,_,_,read_stage);

    // Per-(row, 4-block) amax scratch in smem (broadcast → no cross-thread shuffle),
    // reused across k-blocks. atom-K = SFVecSize * kBPK.
    float* amax_sm = shared_tensors.smem_amax.data();
    const int BPK = kAmaxBlocksPKB;                        // 4 SF-blocks per k-block
    const int kBaseBlk = BPK;                               // global SF-blocks per k-block
    const int n_amax = kAmaxRows * BPK;
    const int n_thr = size(tiled_mma);
    using MMAOp = typename TiledMma::MMA_Op;

    // Quantize the bf16 A tile of `stage` into tCrA(e2m1) + tCrSFA(ue4m3), one MMA
    // k-block at a time (so the amax buffer holds only 4 SF-blocks). IDENTITY norm
    // (no *rstd *norm_w); reproduces quantize_bf16_to_nvfp4 math.
    auto quantize_A_stage = [&](int stage) {
      copy(tCsA_bf16(_,_,_,stage), tArA);                   // bf16 A smem -> rmem (all k-blocks)
      CUTE_UNROLL
      for (int kb = 0; kb < int(K_BLOCK_MAX); ++kb) {
        int blk0 = kb * kBaseBlk;                           // first global SF-block of this k-block
        // (1) zero + cooperative per-block amax for this k-block
        for (int i = thread_idx; i < n_amax; i += n_thr) amax_sm[i] = 0.0f;
        cutlass::arch::NamedBarrier::sync(n_thr, cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        auto tArA_kb = tArA(_,_,kb);
        auto cA_kb   = cA(_,_,kb);
        CUTE_UNROLL
        for (int e = 0; e < size(tArA_kb); ++e) {
          auto mk = cA_kb(e);
          int slot = get<0>(mk) * BPK + ((get<1>(mk) / SFVecSize) % BPK);
          atomicMax(reinterpret_cast<int*>(&amax_sm[slot]), __float_as_int(fabsf(float(tArA_kb(e)))));
        }
        cutlass::arch::NamedBarrier::sync(n_thr, cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        // (2) tCrSFA = ue4m3(amax/6) for this k-block, via the production copy-view
        // (cSFA = identity partitioned by the SFA tiled-copy → copy-order coords).
        // Iterate the full SFA frag, k-guarded to this k-block (atom-K = SFVecSize*BPK).
        auto tCrSFA_raw = cute::recast<uint8_t>(tCrSFA_cv);
        CUTE_UNROLL
        for (int e = 0; e < size(cSFA); ++e) {
          auto mk = cSFA(e);
          int k = get<1>(mk);
          if (k / (SFVecSize * BPK) == kb) {
            float amax = amax_sm[get<0>(mk) * BPK + ((k / SFVecSize) % BPK)];
            tCrSFA_raw(e) = nfd::float_to_ue4m3_ceil(amax / 6.0f);
          }
        }
        // (3) tCrA = e2m1(a / dequant(sf)) for this k-block. Write the RAW 4-bit
        // nibble via a uint4 recast — assigning a float_e2m1_t would re-encode the
        // value (0x07→6.0→int) and corrupt the e2m1 bit pattern.
        auto tCrA_kb  = tCrA(_,_,kb);
        auto tCrA_raw = cute::recast<cute::uint4_t>(tCrA_kb);
        CUTE_UNROLL
        for (int e = 0; e < size(tArA_kb); ++e) {
          auto mk = cA_kb(e);
          float amax = amax_sm[get<0>(mk) * BPK + ((get<1>(mk) / SFVecSize) % BPK)];
          float dq = nfd::ue4m3_to_float(nfd::float_to_ue4m3_ceil(amax / 6.0f));
          float inv = (dq > 0.0f) ? (1.0f / dq) : 0.0f;
          tCrA_raw(e) = cute::uint4_t(nfd::float_to_fp4_e2m1(float(tArA_kb(e)) * inv));
        }
        fp4_shift_A(MMAOp{}, tCrA_kb);
        cutlass::arch::NamedBarrier::sync(n_thr, cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);  // reuse amax_sm next kb
      }
    };

    auto copy_b_kblock = [&](auto k_block) {
      copy(smem_tiled_copy_B, tCsB_stage(_,_,k_block), tCrB_copy_view(_,_,k_block));
      fp4_shift_B(MMAOp{}, tCrB_copy_view(_,_,k_block));
      copy(tCsSFB_stage(_,_,k_block), tCrSFB_copy_view(_,_,k_block));
    };

    auto gemm_kblock = [&](auto k_block) {
      cute::gemm(tiled_mma, make_zip_tensor(tCrA(_,_,k_block), tCrSFA(_,_,k_block)),
                 make_zip_tensor(tCrB(_,_,k_block), tCrSFB(_,_,k_block)), accum);
    };

    // Per-stage (no A cross-stage software pipeline): quantize the WHOLE k-tile's
    // A into tCrA/tCrSFA, run all its k-blocks, THEN release + advance. (A pipeline
    // would overwrite tCrA/tCrSFA — quantize_A_stage fills all k-blocks at once —
    // before the current tile's last gemm_kblock consumed them: cross-tile
    // corruption visible only when consecutive k-tiles differ.)
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      pipeline.consumer_wait(smem_pipe_read);
      read_stage = smem_pipe_read.index();
      tCsB_stage   = tCsB(_,_,_,read_stage);
      tCsSFB_stage = tCsSFB(_,_,_,read_stage);
      quantize_A_stage(read_stage);
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
        copy_b_kblock(k_block);
        gemm_kblock(k_block);
      });
      cutlass::arch::NamedBarrier::sync(
        thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
    }
}

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline, PipelineState, int) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
