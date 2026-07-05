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
//
// PRODUCER-QUANT (PQ) NORM-FOLD — the optimal dataflow (2026-06-27 redesign).
// The bf16a fork (1abaab0) quantized in the CONSUMER's MMA-fragment layout → forced
// cross-thread amax + coord evals = 727us of fragment machinery. PQ instead quantizes
// in the NATURAL layout (per-row, per-16-K-block contiguous → thread-local amax, ZERO
// shuffle, like the 8us standalone quantize) and writes fp4 sA + ue4m3 sSFA THROUGH
// the stock SmemLayoutA / SmemLayoutSFA tensors (CuTe does the swizzle → correct by
// construction). The stock fp4 consumer (copy_kblock + gemm_kblock) reads sA UNCHANGED
// = roofline, zero MMA risk.
//
// Development note: this producer-quant fork is compiled only when the explicit
// SM120 developer-probe CMake option is enabled. The default runtime build does
// not expose the probe bindings or link this translation unit.
//
// INTEGRATION SPECIFICS (worked out 2026-06-27 — edit sites + the tricky bits):
//   • bf16 staging types (add after SmemLayoutB ~L316):
//       using ElementAStage = cutlass::bfloat16_t;
//       using SmemLayoutAtomAStage = decltype(UMMA::Layout_K_SW128_Atom<ElementAStage>());
//       using SmemLayoutAStage = decltype(tile_to_shape(SmemLayoutAtomAStage{},
//           make_shape(shape<0>(TileShape{}),shape<2>(TileShape{}),Int<Stages>{}), Step<_1,_2,_3>{}));
//   • TensorStorage (~L383): KEEP smem_A(fp4 scratch)+smem_SFA+smem_B+smem_SFB; ADD
//       smem_A_bf16 (ElementAStage, cosize SmemLayoutAStage). SMEM BUDGET @BLK_K=128,S=2:
//       smem_A_bf16=64KB dominates → smem_A(fp4)+smem_SFA MUST be SINGLE-buffered (drop
//       PIPE dim) to fit <99KB (bf16a hit 100352 w/o fp4 sA). So sA/sSFA = 1 buffer; the
//       stock consumer's sA(_,_,read_stage) becomes sA(_,_) — touch those few index sites.
//   • TMA_A (~L411) → bf16 TMA into staging: make_tma_copy(GmemTiledCopyA{},
//       make_tensor(recast_ptr<ElementAStage>(nullptr),...StrideA), SmemLayoutAStage{}(_,_,_0{}),
//       make_shape(BLK_M,BLK_K), _1{}); Arguments.ptr_A → ElementAStage const*. tx_bytes_mk
//       = bf16 staging bytes only (drop the SFA term + the fp4-A term). Keep TMA_SFA field
//       but DON'T issue it (or delete). REMOVE the SFA TMA issue in load (~L792).
//   • load (~L757): partition gA via the bf16 TMA → tAsA into smem_A_bf16; B+SFB stock.
//   • mma (~L949): after EACH consumer_wait, build sA_bf16=make_tensor(smem_A_bf16,
//       SmemLayoutAStage)(_,_,read_stage); call quantize_tile_natural<SFVecSize>(sA_bf16,
//       sA, sSFA, thread_idx, ThreadCount); __syncthreads(); then stock copy_kblock loop.
//   • Risk to watch: writing sA (fp4 SmemLayoutA, swizzled) + sSFA from the natural (m,k)
//       loop — quantize_tile_natural indexes sA(m,k)/sSFA(m,blk); CuTe applies the swizzle.
//       The UNIT TEST proved the math+nibble/SF writes on a ROW-MAJOR layout; the SWIZZLED
//       smem write is the one thing E2E cos must confirm (should be correct by construction).
//
namespace flash_rt { namespace normfold { namespace detail {

// bf16/fp32 → NVFP4 e2m1 (threshold-rounded magnitude), raw 4-bit nibble.
CUTLASS_DEVICE uint8_t pq_float_to_fp4_e2m1(float v) {
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
// fp32 scale → UE4M3 (ceil), matches the production quantize_bf16_to_nvfp4 path.
CUTLASS_DEVICE uint8_t pq_float_to_ue4m3_ceil(float v) {
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
CUTLASS_DEVICE float pq_ue4m3_to_float(uint8_t v) {
  int e = (v >> 3) & 0xF;
  int m = v & 0x7;
  if (e == 0) return ldexpf((float)m / 8.0f, -6);
  return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

// Natural-layout quantize of one bf16 A tile → fp4 sA + ue4m3 sSFA. NO cross-thread
// communication: each thread owns whole (row, 16-K-block) groups whose K are
// contiguous, so the per-block amax is a thread-local reduce. CuTe layout tensors
// place the swizzle on write. (Identity norm for now; real RMSNorm = M-FULL-3b: fold
// *rstd*norm_w into the value before pq_float_to_fp4_e2m1.)
//   sA_bf16 : (BLK_M, BLK_K) bf16   — staged bf16 A for this pipe stage
//   sA      : (BLK_M, BLK_K) e2m1   — fp4 out (write via recast<uint4_t>)
//   sSFA    : (BLK_M, n_blk)  ue4m3 — per-16-block scale out
template <int SFVecSize, class TBf16, class TFp4, class TSFA>
CUTLASS_DEVICE void quantize_tile_natural(
    TBf16 sA_bf16, TFp4 sA, TSFA sSFA, int thread_idx, int n_thr) {  // CuTe views = by value
  using namespace cute;
  const int BLK_M = size<0>(sA_bf16);
  const int BLK_K = size<1>(sA_bf16);
  const int n_blk = BLK_K / SFVecSize;
  auto sA_raw   = cute::recast<cute::uint4_t>(sA);          // nibble-addressable view
  auto sSFA_raw = cute::recast<uint8_t>(sSFA);              // raw-byte ue4m3 view
  // grid-stride over (row, 16-K-block) — both owned wholly by one thread.
  for (int rb = thread_idx; rb < BLK_M * n_blk; rb += n_thr) {
    const int m   = rb / n_blk;
    const int blk = rb % n_blk;
    const int k0  = blk * SFVecSize;
    float amax = 0.0f;
    CUTE_UNROLL
    for (int j = 0; j < SFVecSize; ++j)
      amax = fmaxf(amax, fabsf(float(sA_bf16(m, k0 + j))));
    const uint8_t ue = pq_float_to_ue4m3_ceil(amax / 6.0f);
    // SmemLayoutSFA mode1 indexes the SFVecSize-broadcast K (16 K share one slot,
    // blocks at stride 1), NOT n_blk — write at the block's K-position k0, else only
    // block 0 gets written and blocks 1..n_blk-1 stay uninitialized → NaN.
    sSFA_raw(m, k0) = ue;                                    // ue4m3 SF write (raw byte)
    const float dq  = pq_ue4m3_to_float(ue);
    const float inv = (dq > 0.0f) ? (1.0f / dq) : 0.0f;
    CUTE_UNROLL
    for (int j = 0; j < SFVecSize; ++j)
      sA_raw(m, k0 + j) = cute::uint4_t(pq_float_to_fp4_e2m1(float(sA_bf16(m, k0 + j)) * inv));
  }
}

}}}  // namespace flash_rt::normfold::detail

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
// NormFold dispatch tag: a distinct DispatchPolicy so this forked CollectiveMma
// specialization (which will load bf16 A and quantize to NVFP4 inline — folding
// the RMSNorm into the consuming GEMM's A-load) does not ODR-conflict with the
// stock MainloopSm120TmaWarpSpecializedBlockScaled collective. Same 4 params, so
// the builder's arg computation is reused verbatim (only the final tag swaps).
template <int Stages_, int SchedulerPipelineStageCount_, class ClusterShape_,
          class KernelScheduleType_>
struct MainloopSm120NormFoldPQ {
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
    MainloopSm120NormFoldPQ<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>,
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
  using DispatchPolicy = MainloopSm120NormFoldPQ<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>;
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
  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using StrideA  = remove_cvref_t<decltype(get<0>(StridePairA{}))>;

  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using StrideB  = remove_cvref_t<decltype(get<0>(StridePairB{}))>;

  // SFA and SFB
  using ElementSF = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
  using LayoutSFA = remove_cvref_t<decltype(get<1>(StridePairA{}))>;
  using LayoutSFB = remove_cvref_t<decltype(get<1>(StridePairB{}))>;

  using ArrayElementA = ElementA;
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
  // PQ: SINGLE-buffer fp4 sA — it is consumer-written scratch (quantize then immediately
  // ldmatrix, synchronous), so 1 buffer is correct and saves 8KB → K128 fits. Keep RANK 3
  // (the partition machinery requires it) but PIPE=1; the consumer indexes stage 0 always.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<1>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  // ── PQ bf16 staging: a separate bf16 A buffer the producer TMAs into; the consumer
  // then quantizes it (quantize_tile_natural) into smem_A (fp4) + smem_SFA. It needs its
  // own 16-bit SW128 layout — the fp4 SmemLayoutA can't hold bf16. ──
  using ElementAStage = cutlass::bfloat16_t;
  using SmemLayoutAtomAStage = decltype(cute::UMMA::Layout_K_SW128_Atom<ElementAStage>());
  // PQ: SINGLE-buffer the bf16 staging (PIPE=1). The producer fills bf16[0], the consumer
  // quantizes it → fp4 sA before the MMA, then frees bf16[0] via the bf16_empty mbarrier so
  // the producer can refill it for the next K-tile (overlapping with this tile's MMA). This
  // halves the staging from 2×32KB → 32KB, so BLK_K=128 fits the sm120 ~100KB smem cap.
  using SmemLayoutAStage = decltype(tile_to_shape(
      SmemLayoutAtomAStage{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<1>{}),
      Step<_1,_2,_3>{}));

  // SmemLayoutAtomSFA and SmemLayoutAtomSFB are for whole CTA tiles. We add the number of pipeline stages here.
  // The number of pipeline stages is the same as the number of pipeline stages from AB Load <-> MainLoop
  // PQ: SINGLE-buffer SFA (PIPE=1) — consumer-written scratch like fp4 sA; saves ~1KB.
  using SmemLayoutSFA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<1>{}),
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

  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  using TmaInternalElementA = cute::conditional_t<not IsF8F6F4,
                                                  ElementA,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e2m1_t>,
                                                  cutlass::detail::float_e2m1_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e2m3_t>,
                                                cutlass::detail::float_e2m3_unpacksmem_t,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e3m2_t>,
                                                cutlass::detail::float_e3m2_unpacksmem_t,
                                                uint_bit_t<sizeof_bits_v<ElementA>>>>>>;

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

  using SmemAllocTypeA = cute::conditional_t<IsF8F6F4, uint8_t, typename TiledMma::ValTypeA>;
  using SmemAllocTypeB = cute::conditional_t<IsF8F6F4, uint8_t, typename TiledMma::ValTypeB>;

  // Set the bytes transferred in this TMA transaction (may involve multiple issues)
  // PQ: A's TMA carries bf16 staging bytes ONLY (no fp4-A TMA, no SFA TMA — both are
  // produced by the consumer's quantize_tile_natural).
  static constexpr uint32_t TmaTransactionBytesMK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutAStage{})) * sizeof_bits<ElementAStage>::value));

  static constexpr uint32_t TmaTransactionBytesNK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF>) +
    cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutB{})) * sizeof_bits<ElementB>::value));

  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      alignas(1024) cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(1024) cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      alignas(1024) cute::ArrayEngine<ElementAStage, cute::cosize_v<SmemLayoutAStage>> smem_A_bf16;
      alignas(16)   cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
      alignas(16)   cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
      // PQ: 1-deep handoff mbarrier for the single-buffer bf16 staging. The consumer
      // ARRIVEs after quantizing bf16[0] (frees it); the producer WAITs before refilling.
      // Lazily init'd once (producer's first K-tile); the consumer sees it via the
      // pipeline tx-byte fence on its first consumer_wait.
      alignas(8) typename cutlass::arch::ClusterBarrier::ValueType bf16_empty;
    } tensors;
    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    alignas(16) PipelineStorage pipeline_storage;
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side kernel arguments
  struct Arguments {
    ElementAStage const* ptr_A{nullptr};   // PQ: bf16 (un-quantized) A
    StrideA dA{};
    ElementB const* ptr_B{nullptr};
    StrideB dB{};
    ElementSF const* ptr_SFA{nullptr};
    LayoutSFA layout_SFA{};
    ElementSF const* ptr_SFB{nullptr};
    LayoutSFB layout_SFB{};
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy(   // PQ: bf16 A → bf16 staging smem
        GmemTiledCopyA{},
        make_tensor(recast_ptr<ElementAStage>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutAStage{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}));  // No programmatic multicast
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(recast_ptr<TmaInternalElementB>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        _1{}));  // No programmatic multicast

    using TMA_SFA = decltype(make_tma_copy<uint16_t>(
        GmemTiledCopySFA{},
        make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFA{}),
        SmemLayoutSFA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}));  // No programmatic multicast


    using TMA_SFB = decltype(make_tma_copy<uint16_t>(
        GmemTiledCopySFB{},
        make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFB{}),
        SmemLayoutSFB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        _1{}));  // No programmatic multicast

    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_SFA tma_load_sfa;
    TMA_SFB tma_load_sfb;
    LayoutSFA layout_SFA;
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

    auto ptr_A = recast_ptr<ElementAStage>(args.ptr_A);   // PQ: bf16 A
    auto ptr_B = recast_ptr<TmaInternalElementB>(args.ptr_B);

    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));

    Tensor tensor_sfa = make_tensor(args.ptr_SFA, args.layout_SFA);
    Tensor tensor_sfb = make_tensor(args.ptr_SFB, args.layout_SFB);

    typename Params::TMA_A tma_load_a = make_tma_copy(   // PQ: bf16 A → bf16 staging
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutAStage{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}); // No programmatic multicast
    typename Params::TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        _1{}); // No programmatic multicast

    typename Params::TMA_SFA tma_load_sfa = make_tma_copy<uint16_t>(
        GmemTiledCopySFA{},
        tensor_sfa,
        SmemLayoutSFA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
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
      tma_load_sfa,
      tma_load_sfb,
      args.layout_SFA,
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
    cute::prefetch_tma_descriptor(params.tma_load_sfa.get_tma_descriptor());
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
    Tensor mSFA_mkl = params.tma_load_sfa.get_tma_tensor(shape(params.layout_SFA));
    Tensor mSFB_nkl = params.tma_load_sfb.get_tma_tensor(shape(params.layout_SFB));

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k,l)

    Tensor gSFA_mkl = local_tile(mSFA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});    // (TILE_M,TILE_K,m,k,l)
    Tensor gSFB_nkl = local_tile(mSFB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});    // (TILE_N,TILE_K,n,k,l)

    return cute::make_tuple(gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TensorB,
    class TensorSFA, class TensorSFB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorSFA, TensorSFB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {

      // PQ: A TMA dest is the bf16 STAGING buffer (consumer quantizes it → fp4 sA).
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A_bf16.begin()), SmemLayoutAStage{});  // (BLK_M,BLK_K,PIPE) bf16
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)
      Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});  // (BLK_M,BLK_K,PIPE)
      Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});  // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA loads for A, B, SFA and SFB
      //

      auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl] = load_inputs;

      auto block_tma_a = params.tma_load_a.get_slice(0);
      auto block_tma_b = params.tma_load_b.get_slice(0);

      auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
      auto block_tma_sfb = params.tma_load_sfb.get_slice(0);

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

      Tensor gA =   gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
      Tensor gB =   gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)
      Tensor gSFA = gSFA_mkl(_,_,m_coord,_,l_coord);                                                   // (BLK_M,BLK_K,k)
      Tensor gSFB = gSFB_nkl(_,_,n_coord,_,l_coord);                                                   // (BLK_N,BLK_K,k)

      // Partition source and destination tensors for tma copies
      Tensor tAgA = block_tma_a.partition_S(gA);                                              // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                              // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      Tensor tAgSFA = block_tma_sfa.partition_S(gSFA);                                        // (TMA,TMA_M,TMA_K,k)
      Tensor tAsSFA = block_tma_sfa.partition_D(sSFA);                                        // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgSFB = block_tma_sfb.partition_S(gSFB);                                        // (TMA,TMA_N,TMA_K,k)
      Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);                                        // (TMA,TMA_N,TMA_K,PIPE)

      // PQ: lazily init the bf16-staging handoff barrier EXACTLY ONCE (the very first
      // K-tile ever — count()==0; the kernel threads smem_pipe_write persistently across
      // work-tiles, so this never re-fires). arrive_count=1 (one elected consumer thread
      // ARRIVEs per K-tile). The consumer sees the init via the pipeline tx-byte fence.
      if (smem_pipe_write.count() == 0) {
        cutlass::arch::ClusterBarrier::init(&shared_tensors.bf16_empty, 1);
      }

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
        // B / SFB ride the 2-stage pipeline (stage-indexed). PQ: NO SFA TMA — SFA is
        // produced by the consumer's quantize_tile_natural.
        copy(params.tma_load_b.with(*tma_barrier), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        copy(params.tma_load_sfb.with(*tma_barrier), tBgSFB(_,_,_,*k_tile_iter), tBsSFB(_,_,_,write_stage));

        // PQ: bf16 A is SINGLE-buffer (bf16[0]). Wait until the consumer freed it (quantized
        // the previous K-tile), THEN TMA the current tile into bf16[0]. Both writes target
        // the same tx-byte barrier for this stage, so consumer_wait still gates on all bytes.
        uint32_t bf16_phase = (static_cast<uint32_t>(smem_pipe_write.count()) & 1u) ^ 1u;
        cutlass::arch::ClusterBarrier::wait(&shared_tensors.bf16_empty, bf16_phase);
        copy(params.tma_load_a.with(*tma_barrier), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,_0{}));

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
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});  // (BLK_N,BLK_K,PIPE)
    // PQ: bf16 staging the producer TMA'd; the consumer quantizes it → fp4 sA + sSFA.
    Tensor sA_bf16 = make_tensor(make_smem_ptr(shared_tensors.smem_A_bf16.begin()), SmemLayoutAStage{});
    // quantize_stage(stage): bf16 staging[stage] → fp4 sA[stage] + ue4m3 sSFA[stage],
    // natural layout (thread-local block amax). Consumer-scoped NamedBarrier (NOT
    // __syncthreads — this is warp-specialized; the producer warp is in load()).
    auto quantize_stage = [&](int stage) {
      (void)stage;  // PQ: bf16 staging is single-buffer (bf16[0]); stage index unused.
      // Write sA/sSFA through the SAME position-independent swizzle the stock consumer
      // READS them with (partition_S(as_position_independent_swizzle_tensor(...))), else
      // the swizzled write and the ldmatrix read disagree → garbage/NaN. sA_bf16 is read
      // raw (it matches the TMA's own raw write).
      flash_rt::normfold::detail::quantize_tile_natural<SFVecSize>(
          sA_bf16(_,_,_0{}),                // PQ: bf16 staging single-buffer (PIPE=1, stage 0)
          sA(_,_,_0{}),                     // PQ: fp4 sA single-buffer (PIPE=1, stage 0)
          sSFA(_,_,_0{}), thread_idx, ThreadCount);  // PQ SFA single-buffer
      cutlass::arch::NamedBarrier::sync(
          thr_size(TiledMma{}), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
      // bf16[0] is now fully consumed (the NamedBarrier synced all consumer threads) → free
      // it so the producer can refill it for the next K-tile. One elected thread ARRIVEs to
      // match the barrier's arrive_count=1.
      if (thread_idx == 0) {
        cutlass::arch::ClusterBarrier::arrive(&shared_tensors.bf16_empty);
      }
    };

    //
    // Define C accumulators and A/B partitioning
    //

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    // Allocate fragments and descriptors
    Tensor tCrA = thread_mma.partition_fragment_A(sA(_,_,Int<0>{}));                         // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thread_mma.partition_fragment_B(sB(_,_,Int<0>{}));                         // (MMA,MMA_N,MMA_K)

    Tensor tCrSFA = partition_fragment_SFA(sSFA(_,_,Int<0>{}), thread_mma);                  // (MMA,MMA_M,MMA_K)
    Tensor tCrSFB = partition_fragment_SFB(sSFB(_,_,Int<0>{}), thread_mma);                  // (MMA,MMA_N,MMA_K)

    //
    // Copy from smem to registers
    //

    // A
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(
      as_position_independent_swizzle_tensor(sA));                                      // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                            //      (CPY,CPY_M,CPY_K)

    // B
    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(
      as_position_independent_swizzle_tensor(sB));                                      // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                            //      (CPY,CPY_M,CPY_K)

    // SFA
    auto tile_shape_mnk = tile_shape(tiled_mma);
    auto smem_tiled_copy_SFA = make_tiled_copy_impl(SmemCopyAtomSFA{},
                                                    get_layoutSFA_TV(tiled_mma),
                                                    make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk))
                                                  );
    auto smem_thr_copy_SFA   = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
    Tensor tCsSFA            = smem_thr_copy_SFA.partition_S(
        as_position_independent_swizzle_tensor(sSFA));                                      // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrSFA_copy_view  = smem_thr_copy_SFA.retile_D(tCrSFA);                          //      (CPY,CPY_M,CPY_K)

    // SFB
    auto smem_tiled_copy_SFB = make_tiled_copy_impl(SmemCopyAtomSFB{},
                                                    get_layoutSFB_TV(tiled_mma),
                                                    make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk))
                                                  );
    auto smem_thr_copy_SFB   = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
    Tensor tCsSFB            = smem_thr_copy_SFB.partition_S(
      as_position_independent_swizzle_tensor(sSFB));                                       // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrSFB_copy_view  = smem_thr_copy_SFB.retile_D(tCrSFB);                         //      (CPY,CPY_N,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));                        // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));                        // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                  // CPY_K
    // PQ: sA is single-buffer (no PIPE) → the tCsA/sA PIPE asserts below don't apply.
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                    // PIPE

    CUTE_STATIC_ASSERT_V(size<1>(tCsSFA) == size<1>(tCrSFA_copy_view));                    // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsSFA) == size<2>(tCrSFA_copy_view));                    // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrSFA) == size<1>(accum));                               // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrSFB) == size<2>(accum));                               // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsSFA) == size<2>(tCsSFB));                              // CPY_K

    //
    // PIPELINED MAIN LOOP
    //

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);

    int read_stage = smem_pipe_read.index();
    auto tCsA_stage   = tCsA(_,_,_,_0{});  // PQ: sA single-buffer (stage 0)
    auto tCsB_stage   = tCsB(_,_,_,read_stage);
    auto tCsSFA_stage = tCsSFA(_,_,_,_0{});  // PQ SFA single-buffer
    auto tCsSFB_stage = tCsSFB(_,_,_,read_stage);

    auto copy_kblock = [&](auto k_block) {
        // copy smem->rmem for A/B operand
      copy(smem_tiled_copy_A, tCsA_stage(_,_,k_block), tCrA_copy_view(_,_,k_block));
      copy(smem_tiled_copy_B, tCsB_stage(_,_,k_block), tCrB_copy_view(_,_,k_block));

      // Left shift A,B for FP4
      using MMAOp = typename TiledMma::MMA_Op;
      fp4_shift_A(MMAOp{}, tCrA_copy_view(_,_,k_block));
      fp4_shift_B(MMAOp{}, tCrB_copy_view(_,_,k_block));


      // Copy smem->rmem for SFA/SFB operand
      copy(tCsSFA_stage(_,_,k_block), tCrSFA_copy_view(_,_,k_block));
      copy(tCsSFB_stage(_,_,k_block), tCrSFB_copy_view(_,_,k_block));
    };

    auto gemm_kblock = [&](auto k_block) {
      // (V,M) x (V,N) => (V,M,N)
      cute::gemm(tiled_mma, make_zip_tensor(tCrA(_,_,k_block), tCrSFA(_,_,k_block)), make_zip_tensor(tCrB(_,_,k_block), tCrSFB(_,_,k_block)), accum);
    };

    pipeline.consumer_wait(smem_pipe_read);

    quantize_stage(read_stage);   // PQ: bf16 staging → fp4 sA + sSFA before the stock read
    copy_kblock(_0{});
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count) {
      //
      // Compute on k_tile
      //
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {

        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

        if (k_block == K_BLOCK_MAX - 1) {
          cutlass::arch::NamedBarrier::sync(
          thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
          // UNLOCK smem_pipe_read, done _computing_ on it
          pipeline.consumer_release(smem_pipe_read);
          ++smem_pipe_read;
          read_stage = smem_pipe_read.index();
          tCsA_stage   = tCsA(_,_,_,_0{});  // PQ: sA single-buffer (stage 0)
          tCsB_stage   = tCsB(_,_,_,read_stage);
          tCsSFA_stage = tCsSFA(_,_,_,_0{});  // PQ SFA single-buffer
          tCsSFB_stage = tCsSFB(_,_,_,read_stage);
          pipeline.consumer_wait(smem_pipe_read);
          quantize_stage(read_stage);   // PQ: quantize the newly-arrived stage
        }

        copy_kblock(k_block_next);
        gemm_kblock(k_block);

      });
    } // k_tile_count

    //
    // Hoist out last k_tile
    //
    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {

      auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);

      if (k_block == K_BLOCK_MAX - 1) {
        cutlass::arch::NamedBarrier::sync(
        thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        // UNLOCK smem_pipe_read, done _computing_ on it
        pipeline.consumer_release(smem_pipe_read);
        ++smem_pipe_read;
      }

      if (k_block_next > 0) {
        copy_kblock(k_block_next);
      }
      gemm_kblock(k_block);

    });
}

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline, PipelineState, int) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
