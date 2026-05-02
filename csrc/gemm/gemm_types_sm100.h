// ================================================================
// FlashRT — CUTLASS FP8 GEMM Templates for SM100/SM110
// (Jetson AGX Thor, Grace Hopper, etc.)
//
// These templates are compiled with -arch=sm_110a or sm_100a.
// They use FP8 E4M3 inputs → FP16/BF16 output via CUTLASS.
//
// Key difference vs SM120 (gemm_types.h):
//   - SM100 arch target (not SM120)
//   - FP16 output (not BF16) — Thor uses FP16 throughout
//   - T1/T2 use explicit TmaWarpSpecialized2Sm for tactic control
//   - No NVFP4/W4A8 (requires SM120)
// ================================================================
#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// Type aliases
using cutlass_fp8 = cutlass::float_e4m3_t;
using cutlass_fp16 = cutlass::half_t;

// ============================================================
//  PlainGemm: 256×128×64, Cluster 2×2×1
//  Standard FP8→FP16 GEMM (Identity epilogue)
// ============================================================
namespace sm100_plain {
using Tile = Shape<_256, _128, _64>;
using Cluster = Shape<_2, _2, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_fp16, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp16, cutlass::layout::RowMajor, 8,
    cutlass_fp16, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp8, cutlass::layout::RowMajor, 16,
    cutlass_fp8, cutlass::layout::ColumnMajor, 16,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_plain

// ============================================================
//  GeluGemm: 256×128×64 + GELU epilogue
// ============================================================
namespace sm100_gelu {
using Tile = Shape<_256, _128, _64>;
using Cluster = Shape<_2, _2, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::GELU, cutlass_fp16, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp16, cutlass::layout::RowMajor, 8,
    cutlass_fp16, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp8, cutlass::layout::RowMajor, 16,
    cutlass_fp8, cutlass::layout::ColumnMajor, 16,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_gelu

// ============================================================
//  SqGemm: 256×256×128 — deeper K pipeline for large GEMMs
// ============================================================
namespace sm100_sq {
using Tile = Shape<_256, _256, _128>;
using Cluster = Shape<_2, _2, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_fp16, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp16, cutlass::layout::RowMajor, 8,
    cutlass_fp16, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp8, cutlass::layout::RowMajor, 16,
    cutlass_fp8, cutlass::layout::ColumnMajor, 16,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_sq

// ============================================================
//  WideGemm: 256×128×128 — deeper K for FFN down projection
// ============================================================
namespace sm100_wide {
using Tile = Shape<_256, _128, _128>;
using Cluster = Shape<_2, _2, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_fp16, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp16, cutlass::layout::RowMajor, 8,
    cutlass_fp16, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp8, cutlass::layout::RowMajor, 16,
    cutlass_fp8, cutlass::layout::ColumnMajor, 16,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_wide

// ============================================================
//  T1Gemm: 128×256×128, Cluster 2×1×1, TmaWarpSpecialized2Sm
//  EXACT match for Myelin's s128x256 best tactic (2SM)
// ============================================================
namespace sm100_t1 {
using Tile = Shape<_128, _256, _128>;
using Cluster = Shape<_2, _1, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_fp16, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp16, cutlass::layout::RowMajor, 8,
    cutlass_fp16, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized2Sm, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp8, cutlass::layout::RowMajor, 16,
    cutlass_fp8, cutlass::layout::ColumnMajor, 16,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_t1

// ============================================================
//  T2Gemm: 256×256×128, Cluster 2×1×1, TmaWarpSpecialized2Sm
// ============================================================
namespace sm100_t2 {
using Tile = Shape<_256, _256, _128>;
using Cluster = Shape<_2, _1, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_fp16, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp16, cutlass::layout::RowMajor, 8,
    cutlass_fp16, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized2Sm, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp8, cutlass::layout::RowMajor, 16,
    cutlass_fp8, cutlass::layout::ColumnMajor, 16,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_t2

// ============================================================
//  FP32 Output Variants — for models with activations > FP16 range
//  (e.g., Pi0-FAST Gemma 2B deep layers where residual > 65504)
//  Same tile configs, only output dtype changed: cutlass_fp16 → float
// ============================================================

using cutlass_fp32 = float;
using cutlass_bf16 = cutlass::bfloat16_t;

// ============================================================
//  BF16 Output Variants — for models trained in BF16 with large activations
//  Same FP8 inputs/accumulation, BF16 output (range ±3.4e38)
// ============================================================

namespace sm100_sq_bf16out {
using Tile = Shape<_256, _256, _128>;
using Cluster = Shape<_2, _2, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_bf16, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_bf16, cutlass::layout::RowMajor, 8,
    cutlass_bf16, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp8, cutlass::layout::RowMajor, 16,
    cutlass_fp8, cutlass::layout::ColumnMajor, 16,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_sq_bf16out

namespace sm100_wide_bf16out {
using Tile = Shape<_256, _128, _128>;
using Cluster = Shape<_2, _2, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_bf16, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_bf16, cutlass::layout::RowMajor, 8,
    cutlass_bf16, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp8, cutlass::layout::RowMajor, 16,
    cutlass_fp8, cutlass::layout::ColumnMajor, 16,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_wide_bf16out

namespace sm100_t1_bf16out {
using Tile = Shape<_128, _256, _128>;
using Cluster = Shape<_2, _1, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_bf16, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_bf16, cutlass::layout::RowMajor, 8,
    cutlass_bf16, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized2Sm, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp8, cutlass::layout::RowMajor, 16,
    cutlass_fp8, cutlass::layout::ColumnMajor, 16,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_t1_bf16out

namespace sm100_sq_f32out {
using Tile = Shape<_256, _256, _128>;
using Cluster = Shape<_2, _2, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_fp32, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp32, cutlass::layout::RowMajor, 4,
    cutlass_fp32, cutlass::layout::RowMajor, 4,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp8, cutlass::layout::RowMajor, 16,
    cutlass_fp8, cutlass::layout::ColumnMajor, 16,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_sq_f32out

namespace sm100_wide_f32out {
using Tile = Shape<_256, _128, _128>;
using Cluster = Shape<_2, _2, _1>;
using Fusion = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity, cutlass_fp32, float>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, cutlass_fp32, cutlass::layout::RowMajor, 4,
    cutlass_fp32, cutlass::layout::RowMajor, 4,
    cutlass::epilogue::collective::EpilogueScheduleAuto, Fusion>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    cutlass_fp8, cutlass::layout::RowMajor, 16,
    cutlass_fp8, cutlass::layout::ColumnMajor, 16,
    float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main, Epi>>;
}  // namespace sm100_wide_f32out
