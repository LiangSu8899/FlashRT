// Phase 3B-α S3 (CUTLASS path) — Phase 1 scaffolding.
//
// CUTLASS 2.x device::Gemm bf16 × bf16 → bf16 GEMM with fp32 accumulator,
// targeting Sm80 (Ampere TensorCore).  Compatible with sm_120 (RTX 5090
// consumer Blackwell) via forward-compat — Ampere cp.async + bf16 MMA
// instructions are fully supported on every Blackwell SM.
//
// Phase 1: default LinearCombination epilogue — validates framework
// runs at our shapes and matches cuBLAS perf.
//
// Phase 2 (next): custom thread-level epilogue functor that does the
// combine `K_out = norm * (K_pre + coef * rnorm * Sr)` inline,
// eliminating the fp32 K_pre intermediate buffer (the BW bottleneck).

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// Phase 2 — CUTLASS 2.x EVT (Epilogue Visitor Tree) for custom combine.
// Pattern lifted from third_party/cutlass/examples/47_ampere_gemm_universal_streamk.
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace flash_rt {
namespace tq_dequant_cutlass {

namespace {

using ElementA  = cutlass::bfloat16_t;
using LayoutA   = cutlass::layout::RowMajor;
using ElementB  = cutlass::bfloat16_t;
using LayoutB   = cutlass::layout::ColumnMajor;  // TN layout (TC requirement)
using ElementC  = cutlass::bfloat16_t;
using LayoutC   = cutlass::layout::RowMajor;
using ElementAccumulator    = float;
using ElementCompute        = float;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using MMAOp   = cutlass::arch::OpClassTensorOp;
using SmArch  = cutlass::arch::Sm80;

// Tile shapes for our (M up to 524K, N=K=256).
// Threadblock: 128 × 128 × 32  → ~64 KB smem with 4 stages, fits sm120
// Warp: 64 × 64 × 32  →  4 warps per CTA
// MMA atom: 16 × 8 × 16 (bf16)
using ShapeThreadblock = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeWarp        = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAOp       = cutlass::gemm::GemmShape<16, 8, 16>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Default epilogue: D = alpha * (A @ B) + beta * C
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,  // 8 (vec width for bf16)
    ElementAccumulator,
    ElementCompute>;

constexpr int NumStages = 3;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    MMAOp, SmArch,
    ShapeThreadblock, ShapeWarp, ShapeMMAOp,
    EpilogueOp,
    SwizzleThreadBlock,
    NumStages>;

// Per-shape workspace cache
struct ShapeKey {
  int M, N, K;
  bool operator==(const ShapeKey& o) const {
    return M == o.M && N == o.N && K == o.K;
  }
};
struct ShapeKeyHash {
  size_t operator()(const ShapeKey& k) const noexcept {
    return (static_cast<size_t>(k.M) * 1315423911u)
         ^ (static_cast<size_t>(k.N) * 2654435761u)
         ^ static_cast<size_t>(k.K);
  }
};
struct CachedWorkspace { void* ptr = nullptr; size_t size = 0; };
std::unordered_map<ShapeKey, CachedWorkspace, ShapeKeyHash> g_ws_cache;
std::mutex g_ws_mu;

void* get_workspace(int M, int N, int K, size_t needed) {
  std::lock_guard<std::mutex> lk(g_ws_mu);
  ShapeKey key{M, N, K};
  auto it = g_ws_cache.find(key);
  if (it != g_ws_cache.end() && it->second.size >= needed) return it->second.ptr;
  if (it != g_ws_cache.end()) { cudaFree(it->second.ptr); g_ws_cache.erase(it); }
  CachedWorkspace w; w.size = needed;
  if (needed > 0) cudaMalloc(&w.ptr, needed);
  g_ws_cache[key] = w;
  return w.ptr;
}

}  // namespace

// A row-major (M, K), B column-major (K, N) — caller passes B as
// (N, K) row-major-stored so it's effectively column-major (K, N).
// Output D row-major (M, N).
extern "C"
void tq_cutlass_bf16_gemm_launch(
    const void* a_bf16, const void* b_bf16, void* d_bf16,
    int M, int N, int K, cudaStream_t stream)
{
  cutlass::gemm::GemmCoord problem(M, N, K);
  // Strides: A row-major (M, K) → ld=K
  //          B col-major  (K, N) → ld=K  (caller supplies (N, K) row-major)
  //          C row-major  (M, N) → ld=N
  // CUTLASS uses TensorRef which takes raw stride.

  cutlass::TensorRef<ElementA const, LayoutA> A_ref(
      reinterpret_cast<ElementA const*>(a_bf16), LayoutA(K));
  cutlass::TensorRef<ElementB const, LayoutB> B_ref(
      reinterpret_cast<ElementB const*>(b_bf16), LayoutB(K));
  cutlass::TensorRef<ElementC, LayoutC> C_ref(
      reinterpret_cast<ElementC*>(d_bf16), LayoutC(N));
  cutlass::TensorRef<ElementC, LayoutC> D_ref = C_ref;

  typename Gemm::Arguments args{
      problem,
      A_ref, B_ref, C_ref, D_ref,
      {1.0f, 0.0f},  // alpha=1, beta=0
  };

  Gemm gemm;
  size_t ws_size = Gemm::get_workspace_size(args);
  void* ws_ptr = get_workspace(M, N, K, ws_size);

  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[tq_cutlass_bf16] can_implement FAIL M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return;
  }
  status = gemm.initialize(args, ws_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[tq_cutlass_bf16] initialize FAIL M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return;
  }
  status = gemm(stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[tq_cutlass_bf16] launch FAIL M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return;
  }
  cudaError_t cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess) {
    std::fprintf(stderr,
        "[tq_cutlass_bf16] CUDA launch err: %s (M=%d N=%d K=%d)\n",
        cudaGetErrorString(cu_err), M, N, K);
  }
}

// ────────────────────────────────────────────────────────────────────
// Phase 2 — EVT V kernel: D = norm_v[r] * (A @ B)
// One per-row scale + multiply.  Validates the EVT pattern in our
// codebase before tackling the more complex K combine (which needs
// a per-element aux input AND two per-row scales).
// ────────────────────────────────────────────────────────────────────

namespace evt_detail {

using namespace cute;

// Output thread map (used by every EVT visitor for tile partitioning).
// Args mirror the standard CollectiveBuilder / device::Gemm template
// parameters exactly.
constexpr int EVTEpilogueStages = 1;
using OutputTileThreadMap =
    cutlass::epilogue::threadblock::OutputTileThreadLayout<
        ShapeThreadblock,        // 128x128x32
        ShapeWarp,               // 64x64x32
        ElementC,                // bf16
        AlignmentC,              // 8
        EVTEpilogueStages>;

// V-side EVT: D[r,c] = norm_v[r] * acc[r,c].
//   AccFetch                              acc
//   ColBroadcast<float>                   norm_v[r]   (per-M)
//   Compute<multiplies>                   norm_v * acc  (fp32 internal)
//   AuxStore<bf16>                        D[r,c] cast → bf16

using V_Acc = cutlass::epilogue::threadblock::VisitorAccFetch;
using V_NormCol = cutlass::epilogue::threadblock::VisitorColBroadcast<
    OutputTileThreadMap, float,
    cute::Stride<_1, _0, int32_t>>;       // (m, n, l) stride: m=1, n=0, l=M
using V_Mul = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;
using V_EvtMul = cutlass::epilogue::threadblock::Sm80EVT<V_Mul, V_NormCol, V_Acc>;
using V_AuxStoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementC,
    cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, _1, int64_t>>;  // (m, n, l): row-major (M,N)
using V_EVT = cutlass::epilogue::threadblock::Sm80EVT<V_AuxStoreD, V_EvtMul>;

using V_GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
    ElementC, LayoutC, AlignmentC,
    ElementAccumulator, ElementCompute, MMAOp, SmArch,
    ShapeThreadblock, ShapeWarp, ShapeMMAOp,
    V_EVT,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    cutlass::arch::OpMultiplyAdd, EVTEpilogueStages>::GemmKernel;

using V_Gemm = cutlass::gemm::device::GemmUniversalAdapter<V_GemmKernel>;

// ── K-side EVT: D[r,c] = norm_k[r] * (acc[r,c] + (coef*rnorm[r]) * Sr[r,c]) ──
// Precondition: caller pre-computes coef_rnorm[r] = coef * rnorm_k[r]
// (one fmul per row at unpack time → trivial cost, simplifies the EVT).
//
// Tree (bottom-up):
//   AuxLoad<Sr>                            Sr[r,c] fp32
//   ColBroadcast<coef_rnorm>               coef_rnorm[r] fp32
//   Compute<multiplies>                    coef_rnorm * Sr
//   AccFetch                               acc[r,c] (fp32 from MMA)
//   Compute<plus>                          acc + coef_rnorm * Sr
//   ColBroadcast<norm_k>                   norm_k[r] fp32
//   Compute<multiplies>                    norm_k * (acc + coef_rnorm * Sr)
//   AuxStore<bf16>                         D[r,c] cast → bf16

using K_AuxLoadSr = cutlass::epilogue::threadblock::VisitorAuxLoad<
    OutputTileThreadMap, float,
    cute::Stride<int64_t, _1, int64_t>>;  // (M,N) row-major fp32
using K_CoefRnorm = cutlass::epilogue::threadblock::VisitorColBroadcast<
    OutputTileThreadMap, float,
    cute::Stride<_1, _0, int32_t>>;
using K_NormCol = cutlass::epilogue::threadblock::VisitorColBroadcast<
    OutputTileThreadMap, float,
    cute::Stride<_1, _0, int32_t>>;
using K_Mul = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;
using K_Plus = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::plus, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using K_EvtScaledSr = cutlass::epilogue::threadblock::Sm80EVT<
    K_Mul, K_CoefRnorm, K_AuxLoadSr>;
using K_EvtAccPlus = cutlass::epilogue::threadblock::Sm80EVT<
    K_Plus, V_Acc, K_EvtScaledSr>;
using K_EvtNormScaled = cutlass::epilogue::threadblock::Sm80EVT<
    K_Mul, K_NormCol, K_EvtAccPlus>;
using K_AuxStoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementC,
    cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, _1, int64_t>>;
using K_EVT = cutlass::epilogue::threadblock::Sm80EVT<K_AuxStoreD, K_EvtNormScaled>;

using K_GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
    ElementC, LayoutC, AlignmentC,
    ElementAccumulator, ElementCompute, MMAOp, SmArch,
    ShapeThreadblock, ShapeWarp, ShapeMMAOp,
    K_EVT,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    cutlass::arch::OpMultiplyAdd, EVTEpilogueStages>::GemmKernel;

using K_Gemm = cutlass::gemm::device::GemmUniversalAdapter<K_GemmKernel>;

}  // namespace evt_detail

using evt_detail::V_EVT;
using evt_detail::V_Gemm;
using evt_detail::K_EVT;
using evt_detail::K_Gemm;

// EVT V launch: D[r,c] = norm_v[r] * (A @ B)[r,c]
//   A: (M, K) bf16 row-major
//   B: (N, K) bf16 row-major (= col-major (K, N))
//   norm_v: (M,) fp32 device pointer
//   D: (M, N) bf16 row-major
extern "C"
void tq_cutlass_v_combine_launch(
    const void* a_bf16, const void* b_bf16, const void* norm_v_fp32,
    void* d_bf16,
    int M, int N, int K, cudaStream_t stream)
{
  using namespace cute;

  // EVT callback args build bottom-up: AuxStoreD( Compute( ColBroadcast, AccFetch ) )
  typename V_EVT::Arguments callback_args{
      {  // V_EvtMul: Compute(ColBroadcast, AccFetch)
          {                                                       // V_NormCol args
              reinterpret_cast<float const*>(norm_v_fp32),
              0.0f,
              {_1{}, _0{}, M},                                    // stride (m=1, n=0, l=M)
          },
          {},                                                     // V_Acc (no args)
          {},                                                     // V_Mul (no args)
      },
      {  // V_AuxStoreD args
          reinterpret_cast<ElementC*>(d_bf16),
          {static_cast<int64_t>(N), _1{}, static_cast<int64_t>(M) * N},
      },
  };

  // No batch (single problem)
  typename V_Gemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      cutlass::gemm::GemmCoord{M, N, K},
      /*split_k_factor=*/1,
      callback_args,
      reinterpret_cast<ElementA const*>(a_bf16),
      reinterpret_cast<ElementB const*>(b_bf16),
      nullptr, nullptr,
      static_cast<int64_t>(M) * K,                // batch_stride_A
      static_cast<int64_t>(N) * K,                // batch_stride_B
      0, 0,
      K,                                          // stride_a (row-major M, ld=K)
      K,                                          // stride_b (col-major (K,N), ld=K)
      0, 0);

  V_Gemm gemm;
  size_t ws_size = V_Gemm::get_workspace_size(args);
  void* ws_ptr = get_workspace(M, N, K, ws_size);

  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[tq_cutlass_v_combine] can_implement FAIL M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return;
  }
  status = gemm.initialize(args, ws_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[tq_cutlass_v_combine] initialize FAIL M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return;
  }
  status = gemm(stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[tq_cutlass_v_combine] launch FAIL M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return;
  }
  cudaError_t cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess) {
    std::fprintf(stderr,
        "[tq_cutlass_v_combine] CUDA err: %s (M=%d N=%d K=%d)\n",
        cudaGetErrorString(cu_err), M, N, K);
  }
}

// EVT K launch: D[r,c] = norm_k[r] * (A @ B + coef_rnorm[r] * Sr[r,c])
//   A: (M, K) bf16 row-major   (yk pre-cast)
//   B: (N, K) bf16 row-major   (Π^T row-major = Π col-major)
//   Sr: (M, N) fp32 row-major  (precomputed by qjl @ S GEMM)
//   norm_k:     (M,) fp32 device pointer
//   coef_rnorm: (M,) fp32 device pointer (precomputed = coef * rnorm[r])
//   D: (M, N) bf16 row-major
extern "C"
void tq_cutlass_k_combine_launch(
    const void* a_bf16, const void* b_bf16,
    const void* sr_fp32,
    const void* norm_k_fp32, const void* coef_rnorm_fp32,
    void* d_bf16,
    int M, int N, int K, cudaStream_t stream)
{
  using namespace cute;

  // EVT bottom-up:
  //   K_EvtScaledSr (Compute<mul>, ColBroadcast<coef_rnorm>, AuxLoad<Sr>)
  //   K_EvtAccPlus  (Compute<plus>, AccFetch, K_EvtScaledSr)
  //   K_EvtNormScaled (Compute<mul>, ColBroadcast<norm_k>, K_EvtAccPlus)
  //   K_AuxStoreD   (D)
  typename K_EVT::Arguments callback_args{
      {  // K_EvtNormScaled (Compute<mul>, ColBroadcast<norm_k>, K_EvtAccPlus)
          {  // ColBroadcast<norm_k>
              reinterpret_cast<float const*>(norm_k_fp32),
              0.0f,
              {_1{}, _0{}, M},
          },
          {  // K_EvtAccPlus (Compute<plus>, AccFetch, K_EvtScaledSr)
              {},  // AccFetch (no args)
              {  // K_EvtScaledSr (Compute<mul>, ColBroadcast<coef_rnorm>, AuxLoad<Sr>)
                  {  // ColBroadcast<coef_rnorm>
                      reinterpret_cast<float const*>(coef_rnorm_fp32),
                      0.0f,
                      {_1{}, _0{}, M},
                  },
                  {  // AuxLoad<Sr>
                      const_cast<float*>(reinterpret_cast<float const*>(sr_fp32)),
                      0.0f,
                      {static_cast<int64_t>(N), _1{}, static_cast<int64_t>(M) * N},
                  },
                  {},  // K_Mul (no args)
              },
              {},  // K_Plus (no args)
          },
          {},  // K_Mul outer (no args)
      },
      {  // K_AuxStoreD args
          reinterpret_cast<ElementC*>(d_bf16),
          {static_cast<int64_t>(N), _1{}, static_cast<int64_t>(M) * N},
      },
  };

  typename K_Gemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      cutlass::gemm::GemmCoord{M, N, K},
      /*split_k_factor=*/1,
      callback_args,
      reinterpret_cast<ElementA const*>(a_bf16),
      reinterpret_cast<ElementB const*>(b_bf16),
      nullptr, nullptr,
      static_cast<int64_t>(M) * K,
      static_cast<int64_t>(N) * K,
      0, 0,
      K, K, 0, 0);

  K_Gemm gemm;
  size_t ws_size = K_Gemm::get_workspace_size(args);
  void* ws_ptr = get_workspace(M, N, K, ws_size);

  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[tq_cutlass_k_combine] can_implement FAIL M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return;
  }
  status = gemm.initialize(args, ws_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[tq_cutlass_k_combine] initialize FAIL M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return;
  }
  status = gemm(stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr,
        "[tq_cutlass_k_combine] launch FAIL M=%d N=%d K=%d (status=%d)\n",
        M, N, K, static_cast<int>(status));
    return;
  }
  cudaError_t cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess) {
    std::fprintf(stderr,
        "[tq_cutlass_k_combine] CUDA err: %s (M=%d N=%d K=%d)\n",
        cudaGetErrorString(cu_err), M, N, K);
  }
}

}  // namespace tq_dequant_cutlass
}  // namespace flash_rt
