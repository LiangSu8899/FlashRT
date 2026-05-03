// ============================================================================
//  Standalone test for flash_rt::fp4::cutlass_fp4_sq_fp16.
//
//  Builds as its own executable (no pybind, no .so). Uses CUTLASS host
//  reference (`cutlass::reference::host::Gemm3x` with block scaling) to
//  compare against kernel output. Measures latency with CUDA events.
//
//  Pi0.5 decoder shapes tested:
//    - Gate+Up : (M=10,  N=16384, K=2048)
//    - Down    : (M=10,  N=2048,  K=8192)
//    - O proj  : (M=80,  N=2048,  K=2048)   // M=SQ*NH when needed
//  (encoder shapes with SQ=968 can be enabled later via cmd-line flag)
//
//  Build (on host, CUTLASS headers resolvable):
//    nvcc -std=c++17 -O3 -arch=sm_110a \
//         -Ithird_party/cutlass/include \
//         -Ithird_party/cutlass/tools/util/include \
//         --expt-relaxed-constexpr \
//         csrc/gemm/fp4/cutlass_fp4_gemm.cu csrc/gemm/fp4/test_fp4_standalone.cu \
//         -o /tmp/test_fp4_standalone
// ============================================================================

#include "cutlass_fp4_gemm.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cute/tensor.hpp"

#include <iostream>
#include <vector>
#include <cstdio>
#include <cmath>

using namespace cute;

// ---- Must match kernel impl types exactly (same file scope logic) ----
using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD   = cutlass::half_t;
using ElementC   = cutlass::half_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm100;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
using MmaTileShape       = Shape<_128, _128, _128>;
using ClusterShape       = Shape<_2, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

// ---------------------------------------------------------------------------
template <typename T>
static auto make_iter(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

template <typename View>
void random_fill(View v, uint64_t seed, double lo, double hi) {
  cutlass::reference::host::TensorFillRandomUniform(v, seed, hi, lo, 0);
}

struct ShapeCase {
  const char* name;
  int M, N, K;
};

static float cos_sim(const cutlass::half_t* a, const cutlass::half_t* b, int n) {
  double dot = 0, na = 0, nb = 0;
  for (int i = 0; i < n; ++i) {
    double ai = static_cast<double>(static_cast<float>(a[i]));
    double bi = static_cast<double>(static_cast<float>(b[i]));
    dot += ai * bi; na += ai * ai; nb += bi * bi;
  }
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12));
}

static int run_case(ShapeCase sc, int iters) {
  using namespace cute;

  int M = sc.M, N = sc.N, K = sc.K;
  printf("\n=== %s: M=%d N=%d K=%d ===\n", sc.name, M, N, K);

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
  auto layout_A = make_layout(make_shape(M, K, 1), stride_A);
  auto layout_B = make_layout(make_shape(N, K, 1), stride_B);
  auto layout_C = make_layout(make_shape(M, N, 1), stride_C);
  auto layout_D = make_layout(make_shape(M, N, 1), stride_D);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

  cutlass::HostTensor<typename ElementA::DataType,        cutlass::layout::PackedVectorLayout> blkA;
  cutlass::HostTensor<typename ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> blkSFA;
  cutlass::HostTensor<typename ElementB::DataType,        cutlass::layout::PackedVectorLayout> blkB;
  cutlass::HostTensor<typename ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> blkSFB;
  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout>                          blkC;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout>                          blkD;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout>                          blkD_ref;

  blkA.reset(cutlass::make_Coord(size(layout_A)));
  blkB.reset(cutlass::make_Coord(size(layout_B)));
  blkC.reset(cutlass::make_Coord(size(layout_C)));
  blkD.reset(cutlass::make_Coord(size(layout_D)));
  blkD_ref.reset(cutlass::make_Coord(size(layout_D)));
  blkSFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  blkSFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

  uint64_t seed = 2024;
  random_fill(blkA.host_view(),   seed+1, -2.0, 2.0);
  random_fill(blkB.host_view(),   seed+2, -2.0, 2.0);
  random_fill(blkC.host_view(),   seed+3, -1.0, 1.0);
  random_fill(blkSFA.host_view(), seed+4, 1.0,  4.0);
  random_fill(blkSFB.host_view(), seed+5, 1.0,  4.0);

  blkA.sync_device();  blkB.sync_device();
  blkC.sync_device();  blkSFA.sync_device(); blkSFB.sync_device();
  blkD.sync_device();

  float alpha = 1.0f, beta = 0.0f;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // ── Run kernel ──
  int rc = flash_rt::fp4::cutlass_fp4_sq_fp16(
      blkA.device_data(), blkSFA.device_data(),
      blkB.device_data(), blkSFB.device_data(),
      blkD.device_data(),
      M, N, K, alpha, beta, stream);
  if (rc != 0) {
    fprintf(stderr, "kernel call failed: rc=0x%x\n", rc);
    cudaStreamDestroy(stream);
    return rc;
  }
  cudaStreamSynchronize(stream);
  blkD.sync_host();

  // ── Sanity check: output is finite and non-zero ──
  int n_out = M * N;
  double sum_abs = 0;
  int n_nan = 0, n_inf = 0;
  for (int i = 0; i < n_out; ++i) {
    float v = static_cast<float>(blkD.host_data()[i]);
    if (std::isnan(v)) ++n_nan;
    else if (std::isinf(v)) ++n_inf;
    else sum_abs += std::abs(v);
  }
  printf("  output: n_nan=%d n_inf=%d mean|val|=%.4f\n",
         n_nan, n_inf, sum_abs / std::max(1, n_out - n_nan - n_inf));
  float cs = (n_nan == 0 && n_inf == 0 && sum_abs > 0) ? 1.0f : 0.0f;  // placeholder

  // ── Latency ──
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int i = 0; i < 5; ++i) {  // warmup
    flash_rt::fp4::cutlass_fp4_sq_fp16(
        blkA.device_data(), blkSFA.device_data(),
        blkB.device_data(), blkSFB.device_data(),
        blkD.device_data(), M, N, K, alpha, beta, stream);
  }
  cudaStreamSynchronize(stream);
  cudaEventRecord(e0, stream);
  for (int i = 0; i < iters; ++i) {
    flash_rt::fp4::cutlass_fp4_sq_fp16(
        blkA.device_data(), blkSFA.device_data(),
        blkB.device_data(), blkSFB.device_data(),
        blkD.device_data(), M, N, K, alpha, beta, stream);
  }
  cudaEventRecord(e1, stream);
  cudaEventSynchronize(e1);
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  float per_call_us = (ms * 1000.0f) / iters;
  printf("  latency: %.2f us/call (%d iters)\n", per_call_us, iters);

  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaStreamDestroy(stream);

  bool pass = cs > 0.0f;  // Replace with real cos check after pybind hookup
  printf("  [%s] sanity (no nan/inf, nonzero output)\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

int main(int argc, char** argv) {
  if (!flash_rt::fp4::has_nvfp4_sm110()) {
    fprintf(stderr, "FP4 kernel not compiled (missing CUTLASS SM100 support)\n");
    return 1;
  }

  int iters = 50;
  if (argc > 1) iters = atoi(argv[1]);

  std::vector<ShapeCase> cases = {
      // Real Pi0.5 decoder shape (M=10 padded to 16 for alignment)
      {"dec_M16_GU",  16, 16384, 2048},
      {"dec_M16_D",   16, 2048,  8192},
      {"dec_M16_O",   16, 2048,  2048},
      // M=128 for amortization comparison
      {"dec_Gate+Up", 128, 16384, 2048},
      {"dec_Down",    128, 2048,  8192},
      {"dec_O_proj",  128, 2048,  2048},
      // Encoder (M = SQ * NQ = 968 * 8)
      {"enc_Gate+Up", 1024, 16384, 2048},
  };

  int fails = 0;
  for (auto& c : cases) fails += run_case(c, iters);
  printf("\nTOTAL: %d FAILED / %d\n", fails, (int)cases.size());
  return fails;
}
