// ============================================================================
//  FlashVLA — linear scales [rows, D/16] → CUTLASS SFA/SFB tile-interleaved.
//
//  Uses CUTLASS's `Sm1xxBlockScaledConfig<16>::tile_atom_to_shape_SF{A,B}` to
//  derive the target layout at compile time, then a trivial CUDA kernel that
//  maps (row, kblock) → CUTLASS offset and copies the fp8 scale byte.
// ============================================================================
#include "reshape_scales_sfa.cuh"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(__CUDA_ARCH__)
#  include "cutlass/cutlass.h"
#  include "cutlass/detail/sm100_blockscaled_layout.hpp"
#  include "cute/tensor.hpp"
#  define FV_HAVE_CUTLASS 1
#else
#  define FV_HAVE_CUTLASS 0
#endif

namespace flash_rt {
namespace fp4 {

#if FV_HAVE_CUTLASS

using namespace cute;
using Cfg = cutlass::detail::Sm1xxBlockScaledConfig<16>;

// Device-side layout instance computed at call time from problem shape.
// We compute sample offsets with the layout functor and copy bytes.
//
// Important: SFA layout is a function (m, k_block_idx, l=0) -> element offset.
// Our linear source is stored row-major, index (row, k_block) = row*n_kblocks + k_block.
// We need to write src[row, kblock] to dst[layout_sfa(row, kblock, 0)].

template <class LayoutSF>
__global__ void kernel_permute_to_sfa(
    const uint8_t* __restrict__ src_linear,
    uint8_t* __restrict__ dst_sfa,
    LayoutSF layout_sfa,
    int rows, int n_kblocks) {
  int row      = blockIdx.y * blockDim.y + threadIdx.y;
  int k_block  = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || k_block >= n_kblocks) return;
  int src_off = row * n_kblocks + k_block;
  // CUTLASS layouts are indexed by full (M, K, L) — K is the full-K extent,
  // not the block index. The layout knows SFVecSize=16 internally and maps
  // any k in [kblock*16, kblock*16+15] to the same offset.
  int dst_off = layout_sfa(row, k_block * 16, 0);
  dst_sfa[dst_off] = src_linear[src_off];
}

#endif  // FV_HAVE_CUTLASS

int64_t sfa_size_bytes(int rows, int D, bool is_sfb) {
#if FV_HAVE_CUTLASS
  auto shape = cute::make_shape(
      is_sfb ? 1    : rows,
      is_sfb ? rows : 1,
      D, 1);
  if (is_sfb) {
    auto layout = Cfg::tile_atom_to_shape_SFB(shape);
    return static_cast<int64_t>(cute::cosize(layout));
  }
  auto layout = Cfg::tile_atom_to_shape_SFA(shape);
  return static_cast<int64_t>(cute::cosize(layout));
#else
  (void)rows; (void)D; (void)is_sfb;
  return -1;
#endif
}

int reshape_linear_scales_to_sfa(
    const void* src_linear, void* dst_sfa,
    int rows, int D, bool is_sfb, cudaStream_t stream) {
#if FV_HAVE_CUTLASS
  if (D % 16 != 0) return -1;
  const int n_kblocks = D / 16;

  // For SFA: problem (rows=M, N_placeholder, K=D, L=1)
  // For SFB: problem (M_placeholder, rows=N, K=D, L=1)
  // Both use `tile_atom_to_shape_SF{A,B}` producing a Layout over (row, kblk, L).
  // We pass rows and D regardless — CUTLASS ignores the placeholder.
  auto shape = cute::make_shape(
      is_sfb ? 1    : rows,
      is_sfb ? rows : 1,
      D, 1);

  dim3 block(16, 8);
  dim3 grid((n_kblocks + block.x - 1) / block.x,
            (rows      + block.y - 1) / block.y);

  if (is_sfb) {
    auto layout = Cfg::tile_atom_to_shape_SFB(shape);
    kernel_permute_to_sfa<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(src_linear),
        reinterpret_cast<uint8_t*>(dst_sfa),
        layout, rows, n_kblocks);
  } else {
    auto layout = Cfg::tile_atom_to_shape_SFA(shape);
    kernel_permute_to_sfa<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(src_linear),
        reinterpret_cast<uint8_t*>(dst_sfa),
        layout, rows, n_kblocks);
  }
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
#else
  (void)src_linear; (void)dst_sfa; (void)rows; (void)D; (void)is_sfb; (void)stream;
  return -2;
#endif
}

}  // namespace fp4
}  // namespace flash_rt
