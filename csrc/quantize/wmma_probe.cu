// wmma probe: bf16 in × bf16 in → fp32 out matmul.
// Goal: measure if a hand-rolled wmma kernel can approach cuBLAS perf
// at our production sizes (M ∈ {32K, 128K, 524K}, N=K=256).  If yes,
// the α-3-tc fused kernel is feasible.  If not, we need a different
// approach (CUTLASS or full FA2 fuse).
//
// Per CTA layout:
//   M_TILE = 32 rows (kv_seq * nkv direction)
//   N      = 256 cols (output)
//   K      = 256 cols (inner)
//   warps  = 4 (128 threads)
//
// Each warp computes one 16x16 output tile at a time, iterating over
// (M_TILE/16=2) m-tiles × (N/16=16) n-tiles = 32 output tiles total,
// distributed as 8 tiles per warp.  Inner reduction K/16=16 per tile.
// Total wmma_sync per CTA: 32 × 16 = 512.
//
// Smem:
//   sa_bf16: M_TILE × K bf16 = 32 × 256 × 2 = 16 KB  (input matrix A)
//   sb_bf16: K × N    bf16 = 256 × 256 × 2 = 128 KB ← TOO BIG
// → load B fragments directly from gmem (L2-cached after first CTA).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

namespace {

// M_TILE × K × 2 bytes must fit in default static smem (49 KB on sm_120).
// 64 × 256 × 2 = 32 KB — comfortable.
constexpr int M_TILE = 64;
constexpr int N      = 256;
constexpr int K      = 256;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int N_WARPS = 8;                              // 256 threads
constexpr int N_TILES_M = M_TILE / WMMA_M;               // 4
constexpr int N_TILES_N = N / WMMA_N;                    // 16
constexpr int N_TILES_K = K / WMMA_K;                    // 16
constexpr int N_TILES_OUT = N_TILES_M * N_TILES_N;       // 64
constexpr int TILES_PER_WARP = N_TILES_OUT / N_WARPS;    // 8

__global__ __launch_bounds__(256, 1)
void wmma_probe_kernel(
    const __nv_bfloat16* __restrict__ A,  // (M, K)
    const __nv_bfloat16* __restrict__ B,  // (K, N)
    float*               __restrict__ C,  // (M, N)
    int M)
{
    const int cta = blockIdx.x;
    const int row_base = cta * M_TILE;
    if (row_base >= M) return;
    const int warp_id = threadIdx.x / 32;

    __shared__ __nv_bfloat16 sa[M_TILE * K];

    // Cooperative load of A tile (128 × 256 = 32K bf16 = 64 KB).
    // 256 threads, each loads 32K/256 = 128 bf16 elements.  Vectorize
    // as 16 × float4 (8 bf16 per float4) for coalesced 128-byte loads.
    const int tid = threadIdx.x;
    const float4* a_v = reinterpret_cast<const float4*>(A + row_base * K);
    float4*       sa_v = reinterpret_cast<float4*>(sa);
    constexpr int VEC_PER_TID = (M_TILE * K) / 8 / 256;  // 16
    #pragma unroll
    for (int i = 0; i < VEC_PER_TID; ++i) {
        sa_v[tid * VEC_PER_TID + i] = a_v[tid * VEC_PER_TID + i];
    }
    __syncthreads();

    // Each warp handles TILES_PER_WARP output tiles.  No outer-loop
    // unroll: at TILES_PER_WARP=16 with full-unroll, ~16× fragments
    // would be alive simultaneously (~1KB/thread), spilling registers.
    for (int t = 0; t < TILES_PER_WARP; ++t) {
        const int tile_idx = warp_id * TILES_PER_WARP + t;
        const int mt = tile_idx / N_TILES_N;
        const int nt = tile_idx % N_TILES_N;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        // Inner reduction over K
        #pragma unroll
        for (int kt = 0; kt < N_TILES_K; ++kt) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                __nv_bfloat16, wmma::row_major> b_frag;

            // A tile: rows [mt*16:(mt+1)*16], cols [kt*16:(kt+1)*16]
            // smem A is row-major (M_TILE, K)
            const __nv_bfloat16* a_ptr = sa + mt * WMMA_M * K + kt * WMMA_K;
            wmma::load_matrix_sync(a_frag, a_ptr, K);

            // B tile: rows [kt*16:(kt+1)*16], cols [nt*16:(nt+1)*16]
            // gmem B row-major (K, N)
            const __nv_bfloat16* b_ptr = B + kt * WMMA_K * N + nt * WMMA_N;
            wmma::load_matrix_sync(b_frag, b_ptr, N);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store C tile to gmem
        float* c_ptr = C + (row_base + mt * WMMA_M) * N + nt * WMMA_N;
        wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
    }
}

}  // namespace

extern "C"
void wmma_probe_launch(
    const void* a_bf16, const void* b_bf16, void* c_fp32,
    int M, cudaStream_t stream)
{
    dim3 grid((M + M_TILE - 1) / M_TILE);
    dim3 block(N_WARPS * 32);
    wmma_probe_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)a_bf16,
        (const __nv_bfloat16*)b_bf16,
        (float*)c_fp32,
        M);
}
