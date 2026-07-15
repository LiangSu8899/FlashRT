// FlashRT — MiniMax-Remover WanVAE NVFP4 conv3d fprop, sm_120a.
// See nvfp4_conv3d_ndhwc_fp16out.cuh for interface docs.
//
// Adapted from motus_fp4_conv3d_v19sfb_sm120.cu. The compute path (MMA,
// im2col, cp.async pipeline, SF loading) is byte-identical to the motus
// kernel. Only the epilogue changes: fp16 NDHWC output + fp16 bias.
// This eliminates two conversion passes per layer (bf16→fp16 + NCDHW→NDHWC).

#include "nvfp4_conv3d_ndhwc_fp16out.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace minimax_remover {

namespace {

constexpr int BLOCK_M     = 128;
constexpr int BLOCK_N     = 128;
constexpr int BLOCK_K     = 64;    // FP4 MMA k=64
constexpr int N_ATOMS    = BLOCK_N / 8;    // 16
constexpr int N_GROUPS   = N_ATOMS / 4;    // 4
constexpr int NUM_WARPS  = 8;
constexpr int THREADS    = NUM_WARPS * 32; // 256
constexpr int STAGES     = 2;
constexpr int SMEM_K_STRIDE = 48;  // 64/2=32 bytes data + 16 pad (bank conflicts)
constexpr int SF_K_PER_ROW  = 4;   // 4 UE4M3 bytes per K-tile (1 per 16 elem)

// ── NVFP4 MMA: m16n8k64 e2m1 × e2m1, block-scaled ──
// Does 4 N-atoms MMAs (scale_vec::4X) with shared A + SFA, varying B + SFB.
__device__ __forceinline__
void mma_m16n8k64_e2m1_4x(
    float &d0, float &d1, float &d2, float &d3,
    float &d4, float &d5, float &d6, float &d7,
    float &d8, float &d9, float &d10, float &d11,
    float &d12, float &d13, float &d14, float &d15,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    uint32_t b4, uint32_t b5, uint32_t b6, uint32_t b7,
    uint32_t sfa, uint32_t sfb)
{
    constexpr uint16_t bidA = 0, tidA = 0;
    constexpr uint16_t tidB0 = 0, tidB1 = 1, tidB2 = 2, tidB3 = 3;
    // 4 calls, each m16n8k64, sharing A+SFA, varying B+SFB fragments.
    // D layout: {d0,d1,d2,d3} = atom0, {d4,d5,d6,d7} = atom1, etc.
    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
      ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
      "{%14},{%15,%16},{%17},{%18,%19};\n"
      : "+f"(d0), "+f"(d1), "+f"(d8), "+f"(d9)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "f"(d0), "f"(d1), "f"(d8), "f"(d9),
        "r"(sfa), "h"(bidA), "h"(tidA),
        "r"(sfb), "h"(bidA), "h"(tidB0));
    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
      ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
      "{%14},{%15,%16},{%17},{%18,%19};\n"
      : "+f"(d2), "+f"(d3), "+f"(d10), "+f"(d11)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3),
        "f"(d2), "f"(d3), "f"(d10), "f"(d11),
        "r"(sfa), "h"(bidA), "h"(tidA),
        "r"(sfb), "h"(bidA), "h"(tidB1));
    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
      ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
      "{%14},{%15,%16},{%17},{%18,%19};\n"
      : "+f"(d4), "+f"(d5), "+f"(d12), "+f"(d13)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b4), "r"(b5),
        "f"(d4), "f"(d5), "f"(d12), "f"(d13),
        "r"(sfa), "h"(bidA), "h"(tidA),
        "r"(sfb), "h"(bidA), "h"(tidB2));
    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
      ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
      "{%14},{%15,%16},{%17},{%18,%19};\n"
      : "+f"(d6), "+f"(d7), "+f"(d14), "+f"(d15)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b6), "r"(b7),
        "f"(d6), "f"(d7), "f"(d14), "f"(d15),
        "r"(sfa), "h"(bidA), "h"(tidA),
        "r"(sfb), "h"(bidA), "h"(tidB3));
}

// ── im2col address helpers (FP4 packed, same as motus) ──

__device__ __forceinline__
const uint8_t* x_fp4_ptr(const uint8_t* cache_x, const uint8_t* new_x,
                         int m_global, int k_global,
                         int N, int T_cache, int T_new, int H, int W, int Ci) {
    int K_total = 27 * Ci;
    int M_total = N * T_new * H * W;
    if (k_global >= K_total || m_global >= M_total) return nullptr;
    int spatial = T_new * H * W;
    int b_idx = m_global / spatial;
    int rem   = m_global - b_idx * spatial;
    int t_out = rem / (H * W);
    rem      -= t_out * (H * W);
    int h_out = rem / W;
    int w_out = rem - h_out * W;
    int q   = k_global / Ci;
    int ci0 = k_global % Ci;
    int ks  = q % 3; q /= 3;
    int kr  = q % 3;
    int kt  = q / 3;
    int d_in = t_out + kt;
    int h_in = h_out + kr - 1;
    int w_in = w_out + ks - 1;
    if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) return nullptr;
    if (d_in < T_cache) {
        int idx_elem = (((b_idx * T_cache + d_in) * H + h_in) * W + w_in) * Ci + ci0;
        return cache_x + (idx_elem >> 1);
    } else {
        int d_new = d_in - T_cache;
        int idx_elem = (((b_idx * T_new + d_new) * H + h_in) * W + w_in) * Ci + ci0;
        return new_x + (idx_elem >> 1);
    }
}

__device__ __forceinline__
const uint8_t* w_fp4_ptr(const uint8_t* w, int co, int k_global, int Co, int Ci) {
    int K_total = 27 * Ci;
    if (co >= Co || k_global >= K_total) return nullptr;
    int q   = k_global / Ci;
    int ci0 = k_global % Ci;
    int ks  = q % 3; q /= 3;
    int kr  = q % 3;
    int kt  = q / 3;
    int idx_elem = (((co * 3 + kt) * 3 + kr) * 3 + ks) * Ci + ci0;
    return w + (idx_elem >> 1);
}

__device__ __forceinline__
const uint8_t* sfa_ptr(const uint8_t* cache_sfa, const uint8_t* new_sfa,
                       int m_global, int k_base,
                       int N, int T_cache, int T_new, int H, int W, int Ci) {
    int Ci_blk = Ci >> 4;
    int M_total = N * T_new * H * W;
    if (m_global >= M_total) return nullptr;
    int spatial = T_new * H * W;
    int b_idx = m_global / spatial;
    int rem   = m_global - b_idx * spatial;
    int t_out = rem / (H * W);
    rem      -= t_out * (H * W);
    int h_out = rem / W;
    int w_out = rem - h_out * W;
    int kt_iter = k_base / Ci;
    int ci_block_off = (k_base % Ci) >> 4;
    int ks = kt_iter % 3;
    int kr = (kt_iter / 3) % 3;
    int kt = kt_iter / 9;
    int d_in = t_out + kt;
    int h_in = h_out + kr - 1;
    int w_in = w_out + ks - 1;
    if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) return nullptr;
    if (d_in < T_cache) {
        int idx = (((b_idx * T_cache + d_in) * H + h_in) * W + w_in) * Ci_blk + ci_block_off;
        return cache_sfa + idx;
    } else {
        int d_new = d_in - T_cache;
        int idx = (((b_idx * T_new + d_new) * H + h_in) * W + w_in) * Ci_blk + ci_block_off;
        return new_sfa + idx;
    }
}

__device__ __forceinline__
const uint8_t* sfb_ptr(const uint8_t* w_sfb, int co, int k_base, int Co, int Ci) {
    int Ci_blk = Ci >> 4;
    if (co >= Co) return nullptr;
    int kt_iter = k_base / Ci;
    int ci_block_off = (k_base % Ci) >> 4;
    int ks = kt_iter % 3;
    int kr = (kt_iter / 3) % 3;
    int kt = kt_iter / 9;
    int idx = (((co * 3 + kt) * 3 + kr) * 3 + ks) * Ci_blk + ci_block_off;
    return w_sfb + idx;
}

__device__ __forceinline__
void cp_async_16(uint32_t smem_int, const uint8_t* src) {
    int src_bytes = (src == nullptr) ? 0 : 16;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_int), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__
void cp_async_4(uint32_t smem_int, const uint8_t* src) {
    int src_bytes = (src == nullptr) ? 0 : 4;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :: "r"(smem_int), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__
uint32_t to_smem_int(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ── Main kernel ──
__global__ void __launch_bounds__(THREADS, 2)
nvfp4_conv3d_kernel(
    const uint8_t* __restrict__ cache_x,
    const uint8_t* __restrict__ new_x,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ cache_sfa,
    const uint8_t* __restrict__ new_sfa,
    const uint8_t* __restrict__ w_sfb,
    __half* __restrict__ y,                // fp16 NDHWC output
    const __half* __restrict__ bias,       // fp16 per-channel
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha,
    int M_tiles, int N_tiles)
{
    __shared__ __align__(16) uint8_t A_smem [STAGES][BLOCK_M * SMEM_K_STRIDE];
    __shared__ __align__(16) uint8_t B_smem [STAGES][BLOCK_N * SMEM_K_STRIDE];
    __shared__ __align__(16) uint8_t A_sf_smem[STAGES][BLOCK_M * SF_K_PER_ROW];
    __shared__ __align__(16) uint8_t B_sf_smem[STAGES][BLOCK_N * SF_K_PER_ROW];

    const int t       = threadIdx.x;
    const int warp_id = t / 32;
    const int lane    = t % 32;
    const int l       = lane % 4;
    const int h       = lane / 4;

    const int M_total = N * T_new * H * W;
    const int K_total = 27 * Ci;

    const int ld_row_a   = t / 2;
    const int ld_k_off_a = (t & 1) * 32;   // 32 FP4 elements = 16 bytes
    const int ld_row_b   = t / 2;
    const int ld_k_off_b = (t & 1) * 32;

    int tile_idx = blockIdx.x;
    int m_idx  = tile_idx / N_tiles;
    int n_idx  = tile_idx % N_tiles;
    int m_base = m_idx * BLOCK_M;
    int co_base = n_idx * BLOCK_N;
    if (m_base >= M_total || co_base >= Co) return;

    float dA[N_ATOMS] = {0};
    float dB[N_ATOMS] = {0};
    float dC[N_ATOMS] = {0};
    float dD[N_ATOMS] = {0};

    auto issue_load = [&](int stage, int k_base) {
        // FP4 A
        {
            const uint8_t* src = x_fp4_ptr(cache_x, new_x,
                                           m_base + ld_row_a,
                                           k_base + ld_k_off_a,
                                           N, T_cache, T_new, H, W, Ci);
            uint32_t smem_int = to_smem_int(
                &A_smem[stage][ld_row_a * SMEM_K_STRIDE + (ld_k_off_a >> 1)]);
            cp_async_16(smem_int, src);
        }
        // FP4 B
        {
            const uint8_t* src = w_fp4_ptr(w, co_base + ld_row_b,
                                           k_base + ld_k_off_b, Co, Ci);
            uint32_t smem_int = to_smem_int(
                &B_smem[stage][ld_row_b * SMEM_K_STRIDE + (ld_k_off_b >> 1)]);
            cp_async_16(smem_int, src);
        }
        // SF: first 128 threads load A_sf, next 128 load B_sf
        if (t < BLOCK_M) {
            const uint8_t* src = sfa_ptr(cache_sfa, new_sfa,
                                         m_base + t, k_base,
                                         N, T_cache, T_new, H, W, Ci);
            uint32_t smem_int = to_smem_int(&A_sf_smem[stage][t * SF_K_PER_ROW]);
            cp_async_4(smem_int, src);
        } else {
            int n_idx_t = t - BLOCK_M;
            const uint8_t* src = sfb_ptr(w_sfb, co_base + n_idx_t, k_base, Co, Ci);
            uint32_t smem_int = to_smem_int(&B_sf_smem[stage][n_idx_t * SF_K_PER_ROW]);
            cp_async_4(smem_int, src);
        }
    };

    // Prologue
    issue_load(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);

    int compute_stage = 0;

    for (int k_base = 0; k_base < K_total; k_base += BLOCK_K) {
        int next_stage = compute_stage ^ 1;
        int k_next = k_base + BLOCK_K;
        if (k_next < K_total) issue_load(next_stage, k_next);
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();

        const int warp_M_off = warp_id * 16;
        const int kA0 = 4 * l;
        const int kA2 = 4 * l + 16;

        // Load A fragments (4 uint32 per lane)
        int rA0 = warp_M_off + h;
        int rA1 = warp_M_off + h + 8;
        uint32_t A0 = *reinterpret_cast<const uint32_t*>(
            &A_smem[compute_stage][rA0 * SMEM_K_STRIDE + kA0]);
        uint32_t A1 = *reinterpret_cast<const uint32_t*>(
            &A_smem[compute_stage][rA1 * SMEM_K_STRIDE + kA0]);
        uint32_t A2 = *reinterpret_cast<const uint32_t*>(
            &A_smem[compute_stage][rA0 * SMEM_K_STRIDE + kA2]);
        uint32_t A3 = *reinterpret_cast<const uint32_t*>(
            &A_smem[compute_stage][rA1 * SMEM_K_STRIDE + kA2]);

        // SFA: 4 UE4M3 bytes packed in 1 uint32 for this lane's M-row
        int sfa_m_row;
        if ((lane & 3) == 1) {
            sfa_m_row = warp_M_off + (lane >> 2) + 8;
        } else {
            sfa_m_row = warp_M_off + (lane >> 2);
        }
        uint32_t SFA = *reinterpret_cast<const uint32_t*>(
            &A_sf_smem[compute_stage][sfa_m_row * SF_K_PER_ROW]);

        // 4 N-groups, each does 4 atoms (scale_vec::4X)
        #pragma unroll
        for (int g = 0; g < N_GROUPS; ++g) {
            int base = g * 4;
            int co0 = (base + 0) * 8 + h;
            int co1 = (base + 1) * 8 + h;
            int co2 = (base + 2) * 8 + h;
            int co3 = (base + 3) * 8 + h;
            uint32_t B0 = *reinterpret_cast<const uint32_t*>(
                &B_smem[compute_stage][co0 * SMEM_K_STRIDE + kA0]);
            uint32_t B1 = *reinterpret_cast<const uint32_t*>(
                &B_smem[compute_stage][co0 * SMEM_K_STRIDE + kA2]);
            uint32_t B2 = *reinterpret_cast<const uint32_t*>(
                &B_smem[compute_stage][co1 * SMEM_K_STRIDE + kA0]);
            uint32_t B3 = *reinterpret_cast<const uint32_t*>(
                &B_smem[compute_stage][co1 * SMEM_K_STRIDE + kA2]);
            uint32_t B4 = *reinterpret_cast<const uint32_t*>(
                &B_smem[compute_stage][co2 * SMEM_K_STRIDE + kA0]);
            uint32_t B5 = *reinterpret_cast<const uint32_t*>(
                &B_smem[compute_stage][co2 * SMEM_K_STRIDE + kA2]);
            uint32_t B6 = *reinterpret_cast<const uint32_t*>(
                &B_smem[compute_stage][co3 * SMEM_K_STRIDE + kA0]);
            uint32_t B7 = *reinterpret_cast<const uint32_t*>(
                &B_smem[compute_stage][co3 * SMEM_K_STRIDE + kA2]);

            int sfb_n = g * 32 + l * 8 + h;
            uint32_t SFB = *reinterpret_cast<const uint32_t*>(
                &B_sf_smem[compute_stage][sfb_n * SF_K_PER_ROW]);

            mma_m16n8k64_e2m1_4x(
                dA[base+0], dB[base+0], dA[base+1], dB[base+1],
                dA[base+2], dB[base+2], dA[base+3], dB[base+3],
                dC[base+0], dD[base+0], dC[base+1], dD[base+1],
                dC[base+2], dD[base+2], dC[base+3], dD[base+3],
                A0, A1, A2, A3,
                B0, B1, B2, B3, B4, B5, B6, B7,
                SFA, SFB);
        }
        compute_stage = next_stage;
    }
    asm volatile("cp.async.wait_all;\n" ::);

    // ── Epilogue: fp16 NDHWC output ──
    // y[b, t_out, h_out, w_out, co] at offset row * Co + co
    // (row = m_global = b*T_new*H*W + t_out*H*W + h_out*W + w_out)
    const int warp_M_off = warp_id * 16;
    #pragma unroll
    for (int n_atom = 0; n_atom < N_ATOMS; ++n_atom) {
        int co_pair = co_base + n_atom * 8 + 2 * l;
        int row0    = m_base + warp_M_off + h;
        int row1    = m_base + warp_M_off + h + 8;

        float b0 = 0.f, b1 = 0.f;
        if (bias != nullptr && co_pair < Co) {
            b0 = __half2float(bias[co_pair]);
            if (co_pair + 1 < Co) b1 = __half2float(bias[co_pair + 1]);
        }

        if (co_pair + 1 < Co) {
            __half2 packAB, packCD;
            packAB.x = __float2half_rn(dA[n_atom] * alpha + b0);
            packAB.y = __float2half_rn(dB[n_atom] * alpha + b1);
            packCD.x = __float2half_rn(dC[n_atom] * alpha + b0);
            packCD.y = __float2half_rn(dD[n_atom] * alpha + b1);
            if (row0 < M_total) {
                *reinterpret_cast<__half2*>(&y[row0 * Co + co_pair]) = packAB;
            }
            if (row1 < M_total) {
                *reinterpret_cast<__half2*>(&y[row1 * Co + co_pair]) = packCD;
            }
        } else {
            auto store = [&](int row, int co, float v, float bv) {
                if (row < M_total && co < Co) {
                    y[row * Co + co] = __float2half_rn(v * alpha + bv);
                }
            };
            store(row0, co_pair + 0, dA[n_atom], b0);
            store(row0, co_pair + 1, dB[n_atom], b1);
            store(row1, co_pair + 0, dC[n_atom], b0);
            store(row1, co_pair + 1, dD[n_atom], b1);
        }
    }
}

}  // namespace

extern "C" int nvfp4_conv3d_ndhwc_fp16out(
    const void*  cache_x_fp4,
    const void*  new_x_fp4,
    const void*  w_fp4,
    const void*  cache_sfa,
    const void*  new_sfa,
    const void*  w_sfb,
    void*        y_fp16,
    const void*  bias_fp16,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream)
{
    if (Ci % BLOCK_K != 0) {
        std::fprintf(stderr,
            "[nvfp4_conv3d] Ci%%%d (got %d) — must be multiple of 64\n",
            BLOCK_K, Ci);
        return -1;
    }
    if (Co % 8 != 0) {
        std::fprintf(stderr, "[nvfp4_conv3d] Co%%8 (got %d)\n", Co);
        return -2;
    }
    if (T_cache != 2) {
        std::fprintf(stderr, "[nvfp4_conv3d] T_cache must be 2 (got %d)\n", T_cache);
        return -3;
    }
    int M = N * T_new * H * W;
    int M_tiles = (M + BLOCK_M - 1) / BLOCK_M;
    int N_tiles = (Co + BLOCK_N - 1) / BLOCK_N;
    int total_tiles = M_tiles * N_tiles;

    dim3 grid(total_tiles);
    dim3 block(THREADS);
    nvfp4_conv3d_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(cache_x_fp4),
        reinterpret_cast<const uint8_t*>(new_x_fp4),
        reinterpret_cast<const uint8_t*>(w_fp4),
        reinterpret_cast<const uint8_t*>(cache_sfa),
        reinterpret_cast<const uint8_t*>(new_sfa),
        reinterpret_cast<const uint8_t*>(w_sfb),
        reinterpret_cast<__half*>(y_fp16),
        reinterpret_cast<const __half*>(bias_fp16),
        N, T_cache, T_new, H, W, Ci, Co, alpha,
        M_tiles, N_tiles);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[nvfp4_conv3d] launch err: %s\n",
                     cudaGetErrorString(e));
        return -10;
    }
    return 0;
}

}  // namespace minimax_remover
}  // namespace kernels
}  // namespace flash_rt
