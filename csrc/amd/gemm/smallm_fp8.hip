#include "smallm_fp8.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <stdexcept>
#include <string>

// ================================================================
// FlashRT AMD — hand-tuned small-M FP8 GEMM kernels (CDNA4, wave64)
//
// See smallm_fp8.h for the contract. Design notes:
//
// The problem is bandwidth: weight bytes (N*K fp8) dwarf everything
// else (A is M*K <= 64KB, D is M*N*2). The floor is one pass over B
// at DRAM rate, so both kernels stream B exactly once with 16-byte
// (dwordx4) lane loads and keep A resident (LDS for nn, registers
// for nt), accumulating FP32 per lane and folding with wave shuffles.
//
// nn ("kn" weights, B[k*N + n], N contiguous per k-row):
//   Workgroup = 256 threads = 4 waves; each wave owns 16 consecutive
//   output columns, so a WG covers a 64-column tile — the 4 waves
//   together touch 64 consecutive bytes of every k-row (a full 64B
//   DRAM sector). Within a wave, the 64 lanes partition K: lane l
//   streams rows k = l, l+64, ... of the WG's k-slice, loading 16
//   bytes of B per row. A is staged transposed in LDS as
//   aT[k][16 bytes] so a lane fetches all M values for its row with
//   one ds_read_b128. Lane partials fold with a fixed shfl_down tree.
//   Grid = (N/64) x S k-slices. n-parallelism alone (N/64 WGs) cannot
//   fill an MI350X, so S slices of K run in parallel and a second
//   kernel reduces the FP32 partials in ascending slice order
//   (deterministic, no atomics — required inside captured graphs).
//   S is the smallest power of two that reaches ~256 WGs (~512 for
//   the _alt config), capped so it divides K/64 and by the workspace
//   bound SMALLM_FP8_MAX_SPLIT. Split-K costs S*M*N*4 bytes written
//   + read back (~1.3MB for the dominant pi05 shapes at S<=16, i.e.
//   ~25-35% of the 4MB weight stream) — the price of occupancy.
//
// nt ("nk" weights, B[n*K + k], K contiguous per output column):
//   The naturally streaming layout: a wave owns whole output columns,
//   lanes tile K in 16-byte chunks (64 lanes x 16B = 1KB of one B row
//   per pass — perfectly coalesced full cache lines), and column
//   parallelism alone (N/16 WGs at 4 columns/wave) fills the GPU with
//   NO split-K, NO workspace, NO LDS. Each wave processes NPW=4
//   columns per K-chunk so the A-chunk, converted once into
//   registers, is reused across 4 columns. Prefer this path: set the
//   frontend's fp8_layout="nk" so decoder weights are packed (N,K).
//
// fp8 -> f32 conversion: per-byte through __hip_fp8_e4m3's float
// operator, which lowers to the gfx950 native V_CVT_*_FP8 ops. A
// 256-entry LDS LUT was considered and rejected: the native converts
// are cheap on CDNA4 and a LUT would add an LDS gather (64 scattered
// lane reads) per 16 weight bytes plus LDS pressure.
//
// M is dispatched to a compiled tile height M_T in {4, 8, 10, 16}
// (accumulators are register arrays, so the m-loop must be
// compile-time). M=10 (pi05 action chunk) hits its exact tile.
// ================================================================

#define SMALLM_FP8_BLOCK 256   // threads per workgroup (4 waves)

// ── fp8 e4m3 byte -> f32 ──
__device__ __forceinline__ float smallm_fp8_to_f32(unsigned char b) {
    __hip_fp8_e4m3 v;
    v.__x = b;
    return static_cast<float>(v);
}

// Unpack 16 fp8 bytes (one dwordx4 load) to f32.
__device__ __forceinline__ void smallm_fp8x16_to_f32(const uint4& v, float* out) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&v);
    #pragma unroll
    for (int j = 0; j < 16; ++j) out[j] = smallm_fp8_to_f32(b[j]);
}

// ================================================================
// nn kernel: B stored (K, N) row-major.
//   grid = (N/64, S), block = 256. Dynamic LDS = k_slice * 16 bytes.
//   S == 1: writes scaled BF16 straight to D.
//   S  > 1: writes unscaled FP32 partials to ws[s][m][n]; the reduce
//           kernel below finishes deterministically.
// ================================================================
template<int M_T>
__global__ __launch_bounds__(SMALLM_FP8_BLOCK)
void smallm_fp8_nn_kernel(const unsigned char* __restrict__ A,
                          const unsigned char* __restrict__ B,
                          __hip_bfloat16* __restrict__ D,
                          float* __restrict__ ws,
                          const float* __restrict__ d_scale_a,
                          const float* __restrict__ d_scale_b,
                          int M, int N, int K, int k_slice) {
    extern __shared__ unsigned char smallm_lds_aT[];  // aT[k_local][16]
    const int tid  = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int n0   = blockIdx.x * 64 + wave * 16;   // wave's 16-column group
    const int s    = blockIdx.y;                    // k-slice index
    const int k0   = s * k_slice;

    // Stage this k-slice of A transposed: aT[k - k0][m], m zero-padded
    // to 16 so a lane reads its row's M values as one 16B LDS load.
    for (int i = tid; i < k_slice * 16; i += SMALLM_FP8_BLOCK) {
        const int m = i & 15, koff = i >> 4;
        smallm_lds_aT[i] =
            (m < M) ? A[(size_t)m * K + (k0 + koff)] : (unsigned char)0;
    }
    __syncthreads();

    float acc[M_T * 16];
    #pragma unroll
    for (int i = 0; i < M_T * 16; ++i) acc[i] = 0.0f;

    // Lane l streams rows k0 + l, k0 + l + 64, ... — fixed sequential
    // order per lane (deterministic accumulation).
    const int iters = k_slice >> 6;
    #pragma unroll 4
    for (int it = 0; it < iters; ++it) {
        const int kl = (it << 6) + lane;  // row within the slice
        const uint4 bv = *reinterpret_cast<const uint4*>(
            B + (size_t)(k0 + kl) * N + n0);
        float bf[16];
        smallm_fp8x16_to_f32(bv, bf);
        const uint4 av = reinterpret_cast<const uint4*>(smallm_lds_aT)[kl];
        const unsigned char* ab = reinterpret_cast<const unsigned char*>(&av);
        #pragma unroll
        for (int m = 0; m < M_T; ++m) {
            const float a = smallm_fp8_to_f32(ab[m]);
            #pragma unroll
            for (int j = 0; j < 16; ++j) acc[m * 16 + j] += a * bf[j];
        }
    }

    // Fixed-shape wave shuffle tree (order identical every launch).
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        #pragma unroll
        for (int i = 0; i < M_T * 16; ++i)
            acc[i] += __shfl_down(acc[i], off);
    }

    if (lane == 0) {
        if (gridDim.y > 1) {
            float* dst = ws + (size_t)s * M * N;
            #pragma unroll
            for (int m = 0; m < M_T; ++m) {
                if (m < M) {
                    #pragma unroll
                    for (int j = 0; j < 16; ++j)
                        dst[(size_t)m * N + (n0 + j)] = acc[m * 16 + j];
                }
            }
        } else {
            const float sc = d_scale_a[0] * d_scale_b[0];
            #pragma unroll
            for (int m = 0; m < M_T; ++m) {
                if (m < M) {
                    #pragma unroll
                    for (int j = 0; j < 16; ++j)
                        D[(size_t)m * N + (n0 + j)] =
                            __float2bfloat16(acc[m * 16 + j] * sc);
                }
            }
        }
    }
}

// ================================================================
// Split-K reduce: D[i] = bf16(sum_{s ascending} ws[s][i] * scale).
// One thread per 4 outputs (float4 partial reads). The s-loop runs in
// fixed ascending order — the only cross-workgroup reduction in the
// nn path, and it is deterministic by construction.
// ================================================================
__global__ __launch_bounds__(SMALLM_FP8_BLOCK)
void smallm_fp8_split_reduce_kernel(const float* __restrict__ ws,
                                    __hip_bfloat16* __restrict__ D,
                                    const float* __restrict__ d_scale_a,
                                    const float* __restrict__ d_scale_b,
                                    int split, long long mn) {
    const long long i4 =
        ((long long)blockIdx.x * SMALLM_FP8_BLOCK + threadIdx.x) * 4;
    if (i4 >= mn) return;  // mn is a multiple of 4 (N % 64 == 0)
    float sx = 0.0f, sy = 0.0f, sz = 0.0f, sw = 0.0f;
    for (int s = 0; s < split; ++s) {
        const float4 p =
            *reinterpret_cast<const float4*>(ws + (size_t)s * mn + i4);
        sx += p.x; sy += p.y; sz += p.z; sw += p.w;
    }
    const float sc = d_scale_a[0] * d_scale_b[0];
    D[i4 + 0] = __float2bfloat16(sx * sc);
    D[i4 + 1] = __float2bfloat16(sy * sc);
    D[i4 + 2] = __float2bfloat16(sz * sc);
    D[i4 + 3] = __float2bfloat16(sw * sc);
}

// ================================================================
// nt kernel: B stored (N, K) row-major.
//   grid = ceil(N / (4*NPW)), block = 256; wave w owns output columns
//   nb + w*NPW .. +NPW-1. Lanes tile K in 16B chunks (1KB per wave
//   pass); the lane's A window is converted to registers once per
//   chunk and reused across the wave's NPW columns. No LDS, no
//   split-K, no workspace.
// ================================================================
template<int M_T, int NPW>
__global__ __launch_bounds__(SMALLM_FP8_BLOCK)
void smallm_fp8_nt_kernel(const unsigned char* __restrict__ A,
                          const unsigned char* __restrict__ B,
                          __hip_bfloat16* __restrict__ D,
                          const float* __restrict__ d_scale_a,
                          const float* __restrict__ d_scale_b,
                          int M, int N, int K) {
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int nb   = blockIdx.x * (4 * NPW) + wave * NPW;

    float acc[NPW][M_T];
    #pragma unroll
    for (int p = 0; p < NPW; ++p)
        #pragma unroll
        for (int m = 0; m < M_T; ++m) acc[p][m] = 0.0f;

    // K-chunks of 64 lanes x 16B = 1024 elements; sequential chunk
    // loop per lane keeps accumulation order fixed.
    const int chunks = (K + 1023) >> 10;
    for (int c = 0; c < chunks; ++c) {
        const int kw = (c << 10) + (lane << 4);
        if (kw >= K) continue;  // tail guard (K % 16 == 0 keeps loads whole)

        // A window for this lane's 16 k's, all rows, converted once.
        float a_f[M_T][16];
        #pragma unroll
        for (int m = 0; m < M_T; ++m) {
            if (m < M) {
                const uint4 av = *reinterpret_cast<const uint4*>(
                    A + (size_t)m * K + kw);
                smallm_fp8x16_to_f32(av, a_f[m]);
            } else {
                #pragma unroll
                for (int j = 0; j < 16; ++j) a_f[m][j] = 0.0f;
            }
        }

        #pragma unroll
        for (int p = 0; p < NPW; ++p) {
            const int n = nb + p;
            if (n >= N) continue;
            const uint4 bv = *reinterpret_cast<const uint4*>(
                B + (size_t)n * K + kw);
            float bf[16];
            smallm_fp8x16_to_f32(bv, bf);
            #pragma unroll
            for (int m = 0; m < M_T; ++m) {
                #pragma unroll
                for (int j = 0; j < 16; ++j)
                    acc[p][m] += a_f[m][j] * bf[j];
            }
        }
    }

    // Fixed-shape wave shuffle tree.
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        #pragma unroll
        for (int p = 0; p < NPW; ++p)
            #pragma unroll
            for (int m = 0; m < M_T; ++m)
                acc[p][m] += __shfl_down(acc[p][m], off);
    }

    if (lane == 0) {
        const float sc = d_scale_a[0] * d_scale_b[0];
        #pragma unroll
        for (int p = 0; p < NPW; ++p) {
            const int n = nb + p;
            if (n < N) {
                #pragma unroll
                for (int m = 0; m < M_T; ++m)
                    if (m < M)
                        D[(size_t)m * N + n] = __float2bfloat16(acc[p][m] * sc);
            }
        }
    }
}

// ================================================================
// Host side
// ================================================================

size_t smallm_fp8_nn_dev_ws_bytes(int M, int N) {
    return (size_t)SMALLM_FP8_MAX_SPLIT * M * N * sizeof(float);
}

static void smallm_fp8_check(const char* who, int M, int N, int K,
                             int n_align, int k_align) {
    if (M < 1 || M > 16)
        throw std::runtime_error(std::string(who) + ": M must be in [1,16], got " +
                                 std::to_string(M));
    if (N < n_align || (N % n_align) != 0)
        throw std::runtime_error(std::string(who) + ": N must be a multiple of " +
                                 std::to_string(n_align) + ", got " + std::to_string(N));
    if (K < k_align || (K % k_align) != 0)
        throw std::runtime_error(std::string(who) + ": K must be a multiple of " +
                                 std::to_string(k_align) + ", got " + std::to_string(K));
}

// Smallest power-of-two split S with (N/64)*S >= target_wgs, subject to
// S | (K/64), S <= SMALLM_FP8_MAX_SPLIT, and LDS staging <= 64KB/slice.
static int smallm_fp8_pick_split(int N, int K, bool have_ws, int target_wgs) {
    const int k64 = K / 64;
    int s = 1;
    if (have_ws) {
        const int n_wgs = N / 64;
        while (n_wgs * s < target_wgs && (s << 1) <= SMALLM_FP8_MAX_SPLIT) s <<= 1;
        while (s > 1 && (s > k64 || (k64 % s) != 0)) s >>= 1;
    }
    // LDS bound: k_slice * 16 bytes <= 64KB  =>  K/S <= 4096.
    while ((size_t)(K / s) * 16 > 65536) {
        const int s2 = s << 1;
        if (!have_ws || s2 > SMALLM_FP8_MAX_SPLIT || s2 > k64 || (k64 % s2) != 0)
            throw std::runtime_error(
                "smallm_fp8_nn_dev: K too large for LDS staging without a "
                "(deeper) split_ws; K=" + std::to_string(K));
        s = s2;
    }
    return s;
}

static void smallm_fp8_nn_dev_impl(const void* A, const void* B, void* D,
                                   const float* d_scale_a, const float* d_scale_b,
                                   int M, int N, int K, float* split_ws,
                                   hipStream_t stream, int target_wgs) {
    smallm_fp8_check("smallm_fp8_nn_dev", M, N, K, 64, 64);
    const int s = smallm_fp8_pick_split(N, K, split_ws != nullptr, target_wgs);
    const int k_slice = K / s;
    const dim3 grid((unsigned)(N / 64), (unsigned)s);
    const dim3 block(SMALLM_FP8_BLOCK);
    const size_t shmem = (size_t)k_slice * 16;
    const unsigned char* a8 = static_cast<const unsigned char*>(A);
    const unsigned char* b8 = static_cast<const unsigned char*>(B);
    __hip_bfloat16* d16 = static_cast<__hip_bfloat16*>(D);

    #define SMALLM_LAUNCH_NN(MT)                                                \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(smallm_fp8_nn_kernel<MT>),           \
                           grid, block, shmem, stream,                          \
                           a8, b8, d16, split_ws, d_scale_a, d_scale_b,         \
                           M, N, K, k_slice)
    if      (M <= 4)  SMALLM_LAUNCH_NN(4);
    else if (M <= 8)  SMALLM_LAUNCH_NN(8);
    else if (M <= 10) SMALLM_LAUNCH_NN(10);
    else              SMALLM_LAUNCH_NN(16);
    #undef SMALLM_LAUNCH_NN

    if (s > 1) {
        const long long mn = (long long)M * N;
        const unsigned rblocks =
            (unsigned)((mn / 4 + SMALLM_FP8_BLOCK - 1) / SMALLM_FP8_BLOCK);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(smallm_fp8_split_reduce_kernel),
                           dim3(rblocks), block, 0, stream,
                           split_ws, d16, d_scale_a, d_scale_b, s, mn);
    }
}

void smallm_fp8_nn_dev(const void* A, const void* B, void* D,
                       const float* d_scale_a, const float* d_scale_b,
                       int M, int N, int K, float* split_ws,
                       hipStream_t stream) {
    smallm_fp8_nn_dev_impl(A, B, D, d_scale_a, d_scale_b, M, N, K,
                           split_ws, stream, /*target_wgs=*/256);
}

void smallm_fp8_nn_dev_alt(const void* A, const void* B, void* D,
                           const float* d_scale_a, const float* d_scale_b,
                           int M, int N, int K, float* split_ws,
                           hipStream_t stream) {
    smallm_fp8_nn_dev_impl(A, B, D, d_scale_a, d_scale_b, M, N, K,
                           split_ws, stream, /*target_wgs=*/512);
}

template<int NPW>
static void smallm_fp8_nt_dev_impl(const void* A, const void* B, void* D,
                                   const float* d_scale_a, const float* d_scale_b,
                                   int M, int N, int K, hipStream_t stream) {
    smallm_fp8_check("smallm_fp8_nt_dev", M, N, K, 1, 16);
    const int cols_per_wg = 4 * NPW;
    const dim3 grid((unsigned)((N + cols_per_wg - 1) / cols_per_wg));
    const dim3 block(SMALLM_FP8_BLOCK);
    const unsigned char* a8 = static_cast<const unsigned char*>(A);
    const unsigned char* b8 = static_cast<const unsigned char*>(B);
    __hip_bfloat16* d16 = static_cast<__hip_bfloat16*>(D);

    #define SMALLM_LAUNCH_NT(MT)                                                \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(smallm_fp8_nt_kernel<MT, NPW>),      \
                           grid, block, 0, stream,                              \
                           a8, b8, d16, d_scale_a, d_scale_b, M, N, K)
    if      (M <= 4)  SMALLM_LAUNCH_NT(4);
    else if (M <= 8)  SMALLM_LAUNCH_NT(8);
    else if (M <= 10) SMALLM_LAUNCH_NT(10);
    else              SMALLM_LAUNCH_NT(16);
    #undef SMALLM_LAUNCH_NT
}

void smallm_fp8_nt_dev(const void* A, const void* B, void* D,
                       const float* d_scale_a, const float* d_scale_b,
                       int M, int N, int K, float* split_ws,
                       hipStream_t stream) {
    (void)split_ws;  // nt never splits K — accepted for signature symmetry
    smallm_fp8_nt_dev_impl<4>(A, B, D, d_scale_a, d_scale_b, M, N, K, stream);
}

void smallm_fp8_nt_dev_alt(const void* A, const void* B, void* D,
                           const float* d_scale_a, const float* d_scale_b,
                           int M, int N, int K, float* split_ws,
                           hipStream_t stream) {
    (void)split_ws;
    smallm_fp8_nt_dev_impl<2>(A, B, D, d_scale_a, d_scale_b, M, N, K, stream);
}
