// ================================================================
// FlashRT AMD — decoder cross-attention, split-KV flash (CDNA wave64)
//
// One fixed production site: tiny-M GQA cross-attention
//     Q (Sq, Hq, D) bf16   — Sq ~ 10 query tokens, Hq query heads
//     K (Skv, D)    bf16   — single KV head (GQA Hq:1), Skv ~ 590-876
//     V (Skv, D)    bf16
//     O (Sq, Hq, D) bf16   — bidirectional, no mask
//
// Why this shape needs its own kernel: total work is ~80 query
// vectors x <1K keys x 256 dims — library FA tiles (128/256-row Q
// blocks, MFMA pipelines) waste >90% of their lanes here and the
// call is launch/bandwidth bound, not compute bound. MFMA is
// deliberately NOT used: a 10-row score tile cannot feed the
// 16x16x16 matrix pipes; packed FMA from LDS is already >10x above
// the arithmetic demand of this site.
//
// Decomposition (mirrors the FlashRT RTX split-KV win on this site):
//   kernel 1 (partial): grid (ATTN_NSPLIT=32, Hq) — 32*8 = 256
//     workgroups, one per CU on a 256-CU MI350X. Workgroup (i, h)
//     owns KV rows [i*chunk, min((i+1)*chunk, skv)) for head h,
//     chunk = ceil(skv/32) (<= 28 rows at Skv=876). It computes a
//     full online-softmax partial over its slice for ALL Sq queries:
//     row max m[q], sum-of-exp l[q], and the UNNORMALIZED
//     acc[q][d] = sum_j exp(s[q][j] - m[q]) * V[j][d],
//     written to the caller-provided fp32 workspace.
//   kernel 2 (reduce): grid (Sq, Hq) — 80 tiny workgroups combine
//     the 32 partials per (q, h) with the standard flash rescale
//     (global max, exp(m_i - gm) coefficients) and emit bf16 O.
//   Two kernels on one stream: plain stream ordering, no atomics,
//   no cooperative counters — trivially HIP-graph capturable.
//
// K/V are read once per head (8x total through L2; K+V ~ 850 KB,
// so ~7 MB of L2-resident traffic — well under 10 us at any
// realistic L2/HBM bandwidth). The problem is launch bound; both
// kernels are deliberately shallow.
//
// Sequence length modes (matches qkv_split_rope_devpos):
//   exact mode:  seqused == nullptr, Skv host int is the length.
//   fixed mode:  seqused != nullptr, kernel reads seqused[0] on
//     device; host Skv is ignored. Rows >= seqused[0] are NEVER
//     touched (loops bound by the device value), so a fixed-shape
//     graph can run any valid prefix length and padding rows may
//     hold garbage/NaN.
//
// fp32 math order (deterministic, fixed for a given shape):
//   - scores: per-lane partial over dims {lane, lane+64, lane+128,
//     lane+192} in ascending order, then the wave64 shuffle tree of
//     wave_reduce_sum (fixed radix-2 order). Scale applied once to
//     the reduced fp32 score.
//   - softmax: m = serial fmaxf over the slice in ascending j;
//     p = __expf(s - m) (device fast exp, rel err ~2^-22 — far
//     below bf16 input noise); l = serial sum ascending j.
//   - acc: serial ascending j per output dim; fp32 FMA.
//   - reduce: serial ascending split index. No atomics anywhere.
//
// LDS layout (kernel 1, ~20.3 KB static):
//   q_lds  [Sq][D]  fp32 Q for this head (10x256 = 10 KB at site)
//   s_lds  [SQ_MAX][CHUNK_MAX] scores -> exp'd probabilities
//   m_lds  [SQ_MAX] row maxima
//   Bank behavior: score dot reads q_lds[q*D + lane + 64*i] —
//   consecutive lanes hit consecutive banks (conflict-free); the
//   matching K global read is a 2 B/lane contiguous burst (128 B
//   per quarter row per instruction, fully coalesced). Phase C
//   reads s_lds at one address per wave (broadcast) and V rows as
//   contiguous 2 B/lane bursts.
//
// Workspace (partial_ws, caller-provided, NO allocation here):
//   ATTN_NSPLIT * Hq * Sq * (D + 2) floats
//   = 32 * Hq * Sq * (D + 2) * 4 bytes  (pi05 site: 32*8*10*258*4
//   = 2,641,920 B ~ 2.52 MiB). Per-(split,head) block layout:
//   [Sq*D unnormalized acc][Sq m][Sq l]. Never needs zeroing:
//   empty tail slices mark l = 0 and the reducer skips them.
//
// Limits (asserted nowhere — this is a fixed-site kernel; the
// wrapper documents them): Sq <= 16, D <= 256, D % 4 == 0,
// skv <= ATTN_NSPLIT * ATTN_CHUNK_MAX = 2048 (in-kernel clamped),
// single KV head (GQA Hq:1).
// ================================================================

#include "../kernels/common_hip.h"

#include <cfloat>

#define ATTN_NSPLIT     32   // KV splits; 32 * Hq(8) = 256 WGs = 1/CU
#define ATTN_NTHREADS   256  // 4 waves
#define ATTN_NWAVES     (ATTN_NTHREADS / FVK_WAVE_SIZE)
#define ATTN_SQ_MAX     16   // register/LDS sizing bound for Sq
#define ATTN_D_MAX      256  // LDS sizing bound for D
#define ATTN_CHUNK_MAX  64   // max KV rows per split -> skv <= 2048
#define ATTN_KD_PER_LANE (ATTN_D_MAX / FVK_WAVE_SIZE)  // 4 dims/lane

// ── Kernel 1: per-slice flash partials ──
// grid (ATTN_NSPLIT, Hq), block ATTN_NTHREADS.
__global__ __launch_bounds__(ATTN_NTHREADS)
void attention_decoder_gqa_partial_kernel(
    const __hip_bfloat16* __restrict__ Q,
    const __hip_bfloat16* __restrict__ K,
    const __hip_bfloat16* __restrict__ V,
    float* __restrict__ ws,
    const int* __restrict__ seqused,
    int Sq, int Skv, int Hq, int D, float scale) {
    const int split = blockIdx.x;
    const int h     = blockIdx.y;
    const int tid   = threadIdx.x;
    const int lane  = tid & (FVK_WAVE_SIZE - 1);
    const int wave  = tid >> 6;

    __shared__ float q_lds[ATTN_SQ_MAX * ATTN_D_MAX];
    __shared__ float s_lds[ATTN_SQ_MAX * ATTN_CHUNK_MAX];
    __shared__ float m_lds[ATTN_SQ_MAX];

    // Effective sequence length: device seqused[0] wins (fixed-shape
    // graph mode); otherwise the host int (exact mode).
    int skv_eff = (seqused != nullptr) ? seqused[0] : Skv;
    if (skv_eff > ATTN_NSPLIT * ATTN_CHUNK_MAX)
        skv_eff = ATTN_NSPLIT * ATTN_CHUNK_MAX;  // documented hard cap

    const int chunk = (skv_eff + ATTN_NSPLIT - 1) / ATTN_NSPLIT;
    const int start = split * chunk;
    int end = start + chunk;
    if (end > skv_eff) end = skv_eff;
    const int cn = end - start;  // rows in this slice (may be <= 0)

    // Workspace block for (split, head): [Sq*D acc][Sq m][Sq l]
    const size_t base = (size_t)(split * Hq + h) * (size_t)Sq * (size_t)(D + 2);
    float* acc_ws = ws + base;
    float* m_ws   = ws + base + (size_t)Sq * D;
    float* l_ws   = m_ws + Sq;

    if (cn <= 0) {
        // Empty tail slice: l = 0 tells the reducer to skip this block
        // (its acc region is never read), so ws needs no pre-zeroing.
        if (tid < Sq) { m_ws[tid] = -FLT_MAX; l_ws[tid] = 0.0f; }
        return;
    }

    // Stage this head's Q into LDS as fp32 (10x256 = 10 KB at site).
    for (int idx = tid; idx < Sq * D; idx += ATTN_NTHREADS) {
        const int q = idx / D;
        const int d = idx - q * D;
        q_lds[q * D + d] =
            __bfloat162float(Q[((size_t)q * Hq + h) * D + d]);
    }
    __syncthreads();

    // ── Phase A: scores s[q][j] = scale * <Q[q], K[j]> ──
    // Waves interleave over slice rows (wave w -> rows start+w,
    // start+w+4, ...). Lane l owns dims {l, l+64, l+128, l+192}:
    // conflict-free LDS reads of Q, coalesced 2 B/lane K bursts.
    for (int j = start + wave; j < end; j += ATTN_NWAVES) {
        const __hip_bfloat16* krow = K + (size_t)j * D;
        float kd[ATTN_KD_PER_LANE];
        #pragma unroll
        for (int i = 0; i < ATTN_KD_PER_LANE; ++i) {
            const int d = lane + i * FVK_WAVE_SIZE;
            kd[i] = (d < D) ? __bfloat162float(krow[d]) : 0.0f;
        }
        float p[ATTN_SQ_MAX];
        #pragma unroll
        for (int q = 0; q < ATTN_SQ_MAX; ++q) {
            float a = 0.0f;
            if (q < Sq) {
                #pragma unroll
                for (int i = 0; i < ATTN_KD_PER_LANE; ++i) {
                    const int d = lane + i * FVK_WAVE_SIZE;
                    if (d < D) a += kd[i] * q_lds[q * D + d];
                }
            }
            p[q] = a;
        }
        #pragma unroll
        for (int q = 0; q < ATTN_SQ_MAX; ++q) {
            if (q < Sq) {
                const float r = wave_reduce_sum(p[q]);  // fixed radix-2 tree
                if (lane == 0)
                    s_lds[q * ATTN_CHUNK_MAX + (j - start)] = r * scale;
            }
        }
    }
    __syncthreads();

    // ── Phase B: per-slice online softmax over the <= CHUNK_MAX rows ──
    // Row maxima: one thread per query row, serial ascending j
    // (<= 64 iterations — cheaper than another reduction round-trip).
    if (tid < Sq) {
        float m = -FLT_MAX;
        for (int jj = 0; jj < cn; ++jj)
            m = fmaxf(m, s_lds[tid * ATTN_CHUNK_MAX + jj]);
        m_lds[tid] = m;
    }
    __syncthreads();

    // Exponentiate in place, all threads in parallel.
    for (int idx = tid; idx < Sq * cn; idx += ATTN_NTHREADS) {
        const int q  = idx / cn;
        const int jj = idx - q * cn;
        float* sp = &s_lds[q * ATTN_CHUNK_MAX + jj];
        *sp = __expf(*sp - m_lds[q]);
    }
    __syncthreads();

    // l[q]: serial ascending j — fixed summation order.
    if (tid < Sq) {
        float l = 0.0f;
        for (int jj = 0; jj < cn; ++jj)
            l += s_lds[tid * ATTN_CHUNK_MAX + jj];
        m_ws[tid] = m_lds[tid];
        l_ws[tid] = l;
    }

    // ── Phase C: unnormalized acc[q][d] = sum_j p[q][j] * V[j][d] ──
    // Thread t owns output dim t for all Sq queries; V row reads are
    // contiguous 2 B/lane, p reads are LDS broadcasts. Ascending j.
    if (tid < D) {
        float acc[ATTN_SQ_MAX];
        #pragma unroll
        for (int q = 0; q < ATTN_SQ_MAX; ++q) acc[q] = 0.0f;
        for (int jj = 0; jj < cn; ++jj) {
            const float v =
                __bfloat162float(V[(size_t)(start + jj) * D + tid]);
            #pragma unroll
            for (int q = 0; q < ATTN_SQ_MAX; ++q)
                if (q < Sq) acc[q] += s_lds[q * ATTN_CHUNK_MAX + jj] * v;
        }
        #pragma unroll
        for (int q = 0; q < ATTN_SQ_MAX; ++q)
            if (q < Sq) acc_ws[(size_t)q * D + tid] = acc[q];
    }
}

// ── Kernel 2: combine the ATTN_NSPLIT partials per (q, h) ──
// grid (Sq, Hq), block ATTN_NTHREADS (thread t = output dim t).
// Standard flash rescale: gm = max_i m_i, coeff_i = exp(m_i - gm),
// O = (sum_i coeff_i * acc_i) / (sum_i coeff_i * l_i).
__global__ __launch_bounds__(ATTN_NTHREADS)
void attention_decoder_gqa_reduce_kernel(
    const float* __restrict__ ws,
    __hip_bfloat16* __restrict__ O,
    int Sq, int Hq, int D) {
    const int q   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ float coeff[ATTN_NSPLIT];
    __shared__ float s_invL;

    const size_t blk = (size_t)Sq * (size_t)(D + 2);  // floats per (split,head)
    const size_t ml_off = (size_t)Sq * D;

    // Serial over 32 splits on thread 0: ~64 independent fp32 loads,
    // negligible next to launch cost; fixed ascending order.
    if (tid == 0) {
        float gm = -FLT_MAX;
        for (int i = 0; i < ATTN_NSPLIT; ++i) {
            const size_t b = (size_t)(i * Hq + h) * blk;
            const float li = ws[b + ml_off + Sq + q];
            const float mi = ws[b + ml_off + q];
            if (li > 0.0f && mi > gm) gm = mi;
        }
        float L = 0.0f;
        for (int i = 0; i < ATTN_NSPLIT; ++i) {
            const size_t b = (size_t)(i * Hq + h) * blk;
            const float li = ws[b + ml_off + Sq + q];
            const float mi = ws[b + ml_off + q];
            const float c = (li > 0.0f) ? __expf(mi - gm) : 0.0f;
            coeff[i] = c;
            L += c * li;
        }
        s_invL = (L > 0.0f) ? 1.0f / L : 0.0f;  // skv >= 1 => L > 0
    }
    __syncthreads();

    if (tid < D) {
        float s = 0.0f;
        for (int i = 0; i < ATTN_NSPLIT; ++i) {
            const float c = coeff[i];
            // c == 0 also guards the never-written acc of empty slices.
            if (c != 0.0f) {
                const size_t b = (size_t)(i * Hq + h) * blk;
                s += c * ws[b + (size_t)q * D + tid];
            }
        }
        O[((size_t)q * Hq + h) * D + tid] = __float2bfloat16(s * s_invL);
    }
}

// ── Host wrapper ──
// partial_ws: caller-provided device fp32 scratch of at least
//   ATTN_NSPLIT * Hq * Sq * (D + 2) floats
//   (= 32 * Hq * Sq * (D + 2) * 4 bytes; pi05 decoder site
//   32*8*10*258*4 = 2,641,920 bytes). No allocation, no memset, no
//   host sync — safe inside HIP graph capture.
// seqused: nullptr => exact mode (Skv host int is the length);
//   else device int32, seqused[0] read in-kernel and Skv is ignored
//   (fixed-shape graph mode; padding rows beyond seqused[0] are
//   never read).
// Constraints (fixed-site kernel, not checked): 1 <= Sq <= 16,
//   D <= 256, seq length <= 2048, single KV head (K/V are (Skv, D)).
void attention_decoder_gqa(const __hip_bfloat16* Q,
                           const __hip_bfloat16* K,
                           const __hip_bfloat16* V,
                           __hip_bfloat16* O,
                           float* partial_ws,
                           int Sq, int Skv, int Hq, int D,
                           const int* seqused,
                           float softmax_scale,
                           hipStream_t stream) {
    dim3 grid1(ATTN_NSPLIT, Hq);
    attention_decoder_gqa_partial_kernel<<<grid1, ATTN_NTHREADS, 0, stream>>>(Q, K, V, partial_ws, seqused, Sq, Skv, Hq, D, softmax_scale);
    dim3 grid2(Sq, Hq);
    attention_decoder_gqa_reduce_kernel<<<grid2, ATTN_NTHREADS, 0, stream>>>(partial_ws, O, Sq, Hq, D);
}
