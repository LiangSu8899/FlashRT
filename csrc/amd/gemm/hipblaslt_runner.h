#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <functional>

// ================================================================
// GemmRunner (AMD): hipBLASLt-based GEMM for BF16/FP8 on CDNA.
//
// Port of csrc/gemm/gemm_runner.h (cuBLASLt). Method names,
// signatures, and math semantics are identical to the CUDA class
// (hipStream_t replaces cudaStream_t). Only the subset needed by
// the pi05 pipeline is ported; see hipblaslt_runner.hip for the
// layout-convention note (col-major swap instead of ORDER_ROW).
//
// FP8 here is OCP e4m3 (HIP_R_8F_E4M3) — the gfx950 native format.
// The fnuz variant (HIP_R_8F_E4M3_FNUZ, gfx942) is NOT used.
// ================================================================

// Check hipBLASLt status
#define HIPBLASLT_CHECK(expr)                                           \
    do {                                                                \
        hipblasStatus_t status = (expr);                                \
        if (status != HIPBLAS_STATUS_SUCCESS) {                        \
            throw std::runtime_error(                                   \
                std::string("hipBLASLt error at ") + __FILE__ + ":" +   \
                std::to_string(__LINE__) + " code=" +                   \
                std::to_string(static_cast<int>(status)));              \
        }                                                               \
    } while (0)

#define HIP_CHECK(expr)                                                 \
    do {                                                                \
        hipError_t err = (expr);                                        \
        if (err != hipSuccess) {                                        \
            throw std::runtime_error(                                   \
                std::string("HIP error at ") + __FILE__ + ":" +         \
                std::to_string(__LINE__) + ": " +                       \
                hipGetErrorString(err));                                \
        }                                                               \
    } while (0)

class GemmRunner {
public:
    GemmRunner();
    ~GemmRunner();

    // ── Inference (no timing, no sync, stream-based) ──

    // BF16: D = A(M,K) @ B(N,K)^T  (row-major, B transposed)
    // (This is the CUDA class's NT path; the CUDA name is bf16_run.)
    void bf16_run(void* A, void* B, void* D,
                  int M, int N, int K,
                  hipStream_t stream = 0);

    // BF16: D = A(M,K) @ B(K,N)  (row-major, no transpose)
    void bf16_nn(void* A, void* B, void* D,
                 int M, int N, int K,
                 hipStream_t stream = 0);

    // BF16 + BIAS epilogue: D = A(M,K) @ B(K,N) + bias(N)
    void bf16_nn_bias(void* A, void* B, void* D, void* bias,
                       int M, int N, int K,
                       hipStream_t stream = 0);

    // BF16 + BIAS + GELU epilogue: D = GELU(A(M,K) @ B(K,N) + bias(N))
    void bf16_nn_bias_gelu(void* A, void* B, void* D, void* bias,
                            int M, int N, int K,
                            hipStream_t stream = 0);

    // FP8 no-transpose: D_bf16 = A_fp8(M,K) @ B_fp8(K,N) with device scale pointers
    // Matches bf16_nn layout — B stored as (K,N), no transpose.
    // d_scale_a / d_scale_b are device float* holding per-tensor descale
    // factors; hipBLASLt applies scale_a * scale_b to the FP32 accumulator
    // (A/B_SCALE_POINTER semantics, identical to cuBLASLt).
    void fp8_nn_dev(void* A, void* B, void* D,
                    int M, int N, int K,
                    float* d_scale_a, float* d_scale_b,
                    hipStream_t stream = 0);

    // FP8 transpose-B path: D_bf16 = A_fp8(M,K) @ B_fp8(N,K)^T
    // with device scale pointers. B is stored as (N,K) row-major.
    void fp8_nt_dev(void* A, void* B, void* D,
                    int M, int N, int K,
                    float* d_scale_a, float* d_scale_b,
                    hipStream_t stream = 0);

    // MXFP4 a4w4 transpose-B path (OCP MX: E2M1 elements + per-1x32-block
    // UE8M0 scales): D_bf16 = A_fp4(M,K) @ B_fp4(N,K)^T
    //
    //   A        : E2M1 packed 2-per-byte along K → (M, K/2) bytes row-major.
    //              Element 2i in the low nibble, 2i+1 in the high nibble.
    //   A_scales : uint8 UE8M0 (biased-127 exponent), one per 1x32 block
    //              along K → (M, K/32) bytes row-major.
    //   B        : E2M1 packed 2-per-byte along K → (N, K/2) bytes row-major.
    //   B_scales : uint8 UE8M0 → (N, K/32) bytes row-major.
    //   D        : BF16 (M, N) row-major.
    //
    // hipBLASLt contract (see hipblaslt.h): the block-scale tensors are
    // installed via A/B_SCALE_POINTER with A/B_SCALE_MODE =
    // HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 — "an 8-bit R_8F_UE8M0
    // value for each 32-element block in the innermost dimension of the
    // corresponding data tensor". Under the col-major swap used by this
    // class the innermost (stride-1) dimension of BOTH operands is K, so
    // the row-major (rows, K/32) scale tensors above map to the library's
    // expectation with no reshuffling (full derivation in the .hip file).
    // K must be a multiple of 32 (throws otherwise).
    void mxfp4_nt_dev(void* A, void* A_scales,
                      void* B, void* B_scales, void* D,
                      int M, int N, int K,
                      hipStream_t stream = 0);

    // ── Autotune: benchmark top-N algorithms and cache the best ──
    // Call before HIP Graph capture. Uses dummy data at the provided pointers.
    // Matters on gfx950: hipBLASLt FP8 heuristics have known gaps, so the
    // timed selection over N candidates is the production algo picker.
    void autotune_bf16_nn(void* A, void* B, void* D,
                          int M, int N, int K, int num_algos = 16);
    void autotune_fp8_nn_dev(void* A, void* B, void* D,
                             int M, int N, int K,
                             float* d_scale_a, float* d_scale_b,
                             int num_algos = 16);
    void autotune_fp8_nt_dev(void* A, void* B, void* D,
                             int M, int N, int K,
                             float* d_scale_a, float* d_scale_b,
                             int num_algos = 16);
    void autotune_mxfp4_nt_dev(void* A, void* A_scales,
                               void* B, void* B_scales, void* D,
                               int M, int N, int K,
                               int num_algos = 16);

private:
    hipblasLtHandle_t handle_;
    void* workspace_;
    size_t workspace_size_;

    // ── GEMM descriptor + algorithm cache ──
    // Enum values match the CUDA class for the ported subset.
    // MXFP4_NT_DEV is AMD-only (gfx950 native MX support, no CUDA
    // counterpart in the ported class) and uses the next free value.
    enum GemmType { BF16_NN = 0, FP8_NN_DEV = 2, FP8_NT_DEV = 5, MXFP4_NT_DEV = 6 };

    struct GemmKey {
        int type, M, N, K;
        bool operator==(const GemmKey& o) const {
            return type == o.type && M == o.M && N == o.N && K == o.K;
        }
    };

    struct GemmKeyHash {
        size_t operator()(const GemmKey& k) const {
            size_t h = std::hash<int>()(k.type);
            h ^= std::hash<int>()(k.M) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    // NOTE: hipBLASLt matmul requires column-major layouts, so cached
    // descriptors are built with the col-major operand swap (see .hip):
    //   op0_desc describes the FlashRT "B" matrix, op1_desc the "A" matrix.
    struct CachedGemm {
        hipblasLtMatmulDesc_t matmul_desc;
        hipblasLtMatrixLayout_t op0_desc, op1_desc, D_desc;
        hipblasLtMatmulAlgo_t algo;
    };

    std::unordered_map<GemmKey, CachedGemm, GemmKeyHash> gemm_cache_;

    // Setup descriptors for a given GEMM type and shape, store in cache
    CachedGemm& get_or_create_cached(GemmType type, int M, int N, int K);
    // Autotune helper: benchmark algorithms and pick the best.
    // op0/op1 are already in hipBLASLt operand order (op0 = FlashRT B,
    // op1 = FlashRT A); scale_op0/scale_op1 follow the same order.
    void autotune_cached(CachedGemm& entry, void* op0, void* op1, void* D,
                         float alpha, float beta, int num_algos,
                         float* scale_op0 = nullptr, float* scale_op1 = nullptr);
};
