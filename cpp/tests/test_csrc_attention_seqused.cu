#include "attention_cublas.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kGuardElements = 64;
constexpr std::uint16_t kGuardBits = 0xa5a5;

struct Shape {
    int query_rows;
    int key_rows;
    int heads;
    int head_dimension;
    int valid_keys;
};

struct Result {
    Shape shape{};
    int logits_stride = 0;
    std::vector<__half> queries;
    std::vector<__half> keys;
    std::vector<__half> values;
    std::vector<__half> logits;
    std::vector<__half> output;
};

std::uint16_t half_bits(__half value) {
    std::uint16_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

__half half_from_bits(std::uint16_t bits) {
    __half value{};
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

class HalfBuffer {
  public:
    HalfBuffer() = default;
    HalfBuffer(const HalfBuffer&) = delete;
    HalfBuffer& operator=(const HalfBuffer&) = delete;
    ~HalfBuffer() {
        if (storage_) cudaFree(storage_);
    }

    bool allocate(std::size_t payload, bool guarded, const char* name) {
        payload_ = payload;
        guard_ = guarded ? kGuardElements : 0;
        const cudaError_t status = cudaMallocManaged(
            &storage_, (payload_ + 2 * guard_) * sizeof(__half));
        if (status != cudaSuccess) {
            std::fprintf(stderr, "cudaMallocManaged(%s) failed: %s\n", name,
                         cudaGetErrorString(status));
            return false;
        }
        std::fill(storage_, storage_ + payload_ + 2 * guard_,
                  half_from_bits(kGuardBits));
        return true;
    }

    __half* data() const { return storage_ + guard_; }

    bool guards_intact(const char* name) const {
        for (std::size_t i = 0; i < guard_; ++i) {
            if (half_bits(storage_[i]) != kGuardBits ||
                half_bits(storage_[guard_ + payload_ + i]) != kGuardBits) {
                std::fprintf(stderr, "%s guard was overwritten\n", name);
                return false;
            }
        }
        return true;
    }

  private:
    __half* storage_ = nullptr;
    std::size_t payload_ = 0;
    std::size_t guard_ = 0;
};

std::vector<__half> make_values(std::size_t count, int salt) {
    std::vector<__half> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        const int pattern =
            static_cast<int>((i * 37 + static_cast<std::size_t>(salt) * 17) %
                             101) -
            50;
        values[i] = __float2half_rn(static_cast<float>(pattern) / 128.0f);
    }
    return values;
}

bool run_attention(cublasHandle_t handle, const Shape& shape, bool seqused,
                   Result* result) {
    const int rows = shape.query_rows * shape.heads;
    const int stride =
        seqused ? shape.key_rows + (shape.key_rows & 1) : shape.key_rows;
    const std::size_t query_count =
        static_cast<std::size_t>(rows) * shape.head_dimension;
    const std::size_t key_count =
        static_cast<std::size_t>(shape.key_rows) * shape.head_dimension;
    const std::size_t logits_count = static_cast<std::size_t>(rows) * stride;

    Result current;
    current.shape = shape;
    current.logits_stride = stride;
    current.queries = make_values(query_count, 1);
    current.keys = make_values(key_count, 2);
    current.values = make_values(key_count, 3);

    HalfBuffer queries;
    HalfBuffer keys;
    HalfBuffer values;
    HalfBuffer logits;
    HalfBuffer output;
    int* valid_keys = nullptr;
    if (!queries.allocate(query_count, false, "queries") ||
        !keys.allocate(key_count, false, "keys") ||
        !values.allocate(key_count, false, "values") ||
        !logits.allocate(logits_count, true, "logits") ||
        !output.allocate(query_count, true, "output") ||
        !check_cuda(cudaMallocManaged(&valid_keys, sizeof(int)),
                    "cudaMallocManaged(valid_keys)")) {
        if (valid_keys) cudaFree(valid_keys);
        return false;
    }
    std::copy(current.queries.begin(), current.queries.end(), queries.data());
    std::copy(current.keys.begin(), current.keys.end(), keys.data());
    std::copy(current.values.begin(), current.values.end(), values.data());
    *valid_keys = shape.valid_keys;

    const float scale =
        1.0f / std::sqrt(static_cast<float>(shape.head_dimension));
    if (seqused) {
        attention_qkv_fp16_seqused(
            handle, queries.data(), keys.data(), values.data(), logits.data(),
            output.data(), shape.query_rows, shape.key_rows, shape.heads,
            shape.head_dimension, valid_keys, scale);
    } else {
        attention_qkv_fp16(
            handle, queries.data(), keys.data(), values.data(), logits.data(),
            output.data(), shape.query_rows, shape.key_rows, shape.heads,
            shape.head_dimension, scale);
    }
    const bool synchronized =
        check_cuda(cudaGetLastError(), "attention launch") &&
        check_cuda(cudaDeviceSynchronize(), "attention synchronize");
    cudaFree(valid_keys);
    if (!synchronized || !logits.guards_intact("logits") ||
        !output.guards_intact("output")) {
        return false;
    }
    current.logits.assign(logits.data(), logits.data() + logits_count);
    current.output.assign(output.data(), output.data() + query_count);
    *result = std::move(current);
    return true;
}

bool byte_exact(const std::vector<__half>& lhs,
                const std::vector<__half>& rhs) {
    return lhs.size() == rhs.size() &&
           std::memcmp(lhs.data(), rhs.data(),
                       lhs.size() * sizeof(__half)) == 0;
}

bool masked_rows_are_zero(const Result& result) {
    const int rows = result.shape.query_rows * result.shape.heads;
    for (int row = 0; row < rows; ++row) {
        for (int key = result.shape.valid_keys;
             key < result.logits_stride; ++key) {
            if (__half2float(
                    result.logits[static_cast<std::size_t>(row) *
                                      result.logits_stride +
                                  key]) != 0.0f) {
                std::fprintf(stderr,
                             "masked probability is nonzero at row=%d key=%d\n",
                             row, key);
                return false;
            }
        }
    }
    return true;
}

std::vector<__half> reference_output(const Result& result) {
    const Shape& shape = result.shape;
    const int rows = shape.query_rows * shape.heads;
    const float scale =
        1.0f / std::sqrt(static_cast<float>(shape.head_dimension));
    std::vector<__half> expected(
        static_cast<std::size_t>(rows) * shape.head_dimension);
    std::vector<float> logits(static_cast<std::size_t>(shape.valid_keys));
    std::vector<__half> probabilities(logits.size());
    for (int row = 0; row < rows; ++row) {
        float maximum = -INFINITY;
        for (int key = 0; key < shape.valid_keys; ++key) {
            float dot = 0.0f;
            for (int column = 0; column < shape.head_dimension; ++column) {
                dot += __half2float(
                           result.queries[static_cast<std::size_t>(row) *
                                              shape.head_dimension +
                                          column]) *
                       __half2float(
                           result.keys[static_cast<std::size_t>(key) *
                                           shape.head_dimension +
                                       column]);
            }
            logits[static_cast<std::size_t>(key)] =
                __half2float(__float2half_rn(dot * scale));
            maximum = std::max(maximum, logits[static_cast<std::size_t>(key)]);
        }
        float denominator = 0.0f;
        for (float logit : logits) denominator += std::exp(logit - maximum);
        for (int key = 0; key < shape.valid_keys; ++key) {
            probabilities[static_cast<std::size_t>(key)] = __float2half_rn(
                std::exp(logits[static_cast<std::size_t>(key)] - maximum) /
                denominator);
        }
        for (int column = 0; column < shape.head_dimension; ++column) {
            float value = 0.0f;
            for (int key = 0; key < shape.valid_keys; ++key) {
                value += __half2float(
                             result.values[static_cast<std::size_t>(key) *
                                               shape.head_dimension +
                                           column]) *
                         __half2float(
                             probabilities[static_cast<std::size_t>(key)]);
            }
            expected[static_cast<std::size_t>(row) * shape.head_dimension +
                     column] = __float2half_rn(value);
        }
    }
    return expected;
}

bool within_fp16_bound(const Result& result) {
    const std::vector<__half> expected = reference_output(result);
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    float max_abs = 0.0f;
    for (std::size_t i = 0; i < result.output.size(); ++i) {
        const float actual = __half2float(result.output[i]);
        const float reference = __half2float(expected[i]);
        if (!std::isfinite(actual)) return false;
        max_abs = std::max(max_abs, std::abs(actual - reference));
        dot += static_cast<double>(actual) * reference;
        actual_norm += static_cast<double>(actual) * actual;
        expected_norm += static_cast<double>(reference) * reference;
    }
    const double cosine =
        dot / std::sqrt(std::max(actual_norm * expected_norm, 1e-30));
    if (max_abs > 0.001953125f || cosine < 0.9999) {
        std::fprintf(stderr,
                     "attention reference mismatch: max_abs=%g cosine=%.9f\n",
                     max_abs, cosine);
        return false;
    }
    return true;
}

bool test_even_compatibility(cublasHandle_t handle) {
    const Shape shape{2, 8, 2, 16, 8};
    Result legacy;
    Result seqused;
    const bool passed = run_attention(handle, shape, false, &legacy) &&
                        run_attention(handle, shape, true, &seqused) &&
                        byte_exact(legacy.logits, seqused.logits) &&
                        byte_exact(legacy.output, seqused.output);
    if (!passed) std::fprintf(stderr, "even compatibility failed\n");
    return passed;
}

bool test_odd_shapes(cublasHandle_t handle) {
    Result minimum;
    Result zero_valid;
    Result first;
    Result second;
    const Shape realistic{3, 259, 8, 256, 256};
    if (!run_attention(handle, Shape{1, 1, 1, 4, 1}, true, &minimum) ||
        !masked_rows_are_zero(minimum) || !within_fp16_bound(minimum) ||
        !run_attention(handle, Shape{1, 1, 1, 4, 0}, true, &zero_valid) ||
        zero_valid.logits.size() != 2 ||
        __half2float(zero_valid.logits[0]) !=
            __half2float(zero_valid.logits[1]) ||
        !run_attention(handle, realistic, true, &first) ||
        !run_attention(handle, realistic, true, &second) ||
        !masked_rows_are_zero(first) || !within_fp16_bound(first)) {
        return false;
    }
    for (__half value : zero_valid.output) {
        if (!std::isfinite(__half2float(value))) return false;
    }
    const bool deterministic = byte_exact(first.logits, second.logits) &&
                               byte_exact(first.output, second.output);
    if (!deterministic) std::fprintf(stderr, "odd replay is not exact\n");
    return deterministic;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        cudaGetLastError();
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 1;
    const bool passed =
        test_even_compatibility(handle) && test_odd_shapes(handle);
    const bool destroyed = cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS;
    return passed && destroyed ? 0 : 1;
}
