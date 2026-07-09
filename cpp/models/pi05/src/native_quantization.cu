#include "flashrt/cpp/models/pi05/native_quantization.h"

#include <cuda_fp8.h>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <utility>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

bool valid_matrix(const NativeFloatTensor& tensor) {
    if (tensor.shape.size() != 2 || !tensor.shape[0] || !tensor.shape[1]) {
        return false;
    }
    const std::uint64_t rows = tensor.shape[0];
    const std::uint64_t columns = tensor.shape[1];
    return rows <= SIZE_MAX / columns &&
           rows * columns == tensor.values.size();
}

bool finite_values(const NativeFloatTensor& tensor) {
    for (float value : tensor.values) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

}  // namespace

modalities::Status native_quantize_fp8_e4m3(
    const NativeFloatTensor& bf16_weight,
    bool transpose,
    NativeFp8Tensor* out) {
    if (!out || !valid_matrix(bf16_weight) || !finite_values(bf16_weight)) {
        return invalid("FP8 weight must be a finite BF16 matrix");
    }
    NativeFloatTensor arranged;
    if (transpose) {
        const modalities::Status st =
            native_transpose_2d(bf16_weight, &arranged);
        if (!st.ok_status()) return st;
    } else {
        arranged = bf16_weight;
    }
    float amax = 0.0f;
    for (float value : arranged.values) {
        amax = std::max(amax, std::fabs(value));
    }
    NativeFp8Tensor result;
    result.shape = arranged.shape;
    result.scale = std::max(amax / 448.0f, 1.0e-12f);
    result.values.resize(arranged.values.size());
    for (std::size_t i = 0; i < arranged.values.size(); ++i) {
        const float value = std::max(
            -448.0f,
            std::min(448.0f, arranged.values[i] / result.scale));
        result.values[i] = __nv_fp8_e4m3(value).__x;
    }
    *out = std::move(result);
    return modalities::Status::ok();
}

modalities::Status native_quantize_int8_per_output(
    const NativeFloatTensor& bf16_weight,
    NativeInt8Tensor* out) {
    if (!out || !valid_matrix(bf16_weight) || !finite_values(bf16_weight)) {
        return invalid("INT8 weight must be a finite BF16 matrix");
    }
    NativeFloatTensor transposed;
    modalities::Status st = native_transpose_2d(bf16_weight, &transposed);
    if (!st.ok_status()) return st;
    const std::size_t rows = static_cast<std::size_t>(transposed.shape[0]);
    const std::size_t columns =
        static_cast<std::size_t>(transposed.shape[1]);
    NativeInt8Tensor result;
    result.shape = transposed.shape;
    result.values.resize(transposed.values.size());
    result.scales.resize(rows);
    const float inv_int8_max = 1.0f / 127.0f;
    for (std::size_t row = 0; row < rows; ++row) {
        float amax = 0.0f;
        for (std::size_t column = 0; column < columns; ++column) {
            amax = std::max(
                amax, std::fabs(transposed.values[row * columns + column]));
        }
        const float scale = std::max(amax * inv_int8_max, 1.0e-12f);
        result.scales[row] = scale;
        for (std::size_t column = 0; column < columns; ++column) {
            const float scaled =
                transposed.values[row * columns + column] / scale;
            const float rounded = std::nearbyint(scaled);
            result.values[row * columns + column] = static_cast<std::int8_t>(
                std::max(-127.0f, std::min(127.0f, rounded)));
        }
    }
    *out = std::move(result);
    return modalities::Status::ok();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt
