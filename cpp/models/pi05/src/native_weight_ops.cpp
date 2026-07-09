#include "flashrt/cpp/models/pi05/native_weight_ops.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

bool element_count(const std::vector<std::uint64_t>& shape,
                   std::size_t* out) {
    std::size_t count = 1;
    for (std::uint64_t dim : shape) {
        if (dim > std::numeric_limits<std::size_t>::max() ||
            (dim && count > std::numeric_limits<std::size_t>::max() /
                                static_cast<std::size_t>(dim))) {
            return false;
        }
        count *= static_cast<std::size_t>(dim);
    }
    if (out) *out = count;
    return true;
}

bool valid_tensor(const NativeFloatTensor& tensor) {
    std::size_t expected = 0;
    return element_count(tensor.shape, &expected) &&
           expected == tensor.values.size();
}

const loader::SafetensorInfo* find_source_tensor(
    const loader::SafetensorsFile& file,
    const std::string& key) {
    const loader::SafetensorInfo* tensor = file.find(key);
    if (!tensor) tensor = file.find(std::string("model.") + key);
    return tensor;
}

}  // namespace

modalities::Status load_native_float_tensor(
    const loader::SafetensorsFile& file,
    const std::string& key,
    NativeFloatTensor* out) {
    if (!file.is_open() || !out) return invalid("invalid native tensor load");
    const loader::SafetensorInfo* tensor = find_source_tensor(file, key);
    if (!tensor) {
        return modalities::Status::error(modalities::StatusCode::kNotFound,
                                         "native tensor not found: " + key);
    }
    std::size_t count = 0;
    if (!element_count(tensor->shape, &count)) {
        return invalid("native tensor shape overflows size_t");
    }
    const void* data = file.data(*tensor);
    if (!data && tensor->bytes) return invalid("native tensor has no payload");

    NativeFloatTensor loaded;
    loaded.shape = tensor->shape;
    loaded.values.resize(count);
    if (tensor->dtype == "F32") {
        std::memcpy(loaded.values.data(), data, count * sizeof(float));
    } else if (tensor->dtype == "BF16") {
        const auto* src = static_cast<const std::uint16_t*>(data);
        for (std::size_t i = 0; i < count; ++i) {
            loaded.values[i] = modalities::bfloat16_to_float(src[i]);
        }
    } else if (tensor->dtype == "F16") {
        const auto* src = static_cast<const std::uint16_t*>(data);
        for (std::size_t i = 0; i < count; ++i) {
            loaded.values[i] = modalities::float16_to_float(src[i]);
        }
    } else {
        return modalities::Status::error(
            modalities::StatusCode::kUnsupported,
            "native tensor dtype is not a floating-point weight: " +
                tensor->dtype);
    }
    *out = std::move(loaded);
    return modalities::Status::ok();
}

modalities::Status native_to_bf16(const NativeFloatTensor& input,
                                  NativeBf16Tensor* out) {
    if (!out || !valid_tensor(input)) return invalid("invalid BF16 input");
    NativeBf16Tensor converted;
    converted.shape = input.shape;
    converted.values.resize(input.values.size());
    for (std::size_t i = 0; i < input.values.size(); ++i) {
        converted.values[i] = modalities::float_to_bfloat16(input.values[i]);
    }
    *out = std::move(converted);
    return modalities::Status::ok();
}

modalities::Status native_round_to_bf16_float(
    const NativeFloatTensor& input,
    NativeFloatTensor* out) {
    if (!out || !valid_tensor(input)) {
        return invalid("invalid BF16 round-trip input");
    }
    NativeFloatTensor rounded = input;
    for (float& value : rounded.values) {
        value = modalities::bfloat16_to_float(
            modalities::float_to_bfloat16(value));
    }
    *out = std::move(rounded);
    return modalities::Status::ok();
}

modalities::Status native_transpose_2d(const NativeFloatTensor& input,
                                       NativeFloatTensor* out) {
    if (!out || !valid_tensor(input) || input.shape.size() != 2) {
        return invalid("transpose requires a valid rank-2 tensor");
    }
    const std::size_t rows = static_cast<std::size_t>(input.shape[0]);
    const std::size_t cols = static_cast<std::size_t>(input.shape[1]);
    NativeFloatTensor transposed;
    transposed.shape = {input.shape[1], input.shape[0]};
    transposed.values.resize(input.values.size());
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            transposed.values[col * rows + row] =
                input.values[row * cols + col];
        }
    }
    *out = std::move(transposed);
    return modalities::Status::ok();
}

modalities::Status native_patch_oihw_to_hwio(
    const NativeFloatTensor& input,
    NativeFloatTensor* out) {
    if (!out || !valid_tensor(input) || input.shape.size() != 4) {
        return invalid("patch permutation requires a valid rank-4 tensor");
    }
    const std::size_t outputs = static_cast<std::size_t>(input.shape[0]);
    const std::size_t channels = static_cast<std::size_t>(input.shape[1]);
    const std::size_t height = static_cast<std::size_t>(input.shape[2]);
    const std::size_t width = static_cast<std::size_t>(input.shape[3]);
    NativeFloatTensor permuted;
    permuted.shape = {input.shape[2], input.shape[3], input.shape[1],
                      input.shape[0]};
    permuted.values.resize(input.values.size());
    for (std::size_t o = 0; o < outputs; ++o) {
        for (std::size_t c = 0; c < channels; ++c) {
            for (std::size_t h = 0; h < height; ++h) {
                for (std::size_t w = 0; w < width; ++w) {
                    const std::size_t src =
                        ((o * channels + c) * height + h) * width + w;
                    const std::size_t dst =
                        ((h * width + w) * channels + c) * outputs + o;
                    permuted.values[dst] = input.values[src];
                }
            }
        }
    }
    *out = std::move(permuted);
    return modalities::Status::ok();
}

modalities::Status native_interleave_qk_rows(
    const NativeFloatTensor& input,
    std::uint64_t num_heads,
    NativeFloatTensor* out) {
    if (!out || !valid_tensor(input) || input.shape.size() != 2 ||
        !num_heads || input.shape[0] % num_heads != 0) {
        return invalid("Q/K interleave requires divisible rank-2 rows");
    }
    const std::uint64_t head_dim = input.shape[0] / num_heads;
    if (head_dim % 2 != 0) {
        return invalid("Q/K interleave requires an even head dimension");
    }
    const std::size_t cols = static_cast<std::size_t>(input.shape[1]);
    NativeFloatTensor interleaved;
    interleaved.shape = input.shape;
    interleaved.values.resize(input.values.size());
    for (std::uint64_t head = 0; head < num_heads; ++head) {
        for (std::uint64_t pair = 0; pair < head_dim / 2; ++pair) {
            for (std::uint64_t half = 0; half < 2; ++half) {
                const std::uint64_t src_row =
                    head * head_dim + half * (head_dim / 2) + pair;
                const std::uint64_t dst_row =
                    head * head_dim + pair * 2 + half;
                std::memcpy(interleaved.values.data() + dst_row * cols,
                            input.values.data() + src_row * cols,
                            cols * sizeof(float));
            }
        }
    }
    *out = std::move(interleaved);
    return modalities::Status::ok();
}

modalities::Status native_fold_rms_columns(
    const NativeFloatTensor& weight,
    const NativeFloatTensor& norm,
    NativeFloatTensor* out) {
    if (!out || !valid_tensor(weight) || !valid_tensor(norm) ||
        weight.shape.size() != 2 || norm.shape.size() != 1 ||
        weight.shape[1] != norm.shape[0]) {
        return invalid("RMS fold requires weight[out,in] and norm[in]");
    }
    NativeFloatTensor folded = weight;
    const std::size_t rows = static_cast<std::size_t>(weight.shape[0]);
    const std::size_t cols = static_cast<std::size_t>(weight.shape[1]);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            folded.values[row * cols + col] *= 1.0f + norm.values[col];
        }
    }
    *out = std::move(folded);
    return modalities::Status::ok();
}

modalities::Status native_concat_rows_transpose(
    const std::vector<const NativeFloatTensor*>& inputs,
    NativeFloatTensor* out) {
    if (!out || inputs.empty() || !inputs[0] ||
        !valid_tensor(*inputs[0]) || inputs[0]->shape.size() != 2) {
        return invalid("row concat requires rank-2 tensors");
    }
    const std::uint64_t cols = inputs[0]->shape[1];
    std::uint64_t total_rows = 0;
    for (const NativeFloatTensor* input : inputs) {
        if (!input || !valid_tensor(*input) || input->shape.size() != 2 ||
            input->shape[1] != cols ||
            total_rows > std::numeric_limits<std::uint64_t>::max() -
                             input->shape[0]) {
            return invalid("row concat tensors have incompatible shapes");
        }
        total_rows += input->shape[0];
    }
    NativeFloatTensor joined;
    joined.shape = {cols, total_rows};
    std::size_t joined_count = 0;
    if (!element_count(joined.shape, &joined_count)) {
        return invalid("row concat output shape overflows size_t");
    }
    joined.values.resize(joined_count);
    std::uint64_t row_offset = 0;
    for (const NativeFloatTensor* input : inputs) {
        for (std::uint64_t row = 0; row < input->shape[0]; ++row) {
            for (std::uint64_t col = 0; col < cols; ++col) {
                joined.values[static_cast<std::size_t>(col * total_rows +
                                                       row_offset + row)] =
                    input->values[static_cast<std::size_t>(row * cols + col)];
            }
        }
        row_offset += input->shape[0];
    }
    *out = std::move(joined);
    return modalities::Status::ok();
}

modalities::Status native_concat_columns(
    const NativeFloatTensor& left,
    const NativeFloatTensor& right,
    NativeFloatTensor* out) {
    if (!out || !valid_tensor(left) || !valid_tensor(right) ||
        left.shape.size() != 2 || right.shape.size() != 2 ||
        left.shape[0] != right.shape[0]) {
        return invalid("column concat tensors have incompatible shapes");
    }
    const std::size_t rows = static_cast<std::size_t>(left.shape[0]);
    const std::size_t left_cols = static_cast<std::size_t>(left.shape[1]);
    const std::size_t right_cols = static_cast<std::size_t>(right.shape[1]);
    if (left.shape[1] > std::numeric_limits<std::uint64_t>::max() -
                            right.shape[1]) {
        return invalid("column concat output shape overflows uint64");
    }
    NativeFloatTensor joined;
    joined.shape = {left.shape[0], left.shape[1] + right.shape[1]};
    std::size_t joined_count = 0;
    if (!element_count(joined.shape, &joined_count)) {
        return invalid("column concat output shape overflows size_t");
    }
    joined.values.resize(joined_count);
    for (std::size_t row = 0; row < rows; ++row) {
        float* dst = joined.values.data() + row * (left_cols + right_cols);
        std::memcpy(dst, left.values.data() + row * left_cols,
                    left_cols * sizeof(float));
        std::memcpy(dst + left_cols,
                    right.values.data() + row * right_cols,
                    right_cols * sizeof(float));
    }
    *out = std::move(joined);
    return modalities::Status::ok();
}

modalities::Status native_concat_vectors(
    const std::vector<const NativeFloatTensor*>& inputs,
    NativeFloatTensor* out) {
    if (!out || inputs.empty()) return invalid("vector concat has no inputs");
    std::size_t total = 0;
    for (const NativeFloatTensor* input : inputs) {
        if (!input || !valid_tensor(*input) || input->shape.size() != 1 ||
            input->values.size() >
                std::numeric_limits<std::size_t>::max() - total) {
            return invalid("vector concat tensors have incompatible shapes");
        }
        total += input->values.size();
    }
    NativeFloatTensor joined;
    joined.shape = {static_cast<std::uint64_t>(total)};
    joined.values.reserve(total);
    for (const NativeFloatTensor* input : inputs) {
        joined.values.insert(joined.values.end(), input->values.begin(),
                             input->values.end());
    }
    *out = std::move(joined);
    return modalities::Status::ok();
}

modalities::Status native_scale(const NativeFloatTensor& input,
                                float scale,
                                NativeFloatTensor* out) {
    if (!out || !valid_tensor(input)) return invalid("invalid scale input");
    NativeFloatTensor scaled = input;
    for (float& value : scaled.values) value *= scale;
    *out = std::move(scaled);
    return modalities::Status::ok();
}

modalities::Status native_pi05_time_embeddings(
    int num_steps,
    std::uint64_t embedding_dim,
    NativeFloatTensor* out) {
    if (!out || num_steps <= 0 || embedding_dim < 2 ||
        embedding_dim % 2 != 0) {
        return invalid("Pi0.5 time embedding shape is invalid");
    }
    const std::uint64_t half = embedding_dim / 2;
    NativeFloatTensor result;
    result.shape = {static_cast<std::uint64_t>(num_steps), embedding_dim};
    result.values.resize(static_cast<std::size_t>(num_steps) * embedding_dim);
    const float dt = -1.0f / static_cast<float>(num_steps);
    const float min_period = 4.0e-3f;
    const float period_ratio = 1000.0f;
    const float pi = static_cast<float>(3.14159265358979323846);
    const float fraction_step =
        half == 1 ? 0.0f : 1.0f / static_cast<float>(half - 1);
    float t = 1.0f;
    for (int step = 0; step < num_steps; ++step) {
        const std::size_t row = static_cast<std::size_t>(step) * embedding_dim;
        for (std::uint64_t i = 0; i < half; ++i) {
            const float fraction = static_cast<float>(i) * fraction_step;
            const float period =
                min_period * std::pow(period_ratio, fraction);
            float angle = t * (1.0f / period);
            angle *= 2.0f;
            angle *= pi;
            result.values[row + i] = std::sin(angle);
            result.values[row + half + i] = std::cos(angle);
        }
        t += dt;
    }
    *out = std::move(result);
    return modalities::Status::ok();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt
