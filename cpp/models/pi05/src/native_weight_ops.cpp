#include "flashrt/cpp/models/pi05/native_weight_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
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

template <typename Fn>
void parallel_ranges(std::size_t count,
                     std::size_t minimum_items,
                     const Fn& fn) {
    if (!count) return;
    const unsigned available = std::thread::hardware_concurrency();
    const std::size_t wanted =
        (count + minimum_items - 1) / minimum_items;
    const std::size_t workers = std::max<std::size_t>(
        1, std::min<std::size_t>({32, available ? available : 1, wanted}));
    if (workers == 1) {
        fn(0, count);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    for (std::size_t worker = 1; worker < workers; ++worker) {
        const std::size_t begin = count * worker / workers;
        const std::size_t end = count * (worker + 1) / workers;
        threads.emplace_back([begin, end, &fn] { fn(begin, end); });
    }
    fn(0, count / workers);
    for (std::thread& thread : threads) thread.join();
}

template <typename T, typename Fn>
void tiled_transform_transpose(std::size_t rows,
                               std::size_t cols,
                               std::size_t output_rows,
                               std::size_t output_row_offset,
                               T* output,
                               const Fn& transform) {
    constexpr std::size_t kTile = 32;
    const std::size_t row_tiles = (rows + kTile - 1) / kTile;
    parallel_ranges(row_tiles, 2, [&](std::size_t first,
                                      std::size_t last) {
        for (std::size_t tile = first; tile < last; ++tile) {
            const std::size_t row_begin = tile * kTile;
            const std::size_t row_end = std::min(rows, row_begin + kTile);
            T scratch[kTile * kTile];
            for (std::size_t col_begin = 0; col_begin < cols;
                 col_begin += kTile) {
                const std::size_t col_end =
                    std::min(cols, col_begin + kTile);
                for (std::size_t row = row_begin; row < row_end; ++row) {
                    for (std::size_t col = col_begin; col < col_end; ++col) {
                        scratch[(row - row_begin) * kTile +
                                (col - col_begin)] = transform(row, col);
                    }
                }
                for (std::size_t col = col_begin; col < col_end; ++col) {
                    T* destination = output + col * output_rows +
                                     output_row_offset + row_begin;
                    for (std::size_t row = row_begin; row < row_end; ++row) {
                        destination[row - row_begin] =
                            scratch[(row - row_begin) * kTile +
                                    (col - col_begin)];
                    }
                }
            }
        }
    });
}

const loader::SafetensorInfo* find_source_tensor(
    const loader::SafetensorsFile& file,
    const std::string& key) {
    const loader::SafetensorInfo* tensor = file.find(key);
    if (!tensor) tensor = file.find(std::string("model.") + key);
    return tensor;
}

struct F32Reader {
    const float* data;
    float operator[](std::size_t index) const { return data[index]; }
    std::uint16_t bf16(std::size_t index) const {
        return modalities::float_to_bfloat16(data[index]);
    }
};

struct Bf16Reader {
    const std::uint16_t* data;
    float operator[](std::size_t index) const {
        return modalities::bfloat16_to_float(data[index]);
    }
    std::uint16_t bf16(std::size_t index) const { return data[index]; }
};

struct F16Reader {
    const std::uint16_t* data;
    float operator[](std::size_t index) const {
        return modalities::float16_to_float(data[index]);
    }
    std::uint16_t bf16(std::size_t index) const {
        return modalities::float_to_bfloat16(
            modalities::float16_to_float(data[index]));
    }
};

struct UnalignedF32Reader {
    const unsigned char* data;
    float operator[](std::size_t index) const {
        float value = 0.0f;
        std::memcpy(&value, data + index * sizeof(value), sizeof(value));
        return value;
    }
    std::uint16_t bf16(std::size_t index) const {
        return modalities::float_to_bfloat16((*this)[index]);
    }
};

template <bool IsBf16>
struct UnalignedU16Reader {
    const unsigned char* data;
    std::uint16_t bits(std::size_t index) const {
        std::uint16_t value = 0;
        std::memcpy(&value, data + index * sizeof(value), sizeof(value));
        return value;
    }
    float operator[](std::size_t index) const {
        return IsBf16 ? modalities::bfloat16_to_float(bits(index))
                      : modalities::float16_to_float(bits(index));
    }
    std::uint16_t bf16(std::size_t index) const {
        return IsBf16 ? bits(index)
                      : modalities::float_to_bfloat16((*this)[index]);
    }
};

template <typename Fn>
modalities::Status dispatch_source(const NativeSourceTensorView& source,
                                   const Fn& fn) {
    if (!source.data) return invalid("native source tensor has no payload");
    switch (source.dtype) {
        case NativeSourceDType::kF32:
            if (reinterpret_cast<std::uintptr_t>(source.data) %
                    alignof(float) != 0) {
                return fn(UnalignedF32Reader{
                    static_cast<const unsigned char*>(source.data)});
            }
            return fn(F32Reader{static_cast<const float*>(source.data)});
        case NativeSourceDType::kBf16:
            if (reinterpret_cast<std::uintptr_t>(source.data) %
                    alignof(std::uint16_t) != 0) {
                return fn(UnalignedU16Reader<true>{
                    static_cast<const unsigned char*>(source.data)});
            }
            return fn(Bf16Reader{
                static_cast<const std::uint16_t*>(source.data)});
        case NativeSourceDType::kF16:
            if (reinterpret_cast<std::uintptr_t>(source.data) %
                    alignof(std::uint16_t) != 0) {
                return fn(UnalignedU16Reader<false>{
                    static_cast<const unsigned char*>(source.data)});
            }
            return fn(F16Reader{
                static_cast<const std::uint16_t*>(source.data)});
    }
    return invalid("native source tensor dtype is invalid");
}

std::size_t interleaved_source_row(std::size_t output_row,
                                   std::size_t rows,
                                   std::size_t heads) {
    const std::size_t head_dim = rows / heads;
    const std::size_t head = output_row / head_dim;
    const std::size_t within = output_row % head_dim;
    const std::size_t pair = within / 2;
    const std::size_t half = within % 2;
    return head * head_dim + half * (head_dim / 2) + pair;
}

}  // namespace

modalities::Status load_native_source_tensor(
    const loader::SafetensorsFile& file,
    const std::string& key,
    NativeSourceTensorView* out) {
    if (!file.is_open() || !out) return invalid("invalid native tensor view");
    const loader::SafetensorInfo* tensor = find_source_tensor(file, key);
    if (!tensor) {
        return modalities::Status::error(modalities::StatusCode::kNotFound,
                                         "native tensor not found: " + key);
    }
    NativeSourceDType dtype;
    if (tensor->dtype == "F32") {
        dtype = NativeSourceDType::kF32;
    } else if (tensor->dtype == "BF16") {
        dtype = NativeSourceDType::kBf16;
    } else if (tensor->dtype == "F16") {
        dtype = NativeSourceDType::kF16;
    } else {
        return modalities::Status::error(
            modalities::StatusCode::kUnsupported,
            "native tensor dtype is not a floating-point weight: " +
                tensor->dtype);
    }
    std::size_t count = 0;
    if (!element_count(tensor->shape, &count)) {
        return invalid("native tensor shape overflows size_t");
    }
    const void* data = file.data(*tensor);
    if (!data && tensor->bytes) return invalid("native tensor payload is missing");
    *out = NativeSourceTensorView{data, tensor->shape, dtype};
    return modalities::Status::ok();
}

modalities::Status native_source_to_bf16(
    const NativeSourceTensorView& input,
    bool transpose,
    NativeBf16Tensor* out) {
    std::size_t count = 0;
    if (!out || !element_count(input.shape, &count) ||
        (transpose && input.shape.size() != 2)) {
        return invalid("invalid direct source to BF16 input");
    }
    NativeBf16Tensor converted;
    converted.shape = transpose
        ? std::vector<std::uint64_t>{input.shape[1], input.shape[0]}
        : input.shape;
    converted.values.resize(count);
    modalities::Status st = dispatch_source(input, [&](const auto& reader) {
        if (!transpose) {
            parallel_ranges(count, 1 << 18,
                            [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) {
                    converted.values[i] = reader.bf16(i);
                }
            });
        } else {
            const std::size_t rows = static_cast<std::size_t>(input.shape[0]);
            const std::size_t cols = static_cast<std::size_t>(input.shape[1]);
            tiled_transform_transpose(
                rows, cols, rows, 0, converted.values.data(),
                [&](std::size_t row, std::size_t col) {
                    return reader.bf16(row * cols + col);
                });
        }
        return modalities::Status::ok();
    });
    if (!st.ok_status()) return st;
    *out = std::move(converted);
    return modalities::Status::ok();
}

modalities::Status native_source_fold_rms_columns_transpose(
    const NativeSourceTensorView& weight,
    const NativeFloatTensor& norm,
    NativeBf16Tensor* out) {
    std::size_t count = 0;
    if (!out || weight.shape.size() != 2 ||
        !element_count(weight.shape, &count) || !valid_tensor(norm) ||
        norm.shape.size() != 1 || weight.shape[1] != norm.shape[0]) {
        return invalid("invalid direct source RMS fold input");
    }
    const std::size_t rows = static_cast<std::size_t>(weight.shape[0]);
    const std::size_t cols = static_cast<std::size_t>(weight.shape[1]);
    NativeBf16Tensor folded;
    folded.shape = {weight.shape[1], weight.shape[0]};
    folded.values.resize(count);
    modalities::Status st = dispatch_source(weight, [&](const auto& reader) {
        tiled_transform_transpose(
            rows, cols, rows, 0, folded.values.data(),
            [&](std::size_t row, std::size_t col) {
                return modalities::float_to_bfloat16(
                    reader[row * cols + col] * (1.0f + norm.values[col]));
            });
        return modalities::Status::ok();
    });
    if (!st.ok_status()) return st;
    *out = std::move(folded);
    return modalities::Status::ok();
}

modalities::Status native_source_round_scale_to_bf16(
    const NativeSourceTensorView& input,
    float scale,
    bool transpose,
    NativeBf16Tensor* out) {
    std::size_t count = 0;
    if (!out || !element_count(input.shape, &count) ||
        (transpose && input.shape.size() != 2)) {
        return invalid("invalid direct source scale input");
    }
    NativeBf16Tensor scaled;
    scaled.shape = transpose
        ? std::vector<std::uint64_t>{input.shape[1], input.shape[0]}
        : input.shape;
    scaled.values.resize(count);
    modalities::Status st = dispatch_source(input, [&](const auto& reader) {
        const auto convert = [&](std::size_t index) {
            const float rounded = modalities::bfloat16_to_float(
                modalities::float_to_bfloat16(reader[index]));
            return modalities::float_to_bfloat16(rounded * scale);
        };
        if (!transpose) {
            parallel_ranges(count, 1 << 18,
                            [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) {
                    scaled.values[i] = convert(i);
                }
            });
        } else {
            const std::size_t rows = static_cast<std::size_t>(input.shape[0]);
            const std::size_t cols = static_cast<std::size_t>(input.shape[1]);
            tiled_transform_transpose(
                rows, cols, rows, 0, scaled.values.data(),
                [&](std::size_t row, std::size_t col) {
                    return convert(row * cols + col);
                });
        }
        return modalities::Status::ok();
    });
    if (!st.ok_status()) return st;
    *out = std::move(scaled);
    return modalities::Status::ok();
}

modalities::Status native_source_qkv_to_bf16(
    const NativeSourceTensorView& q,
    const NativeSourceTensorView& k,
    const NativeSourceTensorView& v,
    std::uint64_t q_heads,
    std::uint64_t k_heads,
    const NativeFloatTensor* norm,
    NativeBf16Tensor* out) {
    std::size_t q_count = 0;
    std::size_t k_count = 0;
    std::size_t v_count = 0;
    if (!out || q.shape.size() != 2 || k.shape.size() != 2 ||
        v.shape.size() != 2 || q.shape[1] != k.shape[1] ||
        q.shape[1] != v.shape[1] ||
        !element_count(q.shape, &q_count) ||
        !element_count(k.shape, &k_count) ||
        !element_count(v.shape, &v_count) ||
        q.shape[0] > std::numeric_limits<std::uint64_t>::max() - k.shape[0] ||
        q.shape[0] + k.shape[0] >
            std::numeric_limits<std::uint64_t>::max() - v.shape[0] ||
        (q_heads && (q.shape[0] % q_heads ||
                     (q.shape[0] / q_heads) % 2)) ||
        (k_heads && (k.shape[0] % k_heads ||
                     (k.shape[0] / k_heads) % 2)) ||
        (norm && (!valid_tensor(*norm) || norm->shape.size() != 1 ||
                  norm->shape[0] != q.shape[1]))) {
        return invalid("QKV source tensors have incompatible shapes");
    }
    const std::size_t cols = static_cast<std::size_t>(q.shape[1]);
    const std::uint64_t total_rows_u64 =
        q.shape[0] + k.shape[0] + v.shape[0];
    if (total_rows_u64 > std::numeric_limits<std::size_t>::max() ||
        q.shape[1] > std::numeric_limits<std::size_t>::max()) {
        return invalid("QKV output shape overflows size_t");
    }
    const std::size_t total_rows = static_cast<std::size_t>(total_rows_u64);
    NativeBf16Tensor joined;
    joined.shape = {q.shape[1], total_rows_u64};
    std::size_t joined_count = 0;
    if (!element_count(joined.shape, &joined_count)) {
        return invalid("QKV output shape overflows size_t");
    }
    joined.values.resize(joined_count);
    const auto write = [&](const NativeSourceTensorView& source,
                           std::size_t heads, bool interleave,
                           std::size_t offset) {
        return dispatch_source(source, [&](const auto& reader) {
            const std::size_t rows =
                static_cast<std::size_t>(source.shape[0]);
            tiled_transform_transpose(
                rows, cols, total_rows, offset, joined.values.data(),
                [&](std::size_t output_row, std::size_t col) {
                    const std::size_t source_row = interleave
                        ? interleaved_source_row(output_row, rows, heads)
                        : output_row;
                    const std::size_t index = source_row * cols + col;
                    if (!norm) return reader.bf16(index);
                    return modalities::float_to_bfloat16(
                        reader[index] * (1.0f + norm->values[col]));
                });
            return modalities::Status::ok();
        });
    };
    modalities::Status st = write(q, q_heads, q_heads != 0, 0);
    if (!st.ok_status()) return st;
    st = write(k, k_heads, k_heads != 0,
               static_cast<std::size_t>(q.shape[0]));
    if (!st.ok_status()) return st;
    st = write(v, 1, false,
               static_cast<std::size_t>(q.shape[0] + k.shape[0]));
    if (!st.ok_status()) return st;
    *out = std::move(joined);
    return modalities::Status::ok();
}

modalities::Status native_source_concat_vectors_to_bf16(
    const std::vector<const NativeSourceTensorView*>& inputs,
    NativeBf16Tensor* out) {
    if (!out || inputs.empty()) return invalid("vector concat has no inputs");
    std::size_t total = 0;
    for (const NativeSourceTensorView* input : inputs) {
        if (!input || input->shape.size() != 1 ||
            input->shape[0] > std::numeric_limits<std::size_t>::max() ||
            input->shape[0] > std::numeric_limits<std::size_t>::max() - total) {
            return invalid("vector concat tensors have incompatible shapes");
        }
        total += static_cast<std::size_t>(input->shape[0]);
    }
    NativeBf16Tensor joined;
    joined.shape = {static_cast<std::uint64_t>(total)};
    joined.values.resize(total);
    std::size_t offset = 0;
    for (const NativeSourceTensorView* input : inputs) {
        const std::size_t count = static_cast<std::size_t>(input->shape[0]);
        modalities::Status st = dispatch_source(*input, [&](const auto& reader) {
            parallel_ranges(count, 1 << 18,
                            [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) {
                    joined.values[offset + i] = reader.bf16(i);
                }
            });
            return modalities::Status::ok();
        });
        if (!st.ok_status()) return st;
        offset += count;
    }
    *out = std::move(joined);
    return modalities::Status::ok();
}

modalities::Status native_source_patch_oihw_to_hwio_bf16(
    const NativeSourceTensorView& input,
    NativeBf16Tensor* out) {
    std::size_t count = 0;
    if (!out || input.shape.size() != 4 ||
        !element_count(input.shape, &count)) {
        return invalid("patch permutation requires a rank-4 tensor");
    }
    const std::size_t outputs = static_cast<std::size_t>(input.shape[0]);
    const std::size_t channels = static_cast<std::size_t>(input.shape[1]);
    const std::size_t height = static_cast<std::size_t>(input.shape[2]);
    const std::size_t width = static_cast<std::size_t>(input.shape[3]);
    NativeBf16Tensor result;
    result.shape = {input.shape[2], input.shape[3], input.shape[1],
                    input.shape[0]};
    result.values.resize(count);
    modalities::Status st = dispatch_source(input, [&](const auto& reader) {
        parallel_ranges(height * width * channels, 32,
                        [&](std::size_t begin, std::size_t end) {
            for (std::size_t hwc = begin; hwc < end; ++hwc) {
                const std::size_t c = hwc % channels;
                const std::size_t hw = hwc / channels;
                const std::size_t h = hw / width;
                const std::size_t w = hw % width;
                for (std::size_t o = 0; o < outputs; ++o) {
                    const std::size_t src =
                        ((o * channels + c) * height + h) * width + w;
                    result.values[hwc * outputs + o] = reader.bf16(src);
                }
            }
        });
        return modalities::Status::ok();
    });
    if (!st.ok_status()) return st;
    *out = std::move(result);
    return modalities::Status::ok();
}

modalities::Status native_source_pair_transpose_concat_bf16(
    const NativeSourceTensorView& left,
    const NativeSourceTensorView& right,
    NativeBf16Tensor* out) {
    std::size_t source_count = 0;
    if (!out || left.shape.size() != 2 || right.shape.size() != 2 ||
        left.shape != right.shape ||
        !element_count(left.shape, &source_count) ||
        left.shape[0] > std::numeric_limits<std::uint64_t>::max() / 2 ||
        source_count > std::numeric_limits<std::size_t>::max() / 2) {
        return invalid("paired transpose tensors have incompatible shapes");
    }
    const std::size_t source_rows = static_cast<std::size_t>(left.shape[0]);
    const std::size_t output_rows = static_cast<std::size_t>(left.shape[1]);
    NativeBf16Tensor joined;
    joined.shape = {left.shape[1], left.shape[0] + right.shape[0]};
    joined.values.resize(source_count * 2);
    const auto write = [&](const NativeSourceTensorView& source,
                           std::size_t offset) {
        return dispatch_source(source, [&](const auto& reader) {
            tiled_transform_transpose(
                source_rows, output_rows, source_rows * 2, offset,
                joined.values.data(),
                [&](std::size_t row, std::size_t col) {
                    return reader.bf16(row * output_rows + col);
                });
            return modalities::Status::ok();
        });
    };
    modalities::Status st = write(left, 0);
    if (!st.ok_status()) return st;
    st = write(right, source_rows);
    if (!st.ok_status()) return st;
    *out = std::move(joined);
    return modalities::Status::ok();
}

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
        parallel_ranges(count, 1 << 18, [&](std::size_t begin,
                                             std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                loaded.values[i] = modalities::bfloat16_to_float(src[i]);
            }
        });
    } else if (tensor->dtype == "F16") {
        const auto* src = static_cast<const std::uint16_t*>(data);
        parallel_ranges(count, 1 << 18, [&](std::size_t begin,
                                             std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                loaded.values[i] = modalities::float16_to_float(src[i]);
            }
        });
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
    parallel_ranges(input.values.size(), 1 << 18,
                    [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            converted.values[i] =
                modalities::float_to_bfloat16(input.values[i]);
        }
    });
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
    parallel_ranges(rounded.values.size(), 1 << 18,
                    [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            rounded.values[i] = modalities::bfloat16_to_float(
                modalities::float_to_bfloat16(rounded.values[i]));
        }
    });
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
    tiled_transform_transpose(
        rows, cols, rows, 0, transposed.values.data(),
        [&](std::size_t row, std::size_t col) {
            return input.values[row * cols + col];
        });
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
    parallel_ranges(rows, 16, [&](std::size_t begin, std::size_t end) {
        for (std::size_t row = begin; row < end; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                folded.values[row * cols + col] *= 1.0f + norm.values[col];
            }
        }
    });
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
        const std::uint64_t input_rows = input->shape[0];
        const std::uint64_t output_offset = row_offset;
        tiled_transform_transpose(
            static_cast<std::size_t>(input_rows),
            static_cast<std::size_t>(cols),
            static_cast<std::size_t>(total_rows),
            static_cast<std::size_t>(output_offset), joined.values.data(),
            [&](std::size_t row, std::size_t col) {
                return input->values[
                    row * static_cast<std::size_t>(cols) + col];
            });
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
    parallel_ranges(rows, 32, [&](std::size_t begin, std::size_t end) {
        for (std::size_t row = begin; row < end; ++row) {
            float* dst = joined.values.data() + row * (left_cols + right_cols);
            std::memcpy(dst, left.values.data() + row * left_cols,
                        left_cols * sizeof(float));
            std::memcpy(dst + left_cols,
                        right.values.data() + row * right_cols,
                        right_cols * sizeof(float));
        }
    });
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
    parallel_ranges(scaled.values.size(), 1 << 18,
                    [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) scaled.values[i] *= scale;
    });
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
