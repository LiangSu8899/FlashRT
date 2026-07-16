#include "flashrt/cpp/models/pi05/native_weight_ops.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>

namespace {

using flashrt::models::pi05::NativeBf16Tensor;
using flashrt::models::pi05::NativeF16Tensor;
using flashrt::models::pi05::NativeFloatTensor;
using flashrt::models::pi05::NativeSourceDType;
using flashrt::models::pi05::NativeSourceTensorView;

void expect(const NativeFloatTensor& tensor,
            std::initializer_list<std::uint64_t> shape,
            std::initializer_list<float> values) {
    assert(tensor.shape == std::vector<std::uint64_t>(shape));
    assert(tensor.values.size() == values.size());
    std::size_t i = 0;
    for (float value : values) {
        assert(std::fabs(tensor.values[i++] - value) < 1e-6f);
    }
}

void expect_bf16(const NativeBf16Tensor& tensor,
                 std::initializer_list<std::uint64_t> shape,
                 std::initializer_list<float> values) {
    assert(tensor.shape == std::vector<std::uint64_t>(shape));
    assert(tensor.values.size() == values.size());
    std::size_t i = 0;
    for (float value : values) {
        assert(tensor.values[i++] ==
               flashrt::modalities::float_to_bfloat16(value));
    }
}

void expect_f16(const NativeF16Tensor& tensor,
                std::initializer_list<std::uint64_t> shape,
                std::initializer_list<float> values) {
    assert(tensor.shape == std::vector<std::uint64_t>(shape));
    assert(tensor.values.size() == values.size());
    std::size_t i = 0;
    for (float value : values) {
        assert(tensor.values[i++] == flashrt::modalities::float_to_float16(value));
    }
}

std::string temp_path() {
    char path[] = "/tmp/frt_pi05_weight_ops_XXXXXX";
    const int fd = ::mkstemp(path);
    assert(fd >= 0);
    ::close(fd);
    return path;
}

void write_tensor_file(const std::string& path) {
    const std::string header =
        "{\"f32\":{\"dtype\":\"F32\",\"shape\":[2],"
        "\"data_offsets\":[0,8]},"
        "\"bf16\":{\"dtype\":\"BF16\",\"shape\":[2],"
        "\"data_offsets\":[8,12]},"
        "\"f16\":{\"dtype\":\"F16\",\"shape\":[2],"
        "\"data_offsets\":[12,16]}}";
    const float f32[] = {1.25f, -2.5f};
    const std::uint16_t bf16[] = {
        flashrt::modalities::float_to_bfloat16(3.0f),
        flashrt::modalities::float_to_bfloat16(-4.0f)};
    const std::uint16_t f16[] = {
        flashrt::modalities::float_to_float16(5.0f),
        flashrt::modalities::float_to_float16(-6.0f)};
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) {
        const char byte = static_cast<char>((n >> (8 * i)) & 0xffu);
        f.write(&byte, 1);
    }
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    f.write(reinterpret_cast<const char*>(f32), sizeof(f32));
    f.write(reinterpret_cast<const char*>(bf16), sizeof(bf16));
    f.write(reinterpret_cast<const char*>(f16), sizeof(f16));
    assert(f.good());
}

}  // namespace

int main() {
    using namespace flashrt::models::pi05;

    const std::string path = temp_path();
    write_tensor_file(path);
    flashrt::loader::SafetensorsFile file;
    assert(file.open(path));
    NativeFloatTensor loaded;
    assert(load_native_float_tensor(file, "f32", &loaded).ok_status());
    expect(loaded, {2}, {1.25f, -2.5f});
    assert(load_native_float_tensor(file, "bf16", &loaded).ok_status());
    expect(loaded, {2}, {3.0f, -4.0f});
    assert(load_native_float_tensor(file, "f16", &loaded).ok_status());
    expect(loaded, {2}, {5.0f, -6.0f});

    NativeSourceTensorView mapped;
    NativeBf16Tensor mapped_bf16;
    NativeF16Tensor mapped_f16;
    assert(load_native_source_tensor(file, "f32", &mapped).ok_status());
    assert(mapped.dtype == NativeSourceDType::kF32);
    assert(native_source_to_bf16(mapped, false, &mapped_bf16).ok_status());
    expect_bf16(mapped_bf16, {2}, {1.25f, -2.5f});
    assert(native_source_to_f16(mapped, false, &mapped_f16).ok_status());
    expect_f16(mapped_f16, {2}, {1.25f, -2.5f});
    assert(load_native_source_tensor(file, "bf16", &mapped).ok_status());
    assert(mapped.dtype == NativeSourceDType::kBf16);
    assert(native_source_to_bf16(mapped, false, &mapped_bf16).ok_status());
    expect_bf16(mapped_bf16, {2}, {3.0f, -4.0f});
    assert(native_source_to_f16(mapped, false, &mapped_f16).ok_status());
    expect_f16(mapped_f16, {2}, {3.0f, -4.0f});
    assert(load_native_source_tensor(file, "f16", &mapped).ok_status());
    assert(mapped.dtype == NativeSourceDType::kF16);
    assert(native_source_to_bf16(mapped, false, &mapped_bf16).ok_status());
    expect_bf16(mapped_bf16, {2}, {5.0f, -6.0f});
    assert(native_source_to_f16(mapped, false, &mapped_f16).ok_status());
    expect_f16(mapped_f16, {2}, {5.0f, -6.0f});
    assert(::unlink(path.c_str()) == 0);

    NativeFloatTensor matrix{{2, 3}, {1, 2, 3, 4, 5, 6}};
    NativeFloatTensor result;
    assert(native_transpose_2d(matrix, &result).ok_status());
    expect(result, {3, 2}, {1, 4, 2, 5, 3, 6});

    NativeFloatTensor patch{{2, 2, 2, 1}, {0, 1, 2, 3, 4, 5, 6, 7}};
    assert(native_patch_oihw_to_hwio(patch, &result).ok_status());
    expect(result, {2, 1, 2, 2}, {0, 4, 2, 6, 1, 5, 3, 7});

    NativeFloatTensor qk{{8, 1}, {0, 1, 2, 3, 4, 5, 6, 7}};
    assert(native_interleave_qk_rows(qk, 2, &result).ok_status());
    expect(result, {8, 1}, {0, 2, 1, 3, 4, 6, 5, 7});

    NativeFloatTensor norm{{3}, {-0.5f, 0.0f, 1.0f}};
    assert(native_fold_rms_columns(matrix, norm, &result).ok_status());
    expect(result, {2, 3}, {0.5f, 2.0f, 6.0f, 2.0f, 5.0f, 12.0f});

    NativeFloatTensor q{{2, 2}, {1, 2, 3, 4}};
    NativeFloatTensor k{{1, 2}, {5, 6}};
    NativeFloatTensor v{{1, 2}, {7, 8}};
    assert(native_concat_rows_transpose({&q, &k, &v}, &result).ok_status());
    expect(result, {2, 4}, {1, 3, 5, 7, 2, 4, 6, 8});

    NativeFloatTensor left{{2, 2}, {1, 2, 3, 4}};
    NativeFloatTensor right{{2, 2}, {5, 6, 7, 8}};
    assert(native_concat_columns(left, right, &result).ok_status());
    expect(result, {2, 4}, {1, 2, 5, 6, 3, 4, 7, 8});

    NativeFloatTensor a{{2}, {1, 2}};
    NativeFloatTensor b{{3}, {3, 4, 5}};
    assert(native_concat_vectors({&a, &b}, &result).ok_status());
    expect(result, {5}, {1, 2, 3, 4, 5});

    assert(native_scale(matrix, -0.1f, &result).ok_status());
    expect(result, {2, 3}, {-0.1f, -0.2f, -0.3f,
                            -0.4f, -0.5f, -0.6f});

    assert(native_pi05_time_embeddings(2, 4, &result).ok_status());
    assert(result.shape == std::vector<std::uint64_t>({2, 4}));
    assert(result.values.size() == 8);
    assert(!native_pi05_time_embeddings(0, 4, &result).ok_status());
    assert(!native_pi05_time_embeddings(2, 3, &result).ok_status());

    NativeF16Tensor time_f16;
    assert(native_pi05_time_embeddings_f16(2, 4, &time_f16).ok_status());
    assert(time_f16.shape == std::vector<std::uint64_t>({2, 4}));
    assert(time_f16.values.size() == 8);
    const double two_pi = 6.2831853071795864769;
    const double first_angles[] = {two_pi / 4.0e-3, two_pi / 4.0};
    for (std::size_t i = 0; i < 2; ++i) {
        assert(time_f16.values[i] == flashrt::modalities::float_to_float16(
                                          static_cast<float>(std::sin(first_angles[i]))));
        assert(time_f16.values[2 + i] ==
               flashrt::modalities::float_to_float16(
                   static_cast<float>(std::cos(first_angles[i]))));
    }
    assert(native_pi05_time_embeddings_f16(10, 1024, &time_f16).ok_status());
    assert(time_f16.shape == std::vector<std::uint64_t>({10, 1024}));
    assert(time_f16.values[6 * 1024 + 7] == 0xb1c0u);
    assert(time_f16.values[8 * 1024 + 7] == 0x2dc6u);
    assert(time_f16.values[9 * 1024 + 3] == 0x292au);

    NativeFloatTensor unrounded{{2}, {1.003f, -1.003f}};
    assert(native_round_to_bf16_float(unrounded, &result).ok_status());
    assert(result.values[0] == flashrt::modalities::bfloat16_to_float(
                                   flashrt::modalities::float_to_bfloat16(
                                       unrounded.values[0])));
    assert(result.values[1] == flashrt::modalities::bfloat16_to_float(
                                   flashrt::modalities::float_to_bfloat16(
                                       unrounded.values[1])));

    NativeBf16Tensor converted;
    assert(native_to_bf16(matrix, &converted).ok_status());
    assert(converted.shape == matrix.shape);
    for (std::size_t i = 0; i < matrix.values.size(); ++i) {
        assert(converted.values[i] ==
               flashrt::modalities::float_to_bfloat16(matrix.values[i]));
    }

    NativeF16Tensor converted_f16;
    assert(native_to_f16(matrix, &converted_f16).ok_status());
    expect_f16(converted_f16, {2, 3}, {1, 2, 3, 4, 5, 6});

    NativeBf16Tensor direct;
    const float source_f32[] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::uint16_t source_bf16[8];
    std::uint16_t source_f16[8];
    for (std::size_t i = 0; i < 8; ++i) {
        source_bf16[i] = flashrt::modalities::float_to_bfloat16(source_f32[i]);
        source_f16[i] = flashrt::modalities::float_to_float16(source_f32[i]);
    }
    const NativeSourceTensorView source_views[] = {
        {source_f32, {2, 4}, NativeSourceDType::kF32},
        {source_bf16, {2, 4}, NativeSourceDType::kBf16},
        {source_f16, {2, 4}, NativeSourceDType::kF16},
    };
    NativeFloatTensor source_norm{{4}, {0, 0, 0, 0}};
    for (const NativeSourceTensorView& source_view : source_views) {
        assert(native_source_to_bf16(source_view, true, &direct).ok_status());
        expect_bf16(direct, {4, 2}, {1, 5, 2, 6, 3, 7, 4, 8});
        assert(native_source_fold_rms_columns_transpose(
                   source_view, source_norm, &direct).ok_status());
        expect_bf16(direct, {4, 2}, {1, 5, 2, 6, 3, 7, 4, 8});
        assert(native_source_round_scale_to_bf16(
                   source_view, 2.0f, true, &direct).ok_status());
        expect_bf16(direct, {4, 2}, {2, 10, 4, 12, 6, 14, 8, 16});
        assert(native_source_qkv_to_bf16(
                   source_view, source_view, source_view, 1, 1, nullptr,
                   &direct).ok_status());
        expect_bf16(direct, {4, 6},
                    {1, 5, 1, 5, 1, 5,
                     2, 6, 2, 6, 2, 6,
                     3, 7, 3, 7, 3, 7,
                     4, 8, 4, 8, 4, 8});

        NativeF16Tensor direct_f16;
        assert(native_source_to_f16(source_view, true, &direct_f16).ok_status());
        expect_f16(direct_f16, {4, 2}, {1, 5, 2, 6, 3, 7, 4, 8});
        assert(native_source_qkv_to_f16(
                   source_view, source_view, source_view, 1, 1, nullptr,
                   true, &direct_f16).ok_status());
        expect_f16(direct_f16, {4, 6},
                   {1, 5, 1, 5, 1, 5,
                    2, 6, 2, 6, 2, 6,
                    3, 7, 3, 7, 3, 7,
                    4, 8, 4, 8, 4, 8});
        assert(native_source_qkv_to_f16(
                   source_view, source_view, source_view, 1, 1, nullptr,
                   false, &direct_f16).ok_status());
        expect_f16(direct_f16, {6, 4},
                   {1, 2, 3, 4, 5, 6, 7, 8,
                    1, 2, 3, 4, 5, 6, 7, 8,
                    1, 2, 3, 4, 5, 6, 7, 8});
    }

    const float fold_source[] = {1.003f, -2.007f, 3.011f, -4.015f};
    const NativeSourceTensorView fold_view{
        fold_source, {2, 2}, NativeSourceDType::kF32};
    NativeFloatTensor fold_norm{{2}, {0.125f, -0.375f}};
    NativeF16Tensor direct_f16;
    assert(native_source_qkv_to_f16(
               fold_view, fold_view, fold_view, 1, 1, &fold_norm,
               false, &direct_f16).ok_status());
    expect_f16(direct_f16, {6, 2},
               {1.003f * 1.125f, -2.007f * 0.625f,
                3.011f * 1.125f, -4.015f * 0.625f,
                1.003f * 1.125f, -2.007f * 0.625f,
                3.011f * 1.125f, -4.015f * 0.625f,
                1.003f * 1.125f, -2.007f * 0.625f,
                3.011f * 1.125f, -4.015f * 0.625f});

    assert(native_source_pair_to_f16(
               source_views[0], source_views[0], nullptr, false,
               &direct_f16).ok_status());
    expect_f16(direct_f16, {4, 4},
               {1, 2, 3, 4, 5, 6, 7, 8,
                1, 2, 3, 4, 5, 6, 7, 8});
    assert(native_source_pair_to_f16(
               source_views[0], source_views[0], nullptr, true,
               &direct_f16).ok_status());
    expect_f16(direct_f16, {4, 4},
               {1, 5, 1, 5, 2, 6, 2, 6,
                3, 7, 3, 7, 4, 8, 4, 8});

    const float vector_a[] = {1, 2};
    const std::uint16_t vector_b[] = {
        flashrt::modalities::float_to_bfloat16(3),
        flashrt::modalities::float_to_bfloat16(4)};
    const NativeSourceTensorView vector_a_view{
        vector_a, {2}, NativeSourceDType::kF32};
    const NativeSourceTensorView vector_b_view{
        vector_b, {2}, NativeSourceDType::kBf16};
    assert(native_source_concat_vectors_to_f16(
               {&vector_a_view, &vector_b_view}, &direct_f16).ok_status());
    expect_f16(direct_f16, {4}, {1, 2, 3, 4});

    const float patch_source[] = {0, 1, 2, 3, 4, 5, 6, 7};
    const NativeSourceTensorView patch_view{
        patch_source, {2, 2, 2, 1}, NativeSourceDType::kF32};
    assert(native_source_patch_oihw_to_hwio_f16(
               patch_view, &direct_f16).ok_status());
    expect_f16(direct_f16, {2, 1, 2, 2}, {0, 4, 2, 6, 1, 5, 3, 7});

    assert(!native_interleave_qk_rows(matrix, 2, &result).ok_status());
    assert(!native_concat_columns(matrix, k, &result).ok_status());
    std::printf("PASS - Pi0.5 native weight transforms\n");
    return 0;
}
