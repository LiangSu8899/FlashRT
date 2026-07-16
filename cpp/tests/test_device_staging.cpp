#include "flashrt/cpp/modalities/action.h"
#include "flashrt/cpp/modalities/text.h"
#include "flashrt/cpp/modalities/vision.h"
#include "flashrt/cpp/models/pi05/spec.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

using flashrt::modalities::DType;
using flashrt::modalities::ActionStaging;
using flashrt::modalities::Layout;
using flashrt::modalities::MemoryPlace;
using flashrt::modalities::PixelFormat;
using flashrt::modalities::Shape;
using flashrt::modalities::TensorView;
using flashrt::modalities::EmbeddingGatherSpec;
using flashrt::modalities::TextEmbeddingStaging;
using flashrt::modalities::VisionFrame;
using flashrt::modalities::bfloat16_to_float;
using flashrt::modalities::float_to_bfloat16;
using flashrt::modalities::gather_token_embeddings;
using flashrt::modalities::gather_token_embeddings_cpu;
using flashrt::modalities::postprocess_action;
using flashrt::modalities::preprocess_vision_cpu;
using flashrt::modalities::preprocess_vision;
using flashrt::modalities::required_action_output_bytes;
using flashrt::modalities::required_vision_output_bytes;

namespace {

bool has_cuda_device() {
    int n = 0;
    cudaError_t rc = cudaGetDeviceCount(&n);
    if (rc != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return n > 0;
}

std::uint32_t ordered_bf16(std::uint16_t bits) {
    if (bits & 0x8000u) return 0x8000u - (bits & 0x7fffu);
    return 0x8000u + bits;
}

std::uint32_t bf16_ulp_distance(std::uint16_t a, std::uint16_t b) {
    const std::uint32_t ao = ordered_bf16(a);
    const std::uint32_t bo = ordered_bf16(b);
    return ao > bo ? ao - bo : bo - ao;
}

void test_vision_h2d_staging() {
    flashrt::modalities::VisionStaging overflow;
    auto st = flashrt::modalities::vision_staging_create(
        &overflow, 2,
        std::numeric_limits<std::uint64_t>::max() / 2 + 1);
    assert(!st.ok_status());
    assert(!overflow.device && !overflow.host_pinned);

    const auto spec = flashrt::models::pi05::vision_preprocess_spec(1);
    const std::uint64_t bytes = required_vision_output_bytes(spec);

    void* device = nullptr;
    assert(cudaMalloc(&device, bytes) == cudaSuccess);

    const std::uint8_t rgb[] = {
        0, 127, 255, 255, 127, 0,
        10, 20, 30, 40, 50, 60,
    };
    VisionFrame frame;
    frame.name = "image";
    frame.image = {const_cast<std::uint8_t*>(rgb), sizeof(rgb),
                   DType::kUInt8, MemoryPlace::kHost, Layout::kHWC,
                   Shape{2, 2, 3}};
    frame.format = PixelFormat::kRGB8;
    frame.width = 2;
    frame.height = 2;

    TensorView dst{device, bytes, DType::kBFloat16, MemoryPlace::kDevice,
                   Layout::kNHWC, Shape{1, 224, 224, 3}};
    st = preprocess_vision(spec, {frame}, dst);
    assert(st.ok_status());

    std::vector<std::uint16_t> got(bytes / 2);
    assert(cudaMemcpy(got.data(), device, bytes,
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    std::vector<std::uint16_t> ref(bytes / 2);
    TensorView ref_dst{ref.data(), bytes, DType::kBFloat16, MemoryPlace::kHost,
                       Layout::kNHWC, Shape{1, 224, 224, 3}};
    st = preprocess_vision_cpu(spec, {frame}, ref_dst);
    assert(st.ok_status());

    for (std::size_t i = 0; i < got.size(); ++i) {
        assert(std::fabs(bfloat16_to_float(got[i]) -
                         bfloat16_to_float(ref[i])) < 0.01f);
    }
    assert(std::fabs(bfloat16_to_float(got[0]) - (-1.0f)) < 0.01f);
    assert(std::fabs(bfloat16_to_float(got[1]) -
                     (127.0f / 127.5f - 1.0f)) < 0.01f);
    assert(std::fabs(bfloat16_to_float(got[2]) - 1.0f) < 0.01f);

    /* the persistent staging pool (hot path: no per-frame allocation) must
     * produce the same bytes as the allocating dev path, tick after tick */
    flashrt::modalities::VisionStaging pool;
    st = flashrt::modalities::vision_staging_create(&pool, 1, sizeof(rgb));
    assert(st.ok_status() && pool.device && pool.host_pinned);
    std::vector<std::uint16_t> pooled(bytes / 2);
    for (int round = 0; round < 3; ++round) {
        assert(cudaMemset(device, 0, bytes) == cudaSuccess);
        st = preprocess_vision(spec, {frame}, dst, nullptr, &pool);
        assert(st.ok_status());
        assert(cudaMemcpy(pooled.data(), device, bytes,
                          cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(pooled == got);
    }
    /* over-capacity frames are a hard error, never a fallback allocation */
    VisionFrame big = frame;
    big.width = 64; big.height = 64; big.stride_bytes = 64 * 3;
    big.image.bytes = 64ull * 64 * 3;
    std::vector<std::uint8_t> big_pixels(64ull * 64 * 3, 7);
    big.image.data = big_pixels.data();
    big.image.shape = Shape{64, 64, 3};
    st = preprocess_vision(spec, {big}, dst, nullptr, &pool);
    assert(!st.ok_status());
    assert(st.code == flashrt::modalities::StatusCode::kInsufficientStorage);
    flashrt::modalities::vision_staging_destroy(&pool);
    assert(pool.device == nullptr && pool.host_pinned == nullptr);

    cudaFree(device);
}

void test_vision_resize_matrix() {
    struct Case { int width; int height; int padding; };
    const std::array<Case, 7> cases{{
        {1, 1, 0}, {3, 2, 5}, {17, 19, 1}, {63, 47, 7},
        {224, 224, 0}, {321, 181, 3}, {181, 321, 9},
    }};
    std::uint64_t max_frame_bytes = 0;
    for (const auto& item : cases) {
        max_frame_bytes = std::max(
            max_frame_bytes,
            static_cast<std::uint64_t>(item.width * 3 + item.padding) *
                static_cast<std::uint64_t>(item.height));
    }

    const auto spec = flashrt::models::pi05::vision_preprocess_spec(1);
    const std::uint64_t output_bytes = required_vision_output_bytes(spec);
    void* device = nullptr;
    assert(cudaMalloc(&device, output_bytes) == cudaSuccess);
    flashrt::modalities::VisionStaging pool;
    auto st = flashrt::modalities::vision_staging_create(
        &pool, 1, max_frame_bytes);
    assert(st.ok_status());
    std::vector<std::uint16_t> actual(output_bytes / 2);
    std::vector<std::uint16_t> expected(output_bytes / 2);
    std::uint32_t matrix_max_ulp = 0;
    float matrix_max_abs = 0.0f;
    Case worst_case{};
    std::size_t worst_index = 0;
    std::uint16_t worst_actual = 0;
    std::uint16_t worst_expected = 0;

    for (const auto& item : cases) {
        const int stride = item.width * 3 + item.padding;
        std::vector<std::uint8_t> pixels(
            static_cast<std::size_t>(stride) * item.height, 0xa5);
        for (int y = 0; y < item.height; ++y) {
            for (int x = 0; x < item.width; ++x) {
                for (int c = 0; c < 3; ++c) {
                    pixels[static_cast<std::size_t>(y) * stride + x * 3 + c] =
                        static_cast<std::uint8_t>(
                            (x * 13 + y * 17 + c * 71) & 0xff);
                }
            }
        }
        VisionFrame frame;
        frame.name = "image";
        frame.image = {
            pixels.data(), pixels.size(), DType::kUInt8, MemoryPlace::kHost,
            Layout::kHWC,
            Shape{static_cast<std::uint64_t>(item.height),
                  static_cast<std::uint64_t>(item.width), 3}};
        frame.format = PixelFormat::kRGB8;
        frame.width = item.width;
        frame.height = item.height;
        frame.stride_bytes = stride;
        TensorView device_output{
            device, output_bytes, DType::kBFloat16, MemoryPlace::kDevice,
            Layout::kNHWC, Shape{1, 224, 224, 3}};
        st = preprocess_vision(spec, {frame}, device_output, nullptr, &pool);
        assert(st.ok_status());
        assert(cudaMemcpy(actual.data(), device, output_bytes,
                          cudaMemcpyDeviceToHost) == cudaSuccess);
        TensorView host_output{
            expected.data(), output_bytes, DType::kBFloat16,
            MemoryPlace::kHost, Layout::kNHWC, Shape{1, 224, 224, 3}};
        st = preprocess_vision_cpu(spec, {frame}, host_output);
        assert(st.ok_status());
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const std::uint32_t ulp =
                bf16_ulp_distance(actual[i], expected[i]);
            const float absolute = std::fabs(
                bfloat16_to_float(actual[i]) -
                bfloat16_to_float(expected[i]));
            matrix_max_abs = std::max(matrix_max_abs, absolute);
            if (ulp > matrix_max_ulp) {
                matrix_max_ulp = ulp;
                worst_case = item;
                worst_index = i;
                worst_actual = actual[i];
                worst_expected = expected[i];
            }
        }
    }
    if (matrix_max_ulp > 1) {
        std::cerr << "vision resize max_ulp=" << matrix_max_ulp
                  << " max_abs=" << matrix_max_abs
                  << " size=" << worst_case.width << 'x'
                  << worst_case.height << " index=" << worst_index << '\n';
        std::cerr << "vision resize values actual="
                  << bfloat16_to_float(worst_actual) << " expected="
                  << bfloat16_to_float(worst_expected) << '\n';
    }
    std::cout << "vision resize matrix max BF16 ULP: "
              << matrix_max_ulp << '\n';
    assert(matrix_max_ulp <= 1);
    flashrt::modalities::vision_staging_destroy(&pool);
    cudaFree(device);
}

void test_action_d2h_staging() {
    auto spec = flashrt::models::pi05::action_postprocess_spec(
        {10.0f, 20.0f, 30.0f}, {2.0f, 3.0f, 4.0f},
        /*chunk=*/1, /*model_dim=*/4, /*robot_dim=*/3);
    std::vector<std::uint16_t> host(4);
    host[0] = float_to_bfloat16(1.0f);
    host[1] = float_to_bfloat16(-2.0f);
    host[2] = float_to_bfloat16(3.0f);
    host[3] = float_to_bfloat16(99.0f);

    const std::uint64_t bytes = required_action_output_bytes(spec, DType::kBFloat16);
    void* device = nullptr;
    assert(cudaMalloc(&device, bytes) == cudaSuccess);
    assert(cudaMemcpy(device, host.data(), bytes,
                      cudaMemcpyHostToDevice) == cudaSuccess);

    TensorView src{device, bytes, DType::kBFloat16, MemoryPlace::kDevice,
                   Layout::kFlat, Shape{1, 4}};
    std::vector<float> actions;
    auto st = postprocess_action(spec, src, &actions);
    assert(st.ok_status());
    assert(actions.size() == 3);
    assert(std::fabs(actions[0] - 12.0f) < 0.01f);
    assert(std::fabs(actions[1] - 17.0f) < 0.01f);
    assert(std::fabs(actions[2] - 34.0f) < 0.01f);
    ActionStaging staging;
    st = flashrt::modalities::action_staging_create(&staging, bytes);
    assert(st.ok_status() && staging.host_pinned && staging.bytes == bytes);
    const std::size_t action_capacity = actions.capacity();
    for (int round = 0; round < 1000; ++round) {
        st = postprocess_action(spec, src, &actions, nullptr, &staging);
        assert(st.ok_status());
        assert(actions.capacity() == action_capacity);
    }
    ActionStaging too_small;
    st = flashrt::modalities::action_staging_create(&too_small, bytes - 1);
    assert(st.ok_status());
    st = postprocess_action(spec, src, &actions, nullptr, &too_small);
    assert(!st.ok_status());
    assert(st.code == flashrt::modalities::StatusCode::kInsufficientStorage);
    flashrt::modalities::action_staging_destroy(&too_small);
    flashrt::modalities::action_staging_destroy(&staging);
    assert(staging.host_pinned == nullptr && staging.bytes == 0);
    cudaFree(device);
}

void test_text_embedding_device_gather() {
    const std::vector<float> table = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    };
    const std::int32_t ids[] = {2, 0};
    std::vector<float> ref(2 * 4, 0.0f);
    TensorView host_table{const_cast<float*>(table.data()),
                          static_cast<std::uint64_t>(table.size() * 4),
                          DType::kFloat32, MemoryPlace::kHost, Layout::kFlat,
                          Shape{3, 4}};
    TensorView host_out{ref.data(), static_cast<std::uint64_t>(ref.size() * 4),
                        DType::kFloat32, MemoryPlace::kHost, Layout::kFlat,
                        Shape{2, 4}};
    EmbeddingGatherSpec spec{3, 4, 2.0f};
    auto st = gather_token_embeddings_cpu(spec, ids, 2, host_table, host_out);
    assert(st.ok_status());

    void* d_table = nullptr;
    void* d_out = nullptr;
    assert(cudaMalloc(&d_table, table.size() * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_out, ref.size() * sizeof(float)) == cudaSuccess);
    assert(cudaMemcpy(d_table, table.data(), table.size() * sizeof(float),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    TensorView device_table{d_table,
                            static_cast<std::uint64_t>(table.size() * 4),
                            DType::kFloat32, MemoryPlace::kDevice,
                            Layout::kFlat, Shape{3, 4}};
    TensorView device_out{d_out, static_cast<std::uint64_t>(ref.size() * 4),
                          DType::kFloat32, MemoryPlace::kDevice,
                          Layout::kFlat, Shape{2, 4}};

    TextEmbeddingStaging staging;
    st = flashrt::modalities::text_embedding_staging_create(&staging, 2);
    assert(st.ok_status());
    std::vector<float> got(ref.size(), 0.0f);
    for (int round = 0; round < 3; ++round) {
        assert(cudaMemset(d_out, 0, ref.size() * sizeof(float)) == cudaSuccess);
        st = gather_token_embeddings(spec, ids, 2, device_table, device_out,
                                     nullptr, &staging);
        assert(st.ok_status());
        assert(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float),
                          cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(got == ref);
    }
    st = gather_token_embeddings(spec, ids, 3, device_table, device_out,
                                 nullptr, &staging);
    assert(!st.ok_status());
    assert(st.code == flashrt::modalities::StatusCode::kInsufficientStorage);
    flashrt::modalities::text_embedding_staging_destroy(&staging);
    assert(staging.device_token_ids == nullptr);
    cudaFree(d_out);
    cudaFree(d_table);
}

}  // namespace

int main() {
    if (!has_cuda_device()) {
        std::cout << "SKIP - no CUDA device\n";
        return 0;
    }
    test_vision_h2d_staging();
    test_vision_resize_matrix();
    test_action_d2h_staging();
    test_text_embedding_device_gather();
    std::cout << "PASS - CUDA modality kernels/staging\n";
    return 0;
}
