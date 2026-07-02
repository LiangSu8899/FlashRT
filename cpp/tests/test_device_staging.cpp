#include "flashrt/cpp/modalities/action.h"
#include "flashrt/cpp/modalities/vision.h"
#include "flashrt/cpp/models/pi05/spec.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using flashrt::modalities::DType;
using flashrt::modalities::Layout;
using flashrt::modalities::MemoryPlace;
using flashrt::modalities::PixelFormat;
using flashrt::modalities::Shape;
using flashrt::modalities::TensorView;
using flashrt::modalities::VisionFrame;
using flashrt::modalities::bfloat16_to_float;
using flashrt::modalities::float_to_bfloat16;
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

void test_vision_h2d_staging() {
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
    auto st = preprocess_vision(spec, {frame}, dst);
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
    cudaFree(device);
}

}  // namespace

int main() {
    if (!has_cuda_device()) {
        std::cout << "SKIP - no CUDA device\n";
        return 0;
    }
    test_vision_h2d_staging();
    test_action_d2h_staging();
    std::cout << "PASS - CUDA modality kernels/staging\n";
    return 0;
}
