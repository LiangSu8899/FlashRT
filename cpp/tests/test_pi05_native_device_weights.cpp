#include "flashrt/cpp/models/pi05/native_device_weights.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdio>
#include <vector>

namespace {

bool has_cuda_device() {
    int count = 0;
    const cudaError_t rc = cudaGetDeviceCount(&count);
    if (rc != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

}  // namespace

int main() {
    if (!has_cuda_device()) {
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }
    using flashrt::models::pi05::NativeBf16Tensor;
    using flashrt::models::pi05::NativeDeviceWeightStore;
    using flashrt::models::pi05::NativeWeightDType;

    frt_ctx ctx = frt_ctx_create();
    assert(ctx);
    {
        NativeDeviceWeightStore store(ctx);
        NativeBf16Tensor tensor;
        tensor.shape = {2, 3};
        tensor.values = {
            flashrt::modalities::float_to_bfloat16(1.0f),
            flashrt::modalities::float_to_bfloat16(2.0f),
            flashrt::modalities::float_to_bfloat16(3.0f),
            flashrt::modalities::float_to_bfloat16(4.0f),
            flashrt::modalities::float_to_bfloat16(5.0f),
            flashrt::modalities::float_to_bfloat16(6.0f),
        };
        assert(store.upload("encoder.layer0.qkv", tensor).ok_status());
        assert(store.size() == 1);
        const auto* weight = store.find("encoder.layer0.qkv");
        assert(weight && weight->buffer);
        assert(weight->shape == tensor.shape);
        assert(weight->dtype == NativeWeightDType::kBf16);
        assert(frt_buffer_bytes(weight->buffer) ==
               tensor.values.size() * sizeof(std::uint16_t));
        assert(std::string(frt_buffer_name(weight->buffer)) ==
               "encoder.layer0.qkv");

        std::vector<std::uint16_t> copied(tensor.values.size());
        assert(cudaMemcpy(copied.data(), frt_buffer_dptr(weight->buffer),
                          copied.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(copied == tensor.values);

        const std::vector<std::int8_t> int8_values = {-127, -1, 0, 127};
        assert(store.upload_bytes("encoder.layer0.qkv.int8", {2, 2},
                                  NativeWeightDType::kInt8,
                                  int8_values.data(), int8_values.size())
                   .ok_status());
        const auto* int8_weight = store.find("encoder.layer0.qkv.int8");
        assert(int8_weight && int8_weight->dtype == NativeWeightDType::kInt8);
        std::vector<std::int8_t> int8_copied(int8_values.size());
        assert(cudaMemcpy(int8_copied.data(),
                          frt_buffer_dptr(int8_weight->buffer),
                          int8_copied.size(), cudaMemcpyDeviceToHost) ==
               cudaSuccess);
        assert(int8_copied == int8_values);

        const std::vector<float> scales = {0.25f, 0.5f};
        assert(store.upload_bytes("encoder.layer0.qkv.scale", {2},
                                  NativeWeightDType::kFloat32,
                                  scales.data(),
                                  scales.size() * sizeof(float))
                   .ok_status());
        assert(store.find("encoder.layer0.qkv.scale")->dtype ==
               NativeWeightDType::kFloat32);
        assert(!store.upload_bytes("bad.bytes", {3},
                                   NativeWeightDType::kFp8E4M3,
                                   int8_values.data(), int8_values.size())
                    .ok_status());
        assert(!store.upload("encoder.layer0.qkv", tensor).ok_status());
        tensor.shape = {3, 3};
        assert(!store.upload("bad", tensor).ok_status());
    }

    frt_ctx_destroy(ctx);
    std::printf("PASS - Pi0.5 native device weights\n");
    return 0;
}
