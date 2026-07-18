#include "flashrt/cpp/models/pi05/targets/sm120/fp8_weight_packer.h"

#include "flashrt/exec.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace pi05 = flashrt::models::pi05;
namespace sm120 = flashrt::models::pi05::targets::sm120;

namespace {

[[noreturn]] void fail(const char* expression, int line) {
    std::cerr << "FAIL line " << line << ": " << expression << '\n';
    std::abort();
}

#define CHECK(expression)                                      \
    do {                                                       \
        if (!(expression)) fail(#expression, __LINE__);        \
    } while (false)

bool has_cuda_device() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) cudaGetLastError();
    return status == cudaSuccess && count > 0;
}

pi05::Pi05ResolvedWeight upload_bf16(
    frt_ctx context,
    const char* name,
    const std::uint16_t* values,
    std::size_t count,
    const flashrt::modalities::Shape& shape) {
    frt_buffer buffer =
        frt_buffer_alloc(context, name, count * sizeof(std::uint16_t));
    CHECK(buffer != nullptr);
    CHECK(cudaMemcpy(
              frt_buffer_dptr(buffer), values,
              count * sizeof(std::uint16_t), cudaMemcpyHostToDevice) ==
          cudaSuccess);
    pi05::Pi05ResolvedWeight weight;
    weight.device_data = frt_buffer_dptr(buffer);
    weight.bytes = count * sizeof(std::uint16_t);
    weight.storage = pi05::Pi05WeightStorage::kBFloat16;
    weight.shape = shape;
    return weight;
}

void test_single_and_pair_are_exact() {
    frt_ctx context = frt_ctx_create();
    CHECK(context != nullptr);
    const std::array<std::uint16_t, 4> left = {
        0x3f80, 0xc000, 0x4040, 0xc080};
    const std::array<std::uint16_t, 4> right = {
        0x40a0, 0xc0c0, 0x40e0, 0xc100};
    const std::array<std::uint16_t, 8> merged = {
        left[0], left[1], right[0], right[1],
        left[2], left[3], right[2], right[3],
    };
    pi05::Pi05ResolvedWeight gate = upload_bf16(
        context, "fp8_pack_gate", left.data(), left.size(),
        flashrt::modalities::Shape({2, 2}));
    pi05::Pi05ResolvedWeight up = upload_bf16(
        context, "fp8_pack_up", right.data(), right.size(),
        flashrt::modalities::Shape({2, 2}));
    pi05::Pi05ResolvedWeight contiguous = upload_bf16(
        context, "fp8_pack_merged", merged.data(), merged.size(),
        flashrt::modalities::Shape({2, 4}));
    pi05::Pi05ResolvedWeight fused;

    sm120::Sm120Fp8WeightPacker packer(context);
    const pi05::Pi05LinearWeightKey pair_key = {
        pi05::Pi05LinearDomain::kEncoder,
        pi05::Pi05LinearRole::kMlpGateUpGroup, 3};
    CHECK(packer.record({pair_key, &gate, &up, &fused}).ok_status());
    const pi05::Pi05LinearWeightKey single_key = {
        pi05::Pi05LinearDomain::kVision,
        pi05::Pi05LinearRole::kMlpUp, 4};
    CHECK(packer.record(
              {single_key, &contiguous, nullptr, nullptr})
              .ok_status());
    CHECK(packer.finish().ok_status());

    CHECK(packer.size() == 2);
    CHECK(packer.merge_scratch_bytes() == 16);
    CHECK(!gate.device_data && !up.device_data);
    CHECK(fused.device_data && fused.scale_data);
    CHECK(fused.storage == pi05::Pi05WeightStorage::kFp8E4M3);
    CHECK(fused.shape.rank == 2 && fused.shape.dims[0] == 2 &&
          fused.shape.dims[1] == 4 && fused.bytes == 8);
    CHECK(contiguous.device_data && contiguous.scale_data);
    CHECK(contiguous.storage == pi05::Pi05WeightStorage::kFp8E4M3);

    std::array<std::uint8_t, 8> pair_values{};
    std::array<std::uint8_t, 8> single_values{};
    float pair_scale = 0.0f;
    float single_scale = 0.0f;
    CHECK(cudaMemcpy(
              pair_values.data(), fused.device_data, pair_values.size(),
              cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(
              single_values.data(), contiguous.device_data,
              single_values.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(
              &pair_scale, fused.scale_data, sizeof(pair_scale),
              cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(
              &single_scale, contiguous.scale_data, sizeof(single_scale),
              cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(pair_values == single_values);
    CHECK(std::memcmp(&pair_scale, &single_scale, sizeof(float)) == 0);

    const sm120::Sm120Fp8PackedWeight* pair = packer.packed_weight(0);
    const sm120::Sm120Fp8PackedWeight* single = packer.packed_weight(1);
    CHECK(pair && single && !packer.packed_weight(2));
    CHECK(pair->key.domain == pair_key.domain &&
          pair->key.role == pair_key.role &&
          pair->key.layer == pair_key.layer);
    CHECK(single->key.domain == single_key.domain &&
          single->key.role == single_key.role &&
          single->key.layer == single_key.layer);
    CHECK(!packer.finish().ok_status());
    frt_ctx_destroy(context);
}

void test_invalid_group_fails_closed() {
    frt_ctx context = frt_ctx_create();
    CHECK(context != nullptr);
    const std::array<std::uint16_t, 4> values = {
        0x3f80, 0x4000, 0x4040, 0x4080};
    pi05::Pi05ResolvedWeight first = upload_bf16(
        context, "fp8_pack_invalid", values.data(), values.size(),
        flashrt::modalities::Shape({2, 2}));
    pi05::Pi05ResolvedWeight fused;
    sm120::Sm120Fp8WeightPacker packer(context);
    CHECK(!packer.record({{}, &first, nullptr, &fused}).ok_status());
    CHECK(!packer.finish().ok_status());

    const std::array<std::uint16_t, 3> odd_values = {
        0x3f80, 0x4000, 0x4040};
    pi05::Pi05ResolvedWeight odd = upload_bf16(
        context, "fp8_pack_odd", odd_values.data(), odd_values.size(),
        flashrt::modalities::Shape({1, 3}));
    sm120::Sm120Fp8WeightPacker odd_packer(context);
    CHECK(!odd_packer.record({{}, &odd, nullptr, nullptr}).ok_status());
    CHECK(!odd_packer.finish().ok_status());
    frt_ctx_destroy(context);
}

}  // namespace

int main() {
    if (!has_cuda_device()) {
        std::cout << "SKIP - no CUDA device\n";
        return 0;
    }
    test_single_and_pair_are_exact();
    test_invalid_group_fails_closed();
    return 0;
}
