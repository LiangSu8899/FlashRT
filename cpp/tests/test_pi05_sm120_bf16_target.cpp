#include "flashrt/cpp/modalities/types.h"
#include "flashrt/cpp/models/pi05/model/dims.h"
#include "flashrt/cpp/models/pi05/targets/sm120/bf16_linear.h"
#include "flashrt/cpp/models/pi05/targets/sm120/bf16_scratch.h"
#include "flashrt/exec.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

using flashrt::models::pi05::Pi05ResolvedWeight;
using flashrt::models::pi05::targets::sm120::Sm120Bf16Linear;

struct CaptureArgs final {
    const Sm120Bf16Linear* linear = nullptr;
    const Pi05ResolvedWeight* weight = nullptr;
    const void* input = nullptr;
    void* output = nullptr;
    bool recorded = false;
};

void record_linear(void* user, void* stream) {
    auto* args = static_cast<CaptureArgs*>(user);
    args->recorded =
        args->linear
            ->run(*args->weight, args->input, args->output, 2, 2, 2,
                  reinterpret_cast<std::uintptr_t>(stream))
            .ok_status();
}

void expect_shape(
    const flashrt::models::pi05::targets::sm120::Sm120DeviceBuffer& buffer,
    std::uint64_t rows,
    std::uint64_t columns) {
    assert(buffer.buffer);
    assert(buffer.device_data());
    assert(buffer.dtype == flashrt::modalities::DType::kBFloat16);
    assert(buffer.shape.rank == 2);
    assert(buffer.shape.dims[0] == rows);
    assert(buffer.shape.dims[1] == columns);
    assert(buffer.bytes() == rows * columns * sizeof(std::uint16_t));
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || !device_count) {
        cudaGetLastError();
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }
    int device = 0;
    cudaDeviceProp properties{};
    assert(cudaGetDevice(&device) == cudaSuccess);
    assert(cudaGetDeviceProperties(&properties, device) == cudaSuccess);
    if (properties.major != 12 || properties.minor != 0) {
        std::printf("SKIP - SM120 target needs compute capability 12.0\n");
        return 0;
    }

    using namespace flashrt::models::pi05;
    using namespace flashrt::models::pi05::targets::sm120;

    Pi05ShapeConfig config;
    config.num_views = 3;
    config.max_prompt_tokens = 64;
    config.chunk = 10;
    config.num_steps = 10;
    config.vision_pool_factor = 2;
    config.state_dim = 8;
    config.robot_action_dim = 7;
    Pi05ResolvedShape shape;
    assert(resolve_pi05_shape(config, &shape).ok_status());

    Sm120Bf16ScratchBacking missing_context(nullptr);
    assert(!missing_context.allocate(shape).ok_status());
    assert(missing_context.allocation_count() == 0);

    frt_ctx context = frt_ctx_create();
    assert(context);
    {
        Sm120Bf16ScratchBacking scratch(context);
        Pi05ResolvedShape invalid_shape = shape;
        ++invalid_shape.encoder_sequence;
        assert(!scratch.allocate(invalid_shape).ok_status());
        assert(scratch.allocation_count() == 0);
        assert(!scratch.allocated());
        assert(scratch.allocate(shape).ok_status());
        assert(!scratch.allocate(shape).ok_status());
        assert(scratch.allocation_count() == 12);
        assert(scratch.allocated());

        expect_shape(scratch.vision().normalized, shape.vision_sequence,
                     kPi05ModelDims.vision_width);
        expect_shape(scratch.vision().qkv, shape.vision_sequence,
                     3 * kPi05ModelDims.vision_width);
        expect_shape(scratch.vision().hidden, shape.vision_sequence,
                     kPi05ModelDims.vision_hidden);
        expect_shape(scratch.encoder().normalized, shape.encoder_sequence,
                     kPi05ModelDims.encoder_width);
        expect_shape(
            scratch.encoder().qkv, shape.encoder_sequence,
            kPi05ModelDims.encoder_width +
                2 * kPi05ModelDims.encoder_kv_heads *
                    kPi05ModelDims.encoder_head_dim);
        expect_shape(scratch.encoder().gate, shape.encoder_sequence,
                     kPi05ModelDims.encoder_hidden);
        expect_shape(scratch.encoder().hidden, shape.encoder_sequence,
                     kPi05ModelDims.encoder_hidden);
        expect_shape(scratch.decoder().normalized, shape.chunk,
                     kPi05ModelDims.decoder_width);
        expect_shape(scratch.decoder().gate, shape.chunk,
                     kPi05ModelDims.decoder_width);
        expect_shape(
            scratch.decoder().qkv, shape.chunk,
            kPi05ModelDims.decoder_heads * kPi05ModelDims.decoder_head_dim +
                2 * kPi05ModelDims.decoder_kv_heads *
                    kPi05ModelDims.decoder_head_dim);
        expect_shape(scratch.decoder().gate_projection, shape.chunk,
                     kPi05ModelDims.decoder_hidden);
        expect_shape(scratch.decoder().hidden, shape.chunk,
                     kPi05ModelDims.decoder_hidden);

        const std::array<const Sm120DeviceBuffer*, 12> buffers = {
            &scratch.vision().normalized,
            &scratch.vision().qkv,
            &scratch.vision().hidden,
            &scratch.encoder().normalized,
            &scratch.encoder().qkv,
            &scratch.encoder().gate,
            &scratch.encoder().hidden,
            &scratch.decoder().normalized,
            &scratch.decoder().gate,
            &scratch.decoder().qkv,
            &scratch.decoder().gate_projection,
            &scratch.decoder().hidden,
        };
        std::size_t expected_bytes = 0;
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            expected_bytes += buffers[i]->bytes();
            for (std::size_t j = 0; j < i; ++j) {
                assert(buffers[i]->device_data() != buffers[j]->device_data());
            }
        }
        assert(scratch.allocated_bytes() == expected_bytes);

        frt_buffer input_buffer =
            frt_buffer_alloc(context, "sm120_bf16_linear_input", 8);
        frt_buffer weight_buffer =
            frt_buffer_alloc(context, "sm120_bf16_linear_weight", 8);
        frt_buffer output_buffer =
            frt_buffer_alloc(context, "sm120_bf16_linear_output", 8);
        assert(input_buffer && weight_buffer && output_buffer);

        const std::array<std::uint16_t, 4> input = {
            flashrt::modalities::float_to_bfloat16(1.0f),
            flashrt::modalities::float_to_bfloat16(2.0f),
            flashrt::modalities::float_to_bfloat16(3.0f),
            flashrt::modalities::float_to_bfloat16(4.0f),
        };
        const std::array<std::uint16_t, 4> identity = {
            flashrt::modalities::float_to_bfloat16(1.0f),
            flashrt::modalities::float_to_bfloat16(0.0f),
            flashrt::modalities::float_to_bfloat16(0.0f),
            flashrt::modalities::float_to_bfloat16(1.0f),
        };
        assert(cudaMemcpy(frt_buffer_dptr(input_buffer), input.data(), 8,
                          cudaMemcpyHostToDevice) == cudaSuccess);
        assert(cudaMemcpy(frt_buffer_dptr(weight_buffer), identity.data(), 8,
                          cudaMemcpyHostToDevice) == cudaSuccess);

        Pi05ResolvedWeight weight;
        weight.device_data = frt_buffer_dptr(weight_buffer);
        weight.bytes = 8;
        weight.storage = Pi05WeightStorage::kBFloat16;
        weight.shape = flashrt::modalities::Shape({2, 2});

        Sm120Bf16Linear linear;
        assert(linear.status().ok_status());
        assert(!linear.run(weight, nullptr, frt_buffer_dptr(output_buffer), 2,
                           2, 2, 0)
                    .ok_status());
        assert(!linear.run(weight, frt_buffer_dptr(input_buffer), nullptr, 2,
                           2, 2, 0)
                    .ok_status());
        assert(!linear.run(weight, frt_buffer_dptr(input_buffer),
                           frt_buffer_dptr(output_buffer), 0, 2, 2, 0)
                    .ok_status());
        cudaStream_t stream = nullptr;
        assert(cudaStreamCreate(&stream) == cudaSuccess);
        const Pi05Stream native_stream =
            reinterpret_cast<std::uintptr_t>(stream);
        assert(linear.run(weight, frt_buffer_dptr(input_buffer),
                          frt_buffer_dptr(output_buffer), 2, 2, 2,
                          native_stream)
                   .ok_status());

        Pi05ResolvedWeight patch_view = weight;
        patch_view.shape = flashrt::modalities::Shape({1, 1, 2, 2});
        assert(linear.run(patch_view, frt_buffer_dptr(input_buffer),
                          frt_buffer_dptr(output_buffer), 2, 2, 2,
                          native_stream)
                   .ok_status());
        Pi05ResolvedWeight invalid_weight = weight;
        invalid_weight.shape = flashrt::modalities::Shape({1, 2, 2, 1});
        assert(!linear.run(invalid_weight, frt_buffer_dptr(input_buffer),
                           frt_buffer_dptr(output_buffer), 2, 2, 2,
                           native_stream)
                    .ok_status());
        invalid_weight = weight;
        invalid_weight.storage = Pi05WeightStorage::kFp8E4M3;
        assert(!linear.run(invalid_weight, frt_buffer_dptr(input_buffer),
                           frt_buffer_dptr(output_buffer), 2, 2, 2,
                           native_stream)
                    .ok_status());
        invalid_weight = weight;
        invalid_weight.bytes = 6;
        assert(!linear.run(invalid_weight, frt_buffer_dptr(input_buffer),
                           frt_buffer_dptr(output_buffer), 2, 2, 2,
                           native_stream)
                    .ok_status());

        frt_graph graph = frt_graph_create(context, "sm120_bf16_linear", 1);
        assert(graph);
        assert(frt_graph_bind(graph, "input", input_buffer) == FRT_OK);
        assert(frt_graph_bind(graph, "weight", weight_buffer) == FRT_OK);
        assert(frt_graph_bind(graph, "output", output_buffer) == FRT_OK);
        CaptureArgs capture{&linear, &weight, frt_buffer_dptr(input_buffer),
                            frt_buffer_dptr(output_buffer), false};
        assert(frt_graph_capture(graph, 1, record_linear, &capture) == FRT_OK);
        assert(capture.recorded);
        assert(frt_graph_variant_count(graph) == 1);
        const int stream_id = frt_ctx_wrap_stream(context, stream);
        assert(stream_id >= 0);
        for (int replay = 0; replay < 100; ++replay) {
            assert(frt_graph_replay(graph, 1, stream_id) == FRT_OK);
        }
        assert(frt_graph_variant_count(graph) == 1);
        assert(cudaStreamSynchronize(stream) == cudaSuccess);

        std::array<std::uint16_t, 4> output{};
        assert(cudaMemcpy(output.data(), frt_buffer_dptr(output_buffer), 8,
                          cudaMemcpyDeviceToHost) == cudaSuccess);
        for (std::size_t i = 0; i < output.size(); ++i) {
            assert(std::fabs(
                       flashrt::modalities::bfloat16_to_float(output[i]) -
                       flashrt::modalities::bfloat16_to_float(input[i])) <
                   0.001f);
        }
        frt_graph_destroy(graph);
        assert(cudaStreamDestroy(stream) == cudaSuccess);
    }
    frt_ctx_destroy(context);
    std::printf("PASS - PI0.5 SM120 BF16 target foundation\n");
    return 0;
}
