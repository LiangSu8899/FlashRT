#include "flashrt/cpp/modalities/types.h"
#include "flashrt/cpp/models/pi05/model/dims.h"
#include "flashrt/cpp/models/pi05/support/native_resource_resolver.h"
#include "flashrt/cpp/models/pi05/support/native_workspace.h"
#include "flashrt/cpp/models/pi05/targets/sm120/attention.h"
#include "flashrt/cpp/models/pi05/targets/sm120/attention_driver.h"
#include "flashrt/exec.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using flashrt::models::pi05::targets::sm120::Sm120AttentionBuffer;
using flashrt::models::pi05::targets::sm120::Sm120AttentionDriver;

struct CaptureArgs final {
    const Sm120AttentionDriver* driver = nullptr;
    bool recorded = false;
};

void record_attention(void* user, void* stream) {
    auto* args = static_cast<CaptureArgs*>(user);
    const std::uintptr_t native_stream =
        reinterpret_cast<std::uintptr_t>(stream);
    args->recorded =
        args->driver->vision(native_stream).ok_status() &&
        args->driver->encoder(0, native_stream).ok_status() &&
        args->driver->decoder(0, native_stream).ok_status();
}

std::size_t elements(const Sm120AttentionBuffer& buffer) {
    return static_cast<std::size_t>(buffer.shape.elements());
}

void upload_constant(const Sm120AttentionBuffer& buffer, float value) {
    std::vector<std::uint16_t> host(
        elements(buffer), flashrt::modalities::float_to_bfloat16(value));
    assert(cudaMemcpy(buffer.device_data(), host.data(),
                      host.size() * sizeof(host[0]),
                      cudaMemcpyHostToDevice) == cudaSuccess);
}

void upload_cache_rows(const Sm120AttentionBuffer& buffer,
                       int total_keys,
                       bool row_values) {
    using flashrt::models::pi05::kPi05ModelDims;
    std::vector<std::uint16_t> host(elements(buffer), 0);
    const std::size_t layer_elements =
        static_cast<std::size_t>(total_keys) *
        kPi05ModelDims.encoder_head_dim;
    for (int row = 0; row < total_keys; ++row) {
        const float source = row_values ? static_cast<float>(row + 1) : 0.0f;
        const std::uint16_t value =
            flashrt::modalities::float_to_bfloat16(source);
        for (int column = 0; column < kPi05ModelDims.encoder_head_dim;
             ++column) {
            host[static_cast<std::size_t>(row) *
                     kPi05ModelDims.encoder_head_dim + column] = value;
        }
    }
    assert(cudaMemcpy(buffer.device_data(), host.data(), layer_elements *
                          sizeof(host[0]), cudaMemcpyHostToDevice) ==
           cudaSuccess);
}

void expect_constant(void* device, std::size_t count, float expected) {
    std::vector<std::uint16_t> host(count);
    assert(cudaMemcpy(host.data(), device, count * sizeof(host[0]),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    for (std::uint16_t value : host) {
        assert(std::fabs(flashrt::modalities::bfloat16_to_float(value) -
                         expected) < 0.03f);
    }
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || !device_count) {
        cudaGetLastError();
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }

    using namespace flashrt::models::pi05;
    using namespace flashrt::models::pi05::targets::sm120;

    Pi05ShapeConfig shape_config;
    shape_config.num_views = 1;
    shape_config.max_prompt_tokens = 112;
    shape_config.chunk = 2;
    shape_config.num_steps = 1;
    shape_config.vision_pool_factor = 4;
    shape_config.state_dim = 8;
    shape_config.robot_action_dim = 7;
    Pi05ResolvedShape shape;
    assert(resolve_pi05_shape(shape_config, &shape).ok_status());
    assert(shape.encoder_vision_sequence == 16);
    assert(shape.encoder_sequence == 128);
    assert(shape.total_attention_keys == 130);

    frt_ctx context = frt_ctx_create();
    assert(context);
    {
        NativeWorkspaceConfig workspace_config;
        workspace_config.num_views = shape.num_views;
        workspace_config.max_prompt_tokens = shape.max_prompt_tokens;
        workspace_config.chunk_size = shape.chunk;
        workspace_config.num_steps = shape.num_steps;
        workspace_config.vision_pool_factor = shape.vision_pool_factor;
        NativeWorkspaceRequirements requirements;
        requirements.activation_dtype =
            flashrt::modalities::DType::kBFloat16;
        NativeWorkspace workspace(context);
        assert(workspace.allocate(workspace_config, requirements).ok_status());
        assert(workspace.logical_size() == 21);
        assert(workspace.find("decoder_action_buf"));

        Sm120AttentionBacking backing(context);
        Pi05TargetBufferBindings unchanged;
        std::uint8_t sentinel = 0;
        unchanged.key_cache.storage_identity = &sentinel;
        assert(!backing.make_target_bindings(&unchanged).ok_status());
        assert(unchanged.key_cache.storage_identity == &sentinel);
        assert(backing.allocate(shape).ok_status());
        assert(!backing.allocate(shape).ok_status());
        assert(backing.allocation_count() == 20);
        assert(backing.decoder_splits() == 3);
        assert(backing.cache_layer_stride_bytes() == 130 * 256 * 2);
        assert(backing.key_layer_data(0) == backing.cache().key.device_data());
        assert(static_cast<unsigned char*>(backing.key_layer_data(17)) ==
               static_cast<unsigned char*>(backing.cache().key.device_data()) +
                   17 * backing.cache_layer_stride_bytes());
        assert(!backing.key_layer_data(18));

        Pi05TargetBufferBindings target;
        assert(backing.make_target_bindings(&target).ok_status());
        Pi05ResolvedBuffers resolved;
        assert(resolve_pi05_native_buffers(workspace, target, shape, &resolved)
                   .ok_status());
        assert(resolved.key_cache.logical_spec.rank == 3);
        assert(resolved.key_cache.physical_shape.rank == 4);
        assert(resolved.key_cache.physical_shape.dims[2] == 1);
        assert(resolved.action_delta.storage_identity ==
               frt_buffer_dptr(workspace.find("decoder_action_buf")->buffer));

        const void* control_identity =
            backing.controls().encoder_valid_tokens.device_data();
        const std::size_t allocated_bytes = backing.allocated_bytes();
        for (int prompt = 0; prompt <= shape.max_prompt_tokens; ++prompt) {
            assert(backing.set_prompt_length(prompt).ok_status());
            assert(backing.controls().encoder_valid_tokens.device_data() ==
                   control_identity);
            assert(backing.allocated_bytes() == allocated_bytes);
        }
        assert(!backing.set_prompt_length(shape.max_prompt_tokens + 1)
                    .ok_status());
        assert(backing.set_prompt_length(1).ok_status());
        std::int32_t controls[3] = {};
        assert(cudaMemcpy(&controls[0],
                          backing.controls().encoder_valid_tokens.device_data(),
                          sizeof(controls[0]), cudaMemcpyDeviceToHost) ==
               cudaSuccess);
        assert(cudaMemcpy(&controls[1],
                          backing.controls().decoder_valid_tokens.device_data(),
                          sizeof(controls[1]), cudaMemcpyDeviceToHost) ==
               cudaSuccess);
        assert(cudaMemcpy(&controls[2],
                          backing.controls().decoder_position.device_data(),
                          sizeof(controls[2]), cudaMemcpyDeviceToHost) ==
               cudaSuccess);
        assert(controls[0] == 17 && controls[1] == 19 && controls[2] == 17);

        Sm120AttentionDriver invalid_driver(nullptr);
        assert(!invalid_driver.status().ok_status());
        cudaDeviceProp properties{};
        int device = 0;
        assert(cudaGetDevice(&device) == cudaSuccess);
        assert(cudaGetDeviceProperties(&properties, device) == cudaSuccess);
        Sm120AttentionDriver driver(&backing);
        if (properties.major != 12 || properties.minor != 0) {
            assert(!driver.status().ok_status());
            std::printf("SKIP - SM120 target needs compute capability 12.0\n");
            frt_ctx_destroy(context);
            return 0;
        }
        assert(driver.status().ok_status());
        assert(driver.multiprocessor_count() == properties.multiProcessorCount);

        upload_constant(backing.vision().query, 0.0f);
        upload_constant(backing.vision().key, 0.0f);
        upload_constant(backing.vision().value, 2.0f);
        upload_constant(backing.encoder().query, 0.0f);
        upload_constant(backing.decoder().query, 0.0f);
        upload_cache_rows(backing.cache().key, shape.total_attention_keys,
                          false);
        upload_cache_rows(backing.cache().value, shape.total_attention_keys,
                          true);

        cudaStream_t stream = nullptr;
        assert(cudaStreamCreate(&stream) == cudaSuccess);
        const std::uintptr_t native_stream =
            reinterpret_cast<std::uintptr_t>(stream);
        assert(driver.vision(native_stream).ok_status());
        assert(driver.encoder(0, native_stream).ok_status());
        assert(driver.decoder(0, native_stream).ok_status());
        assert(cudaStreamSynchronize(stream) == cudaSuccess);
        expect_constant(driver.vision_output(), 256 * 16 * 72, 2.0f);
        expect_constant(driver.encoder_output(), 128 * 8 * 256, 9.0f);
        expect_constant(driver.decoder_output(), 2 * 8 * 256, 10.0f);

        frt_graph graph = frt_graph_create(context, "sm120_attention", 1);
        assert(graph);
        assert(frt_graph_bind(graph, "vision_query",
                              backing.vision().query.buffer) == FRT_OK);
        assert(frt_graph_bind(graph, "encoder_query",
                              backing.encoder().query.buffer) == FRT_OK);
        assert(frt_graph_bind(graph, "decoder_query",
                              backing.decoder().query.buffer) == FRT_OK);
        CaptureArgs capture{&driver, false};
        assert(frt_graph_capture(graph, 1, record_attention, &capture) ==
               FRT_OK);
        assert(capture.recorded);
        assert(frt_graph_variant_count(graph) == 1);
        const int stream_id = frt_ctx_wrap_stream(context, stream);
        assert(stream_id >= 0);
        assert(backing.set_prompt_length(0).ok_status());
        for (int replay = 0; replay < 100; ++replay) {
            assert(frt_graph_replay(graph, 1, stream_id) == FRT_OK);
        }
        assert(frt_graph_variant_count(graph) == 1);
        assert(backing.allocated_bytes() == allocated_bytes);
        assert(cudaStreamSynchronize(stream) == cudaSuccess);
        expect_constant(driver.encoder_output(), 128 * 8 * 256, 8.5f);
        expect_constant(driver.decoder_output(), 2 * 8 * 256, 9.5f);
        frt_graph_destroy(graph);
        assert(cudaStreamDestroy(stream) == cudaSuccess);
    }
    frt_ctx_destroy(context);
    std::printf("PASS - PI0.5 SM120 attention target\n");
    return 0;
}
