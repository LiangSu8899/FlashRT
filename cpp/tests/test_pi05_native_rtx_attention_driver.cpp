#include "flashrt/cpp/models/pi05/native_rtx_attention_driver.h"
#include "flashrt/cpp/modalities/types.h"
#include "flashrt/exec.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using flashrt::models::pi05::NativeAttentionBuffer;
using flashrt::models::pi05::NativeRtxAttentionDriver;

struct CaptureArgs {
    const NativeRtxAttentionDriver* driver = nullptr;
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

std::size_t elements(const NativeAttentionBuffer* buffer) {
    assert(buffer);
    std::size_t count = 1;
    for (std::uint64_t dim : buffer->shape) count *= dim;
    return count;
}

void upload_constant(const NativeAttentionBuffer* buffer, float value) {
    std::vector<std::uint16_t> host(
        elements(buffer), flashrt::modalities::float_to_bfloat16(value));
    assert(cudaMemcpy(frt_buffer_dptr(buffer->buffer), host.data(),
                      host.size() * sizeof(std::uint16_t),
                      cudaMemcpyHostToDevice) == cudaSuccess);
}

void upload_kv_rows(const NativeAttentionBuffer* buffer, int total_kv) {
    std::vector<std::uint16_t> host(elements(buffer), 0);
    for (int row = 0; row < total_kv; ++row) {
        const std::uint16_t value =
            flashrt::modalities::float_to_bfloat16(float(row + 1));
        for (int column = 0; column < 256; ++column) {
            host[static_cast<std::size_t>(row) * 256 + column] = value;
        }
    }
    assert(cudaMemcpy(frt_buffer_dptr(buffer->buffer), host.data(),
                      host.size() * sizeof(std::uint16_t),
                      cudaMemcpyHostToDevice) == cudaSuccess);
}

void expect_constant(void* device, std::size_t count, float expected) {
    std::vector<std::uint16_t> host(count);
    assert(cudaMemcpy(host.data(), device, count * sizeof(std::uint16_t),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    for (std::uint16_t value : host) {
        assert(std::fabs(flashrt::modalities::bfloat16_to_float(value) -
                         expected) < 0.02f);
    }
}

}  // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        cudaGetLastError();
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }
    cudaDeviceProp properties{};
    assert(cudaGetDeviceProperties(&properties, 0) == cudaSuccess);
    if (properties.major < 8) {
        std::printf("SKIP - BF16 FA2 needs compute capability 8.0+\n");
        return 0;
    }

    using namespace flashrt::models::pi05;
    NativeRtxAttentionDriver invalid_driver(nullptr);
    assert(!invalid_driver.status().ok_status());

    frt_ctx ctx = frt_ctx_create();
    assert(ctx);
    NativeRtxAttentionWorkspace workspace(ctx);
    NativeRtxAttentionConfig config;
    config.num_views = 1;
    config.encoder_sequence = 128;
    config.encoder_vision_sequence = 2;
    config.chunk_size = 2;
    assert(workspace.allocate(config).ok_status());
    assert(workspace.decoder_splits() == 3);
    assert(workspace.set_fixed_prompt_length(1).ok_status());

    NativeRtxAttentionDriver driver(&workspace);
    assert(driver.status().ok_status());
    assert(driver.num_sms() == properties.multiProcessorCount);

    upload_constant(workspace.find("attn_vis_Q"), 0.0f);
    upload_constant(workspace.find("attn_vis_K"), 0.0f);
    upload_constant(workspace.find("attn_vis_V"), 2.0f);
    upload_constant(workspace.find("attn_enc_Q"), 0.0f);
    upload_constant(workspace.find("attn_dec_Q"), 0.0f);
    upload_kv_rows(workspace.find("attn_enc_K"), 130);
    upload_kv_rows(workspace.find("attn_enc_V"), 130);

    cudaStream_t stream = nullptr;
    assert(cudaStreamCreate(&stream) == cudaSuccess);
    const std::uintptr_t native_stream =
        reinterpret_cast<std::uintptr_t>(stream);
    assert(driver.vision(native_stream).ok_status());
    assert(driver.encoder(0, native_stream).ok_status());
    assert(driver.decoder(0, native_stream).ok_status());
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    expect_constant(driver.vision_output(), 256 * 16 * 72, 2.0f);
    expect_constant(driver.encoder_output(), 128 * 8 * 256, 2.0f);
    expect_constant(driver.decoder_output(), 2 * 8 * 256, 3.0f);

    frt_graph graph = frt_graph_create(ctx, "native_rtx_attention", 1);
    assert(graph);
    assert(frt_graph_bind(graph, "vis_q",
                          workspace.find("attn_vis_Q")->buffer) == FRT_OK);
    assert(frt_graph_bind(graph, "enc_q",
                          workspace.find("attn_enc_Q")->buffer) == FRT_OK);
    assert(frt_graph_bind(graph, "dec_q",
                          workspace.find("attn_dec_Q")->buffer) == FRT_OK);
    CaptureArgs capture{&driver, false};
    assert(frt_graph_capture(graph, 1, record_attention, &capture) == FRT_OK);
    assert(capture.recorded);
    assert(frt_graph_variant_count(graph) == 1);
    const int stream_id = frt_ctx_wrap_stream(ctx, stream);
    assert(stream_id >= 0);
    assert(workspace.set_fixed_prompt_length(0).ok_status());
    for (int i = 0; i < 100; ++i) {
        assert(frt_graph_replay(graph, 1, stream_id) == FRT_OK);
    }
    assert(frt_graph_variant_count(graph) == 1);
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    expect_constant(driver.vision_output(), 256 * 16 * 72, 2.0f);
    expect_constant(driver.encoder_output(), 128 * 8 * 256, 1.5f);
    expect_constant(driver.decoder_output(), 2 * 8 * 256, 2.5f);

    frt_graph_destroy(graph);
    assert(cudaStreamDestroy(stream) == cudaSuccess);
    frt_ctx_destroy(ctx);
    std::printf("PASS - Pi0.5 native RTX FA2 driver\n");
    return 0;
}
