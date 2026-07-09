#include "flashrt/cpp/models/pi05/native_bf16_forward.h"
#include "flashrt/cpp/modalities/types.h"
#include "flashrt/exec.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using flashrt::models::pi05::NativeBf16Forward;
using flashrt::models::pi05::NativeDeviceWeightStore;
using flashrt::models::pi05::NativeKernelDriver;
using flashrt::models::pi05::NativeRtxAttentionWorkspace;
using flashrt::models::pi05::NativeWorkspace;

std::vector<std::uint16_t> download(const void* device, std::size_t elements) {
    std::vector<std::uint16_t> result(elements);
    assert(cudaMemcpy(result.data(), device,
                      result.size() * sizeof(std::uint16_t),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    return result;
}

struct CaptureArgs {
    const NativeBf16Forward* forward = nullptr;
    const NativeDeviceWeightStore* weights = nullptr;
    NativeWorkspace* workspace = nullptr;
    NativeRtxAttentionWorkspace* attention = nullptr;
    bool recorded = false;
};

void record_encoder_qkv(void* user, void* stream) {
    auto* args = static_cast<CaptureArgs*>(user);
    args->recorded = args->forward
        ->encoder_qkv(17, *args->weights, args->workspace, args->attention,
                      reinterpret_cast<std::uintptr_t>(stream))
        .ok_status();
}

}  // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) {
        cudaGetLastError();
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }
    using namespace flashrt::models::pi05;
    frt_ctx ctx = frt_ctx_create();
    assert(ctx);
    NativeWorkspace workspace(ctx);
    NativeWorkspaceConfig workspace_config;
    workspace_config.num_views = 1;
    workspace_config.max_prompt_tokens = 1;
    workspace_config.chunk_size = 2;
    workspace_config.num_steps = 2;
    workspace_config.vision_pool_factor = 4;
    assert(workspace.allocate(workspace_config).ok_status());
    assert(workspace.encoder_sequence() == 17);

    NativeRtxAttentionWorkspace attention(ctx);
    NativeRtxAttentionConfig attention_config;
    attention_config.num_views = 1;
    attention_config.encoder_sequence = 17;
    attention_config.encoder_vision_sequence = 16;
    attention_config.chunk_size = 2;
    assert(attention.allocate(attention_config).ok_status());

    NativeKernelDriver driver;
    NativeBf16Forward forward(&driver);
    NativeDeviceWeightStore weights(ctx);
    assert(!forward.encoder_qkv(17, weights, &workspace, &attention, 0)
                .ok_status());

    NativeBf16Tensor qkv_weight;
    qkv_weight.shape = {2048, 2560};
    qkv_weight.values.assign(2048 * 2560, 0);
    const std::uint16_t one =
        flashrt::modalities::float_to_bfloat16(1.0f);
    for (int column = 0; column < 2048; ++column) {
        qkv_weight.values[static_cast<std::size_t>(column) * 2560 + column] =
            one;
    }
    for (int column = 0; column < 256; ++column) {
        qkv_weight.values[static_cast<std::size_t>(column) * 2560 +
                          2048 + column] = one;
        qkv_weight.values[static_cast<std::size_t>(256 + column) * 2560 +
                          2304 + column] = one;
    }
    assert(weights.upload("encoder_attn_qkv_w_17", qkv_weight).ok_status());

    const auto* encoder_x = workspace.find("encoder_x");
    assert(encoder_x);
    std::vector<std::uint16_t> host_x(17 * 2048, 0);
    for (int row = 0; row < 17; ++row) {
        for (int column = 0; column < 512; ++column) {
            const float value = float((row + column) % 15 - 7) / 8.0f;
            host_x[static_cast<std::size_t>(row) * 2048 + column] =
                flashrt::modalities::float_to_bfloat16(value);
        }
    }
    assert(cudaMemcpy(frt_buffer_dptr(encoder_x->buffer), host_x.data(),
                      host_x.size() * sizeof(std::uint16_t),
                      cudaMemcpyHostToDevice) == cudaSuccess);

    cudaStream_t stream = nullptr;
    assert(cudaStreamCreate(&stream) == cudaSuccess);
    const std::uintptr_t native_stream =
        reinterpret_cast<std::uintptr_t>(stream);
    assert(forward.encoder_qkv(17, weights, &workspace, &attention,
                               native_stream)
               .ok_status());
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    const auto* query_buffer = attention.find("attn_enc_Q");
    assert(query_buffer);
    const std::vector<std::uint16_t> expected_q = download(
        frt_buffer_dptr(query_buffer->buffer), 17 * 2048);
    const std::vector<std::uint16_t> expected_k =
        download(attention.encoder_k_layer_dptr(17), 17 * 256);
    const std::vector<std::uint16_t> expected_v =
        download(attention.encoder_v_layer_dptr(17), 17 * 256);

    assert(cudaMemset(frt_buffer_dptr(query_buffer->buffer), 0,
                      expected_q.size() * sizeof(std::uint16_t)) ==
           cudaSuccess);
    assert(cudaMemset(attention.encoder_k_layer_dptr(17), 0,
                      expected_k.size() * sizeof(std::uint16_t)) ==
           cudaSuccess);
    assert(cudaMemset(attention.encoder_v_layer_dptr(17), 0,
                      expected_v.size() * sizeof(std::uint16_t)) ==
           cudaSuccess);
    const auto* x_norm = workspace.find("encoder_x_norm");
    const auto* qkv = workspace.find("encoder_QKV");
    const auto* rms = workspace.find("encoder_rms_ones");
    const auto* rope = workspace.find("encoder_rope_weights");
    const auto* weight = weights.find("encoder_attn_qkv_w_17");
    assert(x_norm && qkv && rms && rope && weight);
    assert(driver.rms_norm_bf16(
                     frt_buffer_dptr(encoder_x->buffer),
                     frt_buffer_dptr(rms->buffer),
                     frt_buffer_dptr(x_norm->buffer), 17, 2048, 1e-6f,
                     native_stream)
               .ok_status());
    assert(driver.bf16_nn(
                     frt_buffer_dptr(x_norm->buffer),
                     frt_buffer_dptr(weight->buffer),
                     frt_buffer_dptr(qkv->buffer), 17, 2560, 2048,
                     native_stream)
               .ok_status());
    assert(driver.qkv_split_rope_bf16(
                     frt_buffer_dptr(qkv->buffer), frt_buffer_dptr(rope->buffer),
                     frt_buffer_dptr(query_buffer->buffer),
                     attention.encoder_k_layer_dptr(17),
                     attention.encoder_v_layer_dptr(17), 17, 2048, 256, 256,
                     256, native_stream)
               .ok_status());
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    assert(download(frt_buffer_dptr(query_buffer->buffer), 17 * 2048) ==
           expected_q);
    assert(download(attention.encoder_k_layer_dptr(17), 17 * 256) ==
           expected_k);
    assert(download(attention.encoder_v_layer_dptr(17), 17 * 256) ==
           expected_v);

    frt_graph graph = frt_graph_create(ctx, "native_encoder_qkv", 17);
    assert(graph);
    assert(frt_graph_bind(graph, "encoder_x", encoder_x->buffer) == FRT_OK);
    assert(frt_graph_bind(graph, "encoder_q", query_buffer->buffer) == FRT_OK);
    CaptureArgs capture{&forward, &weights, &workspace, &attention, false};
    assert(frt_graph_capture(graph, 17, record_encoder_qkv, &capture) == FRT_OK);
    assert(capture.recorded);
    const int stream_id = frt_ctx_wrap_stream(ctx, stream);
    assert(stream_id >= 0);
    for (int i = 0; i < 100; ++i) {
        assert(frt_graph_replay(graph, 17, stream_id) == FRT_OK);
    }
    assert(frt_graph_variant_count(graph) == 1);
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    assert(download(attention.encoder_k_layer_dptr(17), 17 * 256) ==
           expected_k);

    frt_graph_destroy(graph);
    assert(cudaStreamDestroy(stream) == cudaSuccess);
    frt_ctx_destroy(ctx);
    std::printf("PASS - Pi0.5 native BF16 encoder QKV\n");
    return 0;
}
