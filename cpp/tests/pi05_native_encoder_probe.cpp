#include "flashrt/cpp/models/pi05/native_bf16_forward.h"
#include "flashrt/cpp/models/pi05/native_weight_materializer.h"

#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct CaptureArgs {
    const flashrt::models::pi05::NativeBf16Forward* forward = nullptr;
    const flashrt::models::pi05::NativeDeviceWeightStore* weights = nullptr;
    flashrt::models::pi05::NativeWorkspace* workspace = nullptr;
    flashrt::models::pi05::NativeRtxAttentionWorkspace* attention = nullptr;
    const flashrt::models::pi05::NativeRtxAttentionDriver* attention_driver =
        nullptr;
    bool recorded = false;
};

void record_encoder(void* user, void* stream) {
    auto* args = static_cast<CaptureArgs*>(user);
    args->recorded = args->forward
        ->encoder(*args->weights, args->workspace, args->attention,
                  args->attention_driver,
                  reinterpret_cast<std::uintptr_t>(stream))
        .ok_status();
}

bool write_buffer(std::ofstream* file, const void* device, std::size_t elements) {
    std::vector<std::uint16_t> host(elements);
    if (cudaMemcpy(host.data(), device, elements * sizeof(std::uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    file->write(reinterpret_cast<const char*>(host.data()),
                static_cast<std::streamsize>(host.size() *
                                             sizeof(std::uint16_t)));
    return file->good();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: pi05_native_encoder_probe CHECKPOINT OUTPUT\n";
        return 2;
    }
    using namespace flashrt::models::pi05;
    flashrt::loader::SafetensorsFile source;
    if (!source.open(std::string(argv[1]) + "/model.safetensors")) {
        std::cerr << source.error() << '\n';
        return 2;
    }
    frt_ctx ctx = frt_ctx_create();
    if (!ctx) return 1;
    NativeDeviceWeightStore weights(ctx);
    NativeWeightMaterializer materializer(source, &weights);
    flashrt::modalities::Status st = flashrt::modalities::Status::ok();
    for (int layer = 0; layer < 18 && st.ok_status(); ++layer) {
        st = materializer.materialize_encoder_layer(layer);
    }
    NativeWorkspace workspace(ctx);
    NativeRtxAttentionWorkspace attention(ctx);
    if (!st.ok_status() ||
        !workspace.allocate(NativeWorkspaceConfig{}).ok_status() ||
        !attention.allocate(NativeRtxAttentionConfig{}).ok_status() ||
        !attention.set_fixed_prompt_length(200).ok_status()) {
        std::cerr << st.message << '\n';
        frt_ctx_destroy(ctx);
        return 1;
    }
    const auto* encoder_x = workspace.find("encoder_x");
    const auto* encoder_q = attention.find("attn_enc_Q");
    std::vector<std::uint16_t> host_x(712 * 2048, 0);
    for (int row = 0; row < 712; ++row) {
        for (int column = 0; column < 512; ++column) {
            const float value = float((row + column) % 15 - 7) / 8.0f;
            host_x[static_cast<std::size_t>(row) * 2048 + column] =
                flashrt::modalities::float_to_bfloat16(value);
        }
    }
    if (!encoder_x || !encoder_q ||
        cudaMemcpy(frt_buffer_dptr(encoder_x->buffer), host_x.data(),
                   host_x.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        frt_ctx_destroy(ctx);
        return 1;
    }
    NativeKernelDriver driver;
    NativeRtxAttentionDriver attention_driver(&attention);
    NativeBf16Forward forward(&driver);
    frt_graph graph = frt_graph_create(ctx, "native_encoder", 712);
    cudaStream_t stream = nullptr;
    if (!graph || cudaStreamCreate(&stream) != cudaSuccess ||
        frt_graph_bind(graph, "encoder_x", encoder_x->buffer) != FRT_OK ||
        frt_graph_bind(graph, "encoder_q", encoder_q->buffer) != FRT_OK) {
        frt_ctx_destroy(ctx);
        return 1;
    }
    CaptureArgs capture{&forward, &weights, &workspace, &attention,
                        &attention_driver, false};
    if (frt_graph_capture(graph, 712, record_encoder, &capture) != FRT_OK ||
        !capture.recorded) {
        frt_graph_destroy(graph);
        cudaStreamDestroy(stream);
        frt_ctx_destroy(ctx);
        return 1;
    }
    const int stream_id = frt_ctx_wrap_stream(ctx, stream);
    for (int i = 0; i < 100; ++i) {
        if (cudaMemcpyAsync(frt_buffer_dptr(encoder_x->buffer), host_x.data(),
                            host_x.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess ||
            frt_graph_replay(graph, 712, stream_id) != FRT_OK) {
            frt_graph_destroy(graph);
            cudaStreamDestroy(stream);
            frt_ctx_destroy(ctx);
            return 1;
        }
    }
    if (frt_graph_variant_count(graph) != 1 ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        frt_graph_destroy(graph);
        cudaStreamDestroy(stream);
        frt_ctx_destroy(ctx);
        return 1;
    }
    std::ofstream file(argv[2], std::ios::binary | std::ios::trunc);
    const bool ok = file &&
        write_buffer(&file, frt_buffer_dptr(encoder_x->buffer), 712 * 2048) &&
        write_buffer(&file, frt_buffer_dptr(encoder_q->buffer), 712 * 2048) &&
        write_buffer(&file, attention.encoder_k_layer_dptr(17), 712 * 256) &&
        write_buffer(&file, attention.encoder_v_layer_dptr(17), 712 * 256);
    frt_graph_destroy(graph);
    cudaStreamDestroy(stream);
    frt_ctx_destroy(ctx);
    if (!ok) return 1;
    std::cout << "PASS native encoder 18 layers\n";
    return 0;
}
