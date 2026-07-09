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
    std::string error;
};

void record_vision(void* user, void* stream) {
    auto* args = static_cast<CaptureArgs*>(user);
    const flashrt::modalities::Status st = args->forward->vision(
        *args->weights, args->workspace, args->attention,
        args->attention_driver, reinterpret_cast<std::uintptr_t>(stream));
    args->recorded = st.ok_status();
    args->error = st.message;
}

bool write_buffer(std::ofstream* file, const void* device,
                  std::size_t elements) {
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
        std::cerr << "usage: pi05_native_vision_probe CHECKPOINT OUTPUT\n";
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
    flashrt::modalities::Status st = materializer.materialize_vision_globals();
    for (int layer = 0; layer < 27 && st.ok_status(); ++layer) {
        st = materializer.materialize_vision_layer(layer);
    }
    NativeWorkspace workspace(ctx);
    NativeRtxAttentionWorkspace attention(ctx);
    if (!st.ok_status() ||
        !workspace.allocate(NativeWorkspaceConfig{}).ok_status() ||
        !workspace.expand_vision_position_embedding(weights).ok_status() ||
        !attention.allocate(NativeRtxAttentionConfig{}).ok_status()) {
        std::cerr << st.message << '\n';
        frt_ctx_destroy(ctx);
        return 1;
    }
    const auto* images = workspace.find("observation_images_normalized");
    const auto* vision_x = workspace.find("vision_x");
    const auto* encoder_x = workspace.find("encoder_x");
    std::vector<std::uint16_t> host_images(2 * 224 * 224 * 3);
    for (std::size_t i = 0; i < host_images.size(); ++i) {
        const float value = static_cast<float>(static_cast<int>(i % 257) - 128) /
                            128.0f;
        host_images[i] = flashrt::modalities::float_to_bfloat16(value);
    }
    if (!images || !vision_x || !encoder_x ||
        cudaMemcpy(frt_buffer_dptr(images->buffer), host_images.data(),
                   host_images.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        frt_ctx_destroy(ctx);
        return 1;
    }
    NativeKernelDriver driver;
    NativeRtxAttentionDriver attention_driver(&attention);
    NativeBf16Forward forward(&driver);
    frt_graph graph = frt_graph_create(ctx, "native_vision", 512);
    cudaStream_t stream = nullptr;
    if (!graph || cudaStreamCreate(&stream) != cudaSuccess ||
        frt_graph_bind(graph, "images", images->buffer) != FRT_OK ||
        frt_graph_bind(graph, "vision_x", vision_x->buffer) != FRT_OK ||
        frt_graph_bind(graph, "encoder_x", encoder_x->buffer) != FRT_OK) {
        frt_ctx_destroy(ctx);
        return 1;
    }
    CaptureArgs capture{&forward, &weights, &workspace, &attention,
                        &attention_driver, false, {}};
    const int capture_rc = frt_graph_capture(
        graph, 512, record_vision, &capture);
    if (capture_rc != FRT_OK || !capture.recorded) {
        std::cerr << "vision capture failed: rc=" << capture_rc
                  << " status=" << capture.error << '\n';
        frt_graph_destroy(graph);
        cudaStreamDestroy(stream);
        frt_ctx_destroy(ctx);
        return 1;
    }
    const int stream_id = frt_ctx_wrap_stream(ctx, stream);
    for (int i = 0; i < 100; ++i) {
        if (cudaMemcpyAsync(frt_buffer_dptr(images->buffer), host_images.data(),
                            host_images.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess ||
            frt_graph_replay(graph, 512, stream_id) != FRT_OK) {
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
        write_buffer(&file, frt_buffer_dptr(vision_x->buffer), 512 * 1152) &&
        write_buffer(&file, frt_buffer_dptr(encoder_x->buffer), 512 * 2048);
    frt_graph_destroy(graph);
    cudaStreamDestroy(stream);
    frt_ctx_destroy(ctx);
    if (!ok) return 1;
    std::cout << "PASS native vision 27 layers\n";
    return 0;
}
