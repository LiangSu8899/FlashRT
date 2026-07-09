#include "flashrt/cpp/models/pi05/native_bf16_forward.h"
#include "flashrt/cpp/models/pi05/native_style_precompute.h"
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
    int start_step = 0;
    int steps = 10;
    bool recorded = false;
    std::string error;
};

void record_diffusion(void* user, void* stream) {
    auto* args = static_cast<CaptureArgs*>(user);
    flashrt::modalities::Status st = flashrt::modalities::Status::ok();
    const std::uintptr_t native_stream =
        reinterpret_cast<std::uintptr_t>(stream);
    if (args->start_step == 0 && args->steps == 10) {
        st = args->forward->diffusion(
            *args->weights, args->workspace, args->attention,
            args->attention_driver, native_stream);
    } else {
        for (int offset = 0; offset < args->steps && st.ok_status(); ++offset) {
            const int step = args->start_step + offset;
            st = args->forward->diffusion_step(
                step, *args->weights, args->workspace, args->attention,
                args->attention_driver, native_stream);
        }
    }
    args->recorded = st.ok_status();
    args->error = st.message;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "usage: pi05_native_diffusion_probe CHECKPOINT OUTPUT "
                     "[STEPS [START_STEP]]\n";
        return 2;
    }
    const int steps = argc >= 4 ? std::stoi(argv[3]) : 10;
    const int start_step = argc == 5 ? std::stoi(argv[4]) : 0;
    if (steps < 1 || start_step < 0 || start_step + steps > 10) return 2;
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
    flashrt::modalities::Status st = materializer.materialize_decoder_globals(10);
    for (int layer = 0; layer < 18 && st.ok_status(); ++layer) {
        st = materializer.materialize_decoder_layer(layer, false);
    }
    NativeWorkspace workspace(ctx);
    NativeRtxAttentionWorkspace attention(ctx);
    if (!st.ok_status() ||
        !workspace.allocate(NativeWorkspaceConfig{}).ok_status() ||
        !workspace.update_decoder_rope(200).ok_status() ||
        !attention.allocate(NativeRtxAttentionConfig{}).ok_status() ||
        !attention.set_fixed_prompt_length(200).ok_status()) {
        std::cerr << st.message << '\n';
        frt_ctx_destroy(ctx);
        return 1;
    }
    NativeKernelDriver driver;
    NativeStylePrecomputer precomputer(&driver);
    st = precomputer.run(weights, &workspace, 0);
    if (!st.ok_status()) {
        std::cerr << st.message << '\n';
        frt_ctx_destroy(ctx);
        return 1;
    }
    const auto* noise = workspace.find("diffusion_noise");
    const auto* cache_k = attention.find("attn_enc_K");
    const auto* cache_v = attention.find("attn_enc_V");
    std::vector<std::uint16_t> host_noise(10 * 32);
    for (std::size_t i = 0; i < host_noise.size(); ++i) {
        const float value = static_cast<float>(static_cast<int>(i % 23) - 11) /
                            12.0f;
        host_noise[i] = flashrt::modalities::float_to_bfloat16(value);
    }
    std::vector<std::uint16_t> host_k(18 * 722 * 256);
    std::vector<std::uint16_t> host_v(host_k.size());
    for (int layer = 0; layer < 18; ++layer) {
        for (int row = 0; row < 722; ++row) {
            for (int column = 0; column < 256; ++column) {
                const std::size_t offset =
                    (static_cast<std::size_t>(layer) * 722 + row) * 256 +
                    column;
                host_k[offset] = flashrt::modalities::float_to_bfloat16(
                    static_cast<float>((layer + row + column) % 17 - 8) /
                    16.0f);
                host_v[offset] = flashrt::modalities::float_to_bfloat16(
                    static_cast<float>((2 * layer + row + 3 * column) % 19 -
                                        9) /
                    16.0f);
            }
        }
    }
    if (!noise || !cache_k || !cache_v ||
        cudaMemcpy(frt_buffer_dptr(noise->buffer), host_noise.data(),
                   host_noise.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(frt_buffer_dptr(cache_k->buffer), host_k.data(),
                   host_k.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(frt_buffer_dptr(cache_v->buffer), host_v.data(),
                   host_v.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        frt_ctx_destroy(ctx);
        return 1;
    }
    NativeRtxAttentionDriver attention_driver(&attention);
    NativeBf16Forward forward(&driver);
    frt_graph graph = frt_graph_create(ctx, "native_diffusion", 10);
    cudaStream_t stream = nullptr;
    if (!graph || cudaStreamCreate(&stream) != cudaSuccess ||
        frt_graph_bind(graph, "noise", noise->buffer) != FRT_OK ||
        frt_graph_bind(graph, "encoder_k", cache_k->buffer) != FRT_OK ||
        frt_graph_bind(graph, "encoder_v", cache_v->buffer) != FRT_OK) {
        frt_ctx_destroy(ctx);
        return 1;
    }
    CaptureArgs capture{&forward, &weights, &workspace, &attention,
                        &attention_driver, start_step, steps, false, {}};
    const int capture_rc = frt_graph_capture(
        graph, 10, record_diffusion, &capture);
    if (capture_rc != FRT_OK || !capture.recorded) {
        std::cerr << "diffusion capture failed: rc=" << capture_rc
                  << " status=" << capture.error << '\n';
        frt_graph_destroy(graph);
        cudaStreamDestroy(stream);
        frt_ctx_destroy(ctx);
        return 1;
    }
    const int stream_id = frt_ctx_wrap_stream(ctx, stream);
    for (int i = 0; i < 100; ++i) {
        if (cudaMemcpyAsync(frt_buffer_dptr(noise->buffer), host_noise.data(),
                            host_noise.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess ||
            frt_graph_replay(graph, 10, stream_id) != FRT_OK) {
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
    std::vector<std::uint16_t> output(host_noise.size());
    if (cudaMemcpy(output.data(), frt_buffer_dptr(noise->buffer),
                   output.size() * sizeof(std::uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        frt_graph_destroy(graph);
        cudaStreamDestroy(stream);
        frt_ctx_destroy(ctx);
        return 1;
    }
    std::ofstream file(argv[2], std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(output.data()),
               static_cast<std::streamsize>(output.size() *
                                            sizeof(std::uint16_t)));
    const bool ok = file.good();
    frt_graph_destroy(graph);
    cudaStreamDestroy(stream);
    frt_ctx_destroy(ctx);
    if (!ok) return 1;
    std::cout << "PASS native diffusion steps " << start_step << ".."
              << start_step + steps - 1 << '\n';
    return 0;
}
