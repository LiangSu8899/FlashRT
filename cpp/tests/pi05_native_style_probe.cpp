#include "flashrt/cpp/models/pi05/native_style_precompute.h"
#include "flashrt/cpp/models/pi05/native_weight_materializer.h"

#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <string>

namespace {

bool write_buffer(const std::string& path,
                  const flashrt::models::pi05::NativeWorkspaceBuffer& buffer) {
    const std::size_t bytes = frt_buffer_bytes(buffer.buffer);
    std::vector<unsigned char> host(bytes);
    if (cudaMemcpy(host.data(), frt_buffer_dptr(buffer.buffer), bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(host.data()),
               static_cast<std::streamsize>(host.size()));
    return file.good();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: pi05_native_style_probe CHECKPOINT OUTPUT_PREFIX\n";
        return 2;
    }
    flashrt::loader::SafetensorsFile source;
    if (!source.open(std::string(argv[1]) + "/model.safetensors")) {
        std::cerr << source.error() << '\n';
        return 2;
    }
    frt_ctx ctx = frt_ctx_create();
    if (!ctx) return 1;
    flashrt::models::pi05::NativeDeviceWeightStore weights(ctx);
    flashrt::models::pi05::NativeWeightMaterializer materializer(source,
                                                                  &weights);
    for (int layer = 0; layer < 18; ++layer) {
        const flashrt::modalities::Status st =
            materializer.materialize_decoder_layer(layer, false);
        if (!st.ok_status()) {
            std::cerr << st.message << '\n';
            frt_ctx_destroy(ctx);
            return 1;
        }
    }
    flashrt::modalities::Status st =
        materializer.materialize_decoder_globals(10);
    if (!st.ok_status()) {
        std::cerr << st.message << '\n';
        frt_ctx_destroy(ctx);
        return 1;
    }
    flashrt::models::pi05::NativeWorkspace workspace(ctx);
    flashrt::models::pi05::NativeWorkspaceConfig config;
    if (!workspace.allocate(config).ok_status()) {
        frt_ctx_destroy(ctx);
        return 1;
    }
    flashrt::models::pi05::NativeKernelDriver driver;
    flashrt::models::pi05::NativeStylePrecomputer precomputer(&driver);
    st = precomputer.run(weights, &workspace, 0);
    if (!st.ok_status()) {
        std::cerr << st.message << '\n';
        frt_ctx_destroy(ctx);
        return 1;
    }
    const std::string prefix = argv[2];
    for (const char* name : {"decoder_time_emb", "decoder_style_attn",
                             "decoder_style_ffn", "decoder_style_final"}) {
        const auto* buffer = workspace.find(name);
        if (!buffer || !write_buffer(prefix + "." + name + ".bin", *buffer)) {
            frt_ctx_destroy(ctx);
            return 1;
        }
    }
    std::cout << "PASS native decoder style precompute\n";
    frt_ctx_destroy(ctx);
    return 0;
}
