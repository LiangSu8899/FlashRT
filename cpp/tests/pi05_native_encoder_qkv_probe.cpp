#include "flashrt/cpp/models/pi05/native_bf16_forward.h"
#include "flashrt/cpp/models/pi05/native_weight_materializer.h"

#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool write_device(const std::string& path, const void* device,
                  std::size_t elements) {
    std::vector<std::uint16_t> host(elements);
    if (cudaMemcpy(host.data(), device, host.size() * sizeof(std::uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(host.data()),
               static_cast<std::streamsize>(host.size() * sizeof(std::uint16_t)));
    return file.good();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: pi05_native_encoder_qkv_probe CHECKPOINT OUTPUT\n";
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
    flashrt::modalities::Status st = materializer.materialize_encoder_layer(17);
    if (!st.ok_status()) {
        std::cerr << st.message << '\n';
        frt_ctx_destroy(ctx);
        return 1;
    }
    NativeWorkspace workspace(ctx);
    if (!workspace.allocate(NativeWorkspaceConfig{}).ok_status()) {
        frt_ctx_destroy(ctx);
        return 1;
    }
    NativeRtxAttentionWorkspace attention(ctx);
    if (!attention.allocate(NativeRtxAttentionConfig{}).ok_status()) {
        frt_ctx_destroy(ctx);
        return 1;
    }
    const auto* encoder_x = workspace.find("encoder_x");
    if (!encoder_x) {
        frt_ctx_destroy(ctx);
        return 1;
    }
    std::vector<std::uint16_t> host_x(712 * 2048, 0);
    for (int row = 0; row < 712; ++row) {
        for (int column = 0; column < 512; ++column) {
            const float value = float((row + column) % 15 - 7) / 8.0f;
            host_x[static_cast<std::size_t>(row) * 2048 + column] =
                flashrt::modalities::float_to_bfloat16(value);
        }
    }
    if (cudaMemcpy(frt_buffer_dptr(encoder_x->buffer), host_x.data(),
                   host_x.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        frt_ctx_destroy(ctx);
        return 1;
    }
    NativeKernelDriver driver;
    NativeBf16Forward forward(&driver);
    st = forward.encoder_qkv(17, weights, &workspace, &attention, 0);
    if (!st.ok_status() || cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << st.message << '\n';
        frt_ctx_destroy(ctx);
        return 1;
    }
    const auto* query = attention.find("attn_enc_Q");
    const std::string prefix = argv[2];
    const bool ok = query &&
        write_device(prefix + ".q.bin", frt_buffer_dptr(query->buffer),
                     712 * 2048) &&
        write_device(prefix + ".k.bin", attention.encoder_k_layer_dptr(17),
                     712 * 256) &&
        write_device(prefix + ".v.bin", attention.encoder_v_layer_dptr(17),
                     712 * 256);
    frt_ctx_destroy(ctx);
    if (!ok) return 1;
    std::cout << "PASS native encoder QKV layer 17\n";
    return 0;
}
