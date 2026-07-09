#include "flashrt/cpp/models/pi05/native_weight_materializer.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

struct Entry {
    std::string key;
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};

bool has_cuda_device() {
    int count = 0;
    const cudaError_t rc = cudaGetDeviceCount(&count);
    if (rc != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

std::string temp_path() {
    char path[] = "/tmp/frt_pi05_materializer_XXXXXX";
    const int fd = ::mkstemp(path);
    assert(fd >= 0);
    ::close(fd);
    return path;
}

std::vector<float> sequence(std::size_t count, float start) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = start + static_cast<float>(i) * 0.01f;
    }
    return values;
}

void write_checkpoint(const std::string& path,
                      const std::vector<Entry>& entries) {
    std::string header = "{";
    std::uint64_t offset = 0;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const Entry& entry = entries[i];
        if (i) header += ',';
        header += '"' + entry.key + "\":{\"dtype\":\"F32\",\"shape\":[";
        for (std::size_t d = 0; d < entry.shape.size(); ++d) {
            if (d) header += ',';
            header += std::to_string(entry.shape[d]);
        }
        header += "],\"data_offsets\":[" + std::to_string(offset) + ',';
        offset += entry.values.size() * sizeof(float);
        header += std::to_string(offset) + "]}";
    }
    header += '}';
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) {
        const char byte = static_cast<char>((n >> (8 * i)) & 0xffu);
        file.write(&byte, 1);
    }
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const Entry& entry : entries) {
        file.write(reinterpret_cast<const char*>(entry.values.data()),
                   static_cast<std::streamsize>(entry.values.size() *
                                                sizeof(float)));
    }
    assert(file.good());
}

}  // namespace

int main() {
    if (!has_cuda_device()) {
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }
    const std::string prefix =
        "paligemma_with_expert.paligemma.model.language_model.layers.0";
    const std::vector<Entry> entries = {
        {prefix + ".input_layernorm.weight", {4}, {-0.5f, 0.0f, 0.5f, 1.0f}},
        {prefix + ".self_attn.q_proj.weight", {16, 4}, sequence(64, 0.1f)},
        {prefix + ".self_attn.k_proj.weight", {4, 4}, sequence(16, 1.0f)},
        {prefix + ".self_attn.v_proj.weight", {4, 4}, sequence(16, 2.0f)},
        {prefix + ".self_attn.o_proj.weight", {4, 8}, sequence(32, 3.0f)},
        {prefix + ".post_attention_layernorm.weight", {4},
         {-0.25f, 0.0f, 0.25f, 0.5f}},
        {prefix + ".mlp.gate_proj.weight", {6, 4}, sequence(24, 4.0f)},
        {prefix + ".mlp.up_proj.weight", {6, 4}, sequence(24, 5.0f)},
        {prefix + ".mlp.down_proj.weight", {4, 6}, sequence(24, 6.0f)},
    };
    const std::string path = temp_path();
    write_checkpoint(path, entries);

    flashrt::loader::SafetensorsFile source;
    assert(source.open(path));
    frt_ctx ctx = frt_ctx_create();
    assert(ctx);
    {
        flashrt::models::pi05::NativeDeviceWeightStore destination(ctx);
        flashrt::models::pi05::NativeWeightMaterializer materializer(
            source, &destination);
        assert(materializer.materialize_encoder_layer(0).ok_status());
        assert(destination.size() == 5);
        const auto* qkv = destination.find("encoder_attn_qkv_w_0");
        assert(qkv && qkv->shape == std::vector<std::uint64_t>({4, 24}));
        const auto* gate = destination.find("encoder_ffn_gate_w_0");
        assert(gate && gate->shape == std::vector<std::uint64_t>({4, 6}));
        const auto* down = destination.find("encoder_ffn_down_w_0");
        assert(down && down->shape == std::vector<std::uint64_t>({6, 4}));
        assert(!materializer.materialize_encoder_layer(0).ok_status());
        assert(!materializer.materialize_encoder_layer(18).ok_status());
    }
    frt_ctx_destroy(ctx);
    assert(::unlink(path.c_str()) == 0);

    const char* real_checkpoint = std::getenv("FLASH_RT_PI05_CHECKPOINT");
    if (real_checkpoint && real_checkpoint[0]) {
        flashrt::loader::SafetensorsFile real_source;
        assert(real_source.open(std::string(real_checkpoint) +
                                "/model.safetensors"));
        frt_ctx real_ctx = frt_ctx_create();
        assert(real_ctx);
        {
            flashrt::models::pi05::NativeDeviceWeightStore destination(
                real_ctx);
            flashrt::models::pi05::NativeWeightMaterializer materializer(
                real_source, &destination);
            assert(materializer.materialize_encoder_layer(0).ok_status());
            assert(destination.size() == 5);
            assert(destination.find("encoder_attn_qkv_w_0")->shape ==
                   std::vector<std::uint64_t>({2048, 2560}));
            assert(destination.find("encoder_ffn_gate_w_0")->shape ==
                   std::vector<std::uint64_t>({2048, 16384}));
        }
        frt_ctx_destroy(real_ctx);
    }
    std::printf("PASS - Pi0.5 encoder weight materializer\n");
    return 0;
}
