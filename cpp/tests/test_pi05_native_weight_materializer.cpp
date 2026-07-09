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
    const std::string decoder =
        "paligemma_with_expert.gemma_expert.model.layers.0";
    const std::string vision =
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model";
    const std::string vision_layer = vision + ".encoder.layers.0";
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
        {decoder + ".self_attn.q_proj.weight", {16, 4}, sequence(64, 7.0f)},
        {decoder + ".self_attn.k_proj.weight", {4, 4}, sequence(16, 8.0f)},
        {decoder + ".self_attn.v_proj.weight", {4, 4}, sequence(16, 9.0f)},
        {decoder + ".self_attn.o_proj.weight", {4, 16}, sequence(64, 10.0f)},
        {decoder + ".mlp.gate_proj.weight", {6, 4}, sequence(24, 11.0f)},
        {decoder + ".mlp.up_proj.weight", {6, 4}, sequence(24, 12.0f)},
        {decoder + ".mlp.down_proj.weight", {4, 6}, sequence(24, 13.0f)},
        {decoder + ".input_layernorm.dense.weight", {12, 4},
         sequence(48, 14.0f)},
        {decoder + ".input_layernorm.dense.bias", {12}, sequence(12, 15.0f)},
        {decoder + ".post_attention_layernorm.dense.weight", {12, 4},
         sequence(48, 16.0f)},
        {decoder + ".post_attention_layernorm.dense.bias", {12},
         sequence(12, 17.0f)},
        {vision + ".embeddings.patch_embedding.weight", {2, 2, 2, 1},
         sequence(8, 18.0f)},
        {vision + ".embeddings.patch_embedding.bias", {2},
         sequence(2, 19.0f)},
        {vision + ".embeddings.position_embedding.weight", {3, 2},
         sequence(6, 20.0f)},
        {vision + ".post_layernorm.weight", {2}, sequence(2, 21.0f)},
        {vision + ".post_layernorm.bias", {2}, sequence(2, 22.0f)},
        {"paligemma_with_expert.paligemma.model.multi_modal_projector.linear."
         "weight", {4, 2}, sequence(8, 23.0f)},
        {"paligemma_with_expert.paligemma.model.multi_modal_projector.linear."
         "bias", {4}, sequence(4, 24.0f)},
        {vision_layer + ".self_attn.q_proj.weight", {2, 2},
         sequence(4, 25.0f)},
        {vision_layer + ".self_attn.q_proj.bias", {2},
         sequence(2, 26.0f)},
        {vision_layer + ".self_attn.k_proj.weight", {2, 2},
         sequence(4, 27.0f)},
        {vision_layer + ".self_attn.k_proj.bias", {2},
         sequence(2, 28.0f)},
        {vision_layer + ".self_attn.v_proj.weight", {2, 2},
         sequence(4, 29.0f)},
        {vision_layer + ".self_attn.v_proj.bias", {2},
         sequence(2, 30.0f)},
        {vision_layer + ".self_attn.out_proj.weight", {2, 2},
         sequence(4, 31.0f)},
        {vision_layer + ".self_attn.out_proj.bias", {2},
         sequence(2, 32.0f)},
        {vision_layer + ".mlp.fc1.weight", {3, 2},
         sequence(6, 33.0f)},
        {vision_layer + ".mlp.fc1.bias", {3}, sequence(3, 34.0f)},
        {vision_layer + ".mlp.fc2.weight", {2, 3},
         sequence(6, 35.0f)},
        {vision_layer + ".mlp.fc2.bias", {2}, sequence(2, 36.0f)},
        {vision_layer + ".layer_norm1.weight", {2}, sequence(2, 37.0f)},
        {vision_layer + ".layer_norm1.bias", {2}, sequence(2, 38.0f)},
        {vision_layer + ".layer_norm2.weight", {2}, sequence(2, 39.0f)},
        {vision_layer + ".layer_norm2.bias", {2}, sequence(2, 40.0f)},
        {"paligemma_with_expert.gemma_expert.model.norm.dense.weight",
         {3, 2}, sequence(6, 41.0f)},
        {"paligemma_with_expert.gemma_expert.model.norm.dense.bias",
         {3}, sequence(3, 42.0f)},
        {"time_mlp_in.weight", {2, 2}, sequence(4, 43.0f)},
        {"time_mlp_in.bias", {2}, sequence(2, 44.0f)},
        {"time_mlp_out.weight", {2, 2}, sequence(4, 45.0f)},
        {"time_mlp_out.bias", {2}, sequence(2, 46.0f)},
        {"action_in_proj.weight", {2, 1}, sequence(2, 47.0f)},
        {"action_in_proj.bias", {2}, sequence(2, 48.0f)},
        {"action_out_proj.weight", {1, 2}, sequence(2, 49.0f)},
        {"action_out_proj.bias", {1}, sequence(1, 50.0f)},
        {"paligemma_with_expert.paligemma.lm_head.weight",
         {4, 2}, sequence(8, 51.0f)},
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
        assert(materializer.materialize_decoder_layer(0, true).ok_status());
        assert(destination.size() == 15);
        const auto* decoder_qkv = destination.find("decoder_attn_qkv_w_0");
        assert(decoder_qkv &&
               decoder_qkv->shape == std::vector<std::uint64_t>({4, 24}));
        const auto* gate_up = destination.find("decoder_ffn_gate_up_w_0");
        assert(gate_up &&
               gate_up->shape == std::vector<std::uint64_t>({4, 12}));
        const auto* attn_mod =
            destination.find("decoder_pre_attn_norm_mod_w_0");
        assert(attn_mod &&
               attn_mod->shape == std::vector<std::uint64_t>({4, 12}));
        assert(!materializer.materialize_decoder_layer(0, true).ok_status());
        assert(!materializer.materialize_decoder_layer(18, true).ok_status());
        assert(materializer.materialize_vision_globals().ok_status());
        assert(materializer.materialize_vision_layer(0).ok_status());
        assert(destination.size() == 34);
        const auto* patch = destination.find("vision_patch_embedding_w");
        assert(patch && patch->shape ==
                            std::vector<std::uint64_t>({2, 1, 2, 2}));
        const auto* vision_qkv = destination.find("vision_attn_qkv_w_0");
        assert(vision_qkv &&
               vision_qkv->shape == std::vector<std::uint64_t>({2, 6}));
        assert(!materializer.materialize_vision_layer(27).ok_status());
        assert(!materializer.materialize_decoder_globals(0).ok_status());
        assert(materializer.materialize_decoder_globals(10).ok_status());
        assert(destination.size() == 45);
        assert(destination.find("decoder_final_norm_mod_w")->shape ==
               std::vector<std::uint64_t>({2, 3}));
        assert(destination.find("decoder_time_embeds")->shape ==
               std::vector<std::uint64_t>({10, 1024}));
        assert(destination.find("decoder_action_out_proj_w")->shape ==
               std::vector<std::uint64_t>({2, 1}));
        assert(materializer.materialize_embedding().ok_status());
        assert(destination.size() == 46);
        assert(destination.find("embedding_weight")->shape ==
               std::vector<std::uint64_t>({4, 2}));
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
            assert(materializer.materialize_decoder_layer(0, true).ok_status());
            assert(destination.size() == 15);
            assert(destination.find("decoder_attn_qkv_w_0")->shape ==
                   std::vector<std::uint64_t>({1024, 2560}));
            assert(destination.find("decoder_ffn_gate_up_w_0")->shape ==
                   std::vector<std::uint64_t>({1024, 8192}));
            assert(materializer.materialize_vision_globals().ok_status());
            assert(materializer.materialize_vision_layer(0).ok_status());
            assert(destination.size() == 34);
            assert(destination.find("vision_patch_embedding_w")->shape ==
                   std::vector<std::uint64_t>({14, 14, 3, 1152}));
            assert(destination.find("vision_attn_qkv_w_0")->shape ==
                   std::vector<std::uint64_t>({1152, 3456}));
            assert(materializer.materialize_decoder_globals(10).ok_status());
            assert(destination.size() == 45);
            assert(destination.find("decoder_final_norm_mod_w")->shape ==
                   std::vector<std::uint64_t>({1024, 3072}));
            assert(destination.find("decoder_time_embeds")->shape ==
                   std::vector<std::uint64_t>({10, 1024}));
            assert(destination.find("decoder_action_in_proj_w")->shape ==
                   std::vector<std::uint64_t>({32, 1024}));
            assert(destination.find("decoder_action_out_proj_w")->shape ==
                   std::vector<std::uint64_t>({1024, 32}));
        }
        frt_ctx_destroy(real_ctx);
    }
    std::printf("PASS - Pi0.5 native layer materializer\n");
    return 0;
}
