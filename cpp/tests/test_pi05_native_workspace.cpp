#include "flashrt/cpp/models/pi05/native_workspace.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

bool has_cuda_device() {
    int count = 0;
    const cudaError_t rc = cudaGetDeviceCount(&count);
    if (rc != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

void check_ones(const flashrt::models::pi05::NativeWorkspaceBuffer& buffer) {
    std::vector<std::uint16_t> values(buffer.shape[0]);
    assert(cudaMemcpy(values.data(), frt_buffer_dptr(buffer.buffer),
                      values.size() * sizeof(std::uint16_t),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    const std::uint16_t expected =
        buffer.dtype == flashrt::modalities::DType::kFloat16
            ? flashrt::modalities::float_to_float16(1.0f)
            : flashrt::modalities::float_to_bfloat16(1.0f);
    for (std::uint16_t value : values) assert(value == expected);
}

}  // namespace

int main() {
    if (!has_cuda_device()) {
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }
    using namespace flashrt::models::pi05;
    frt_ctx ctx = frt_ctx_create();
    assert(ctx);
    {
        NativeWorkspace workspace(ctx);
        NativeWorkspaceConfig invalid;
        invalid.vision_pool_factor = 3;
        assert(!workspace.allocate(invalid).ok_status());
        invalid.vision_pool_factor = 1;
        invalid.num_views = 3;
        invalid.max_prompt_tokens = std::numeric_limits<int>::max() - 769;
        invalid.chunk_size = 2;
        assert(!workspace.allocate(invalid).ok_status());
        NativeWorkspaceConfig config;
        assert(workspace.allocate(config).ok_status());
        assert(workspace.logical_size() == 35);
        assert(workspace.allocation_count() == 34);
        assert(workspace.allocated_bytes() > 0);
        assert(workspace.vision_sequence() == 512);
        assert(workspace.encoder_vision_sequence() == 512);
        assert(workspace.encoder_sequence() == 712);
        assert(workspace.find("prompt_embedding")->shape ==
               std::vector<std::uint64_t>({200, 2048}));
        const auto* vision_x = workspace.find("vision_x");
        const auto* pooled = workspace.find("vision_x_pooled");
        assert(vision_x && pooled && pooled->alias);
        assert(vision_x->buffer == pooled->buffer);
        assert(workspace.find("decoder_style_attn")->shape ==
               std::vector<std::uint64_t>({10, 18, 10, 3072}));
        assert(workspace.find("rtc_prefix_weights")->dtype ==
               flashrt::modalities::DType::kFloat32);
        check_ones(*workspace.find("encoder_rms_ones"));
        check_ones(*workspace.find("decoder_rms_ones"));
        assert(workspace.update_decoder_rope(37).ok_status());
        assert(!workspace.update_decoder_rope(201).ok_status());
        void* decoder_rope_ptr =
            frt_buffer_dptr(workspace.find("decoder_rope_weights")->buffer);
        const std::size_t allocation_count = workspace.allocation_count();
        const std::size_t allocated_bytes = workspace.allocated_bytes();
        for (int i = 0; i < 1000; ++i) {
            assert(workspace.update_decoder_rope(i % 201).ok_status());
            assert(frt_buffer_dptr(
                       workspace.find("decoder_rope_weights")->buffer) ==
                   decoder_rope_ptr);
            assert(workspace.allocation_count() == allocation_count);
            assert(workspace.allocated_bytes() == allocated_bytes);
        }

        NativeDeviceWeightStore weights(ctx);
        NativeBf16Tensor position;
        position.shape = {256, 1152};
        position.values.resize(256 * 1152);
        for (std::size_t i = 0; i < position.values.size(); ++i) {
            position.values[i] = flashrt::modalities::float_to_bfloat16(
                static_cast<float>(i % 97) / 97.0f);
        }
        assert(weights.upload("vision_position_embedding", position)
                   .ok_status());
        assert(workspace.expand_vision_position_embedding(weights)
                   .ok_status());
        const auto* expanded = workspace.find("vision_pos_embed_expanded");
        std::vector<std::uint16_t> expanded_values(position.values.size() * 2);
        assert(cudaMemcpy(expanded_values.data(),
                          frt_buffer_dptr(expanded->buffer),
                          expanded_values.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(std::vector<std::uint16_t>(expanded_values.begin(),
                                         expanded_values.begin() +
                                             position.values.size()) ==
               position.values);
        assert(std::vector<std::uint16_t>(
                   expanded_values.begin() + position.values.size(),
                   expanded_values.end()) == position.values);
        assert(!workspace.allocate(config).ok_status());
    }
    frt_ctx_destroy(ctx);

    ctx = frt_ctx_create();
    assert(ctx);
    {
        NativeWorkspace workspace(ctx);
        NativeWorkspaceConfig config;
        config.num_views = 3;
        config.max_prompt_tokens = 256;
        config.chunk_size = 50;
        config.num_steps = 5;
        config.vision_pool_factor = 2;
        assert(workspace.allocate(config).ok_status());
        assert(workspace.logical_size() == 35);
        assert(workspace.allocation_count() == 35);
        assert(workspace.vision_sequence() == 768);
        assert(workspace.encoder_vision_sequence() == 192);
        assert(workspace.encoder_sequence() == 448);
        const auto* pooled = workspace.find("vision_x_pooled");
        assert(pooled && !pooled->alias);
        assert(pooled->shape == std::vector<std::uint64_t>({192, 1152}));
        assert(pooled->buffer != workspace.find("vision_x")->buffer);
        assert(workspace.find("decoder_time_emb")->shape ==
               std::vector<std::uint64_t>({5, 50, 1024}));
    }
    frt_ctx_destroy(ctx);

    ctx = frt_ctx_create();
    assert(ctx);
    {
        NativeWorkspace workspace(ctx);
        NativeWorkspaceConfig invalid;
        invalid.flavor = NativeWorkspaceFlavor::kThorFp8;
        invalid.vision_pool_factor = 2;
        assert(!workspace.allocate(invalid).ok_status());
        invalid.vision_pool_factor = 1;
        invalid.max_prompt_tokens = 199;
        assert(!workspace.allocate(invalid).ok_status());

        NativeWorkspaceConfig config;
        config.flavor = NativeWorkspaceFlavor::kThorFp8;
        config.enable_calibration = true;
        assert(workspace.allocate(config).ok_status());
        assert(workspace.vision_sequence() == 512);
        assert(workspace.encoder_sequence() == 712);
        assert(workspace.total_keys() == 722);
        assert(workspace.find("observation_images_normalized")->dtype ==
               flashrt::modalities::DType::kFloat16);
        assert(workspace.find("encoder_x_fp8")->dtype ==
               flashrt::modalities::DType::kUInt8);
        assert(workspace.find("encoder_logits")->shape ==
               std::vector<std::uint64_t>({712 * 8, 722}));
        assert(workspace.find("encoder_k_cache")->shape ==
               std::vector<std::uint64_t>({18, 722, 256}));
        assert(workspace.find("decoder_activation_scales")->shape ==
               std::vector<std::uint64_t>({10, 18, 4}));
        assert(workspace.find("encoder_sample_scales"));
        assert(workspace.find("decoder_sample_scales"));
        check_ones(*workspace.find("encoder_rms_ones"));
        check_ones(*workspace.find("decoder_rms_ones"));
        std::vector<std::uint16_t> encoder_rope(712 * 256);
        assert(cudaMemcpy(
                   encoder_rope.data(),
                   frt_buffer_dptr(
                       workspace.find("encoder_rope_weights")->buffer),
                   encoder_rope.size() * sizeof(encoder_rope[0]),
                   cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(encoder_rope[468 * 256 + 2 * 30 + 1] == 0xb7fdu);
        assert(encoder_rope[624 * 256 + 2 * 34 + 1] == 0xb7fdu);

        const auto* prompt = workspace.find("prompt_embedding");
        std::vector<std::uint16_t> row(2048);
        for (std::size_t i = 0; i < row.size(); ++i) {
            row[i] = flashrt::modalities::float_to_float16(
                static_cast<float>(i % 31));
        }
        auto* prompt_base = static_cast<unsigned char*>(
            frt_buffer_dptr(prompt->buffer));
        assert(cudaMemcpy(prompt_base + 4 * row.size() * sizeof(row[0]),
                          row.data(), row.size() * sizeof(row[0]),
                          cudaMemcpyHostToDevice) == cudaSuccess);
        assert(workspace.set_fixed_prompt_length(5).ok_status());
        std::vector<std::uint16_t> padded(row.size());
        assert(cudaMemcpy(padded.data(),
                          prompt_base + 5 * row.size() * sizeof(row[0]),
                          padded.size() * sizeof(padded[0]),
                          cudaMemcpyDeviceToHost) == cudaSuccess);
        assert(padded == row);
        const char* control_names[] = {
            "attn_enc_seqused", "attn_dec_seqused", "attn_dec_devpos"};
        const std::int32_t expected_controls[] = {518, 528, 518};
        for (int i = 0; i < 3; ++i) {
            std::int32_t value = 0;
            assert(cudaMemcpy(
                       &value,
                       frt_buffer_dptr(workspace.find(control_names[i])->buffer),
                       sizeof(value), cudaMemcpyDeviceToHost) == cudaSuccess);
            assert(value == expected_controls[i]);
        }
        const std::size_t allocations = workspace.allocation_count();
        for (int i = 0; i < 1000; ++i) {
            assert(workspace.set_fixed_prompt_length(i % 200).ok_status());
            assert(workspace.allocation_count() == allocations);
        }

        NativeDeviceWeightStore weights(ctx);
        NativeF16Tensor position;
        position.shape = {256, 1152};
        position.values.resize(256 * 1152);
        for (std::size_t i = 0; i < position.values.size(); ++i) {
            position.values[i] = flashrt::modalities::float_to_float16(
                static_cast<float>(i % 97) / 97.0f);
        }
        assert(weights.upload("vision_position_embedding", position)
                   .ok_status());
        assert(workspace.expand_vision_position_embedding(weights)
                   .ok_status());
    }
    frt_ctx_destroy(ctx);
    std::printf("PASS - Pi0.5 native workspace\n");
    return 0;
}
