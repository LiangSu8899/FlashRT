#include "flashrt/cpp/models/pi05/native_workspace.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdio>
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
    for (std::uint16_t value : values) {
        assert(value == flashrt::modalities::float_to_bfloat16(1.0f));
    }
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
        NativeWorkspaceConfig config;
        assert(workspace.allocate(config).ok_status());
        assert(workspace.logical_size() == 34);
        assert(workspace.allocation_count() == 33);
        assert(workspace.allocated_bytes() > 0);
        assert(workspace.vision_sequence() == 512);
        assert(workspace.encoder_vision_sequence() == 512);
        assert(workspace.encoder_sequence() == 712);
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
        assert(workspace.logical_size() == 34);
        assert(workspace.allocation_count() == 34);
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
    std::printf("PASS - Pi0.5 native workspace\n");
    return 0;
}
