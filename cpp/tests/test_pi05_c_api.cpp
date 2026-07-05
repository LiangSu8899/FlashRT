#include "flashrt/cpp/models/pi05/c_api.h"
#include "flashrt/exec.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

struct Owner {
    int retain = 0;
    int release = 0;
};

extern "C" void retain_owner(void* p) {
    static_cast<Owner*>(p)->retain += 1;
}

extern "C" void release_owner(void* p) {
    static_cast<Owner*>(p)->release += 1;
}

std::uint16_t float_to_bfloat16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(bits >> 16);
}

bool has_cuda_device() {
    int n = 0;
    cudaError_t rc = cudaGetDeviceCount(&n);
    if (rc != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return n > 0;
}

}  // namespace

int main() {
    if (!has_cuda_device()) {
        std::cout << "SKIP - no CUDA device\n";
        return 0;
    }

    frt_ctx ctx = frt_ctx_create();
    assert(ctx);
    int sid = frt_ctx_stream(ctx, 0);
    assert(sid >= 0);
    frt_graph graph = frt_graph_create(ctx, "infer", 1);
    assert(graph);

    const std::uint64_t image_bytes = 1ull * 224ull * 224ull * 3ull * 2ull;
    const std::uint64_t action_bytes = 1ull * 4ull * 2ull;
    frt_buffer image = frt_buffer_alloc(ctx, "observation_images_normalized",
                                        image_bytes);
    frt_buffer action = frt_buffer_alloc(ctx, "diffusion_noise", action_bytes);
    assert(image);
    assert(action);

    std::vector<std::uint16_t> action_host(4);
    action_host[0] = float_to_bfloat16(1.0f);
    action_host[1] = float_to_bfloat16(-2.0f);
    action_host[2] = float_to_bfloat16(3.0f);
    action_host[3] = float_to_bfloat16(99.0f);
    assert(cudaMemcpy(frt_buffer_dptr(action), action_host.data(), action_bytes,
                      cudaMemcpyHostToDevice) == cudaSuccess);

    frt_runtime_stream_desc stream_desc{};
    stream_desc.name = "main";
    stream_desc.stream_id = sid;
    stream_desc.priority = 0;

    frt_runtime_graph_desc graph_desc{};
    graph_desc.name = "infer";
    graph_desc.handle = graph;
    graph_desc.default_key = 0;
    graph_desc.stream_id = sid;

    frt_runtime_buffer_desc buffers[2]{};
    buffers[0].name = "observation_images_normalized";
    buffers[0].handle = image;
    buffers[0].bytes = image_bytes;
    buffers[0].role = FRT_RT_ROLE_INPUT;
    buffers[1].name = "diffusion_noise";
    buffers[1].handle = action;
    buffers[1].bytes = action_bytes;
    buffers[1].role = FRT_RT_ROLE_INPUT | FRT_RT_ROLE_OUTPUT;

    Owner owner;
    frt_runtime_export_v1 exp{};
    exp.abi_version = FRT_RUNTIME_ABI_VERSION;
    exp.struct_size = sizeof(exp);
    exp.ctx = ctx;
    exp.streams = &stream_desc;
    exp.n_streams = 1;
    exp.graphs = &graph_desc;
    exp.n_graphs = 1;
    exp.buffers = buffers;
    exp.n_buffers = 2;
    exp.identity = "pi05-c-api-test";
    exp.owner = &owner;
    exp.retain = retain_owner;
    exp.release = release_owner;

    const float mean[] = {10.0f, 20.0f, 30.0f};
    const float stddev[] = {2.0f, 3.0f, 4.0f};
    frt_pi05_runtime_config cfg{};
    cfg.struct_size = sizeof(cfg);
    cfg.num_views = 1;
    cfg.chunk = 1;
    cfg.model_action_dim = 4;
    cfg.robot_action_dim = 3;
    cfg.action_mean = mean;
    cfg.n_action_mean = 3;
    cfg.action_stddev = stddev;
    cfg.n_action_stddev = 3;

    frt_pi05_runtime* rt = nullptr;
    int rc = frt_pi05_runtime_create(&exp, &cfg, &rt);
    assert(rc == 0);
    assert(rt);
    assert(owner.retain == 1);

    const std::uint8_t rgb[] = {
        0, 127, 255, 255, 127, 0,
        10, 20, 30, 40, 50, 60,
    };
    frt_pi05_vision_frame frame{};
    frame.struct_size = sizeof(frame);
    frame.name = "image";
    frame.data = rgb;
    frame.bytes = sizeof(rgb);
    frame.width = 2;
    frame.height = 2;
    frame.pixel_format = FRT_PI05_PIXEL_RGB8;
    rc = frt_pi05_runtime_prepare_vision(rt, &frame, 1);
    assert(rc == 0);

    float out[3] = {};
    uint64_t n_written = 0;
    rc = frt_pi05_runtime_read_actions(rt, out, 3, &n_written);
    assert(rc == 0);
    assert(n_written == 3);
    assert(std::fabs(out[0] - 12.0f) < 0.01f);
    assert(std::fabs(out[1] - 17.0f) < 0.01f);
    assert(std::fabs(out[2] - 34.0f) < 0.01f);

    frt_pi05_runtime_destroy(rt);
    assert(owner.release == 1);
    frt_graph_destroy(graph);
    frt_ctx_destroy(ctx);
    std::cout << "PASS - Pi05 C API\n";
    return 0;
}
