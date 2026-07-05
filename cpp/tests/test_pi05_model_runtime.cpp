/* test_pi05_model_runtime.cpp — Pi0.5 through the GENERIC model-runtime face.
 *
 * Same fabricated-export setup as test_pi05_c_api.cpp, but driven exclusively
 * through frt_model_runtime_v1: ports resolve, staged image input lands in the
 * device buffer, SWAP ports refuse the verb (write-the-window discipline),
 * step replays a real captured graph, decoded actions come back through
 * get_output, and the wrapper lifetime releases the export exactly once.
 */
#include "flashrt/cpp/models/pi05/model_runtime.h"
#include "flashrt/exec.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", (msg)); g_fail = 1; } \
    else { std::printf("ok  : %s\n", (msg)); } \
} while (0)

namespace {

struct Owner { int retain = 0, release = 0; };
extern "C" void retain_owner(void* p) { static_cast<Owner*>(p)->retain += 1; }
extern "C" void release_owner(void* p) { static_cast<Owner*>(p)->release += 1; }

std::uint16_t f2bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(bits >> 16);
}
float bf162f(std::uint16_t v) {
    std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

/* graph body: overwrite the action buffer with fixed model outputs */
struct CopyRec { void* dst; const void* src; size_t n; };
void record_copy(void* user, void* stream) {
    auto* r = static_cast<CopyRec*>(user);
    cudaMemcpyAsync(r->dst, r->src, r->n, cudaMemcpyDeviceToDevice,
                    (cudaStream_t)stream);
}

bool has_cuda_device() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return n > 0;
}

}  // namespace

int main() {
    if (!has_cuda_device()) {
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }

    frt_ctx ctx = frt_ctx_create();
    CHECK(ctx != nullptr, "frt_ctx_create");
    int sid = frt_ctx_stream(ctx, 0);

    const std::uint64_t image_bytes = 1ull * 224 * 224 * 3 * 2;
    const std::uint64_t action_bytes = 1ull * 4 * 2;
    frt_buffer image = frt_buffer_alloc(ctx, "observation_images_normalized",
                                        image_bytes);
    frt_buffer action = frt_buffer_alloc(ctx, "diffusion_noise", action_bytes);
    frt_buffer model_out = frt_buffer_alloc(ctx, "model_out", action_bytes);

    /* "model outputs" the graph writes into the action buffer on replay */
    std::uint16_t out_host[4] = {f2bf16(1.0f), f2bf16(-2.0f), f2bf16(3.0f),
                                 f2bf16(99.0f)};
    cudaMemcpy(frt_buffer_dptr(model_out), out_host, action_bytes,
               cudaMemcpyHostToDevice);
    cudaMemset(frt_buffer_dptr(action), 0, action_bytes);

    frt_graph graph = frt_graph_create(ctx, "infer", 1);
    CopyRec rec{frt_buffer_dptr(action), frt_buffer_dptr(model_out),
                action_bytes};
    CHECK(frt_graph_capture(graph, 0, record_copy, &rec) == FRT_OK,
          "capture the infer graph");

    frt_runtime_stream_desc stream_desc{};
    stream_desc.name = "main";
    stream_desc.stream_id = sid;
    frt_runtime_graph_desc graph_desc[2]{};
    graph_desc[0].name = "infer";
    graph_desc[0].handle = graph;
    graph_desc[0].default_key = 0;
    graph_desc[0].stream_id = sid;
    graph_desc[1].name = "decode_only";
    graph_desc[1].handle = graph;
    graph_desc[1].default_key = 0;
    graph_desc[1].stream_id = sid;
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
    exp.streams = &stream_desc; exp.n_streams = 1;
    exp.graphs = graph_desc;    exp.n_graphs = 2;
    exp.buffers = buffers;      exp.n_buffers = 2;
    exp.identity = "pi05-model-runtime-test";
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
    cfg.action_mean = mean;    cfg.n_action_mean = 3;
    cfg.action_stddev = stddev; cfg.n_action_stddev = 3;

    frt_model_runtime_v1* m = nullptr;
    CHECK(frt_pi05_model_runtime_create(&exp, &cfg, &m) == 0 && m,
          "frt_pi05_model_runtime_create");
    CHECK(owner.retain >= 1, "export retained");
    CHECK(m->exp == &exp, "model runtime carries the export");
    CHECK(m->n_ports == 3 && m->n_stages == 1, "port/stage counts");
    CHECK(std::strcmp(m->ports[0].name, "images") == 0 &&
              m->ports[0].update == FRT_RT_PORT_STAGED &&
              m->ports[0].shape[1] == 224,
          "images port schema");
    CHECK(std::strcmp(m->ports[1].name, "noise") == 0 &&
              m->ports[1].update == FRT_RT_PORT_SWAP &&
              m->ports[1].buffer == action,
          "noise SWAP port exposes the device window");
    CHECK(m->stages[0].graph == 0, "stage resolves the infer graph");

    /* staged image input lands in the device buffer */
    const std::uint8_t rgb[] = {0, 127, 255, 255, 127, 0,
                                10, 20, 30, 40, 50, 60};
    frt_image_view view{};
    view.struct_size = sizeof(view);
    view.pixel_format = FRT_RT_PIXEL_RGB8;
    view.data = rgb;
    view.bytes = sizeof(rgb);
    view.width = 2;
    view.height = 2;
    CHECK(m->verbs.set_input(m->self, 0, &view, sizeof(view), -1) == 0,
          "set_input(images) accepts frt_image_view[]");
    std::vector<std::uint16_t> img_host(image_bytes / 2);
    cudaMemcpy(img_host.data(), frt_buffer_dptr(image), image_bytes,
               cudaMemcpyDeviceToHost);
    bool nonzero = false;
    for (std::uint16_t v : img_host) if (v) { nonzero = true; break; }
    CHECK(nonzero, "staged image input reached the device buffer");

    CHECK(m->verbs.set_input(m->self, 1, rgb, 4, -1) < 0,
          "SWAP port refuses the staged verb (write the window instead)");
    bool has_prompt = false;
    for (std::uint64_t i = 0; i < m->n_ports; ++i)
        if (std::strcmp(m->ports[i].name, "prompt") == 0) has_prompt = true;
    CHECK(!has_prompt,
          "no prompt port on the adopted-export face (no advertise-and-refuse)");

    /* step = replay the captured graph; decoded actions come back */
    CHECK(m->verbs.step(m->self) == 0, "step replays the infer graph");
    cudaDeviceSynchronize();
    std::uint16_t acted[4] = {};
    cudaMemcpy(acted, frt_buffer_dptr(action), action_bytes,
               cudaMemcpyDeviceToHost);
    CHECK(std::fabs(bf162f(acted[0]) - 1.0f) < 1e-3, "replay wrote the action buffer");

    float out[3] = {};
    std::uint64_t written = 0;
    CHECK(m->verbs.get_output(m->self, 2, out, sizeof(out), &written, -1) == 0
              && written == sizeof(out),
          "get_output(actions) in bytes");
    CHECK(std::fabs(out[0] - 12.0f) < 0.01f &&
              std::fabs(out[1] - 17.0f) < 0.01f &&
              std::fabs(out[2] - 34.0f) < 0.01f,
          "actions are unnormalized (clip to [-1,1], * stddev, + mean)");

    float small[1] = {};
    CHECK(m->verbs.get_output(m->self, 2, small, sizeof(small), &written, -1)
              == -5 && written == sizeof(out),
          "get_output reports the needed size on short buffers");

    /* wrapper lifetime: one release tears down adapter + export refs */
    const int retains = owner.retain;
    m->retain(m->owner);
    m->release(m->owner);
    CHECK(owner.release == 0, "alive after paired retain/release");
    m->release(m->owner);
    CHECK(owner.release == retains, "all export references dropped");

    /* production path: inherit producer declarations, override only verbs */
    cudaMemset(frt_buffer_dptr(action), 0, action_bytes);
    frt_runtime_port_desc ports[3] = {};
    const int64_t image_shape[4] = {1, 224, 224, 3};
    const int64_t noise_shape[2] = {1, 4};
    const int64_t action_shape[2] = {1, 3};
    ports[0].name = "images";
    ports[0].modality = FRT_RT_MOD_IMAGE;
    ports[0].dtype = FRT_RT_DTYPE_BF16;
    ports[0].layout = FRT_RT_LAYOUT_NHWC;
    ports[0].direction = FRT_RT_PORT_IN;
    ports[0].update = FRT_RT_PORT_STAGED;
    ports[0].required = 1;
    ports[0].shape = image_shape;
    ports[0].rank = 4;
    ports[0].buffer = image;
    ports[0].bytes = image_bytes;
    ports[1].name = "noise";
    ports[1].modality = FRT_RT_MOD_TENSOR;
    ports[1].dtype = FRT_RT_DTYPE_BF16;
    ports[1].layout = FRT_RT_LAYOUT_FLAT;
    ports[1].direction = FRT_RT_PORT_IN;
    ports[1].update = FRT_RT_PORT_SWAP;
    ports[1].shape = noise_shape;
    ports[1].rank = 2;
    ports[1].buffer = action;
    ports[1].bytes = action_bytes;
    ports[2].name = "actions";
    ports[2].modality = FRT_RT_MOD_ACTION;
    ports[2].dtype = FRT_RT_DTYPE_BF16;
    ports[2].layout = FRT_RT_LAYOUT_FLAT;
    ports[2].direction = FRT_RT_PORT_OUT;
    ports[2].update = FRT_RT_PORT_STAGED;
    ports[2].shape = action_shape;
    ports[2].rank = 2;
    ports[2].buffer = action;
    ports[2].bytes = action_bytes;
    uint32_t after_action[1] = {0};
    frt_runtime_stage_desc stages[2]{};
    stages[0].graph = 0;
    stages[1].graph = 1;
    stages[1].after = after_action;
    stages[1].n_after = 1;

    frt_model_runtime_v1* producer = frt_model_runtime_wrap(
        &exp, ports, 3, stages, 2, nullptr, nullptr, nullptr, nullptr);
    CHECK(producer != nullptr, "producer model declaration for create_over");

    frt_model_runtime_v1* over = nullptr;
    CHECK(frt_pi05_model_runtime_create_over(producer, &cfg, &over) == 0 &&
              over,
          "frt_pi05_model_runtime_create_over");
    CHECK(over->exp == producer->exp && over->ports == producer->ports &&
              over->stages == producer->stages,
          "create_over inherits producer declarations");
    CHECK(over->n_stages == 2 && over->stages[1].graph == 1 &&
              over->stages[1].after[0] == 0,
          "create_over preserves a producer-declared two-stage DAG");
    producer->release(producer->owner);
    CHECK(owner.release == retains,
          "producer declaration stays alive through the override");

    CHECK(over->verbs.set_input(over->self, 0, &view, sizeof(view), -1) == 0,
          "create_over set_input(images)");
    CHECK(over->verbs.step(over->self) == 0,
          "create_over step replays the declared stage");
    cudaDeviceSynchronize();
    float over_out[3] = {};
    written = 0;
    CHECK(over->verbs.get_output(over->self, 2, over_out, sizeof(over_out),
                                 &written, -1) == 0 &&
              std::fabs(over_out[0] - 12.0f) < 0.01f,
          "create_over get_output(actions)");
    over->release(over->owner);
    CHECK(owner.release == owner.retain,
          "create_over releases its native runtime and inherited producer");

    frt_graph_destroy(graph);
    frt_ctx_destroy(ctx);
    std::printf(g_fail ? "\n== PI05 MODEL RUNTIME FAILED ==\n"
                       : "\n== PI05 MODEL RUNTIME PASSED ==\n");
    return g_fail;
}
