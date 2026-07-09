#include "flashrt/cpp/models/pi05/runtime.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>

using flashrt::modalities::DType;
using flashrt::modalities::Layout;
using flashrt::modalities::MemoryPlace;
using flashrt::modalities::PixelFormat;
using flashrt::modalities::Shape;
using flashrt::modalities::TensorView;
using flashrt::modalities::VisionFrame;

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

struct ReplayProbe {
    int calls = 0;
    frt_graph expected_graph = nullptr;
    frt_shape_key expected_key = 0;
    int expected_stream = -1;
};

int fake_replay(frt_graph graph, frt_shape_key key, int stream_id, void* user) {
    auto* p = static_cast<ReplayProbe*>(user);
    p->calls += 1;
    assert(graph == p->expected_graph);
    assert(key == p->expected_key);
    assert(stream_id == p->expected_stream);
    return 0;
}

frt_runtime_export_v1 make_export(Owner* owner,
                                  frt_runtime_graph_desc* graph_desc) {
    frt_runtime_export_v1 exp{};
    exp.abi_version = FRT_RUNTIME_ABI_VERSION;
    exp.struct_size = sizeof(frt_runtime_export_v1);
    exp.graphs = graph_desc;
    exp.n_graphs = 1;
    exp.fingerprint = 0x1234;
    exp.identity = "test-pi05-runtime";
    exp.owner = owner;
    exp.retain = retain_owner;
    exp.release = release_owner;
    return exp;
}

void test_adopted_export_runtime_flow() {
    Owner owner;
    frt_runtime_graph_desc graph{};
    graph.name = "infer";
    graph.handle = reinterpret_cast<frt_graph>(0x1000);
    graph.default_key = 7;
    graph.stream_id = 3;
    auto exp = make_export(&owner, &graph);

    const auto vision_spec = flashrt::models::pi05::vision_preprocess_spec(1);
    std::vector<std::uint16_t> image_input(
        flashrt::modalities::required_vision_output_bytes(vision_spec) / 2);
    TensorView image_view{image_input.data(),
                          static_cast<std::uint64_t>(image_input.size() * 2),
                          DType::kBFloat16, MemoryPlace::kHost, Layout::kNHWC,
                          Shape{1, 224, 224, 3}};

    std::vector<float> action_model(1 * 4);
    action_model[0] = 2.0f;
    action_model[1] = -1.0f;
    action_model[2] = 0.5f;
    action_model[3] = 99.0f;
    TensorView action_view{action_model.data(),
                           static_cast<std::uint64_t>(action_model.size() * 4),
                           DType::kFloat32, MemoryPlace::kHost, Layout::kFlat,
                           Shape{1, 4}};

    ReplayProbe probe;
    probe.expected_graph = graph.handle;
    probe.expected_key = graph.default_key;
    probe.expected_stream = graph.stream_id;

    flashrt::models::pi05::RuntimeConfig cfg;
    cfg.num_views = 1;
    cfg.chunk = 1;
    cfg.model_action_dim = 4;
    cfg.robot_action_dim = 3;
    cfg.action_mean = {10.0f, 20.0f, 30.0f};
    cfg.action_stddev = {2.0f, 3.0f, 4.0f};
    cfg.image_input_override = image_view;
    cfg.action_output_override = action_view;
    cfg.replay_fn = fake_replay;
    cfg.replay_user = &probe;

    {
        flashrt::models::pi05::Runtime runtime(&exp, cfg);
        assert(runtime.ok());
        assert(owner.retain == 1);
        assert(runtime.export_runtime() == &exp);
        assert(runtime.manifest().vision.view_order.size() == 1);
        assert(runtime.manifest().graphs.infer == "infer");
        assert(runtime.set_prompt("pick up the cube") != 0);

        const std::uint8_t image_rgb[] = {
            0, 127, 255, 255, 127, 0,
            10, 20, 30, 40, 50, 60,
        };
        VisionFrame image;
        image.name = "image";
        image.image = {const_cast<std::uint8_t*>(image_rgb), sizeof(image_rgb),
                       DType::kUInt8, MemoryPlace::kHost, Layout::kHWC,
                       Shape{2, 2, 3}};
        image.format = PixelFormat::kRGB8;
        image.width = 2;
        image.height = 2;

        auto st = runtime.prepare_vision({image});
        assert(st.ok_status());
        VisionFrame bgr = image;
        bgr.format = PixelFormat::kBGR8;
        st = runtime.prepare_vision({bgr});
        assert(!st.ok_status());
        assert(st.code == flashrt::modalities::StatusCode::kShapeMismatch);
        assert(runtime.replay_tick() == 0);
        assert(probe.calls == 1);

        std::vector<float> actions;
        st = runtime.read_actions(&actions);
        assert(st.ok_status());
        assert(actions.size() == 3);
        assert(std::fabs(actions[0] - 12.0f) < 0.01f);
        assert(std::fabs(actions[1] - 17.0f) < 0.01f);
        assert(std::fabs(actions[2] - 32.0f) < 0.01f);
    }

    assert(owner.release == 1);
}

void test_prompt_staging_when_configured() {
#ifdef FLASHRT_CPP_HAS_SENTENCEPIECE
    const char* tokenizer = std::getenv("FLASH_RT_PALIGEMMA_TOKENIZER");
    if (!tokenizer || tokenizer[0] == '\0') {
        std::cout << "SKIP - FLASH_RT_PALIGEMMA_TOKENIZER not set\n";
        return;
    }

    Owner owner;
    frt_runtime_graph_desc graph{};
    graph.name = "infer";
    graph.handle = reinterpret_cast<frt_graph>(0x2000);
    graph.default_key = 9;
    graph.stream_id = 5;
    auto exp = make_export(&owner, &graph);

    constexpr std::uint64_t vocab = 257152;
    constexpr std::uint64_t hidden = 2;
    constexpr std::uint64_t max_tokens = 32;
    std::vector<float> table(vocab * hidden);
    for (std::uint64_t i = 0; i < vocab; ++i) {
        table[i * hidden + 0] = static_cast<float>(i);
        table[i * hidden + 1] = -static_cast<float>(i);
    }
    std::vector<float> prompt(max_tokens * hidden, 3.0f);
    std::vector<std::uint16_t> image_input(1 * 224 * 224 * 3);
    std::vector<float> action_model(4, 0.0f);

    flashrt::models::pi05::RuntimeConfig cfg;
    cfg.num_views = 1;
    cfg.chunk = 1;
    cfg.model_action_dim = 4;
    cfg.robot_action_dim = 3;
    cfg.image_input_override = TensorView{
        image_input.data(), static_cast<std::uint64_t>(image_input.size() * 2),
        DType::kBFloat16, MemoryPlace::kHost, Layout::kNHWC,
        Shape{1, 224, 224, 3}};
    cfg.action_output_override = TensorView{
        action_model.data(), static_cast<std::uint64_t>(action_model.size() * 4),
        DType::kFloat32, MemoryPlace::kHost, Layout::kFlat, Shape{1, 4}};
    cfg.prompt_tokenizer_model_path = tokenizer;
    cfg.prompt_vocab_size = vocab;
    cfg.prompt_hidden_dim = hidden;
    cfg.prompt_max_tokens = max_tokens;
    cfg.prompt_embedding_scale = 0.5f;
    cfg.prompt_embedding_table = TensorView{
        table.data(), static_cast<std::uint64_t>(table.size() * 4),
        DType::kFloat32, MemoryPlace::kHost, Layout::kFlat,
        Shape{vocab, hidden}};
    cfg.prompt_embedding_output = TensorView{
        prompt.data(), static_cast<std::uint64_t>(prompt.size() * 4),
        DType::kFloat32, MemoryPlace::kHost, Layout::kFlat,
        Shape{max_tokens, hidden}};

    flashrt::models::pi05::Runtime runtime(&exp, cfg);
    assert(runtime.ok());
    const float state[] = {0.0f, 1.0f, -1.0f};
    assert(runtime.set_prompt_state("pick_up_cube", state, 3) == 0);
    assert(runtime.current_prompt_len() == 24);
    assert(std::fabs(prompt[0] - 1.0f) < 0.001f);
    assert(std::fabs(prompt[1] + 1.0f) < 0.001f);
    assert(prompt[24 * hidden] == 0.0f);
#endif
}

}  // namespace

int main() {
    test_adopted_export_runtime_flow();
    test_prompt_staging_when_configured();
    std::cout << "PASS - Pi05 C++ runtime flow\n";
    return 0;
}
