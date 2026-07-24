#include "flashrt/providers/llama_cpp/c_api.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", (msg)); g_fail = 1; } \
    else { std::printf("ok  : %s\n", (msg)); } \
} while (0)

namespace {

struct FakeEngine {
    int retains = 0;
    int releases = 0;
    int infer = 0;
    int stage_calls = 0;
    uint32_t last_stage = UINT32_MAX;
    int set_input_calls = 0;
    uint32_t last_port = 999;
    float actions[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const char* last_error = "";
};

struct FakeFactory {
    FakeEngine* engine = nullptr;
    int creates = 0;
    frt_llama_cpp_pi0_config seen{};
    std::string seen_model_path;
    std::string seen_mmproj_path;
    std::string seen_backend;
    const char* last_error = "";
    bool return_borrowed = false;
    bool return_retain_only = false;
    bool return_undersized = false;
    bool return_null_self = false;
    bool return_missing_set_input = false;
    bool fail_after_engine = false;
};

void retain_engine(void* p) {
    static_cast<FakeEngine*>(p)->retains += 1;
}

void release_engine(void* p) {
    static_cast<FakeEngine*>(p)->releases += 1;
}

int set_input(void* p, uint32_t port, const void* data, uint64_t bytes,
              int stream) {
    (void)data;
    (void)bytes;
    (void)stream;
    auto* engine = static_cast<FakeEngine*>(p);
    engine->set_input_calls += 1;
    engine->last_port = port;
    return 0;
}

int run_infer(void* p) {
    auto* engine = static_cast<FakeEngine*>(p);
    engine->infer += 1;
    engine->actions[0] += 10.0f;
    return 0;
}

int run_stage(void* p, uint32_t stage) {
    auto* engine = static_cast<FakeEngine*>(p);
    engine->stage_calls += 1;
    engine->last_stage = stage;
    return 0;
}

int get_output(void* p, uint32_t port, void* out, uint64_t capacity,
               uint64_t* written, int stream) {
    (void)stream;
    auto* engine = static_cast<FakeEngine*>(p);
    if (port != FRT_LLAMA_CPP_PI0_PORT_ACTIONS) {
        engine->last_error = "unknown output port";
        return -1;
    }
    const uint64_t need = sizeof(engine->actions);
    if (written) *written = need;
    if (capacity < need) {
        engine->last_error = "action output buffer is too small";
        return -5;
    }
    std::memcpy(out, engine->actions, need);
    engine->last_error = "";
    return 0;
}

const char* last_error(void* p) {
    return static_cast<FakeEngine*>(p)->last_error;
}

const char* null_last_error(void*) {
    return nullptr;
}

int create_pi0_engine(void* p, const frt_llama_cpp_pi0_config* config,
                      frt_llama_cpp_engine_v1* out) {
    auto* factory = static_cast<FakeFactory*>(p);
    factory->creates += 1;
    if (!config || !out || !factory->engine) {
        factory->last_error = "invalid factory input";
        return -1;
    }
    if (factory->fail_after_engine) {
        out->struct_size = sizeof(*out);
        out->self = factory->engine;
        out->retain = retain_engine;
        out->release = release_engine;
        factory->last_error = "factory failed";
        return -7;
    }
    factory->seen = *config;
    factory->seen_model_path = config->model_path ? config->model_path : "";
    factory->seen_mmproj_path = config->mmproj_path ? config->mmproj_path : "";
    factory->seen_backend = config->backend ? config->backend : "";
    out->struct_size = factory->return_undersized ? 8u : sizeof(*out);
    out->self = factory->return_null_self ? nullptr : factory->engine;
    out->retain = (factory->return_borrowed ? nullptr : retain_engine);
    out->release = (factory->return_borrowed || factory->return_retain_only)
                       ? nullptr
                       : release_engine;
    out->set_input = factory->return_missing_set_input ? nullptr : set_input;
    out->run_infer = run_infer;
    out->get_output = get_output;
    out->last_error = last_error;
    return 0;
}

const char* factory_last_error(void* p) {
    return static_cast<FakeFactory*>(p)->last_error;
}

const frt_generic_stage_plan_ext_v1* generic_plan(
        const frt_model_runtime_v1* model) {
    if (!model ||
        model->struct_size < FRT_MODEL_RUNTIME_V1_QUERY_EXTENSION_SIZE ||
        !model->query_extension) return nullptr;
    const void* extension = nullptr;
    if (model->query_extension(
            model, FRT_EXT_GENERIC_STAGE_PLAN_V1, 1, &extension) != 0) {
        return nullptr;
    }
    return static_cast<const frt_generic_stage_plan_ext_v1*>(extension);
}

}  // namespace

int main() {
    frt_model_runtime_v1* bad = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(nullptr, nullptr, &bad)
              == -1 && bad == nullptr,
          "create rejects missing config and engine");

    FakeEngine engine;
    frt_llama_cpp_engine_v1 engine_api{};
    engine_api.struct_size = sizeof(engine_api);
    engine_api.self = &engine;
    engine_api.retain = retain_engine;
    engine_api.release = release_engine;
    engine_api.set_input = set_input;
    engine_api.run_infer = run_infer;
    engine_api.get_output = get_output;
    engine_api.last_error = last_error;

    FakeEngine old_engine;
    frt_llama_cpp_engine_v1 old_engine_api = engine_api;
    old_engine_api.self = &old_engine;
    old_engine_api.struct_size = FRT_LLAMA_CPP_ENGINE_V1_BASE_SIZE;
    old_engine_api.run_stage = nullptr;

    frt_model_runtime_v1* old_pi0 = nullptr;

    frt_llama_cpp_llm_config llm_cfg{};
    llm_cfg.struct_size = FRT_LLAMA_CPP_LLM_CONFIG_BASE_SIZE;
    llm_cfg.model_path = "/models/llm.gguf";
    llm_cfg.backend = "cpu";
    llm_cfg.n_ctx = 2048;
    llm_cfg.max_tokens = 16;
    frt_model_runtime_v1* old_llm = nullptr;
    CHECK(frt_llama_cpp_llm_runtime_create_with_engine(
              &llm_cfg, &old_engine_api, &old_llm) == 0 && old_llm &&
              old_llm->n_ports == 2 && generic_plan(old_llm) &&
              generic_plan(old_llm)->n_stages == 1,
          "old-prefix LLM engine exposes only infer schema");
    if (old_llm) old_llm->release(old_llm->owner);

    frt_llama_cpp_mllm_config mllm_cfg{};
    mllm_cfg.struct_size = FRT_LLAMA_CPP_MLLM_CONFIG_BASE_SIZE;
    mllm_cfg.model_path = "/models/mllm.gguf";
    mllm_cfg.mmproj_path = "/models/mllm-mmproj.gguf";
    mllm_cfg.backend = "cpu";
    mllm_cfg.n_ctx = 2048;
    mllm_cfg.max_tokens = 16;
    frt_model_runtime_v1* old_mllm = nullptr;
    CHECK(frt_llama_cpp_mllm_runtime_create_with_engine(
              &mllm_cfg, &old_engine_api, &old_mllm) == 0 && old_mllm &&
              old_mllm->n_ports == 3 && generic_plan(old_mllm) &&
              generic_plan(old_mllm)->n_stages == 1,
          "old-prefix MLLM engine exposes only infer schema");
    if (old_mllm) old_mllm->release(old_mllm->owner);

    frt_llama_cpp_pi0_config cfg{};
    cfg.struct_size = FRT_LLAMA_CPP_PI0_CONFIG_BASE_SIZE;
    cfg.model_path = "/models/pi0.gguf";
    cfg.mmproj_path = "/models/pi0-mmproj.gguf";
    cfg.backend = "cpu";
    cfg.n_views = 2;
    cfg.image_height = 224;
    cfg.image_width = 224;
    cfg.image_channels = 3;
    cfg.action_steps = 2;
    cfg.action_dim = 2;

    cfg.n_views = 4;
    bad = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(
              &cfg, &old_engine_api, &bad) == -1 && bad == nullptr,
          "create rejects more than three Pi0 views");
    cfg.n_views = 2;

    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(
              &cfg, &old_engine_api, &old_pi0) == 0 && old_pi0 &&
              generic_plan(old_pi0) &&
              generic_plan(old_pi0)->n_stages == 1,
          "old-prefix Pi0 engine exposes only infer schema");
    if (old_pi0) old_pi0->release(old_pi0->owner);

    frt_llama_cpp_engine_v1 asymmetric = engine_api;
    asymmetric.retain = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(&cfg, &asymmetric,
                                                       &bad) == -1 &&
              bad == nullptr,
          "create rejects release without retain");

    asymmetric = engine_api;
    asymmetric.release = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(&cfg, &asymmetric,
                                                       &bad) == -1 &&
              bad == nullptr,
          "create rejects retain without release");

    frt_llama_cpp_engine_v1 borrowed = engine_api;
    borrowed.retain = nullptr;
    borrowed.release = nullptr;
    frt_model_runtime_v1* borrowed_model = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(&cfg, &borrowed,
                                                       &borrowed_model) == 0 &&
              borrowed_model,
          "create accepts a borrowed explicit engine");
    borrowed_model->release(borrowed_model->owner);

    frt_model_runtime_v1* model = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(&cfg, &engine_api,
                                                       &model) == 0 && model,
          "create llama_cpp Pi0 runtime with explicit engine");
    CHECK(engine.retains == 1, "engine retained once");
    const frt_generic_stage_plan_ext_v1* plan = generic_plan(model);
    CHECK(model->abi_version == FRT_MODEL_RUNTIME_ABI_VERSION &&
              model->n_ports == 4 && model->n_stages == 0 && plan &&
              plan->n_stages == 1,
          "runtime exposes v1 metadata and one generic authority");
    CHECK(model->exp && model->exp->ctx == nullptr &&
              model->exp->n_graphs == 0,
          "provider runtime has no FlashRT exec graph");
    CHECK(std::strcmp(model->ports[FRT_LLAMA_CPP_PI0_PORT_IMAGES].name,
                      "images") == 0 &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_IMAGES].update ==
                  FRT_RT_PORT_STAGED &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_IMAGES].shape[0] == 2 &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_IMAGES].shape[3] == 3,
          "images port schema");
    CHECK(std::strcmp(model->ports[FRT_LLAMA_CPP_PI0_PORT_PROMPT].name,
                      "prompt") == 0 &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_PROMPT].modality ==
                  FRT_RT_MOD_TEXT &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_PROMPT].shape[0] == -1,
          "prompt port schema");
    CHECK(std::strcmp(model->ports[FRT_LLAMA_CPP_PI0_PORT_STATE].name,
                      "state") == 0 &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_STATE].dtype ==
                  FRT_RT_DTYPE_F32 &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_STATE].shape[0] == -1,
          "state port schema is model-variable");
    CHECK(std::strcmp(model->ports[FRT_LLAMA_CPP_PI0_PORT_ACTIONS].name,
                      "actions") == 0 &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_ACTIONS].direction ==
                  FRT_RT_PORT_OUT &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_ACTIONS].shape[0] == 2 &&
              model->ports[FRT_LLAMA_CPP_PI0_PORT_ACTIONS].shape[1] == 2,
          "actions port schema");
    CHECK(std::strcmp(plan->stages[0].name, "infer") == 0 &&
              plan->stages[0].executor_kind == FRT_GENERIC_STAGE_OPAQUE &&
              plan->stages[0].executor_ref == 37,
          "infer generic stage schema");
    CHECK(std::strstr(model->exp->identity, "provider=llama_cpp\n") &&
              std::strstr(model->exp->identity, "model_family=pi0\n") &&
              std::strstr(model->exp->identity,
                          "gstage-v1:0:5:infer:1:37:0:"),
          "identity carries provider, model family, and generic stage");

    CHECK(model->verbs.set_input(model->self, FRT_LLAMA_CPP_PI0_PORT_PROMPT,
                                    "hello", 5, -1) == 0 &&
              engine.set_input_calls == 1 &&
              engine.last_port == FRT_LLAMA_CPP_PI0_PORT_PROMPT,
          "set_input delegates to engine");
    CHECK(plan->run_opaque(plan->stage_self,
                           plan->stages[0].executor_ref) == 0 &&
              engine.infer == 1,
          "generic runner delegates executor ref to the engine");
    float out[4] = {};
    uint64_t written = 0;
    CHECK(model->verbs.get_output(model->self,
                                     FRT_LLAMA_CPP_PI0_PORT_ACTIONS,
                                     out, sizeof(out), &written, -1) == 0 &&
              written == sizeof(out) && std::fabs(out[0] - 11.0f) < 0.01f,
          "get_output delegates to engine");
    CHECK(model->verbs.prepare(model->self, 0, 0) == -3 &&
              std::strstr(model->verbs.last_error(model->self),
                          "no graph variants"),
          "prepare hard-errors instead of fabricating graph variants");
    model->release(model->owner);
    CHECK(engine.releases == 1, "engine released once");

    // A callback without the explicit capability may be the old cached-action
    // shim, so it must not publish a staged plan.
    FakeEngine staged_engine;
    frt_llama_cpp_engine_v1 staged_engine_api = engine_api;
    staged_engine_api.self = &staged_engine;
    staged_engine_api.run_stage = run_stage;
    staged_engine_api.struct_size = FRT_LLAMA_CPP_ENGINE_V1_RUN_STAGE_SIZE;
    frt_model_runtime_v1* staged_model = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(
              &cfg, &staged_engine_api, &staged_model) == 0 && staged_model,
          "create staged Pi0 runtime");
    const auto* staged_plan = generic_plan(staged_model);
    CHECK(staged_plan && staged_plan->n_stages == 1 &&
              std::strcmp(staged_plan->stages[0].name, "infer") == 0 &&
              staged_plan->stages[0].executor_ref == 37,
          "uncapable Pi0 runtime publishes a single infer authority");
    CHECK(staged_model->verbs.step(staged_model->self) == 0 &&
              staged_engine.infer == 1,
          "whole-model step delegates to infer (run_stage not used)");
    if (staged_model) staged_model->release(staged_model->owner);

    FakeEngine capable_engine;
    frt_llama_cpp_engine_v1 capable_engine_api = staged_engine_api;
    capable_engine_api.self = &capable_engine;
    capable_engine_api.reserved =
        FRT_LLAMA_CPP_ENGINE_CAP_PI0_REAL_CONTEXT_ACTION;
    frt_model_runtime_v1* capable_model = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(
              &cfg, &capable_engine_api, &capable_model) == 0 && capable_model,
          "create real-split Pi0 runtime");
    const auto* capable_plan = generic_plan(capable_model);
    CHECK(capable_plan && capable_plan->n_stages == 2 &&
              std::strcmp(capable_plan->stages[0].name, "context") == 0 &&
              std::strcmp(capable_plan->stages[1].name, "action") == 0 &&
              capable_plan->stages[1].n_after == 1 &&
              capable_plan->stages[1].after[0] == 0,
          "capable Pi0 runtime publishes context then action");
    CHECK(capable_plan->run_opaque(
              capable_plan->stage_self,
              capable_plan->stages[0].executor_ref) == 0 &&
              capable_engine.last_stage == FRT_LLAMA_CPP_PI0_STAGE_INDEX_CONTEXT,
          "context authority delegates to the real split callback");
    CHECK(capable_plan->run_opaque(
              capable_plan->stage_self,
              capable_plan->stages[1].executor_ref) == 0 &&
              capable_engine.last_stage == FRT_LLAMA_CPP_PI0_STAGE_INDEX_ACTION,
          "action authority delegates to the real split callback");
    CHECK(capable_model->verbs.step(capable_model->self) == 0 &&
              capable_engine.last_stage == FRT_LLAMA_CPP_PI0_STAGE_INDEX_INFER,
          "whole-model step remains available on a split-capable engine");
    if (capable_model) capable_model->release(capable_model->owner);

    FakeEngine null_error_engine;
    frt_llama_cpp_engine_v1 null_error_api = engine_api;
    null_error_api.self = &null_error_engine;
    null_error_api.last_error = null_last_error;
    frt_model_runtime_v1* null_error_model = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_create_with_engine(
              &cfg, &null_error_api, &null_error_model) == 0 &&
              null_error_model,
          "create accepts engine with nullable error implementation");
    CHECK(null_error_model->verbs.get_output(
              null_error_model->self, FRT_LLAMA_CPP_PI0_PORT_PROMPT,
              out, sizeof(out), &written, -1) == -1 &&
              std::strstr(null_error_model->verbs.last_error(
                              null_error_model->self),
                          "null error"),
          "null engine errors are reported without crashing");
    null_error_model->release(null_error_model->owner);

    FakeEngine factory_engine;
    FakeFactory factory;
    factory.engine = &factory_engine;
    frt_llama_cpp_engine_factory_v1 factory_api{};
    factory_api.struct_size = sizeof(factory_api);
    factory_api.self = &factory;
    factory_api.create_pi0 = create_pi0_engine;
    factory_api.last_error = factory_last_error;

    const std::string identity_prefix =
        "/tmp/flashrt-identity-" + std::to_string(getpid());
    const std::string identity_model = identity_prefix + "-model.gguf";
    const std::string identity_mmproj = identity_prefix + "-mmproj.gguf";
    {
        std::string model_bytes(64 * 1024 + 16, 'a');
        model_bytes.back() = 'x';
        std::ofstream(identity_model, std::ios::binary)
            .write(model_bytes.data(), model_bytes.size());
        std::ofstream(identity_mmproj, std::ios::binary) << "mmproj-a";
    }
    const std::string open_json =
        "{\"model_family\":\"pi0\",\"model_path\":\"" +
        identity_model + "\",\"mmproj_path\":\"" + identity_mmproj +
        "\",\"backend\":\"cpu\",\"n_views\":2,"
        "\"image_height\":224,\"image_width\":224,"
        "\"image_channels\":3,\"action_steps\":2,\"action_dim\":2}";
    frt_model_runtime_v1* opened = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              open_json.c_str(), &factory_api, &opened) == 0 &&
              opened && factory.creates == 1,
          "open_with_engine_factory creates a Pi0 runtime from JSON");
    CHECK(factory.seen_model_path == identity_model &&
              factory.seen_mmproj_path == identity_mmproj &&
              factory.seen_backend == "cpu" &&
              factory.seen.n_views == 2 &&
              factory.seen.action_steps == 2 &&
              factory.seen.action_dim == 2,
          "factory receives parsed Pi0 config");
    CHECK(factory_engine.retains == 1 && factory_engine.releases == 1,
          "factory engine reference is transferred to the runtime");
    CHECK(opened->verbs.step(opened->self) == 0 &&
              factory_engine.infer == 1,
          "opened runtime step delegates whole-model infer");
    const uint64_t first_fingerprint = opened->exp->fingerprint;
    opened->release(opened->owner);
    CHECK(factory_engine.releases == 2,
          "opened runtime releases retained factory engine");
    {
        std::string model_bytes(64 * 1024 + 16, 'a');
        model_bytes.back() = 'y';
        std::ofstream(identity_model, std::ios::binary | std::ios::trunc)
            .write(model_bytes.data(), model_bytes.size());
    }
    frt_model_runtime_v1* changed_checkpoint = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              open_json.c_str(), &factory_api, &changed_checkpoint) == 0 &&
              changed_checkpoint &&
              changed_checkpoint->exp->fingerprint != first_fingerprint &&
              std::strstr(changed_checkpoint->exp->identity,
                          "weights_sha256=ee5c5dda1cef40957d80fafab7b0eeddeea052e221a311ee80ec4972cbdc5d5f"),
          "same-size checkpoint tail change alters deployment fingerprint");
    const uint64_t model_changed_fingerprint =
        changed_checkpoint ? changed_checkpoint->exp->fingerprint : 0;
    if (changed_checkpoint) changed_checkpoint->release(changed_checkpoint->owner);
    {
        std::ofstream(identity_mmproj, std::ios::binary | std::ios::trunc)
            << "mmproj-b";
    }
    frt_model_runtime_v1* changed_mmproj = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              open_json.c_str(), &factory_api, &changed_mmproj) == 0 &&
              changed_mmproj &&
              changed_mmproj->exp->fingerprint != model_changed_fingerprint &&
              std::strstr(changed_mmproj->exp->identity,
                          "mmproj_sha256=d772e18bbe6501374cfab6a5d76a93f9288b8f2bcb4ec9694356b32e3bcf4c1c"),
          "mmproj prefix changes deployment fingerprint");

    const std::string cuda_json =
        "{\"model_family\":\"pi0\",\"model_path\":\"" +
        identity_model + "\",\"mmproj_path\":\"" + identity_mmproj +
        "\",\"backend\":\"cuda\",\"n_views\":2,"
        "\"image_height\":224,\"image_width\":224,"
        "\"image_channels\":3,\"action_steps\":2,\"action_dim\":2}";
    frt_model_runtime_v1* cuda_identity = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              cuda_json.c_str(), &factory_api, &cuda_identity) == 0 &&
              cuda_identity &&
              changed_mmproj &&
              cuda_identity->exp->fingerprint !=
                  changed_mmproj->exp->fingerprint,
          "backend change alters deployment fingerprint");
    bool same_port_schema = cuda_identity && changed_mmproj &&
                            cuda_identity->n_ports == changed_mmproj->n_ports;
    if (same_port_schema) {
        for (uint64_t i = 0; i < cuda_identity->n_ports; ++i) {
            const frt_runtime_port_desc& lhs = cuda_identity->ports[i];
            const frt_runtime_port_desc& rhs = changed_mmproj->ports[i];
            same_port_schema &= std::strcmp(lhs.name, rhs.name) == 0 &&
                                lhs.modality == rhs.modality &&
                                lhs.dtype == rhs.dtype &&
                                lhs.layout == rhs.layout &&
                                lhs.direction == rhs.direction &&
                                lhs.update == rhs.update &&
                                lhs.required == rhs.required &&
                                lhs.rank == rhs.rank;
            for (uint32_t dim = 0; same_port_schema && dim < lhs.rank; ++dim) {
                same_port_schema &= lhs.shape[dim] == rhs.shape[dim];
            }
        }
    }
    CHECK(same_port_schema, "backend switch preserves port schema");
    if (cuda_identity) cuda_identity->release(cuda_identity->owner);
    if (changed_mmproj) changed_mmproj->release(changed_mmproj->owner);

    const int creates_before_missing_checkpoint = factory.creates;
    const std::string missing_checkpoint_json =
        "{\"model_family\":\"pi0\",\"model_path\":\"" +
        identity_prefix + "-missing.gguf\",\"mmproj_path\":\"" +
        identity_mmproj +
        "\",\"backend\":\"cpu\",\"n_views\":2,"
        "\"image_height\":224,\"image_width\":224,"
        "\"image_channels\":3,\"action_steps\":2,\"action_dim\":2}";
    frt_model_runtime_v1* missing_checkpoint = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              missing_checkpoint_json.c_str(), &factory_api,
              &missing_checkpoint) == -1 &&
              missing_checkpoint == nullptr &&
              factory.creates == creates_before_missing_checkpoint &&
              std::strstr(frt_llama_cpp_runtime_open_error(),
                          "failed to open checkpoint for identity"),
          "missing checkpoint reports identity error before factory creation");

    frt_model_runtime_v1* missing = nullptr;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              "{\"model_family\":\"pi0\",\"model_path\":\"/models/pi0.gguf\"}",
              &factory_api, &missing) == -1 &&
              missing == nullptr,
          "open rejects incomplete JSON config");
    CHECK(std::strstr(frt_llama_cpp_runtime_open_error(),
                      "invalid Pi0 runtime open arguments or JSON config"),
          "incomplete JSON reports runtime-open validation error");
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              "{\"model_family\":\"pi0\",\"model_path\":\"/models/pi0.gguf\","
              "\"mmproj_path\":\"/models/pi0-mmproj.gguf\","
              "\"backend\":\"cpu\",\"n_views\":2,\"image_height\":224,"
              "\"image_width\":224,\"image_channels\":3,"
              "\"action_steps\":2,\"action_dim\":2,\"unexpected\":1}",
              &factory_api, &missing) == -1 &&
              missing == nullptr,
          "open rejects unknown JSON fields");
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              "{\"model_family\":\"pi0\",\"model_family\":\"pi0\","
              "\"model_path\":\"/models/pi0.gguf\","
              "\"mmproj_path\":\"/models/pi0-mmproj.gguf\","
              "\"backend\":\"cpu\",\"n_views\":2,\"image_height\":224,"
              "\"image_width\":224,\"image_channels\":3,"
              "\"action_steps\":2,\"action_dim\":2}",
              &factory_api, &missing) == -1 &&
              missing == nullptr,
          "open rejects duplicate JSON fields");
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              "{\"model_family\":\"llm\",\"model_path\":\"/models/pi0.gguf\","
              "\"mmproj_path\":\"/models/pi0-mmproj.gguf\","
              "\"backend\":\"cpu\",\"n_views\":2,\"image_height\":224,"
              "\"image_width\":224,\"image_channels\":3,"
              "\"action_steps\":2,\"action_dim\":2}",
              &factory_api, &missing) == -1 &&
              missing == nullptr,
          "open rejects non-Pi0 model family");

    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              "{\"model_family\":\"pi0\",\"model_path\":\"/models/pi0.gguf\","
              "\"mmproj_path\":\"/models/pi0-mmproj.gguf\","
              "\"backend\":\"cpu\",\"n_views\":0,\"image_height\":224,"
              "\"image_width\":224,\"image_channels\":3,"
              "\"action_steps\":2,\"action_dim\":2}",
              &factory_api, &missing) == -1 &&
              missing == nullptr,
          "open rejects zero-sized numeric config fields");
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              "{\"model_family\":\"pi0\",\"model_path\":\"/models/pi0.gguf\","
              "\"mmproj_path\":\"/models/pi0-mmproj.gguf\","
              "\"backend\":\"cpu\",\"n_views\":4,\"image_height\":224,"
              "\"image_width\":224,\"image_channels\":3,"
              "\"action_steps\":2,\"action_dim\":2}",
              &factory_api, &missing) == -1 &&
              missing == nullptr,
          "open rejects more than three Pi0 views");
    FakeFactory borrowed_factory = factory;
    borrowed_factory.engine = &factory_engine;
    borrowed_factory.return_borrowed = true;
    frt_llama_cpp_engine_factory_v1 borrowed_factory_api = factory_api;
    borrowed_factory_api.self = &borrowed_factory;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              open_json.c_str(), &borrowed_factory_api, &missing) == -1 &&
              missing == nullptr,
          "open rejects borrowed engines from factories");
    const int releases_before_invalid_engine = factory_engine.releases;
    FakeFactory invalid_factory = factory;
    invalid_factory.engine = &factory_engine;
    invalid_factory.return_undersized = true;
    frt_llama_cpp_engine_factory_v1 invalid_factory_api = factory_api;
    invalid_factory_api.self = &invalid_factory;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              open_json.c_str(), &invalid_factory_api, &missing) == -1 &&
              missing == nullptr &&
              factory_engine.releases == releases_before_invalid_engine,
          "open rejects undersized factory engines without calling release");
    invalid_factory = factory;
    invalid_factory.engine = &factory_engine;
    invalid_factory.return_null_self = true;
    invalid_factory_api.self = &invalid_factory;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              open_json.c_str(), &invalid_factory_api, &missing) == -1 &&
              missing == nullptr &&
              factory_engine.releases == releases_before_invalid_engine,
          "open rejects null factory engine self without calling release");
    invalid_factory = factory;
    invalid_factory.engine = &factory_engine;
    invalid_factory.return_retain_only = true;
    invalid_factory_api.self = &invalid_factory;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              open_json.c_str(), &invalid_factory_api, &missing) == -1 &&
              missing == nullptr &&
              factory_engine.releases == releases_before_invalid_engine,
          "open rejects asymmetric factory engines without calling release");
    invalid_factory = factory;
    invalid_factory.engine = &factory_engine;
    invalid_factory.return_missing_set_input = true;
    invalid_factory_api.self = &invalid_factory;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              open_json.c_str(), &invalid_factory_api, &missing) == -1 &&
              missing == nullptr &&
              factory_engine.releases == releases_before_invalid_engine + 1,
          "open releases owned factory engines missing hot-path hooks");
    invalid_factory = factory;
    invalid_factory.engine = &factory_engine;
    invalid_factory.fail_after_engine = true;
    invalid_factory_api.self = &invalid_factory;
    CHECK(frt_llama_cpp_pi0_runtime_open_with_engine_factory(
              open_json.c_str(), &invalid_factory_api, &missing) == -7 &&
              missing == nullptr &&
              factory_engine.releases == releases_before_invalid_engine + 1,
          "open ignores out_engine when factory create fails");

    std::remove(identity_model.c_str());
    std::remove(identity_mmproj.c_str());

    std::printf(g_fail ? "\n== LLAMA_CPP PROVIDER FAILED ==\n"
                       : "\n== LLAMA_CPP PROVIDER PASSED ==\n");
    return g_fail;
}
