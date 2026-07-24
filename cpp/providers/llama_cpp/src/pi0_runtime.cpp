#include "flashrt/providers/llama_cpp/c_api.h"
#include "flashrt/providers/llama_cpp/jetson_pi_engine.h"
#include "checkpoint_identity.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>

namespace {

struct RuntimeOwner {
    frt_llama_cpp_engine_v1 engine{};
    std::string last_error;
    int64_t image_shape[4] = {};
    int64_t state_shape[1] = {};
    int64_t action_shape[2] = {};
    bool staged_context_action = false;
};

constexpr uint32_t kInferExecutor = 37;
constexpr uint32_t kContextExecutor = 4;
constexpr uint32_t kActionExecutor = 91;

int unsupported_prepare(void* self, uint32_t, frt_shape_key) {
    auto* owner = static_cast<RuntimeOwner*>(self);
    if (owner) owner->last_error = "llama_cpp Pi0 provider has no graph variants";
    return -3;
}

const char* engine_error(RuntimeOwner* owner) {
    const char* err = owner->engine.last_error(owner->engine.self);
    return err ? err : "llama_cpp Pi0 engine returned a null error";
}

int set_input(void* self, uint32_t port, const void* data, uint64_t bytes,
              int stream) {
    auto* owner = static_cast<RuntimeOwner*>(self);
    if (!owner) return -1;
    const int rc = owner->engine.set_input(owner->engine.self, port, data,
                                           bytes, stream);
    if (rc != 0) {
        owner->last_error = engine_error(owner);
    } else {
        owner->last_error.clear();
    }
    return rc;
}

int get_output(void* self, uint32_t port, void* out, uint64_t capacity,
               uint64_t* written, int stream) {
    auto* owner = static_cast<RuntimeOwner*>(self);
    if (!owner) return -1;
    const int rc = owner->engine.get_output(owner->engine.self, port, out,
                                            capacity, written, stream);
    if (rc != 0) {
        owner->last_error = engine_error(owner);
    } else {
        owner->last_error.clear();
    }
    return rc;
}

int run_engine_stage(void* self, uint32_t stage) {
    auto* owner = static_cast<RuntimeOwner*>(self);
    if (!owner) return -1;
    const int rc = owner->staged_context_action
        ? owner->engine.run_stage(owner->engine.self, stage)
        : (stage == FRT_LLAMA_CPP_PI0_STAGE_INDEX_INFER
               ? owner->engine.run_infer(owner->engine.self) : -1);
    if (rc != 0) {
        owner->last_error = engine_error(owner);
    } else {
        owner->last_error.clear();
    }
    return rc;
}

int step(void* self) {
    return run_engine_stage(self, FRT_LLAMA_CPP_PI0_STAGE_INDEX_INFER);
}

int run_opaque(void* self, uint32_t executor_ref) {
    switch (executor_ref) {
        case kInferExecutor:
            return run_engine_stage(
                self, FRT_LLAMA_CPP_PI0_STAGE_INDEX_INFER);
        case kContextExecutor:
            return run_engine_stage(
                self, FRT_LLAMA_CPP_PI0_STAGE_INDEX_CONTEXT);
        case kActionExecutor:
            return run_engine_stage(
                self, FRT_LLAMA_CPP_PI0_STAGE_INDEX_ACTION);
        default:
            static_cast<RuntimeOwner*>(self)->last_error =
                "unknown llama_cpp Pi0 executor ref";
            return -2;
    }
}

const char* last_error(void* self) {
    auto* owner = static_cast<RuntimeOwner*>(self);
    if (!owner) return "null llama_cpp Pi0 runtime";
    if (!owner->last_error.empty()) return owner->last_error.c_str();
    return engine_error(owner);
}

void destroy_owner(void* self) {
    auto* owner = static_cast<RuntimeOwner*>(self);
    if (owner->engine.release) owner->engine.release(owner->engine.self);
    delete owner;
}

}  // namespace

extern "C" int frt_llama_cpp_pi0_runtime_create_with_engine(
        const frt_llama_cpp_pi0_config* config,
        const frt_llama_cpp_engine_v1* engine,
        frt_model_runtime_v1** out) {
    if (!out) return -1;
    *out = nullptr;
    if (!config || config->struct_size < FRT_LLAMA_CPP_PI0_CONFIG_BASE_SIZE ||
        !config->model_path || !config->model_path[0] ||
        !config->mmproj_path || !config->mmproj_path[0] ||
        !config->backend || !config->backend[0] || !config->n_views ||
        config->n_views > 3 ||
        !config->image_height || !config->image_width ||
        !config->image_channels ||
        !config->action_steps || !config->action_dim) {
        return -1;
    }
    if (!engine || engine->struct_size < FRT_LLAMA_CPP_ENGINE_V1_BASE_SIZE ||
        !engine->self || !engine->set_input || !engine->run_infer ||
        !engine->get_output || !engine->last_error ||
        static_cast<bool>(engine->retain) !=
            static_cast<bool>(engine->release)) {
        return -1;
    }

    auto* owner = new (std::nothrow) RuntimeOwner();
    if (!owner) return -5;
    std::memcpy(&owner->engine, engine,
                std::min<size_t>(engine->struct_size, sizeof(owner->engine)));
    owner->staged_context_action =
        engine->struct_size >= FRT_LLAMA_CPP_ENGINE_V1_RUN_STAGE_SIZE &&
        owner->engine.run_stage &&
        (owner->engine.reserved &
         FRT_LLAMA_CPP_ENGINE_CAP_PI0_REAL_CONTEXT_ACTION) != 0;
    if (owner->engine.retain) owner->engine.retain(owner->engine.self);

    owner->image_shape[0] = static_cast<int64_t>(config->n_views);
    owner->image_shape[1] = static_cast<int64_t>(config->image_height);
    owner->image_shape[2] = static_cast<int64_t>(config->image_width);
    owner->image_shape[3] = static_cast<int64_t>(config->image_channels);
    // State width is model-specific: PI0.5 accepts 1..8 values, while legacy
    // Pi0 accepts 1..action_dim. Publish a bucket-variable dimension and let
    // the backend enforce the selected model's bound.
    owner->state_shape[0] = -1;
    owner->action_shape[0] = static_cast<int64_t>(config->action_steps);
    owner->action_shape[1] = static_cast<int64_t>(config->action_dim);

    frt_runtime_builder b = frt_model_runtime_builder_create_metadata();
    if (!b) {
        destroy_owner(owner);
        return -5;
    }

    int rc = 0;
    const int64_t prompt_shape[1] = {-1};
    rc |= frt_runtime_builder_add_port(
        b, "images", FRT_RT_MOD_IMAGE, FRT_RT_DTYPE_U8, FRT_RT_LAYOUT_NHWC,
        FRT_RT_PORT_IN, FRT_RT_PORT_STAGED, 1, owner->image_shape, 4, 0,
        nullptr, 0, 0);
    rc |= frt_runtime_builder_add_port(
        b, "prompt", FRT_RT_MOD_TEXT, FRT_RT_DTYPE_U8, FRT_RT_LAYOUT_FLAT,
        FRT_RT_PORT_IN, FRT_RT_PORT_STAGED, 1, prompt_shape, 1, 0,
        nullptr, 0, 0);
    rc |= frt_runtime_builder_add_port(
        b, "state", FRT_RT_MOD_STATE, FRT_RT_DTYPE_F32, FRT_RT_LAYOUT_FLAT,
        FRT_RT_PORT_IN, FRT_RT_PORT_STAGED, 1, owner->state_shape, 1, 0,
        nullptr, 0, 0);
    rc |= frt_runtime_builder_add_port(
        b, "actions", FRT_RT_MOD_ACTION, FRT_RT_DTYPE_F32, FRT_RT_LAYOUT_FLAT,
        FRT_RT_PORT_OUT, FRT_RT_PORT_STAGED, 0, owner->action_shape, 2, 0,
        nullptr, 0, 0);
    if (owner->staged_context_action) {
        rc |= frt_runtime_builder_add_generic_stage(
            b, "context", FRT_GENERIC_STAGE_OPAQUE, kContextExecutor,
            nullptr, 0);
        const uint32_t action_after[1] = {0};
        rc |= frt_runtime_builder_add_generic_stage(
            b, "action", FRT_GENERIC_STAGE_OPAQUE, kActionExecutor,
            action_after, 1);
    } else {
        rc |= frt_runtime_builder_add_generic_stage(
            b, "infer", FRT_GENERIC_STAGE_OPAQUE, kInferExecutor,
            nullptr, 0);
    }
    rc |= frt_runtime_builder_set_generic_stage_runner(b, owner, run_opaque);
    rc |= frt_runtime_builder_add_identity(b, "provider", "llama_cpp");
    rc |= frt_runtime_builder_add_identity(b, "model_family", "pi0");
    rc |= frt_runtime_builder_add_identity(b, "model_path", config->model_path);
    rc |= frt_runtime_builder_add_identity(b, "mmproj_path", config->mmproj_path);
    if (config->struct_size >= sizeof(*config) && config->model_identity &&
        config->model_identity[0] && config->mmproj_identity &&
        config->mmproj_identity[0]) {
        rc |= frt_runtime_builder_add_identity(
            b, "weights_sha256", config->model_identity);
        rc |= frt_runtime_builder_add_identity(
            b, "mmproj_sha256", config->mmproj_identity);
    }
    rc |= frt_runtime_builder_add_identity(b, "backend", config->backend);
    rc |= frt_runtime_builder_add_identity(
        b, "stage_plan",
        owner->staged_context_action ? "context_action" : "full");
    if (rc != 0) {
        (void)frt_runtime_builder_finish(b, nullptr, nullptr, nullptr);
        destroy_owner(owner);
        return -1;
    }

    frt_model_runtime_verbs verbs{};
    verbs.struct_size = sizeof(verbs);
    verbs.set_input = set_input;
    verbs.get_output = get_output;
    verbs.prepare = unsupported_prepare;
    verbs.step = step;
    verbs.last_error = last_error;

    frt_model_runtime_v1* model = frt_runtime_builder_finish_model(
        b, &verbs, owner, owner, nullptr, destroy_owner);
    if (!model) {
        (void)frt_runtime_builder_finish(b, nullptr, nullptr, nullptr);
        destroy_owner(owner);
        return -1;
    }
    *out = model;
    return 0;
}

extern "C" int frt_llama_cpp_pi0_runtime_open_with_engine_factory(
        const char* config_json,
        const frt_llama_cpp_engine_factory_v1* factory,
        frt_model_runtime_v1** out) {
    flashrt::providers::llama_cpp::clear_runtime_open_error();
    flashrt::providers::llama_cpp::set_runtime_open_error(
        "invalid Pi0 runtime open arguments or JSON config");
    if (!out) return -1;
    *out = nullptr;
    if (!factory ||
        factory->struct_size < sizeof(frt_llama_cpp_engine_factory_v1) ||
        !factory->create_pi0 || !factory->last_error) {
        return -1;
    }
    if (!config_json) {
        return -1;
    }

    frt_llama_cpp_pi0_config config{};
    config.struct_size = sizeof(config);
    std::string model_family;
    std::string model_path;
    std::string mmproj_path;
    std::string backend;
    std::string model_identity;
    std::string mmproj_identity;
    bool seen_model_family = false;
    bool seen_model_path = false;
    bool seen_mmproj_path = false;
    bool seen_backend = false;
    bool seen_n_views = false;
    bool seen_image_height = false;
    bool seen_image_width = false;
    bool seen_image_channels = false;
    bool seen_action_steps = false;
    bool seen_action_dim = false;
    const char* p = config_json;
    auto skip_ws = [&p]() {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    };
    auto parse_string = [&p](std::string* out) {
        if (*p != '"') return false;
        ++p;
        out->clear();
        while (*p && *p != '"') {
            if (*p == '\\') {
                ++p;
                switch (*p) {
                    case '"':
                    case '\\':
                    case '/':
                        out->push_back(*p++);
                        break;
                    case 'b':
                        out->push_back('\b');
                        ++p;
                        break;
                    case 'f':
                        out->push_back('\f');
                        ++p;
                        break;
                    case 'n':
                        out->push_back('\n');
                        ++p;
                        break;
                    case 'r':
                        out->push_back('\r');
                        ++p;
                        break;
                    case 't':
                        out->push_back('\t');
                        ++p;
                        break;
                    default:
                        return false;
                }
            } else {
                const unsigned char ch = static_cast<unsigned char>(*p);
                if (ch < 0x20) return false;
                out->push_back(*p++);
            }
        }
        if (*p != '"') return false;
        ++p;
        return true;
    };
    auto parse_u32 = [&p](uint32_t* out) {
        uint32_t value = 0;
        if (*p < '0' || *p > '9') return false;
        if (*p == '0') {
            ++p;
        } else {
            while (*p >= '0' && *p <= '9') {
                const uint32_t digit = static_cast<uint32_t>(*p - '0');
                if (value > (UINT32_MAX - digit) / 10u) return false;
                value = value * 10u + digit;
                ++p;
            }
        }
        if (value == 0) return false;
        *out = value;
        return true;
    };
    skip_ws();
    if (*p != '{') return -1;
    ++p;
    skip_ws();
    if (*p == '}') return -1;
    for (;;) {
        std::string key;
        if (!parse_string(&key)) return -1;
        skip_ws();
        if (*p != ':') return -1;
        ++p;
        skip_ws();
        if (key == "model_family") {
            if (seen_model_family || !parse_string(&model_family) ||
                model_family.empty()) {
                return -1;
            }
            seen_model_family = true;
        } else if (key == "model_path") {
            if (seen_model_path || !parse_string(&model_path) ||
                model_path.empty()) {
                return -1;
            }
            seen_model_path = true;
        } else if (key == "mmproj_path") {
            if (seen_mmproj_path || !parse_string(&mmproj_path) ||
                mmproj_path.empty()) {
                return -1;
            }
            seen_mmproj_path = true;
        } else if (key == "backend") {
            if (seen_backend || !parse_string(&backend) || backend.empty()) {
                return -1;
            }
            seen_backend = true;
        } else if (key == "n_views") {
            if (seen_n_views || !parse_u32(&config.n_views)) return -1;
            seen_n_views = true;
        } else if (key == "image_height") {
            if (seen_image_height || !parse_u32(&config.image_height)) {
                return -1;
            }
            seen_image_height = true;
        } else if (key == "image_width") {
            if (seen_image_width || !parse_u32(&config.image_width)) return -1;
            seen_image_width = true;
        } else if (key == "image_channels") {
            if (seen_image_channels || !parse_u32(&config.image_channels)) {
                return -1;
            }
            seen_image_channels = true;
        } else if (key == "action_steps") {
            if (seen_action_steps || !parse_u32(&config.action_steps)) {
                return -1;
            }
            seen_action_steps = true;
        } else if (key == "action_dim") {
            if (seen_action_dim || !parse_u32(&config.action_dim)) return -1;
            seen_action_dim = true;
        } else {
            return -1;
        }
        skip_ws();
        if (*p == '}') {
            ++p;
            break;
        }
        if (*p != ',') return -1;
        ++p;
        skip_ws();
    }
    skip_ws();
    if (*p != '\0') return -1;
    if (!seen_model_family || model_family != "pi0" || !seen_model_path ||
        !seen_mmproj_path || !seen_backend || !seen_n_views ||
        !seen_image_height || !seen_image_width || !seen_image_channels ||
        !seen_action_steps || !seen_action_dim || config.n_views > 3) {
        return -1;
    }
    config.model_path = model_path.c_str();
    config.mmproj_path = mmproj_path.c_str();
    config.backend = backend.c_str();
    std::string identity_error;
    if (!flashrt::providers::llama_cpp::checkpoint_identity(
            config.model_path, &model_identity, &identity_error) ||
        !flashrt::providers::llama_cpp::checkpoint_identity(
            config.mmproj_path, &mmproj_identity, &identity_error)) {
        flashrt::providers::llama_cpp::set_runtime_open_error(identity_error);
        return -1;
    }
    config.model_identity = model_identity.c_str();
    config.mmproj_identity = mmproj_identity.c_str();

    frt_llama_cpp_engine_v1 engine{};
    const int rc = factory->create_pi0(factory->self, &config, &engine);
    if (rc != 0) {
        const char* error = factory->last_error(factory->self);
        flashrt::providers::llama_cpp::set_runtime_open_error(
            error ? error : "Pi0 engine factory failed without an error");
        return rc;
    }
    if (engine.struct_size < FRT_LLAMA_CPP_ENGINE_V1_BASE_SIZE ||
        !engine.self || !engine.retain || !engine.release ||
        !engine.set_input || !engine.run_infer || !engine.get_output ||
        !engine.last_error) {
        if (engine.struct_size >=
                offsetof(frt_llama_cpp_engine_v1, release) + sizeof(engine.release) &&
            engine.self && engine.release) {
            engine.release(engine.self);
        }
        flashrt::providers::llama_cpp::set_runtime_open_error(
            "factory returned an invalid Pi0 engine");
        return -1;
    }
    frt_model_runtime_v1* model = nullptr;
    const int create_rc =
        frt_llama_cpp_pi0_runtime_create_with_engine(&config, &engine,
                                                     &model);
    engine.release(engine.self);
    if (create_rc != 0) {
        flashrt::providers::llama_cpp::set_runtime_open_error(
            "failed to create Pi0 model runtime");
        return create_rc;
    }
    *out = model;
    flashrt::providers::llama_cpp::clear_runtime_open_error();
    return 0;
}
