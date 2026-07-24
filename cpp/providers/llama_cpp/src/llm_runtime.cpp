// frt_llama_cpp_llm_runtime — v1 runtime wrapper for a generic GGUF LLM
// engine. Mirrors pi0_runtime.cpp's shape: 2 STAGED ports (prompt TEXT in,
// text TEXT out) + 1 callback "infer" stage. Strict JSON open path.
//
// The engine vtable (frt_llama_cpp_engine_v1) is reused: set_input(PROMPT),
// run_infer(), get_output(TEXT). No GGML types appear here.

#include "flashrt/providers/llama_cpp/c_api.h"
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
    int64_t prompt_shape[1] = {-1};   // variable-length UTF-8
    int64_t text_shape[1] = {-1};     // variable-length UTF-8 output
    int64_t token_shape[1] = {1};
    int64_t eog_shape[1] = {1};
    int64_t logits_shape[1] = {-1};
    int64_t tokens_shape[1] = {-1};
    bool staged_decode = false;
};

constexpr uint32_t kInferExecutor = 37;
constexpr uint32_t kResetExecutor = 4;
constexpr uint32_t kPrefillExecutor = 91;
constexpr uint32_t kDecodeExecutor = 113;

int unsupported_prepare(void* self, uint32_t, frt_shape_key) {
    auto* owner = static_cast<RuntimeOwner*>(self);
    if (owner) owner->last_error = "llama_cpp LLM provider has no graph variants";
    return -3;
}

const char* engine_error(RuntimeOwner* owner) {
    const char* err = owner->engine.last_error(owner->engine.self);
    return err ? err : "llama_cpp LLM engine returned a null error";
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
    int rc = 0;
    if (stage == FRT_LLAMA_CPP_LLM_STAGE_INDEX_INFER) {
        rc = owner->engine.run_infer(owner->engine.self);
    } else if (owner->staged_decode && owner->engine.run_stage &&
               (stage == FRT_LLAMA_CPP_LLM_STAGE_INDEX_RESET ||
                stage == FRT_LLAMA_CPP_LLM_STAGE_INDEX_PREFILL ||
                stage == FRT_LLAMA_CPP_LLM_STAGE_INDEX_DECODE)) {
        rc = owner->engine.run_stage(owner->engine.self, stage);
    } else {
        owner->last_error = "unknown or unsupported llama_cpp LLM stage";
        return -1;
    }
    if (rc != 0) {
        owner->last_error = engine_error(owner);
    } else {
        owner->last_error.clear();
    }
    return rc;
}

int step(void* self) {
    return run_engine_stage(self, FRT_LLAMA_CPP_LLM_STAGE_INDEX_INFER);
}

int run_opaque(void* self, uint32_t executor_ref) {
    switch (executor_ref) {
        case kInferExecutor:
            return run_engine_stage(
                self, FRT_LLAMA_CPP_LLM_STAGE_INDEX_INFER);
        case kResetExecutor:
            return run_engine_stage(
                self, FRT_LLAMA_CPP_LLM_STAGE_INDEX_RESET);
        case kPrefillExecutor:
            return run_engine_stage(
                self, FRT_LLAMA_CPP_LLM_STAGE_INDEX_PREFILL);
        case kDecodeExecutor:
            return run_engine_stage(
                self, FRT_LLAMA_CPP_LLM_STAGE_INDEX_DECODE);
        default:
            static_cast<RuntimeOwner*>(self)->last_error =
                "unknown llama_cpp LLM executor ref";
            return -2;
    }
}

const char* last_error(void* self) {
    auto* owner = static_cast<RuntimeOwner*>(self);
    if (!owner) return "null llama_cpp LLM runtime";
    if (!owner->last_error.empty()) return owner->last_error.c_str();
    return engine_error(owner);
}

void destroy_owner(void* self) {
    auto* owner = static_cast<RuntimeOwner*>(self);
    if (owner->engine.release) owner->engine.release(owner->engine.self);
    delete owner;
}

}  // namespace

extern "C" int frt_llama_cpp_llm_runtime_create_with_engine(
        const frt_llama_cpp_llm_config* config,
        const frt_llama_cpp_engine_v1* engine,
        frt_model_runtime_v1** out) {
    if (!out) return -1;
    *out = nullptr;
    if (!config || config->struct_size < FRT_LLAMA_CPP_LLM_CONFIG_BASE_SIZE ||
        !config->model_path || !config->model_path[0] ||
        !config->backend || !config->backend[0]) {
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
    owner->staged_decode =
        engine->struct_size >= FRT_LLAMA_CPP_ENGINE_V1_RUN_STAGE_SIZE &&
        owner->engine.run_stage;
    if (owner->engine.retain) owner->engine.retain(owner->engine.self);

    frt_runtime_builder b = frt_model_runtime_builder_create_metadata();
    if (!b) {
        destroy_owner(owner);
        return -5;
    }

    int rc = 0;
    rc |= frt_runtime_builder_add_port(
        b, "prompt", FRT_RT_MOD_TEXT, FRT_RT_DTYPE_U8, FRT_RT_LAYOUT_FLAT,
        FRT_RT_PORT_IN, FRT_RT_PORT_STAGED, 1, owner->prompt_shape, 1, 0,
        nullptr, 0, 0);
    rc |= frt_runtime_builder_add_port(
        b, "text", FRT_RT_MOD_TEXT, FRT_RT_DTYPE_U8, FRT_RT_LAYOUT_FLAT,
        FRT_RT_PORT_OUT, FRT_RT_PORT_STAGED, 0, owner->text_shape, 1, 0,
        nullptr, 0, 0);
    if (owner->staged_decode) {
        rc |= frt_runtime_builder_add_port(
            b, "next_token", FRT_RT_MOD_TENSOR, FRT_RT_DTYPE_I32,
            FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_OUT, FRT_RT_PORT_STAGED, 0,
            owner->token_shape, 1, 0, nullptr, 0, 0);
        rc |= frt_runtime_builder_add_port(
            b, "logits", FRT_RT_MOD_TENSOR, FRT_RT_DTYPE_F32,
            FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_OUT, FRT_RT_PORT_STAGED, 0,
            owner->logits_shape, 1, 0, nullptr, 0, 0);
        rc |= frt_runtime_builder_add_port(
            b, "is_eog", FRT_RT_MOD_TENSOR, FRT_RT_DTYPE_I32,
            FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_OUT, FRT_RT_PORT_STAGED, 0,
            owner->eog_shape, 1, 0, nullptr, 0, 0);
        rc |= frt_runtime_builder_add_port(
            b, "tokens", FRT_RT_MOD_TENSOR, FRT_RT_DTYPE_I32,
            FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_IN, FRT_RT_PORT_STAGED, 0,
            owner->tokens_shape, 1, 0, nullptr, 0, 0);
        rc |= frt_runtime_builder_add_generic_stage(
            b, "reset", FRT_GENERIC_STAGE_OPAQUE, kResetExecutor,
            nullptr, 0);
        const uint32_t prefill_after[1] = {0};
        rc |= frt_runtime_builder_add_generic_stage(
            b, "prefill", FRT_GENERIC_STAGE_OPAQUE, kPrefillExecutor,
            prefill_after, 1);
        const uint32_t decode_after[1] = {1};
        rc |= frt_runtime_builder_add_generic_stage(
            b, "decode", FRT_GENERIC_STAGE_OPAQUE, kDecodeExecutor,
            decode_after, 1);
    } else {
        rc |= frt_runtime_builder_add_generic_stage(
            b, "infer", FRT_GENERIC_STAGE_OPAQUE, kInferExecutor,
            nullptr, 0);
    }
    rc |= frt_runtime_builder_set_generic_stage_runner(b, owner, run_opaque);
    rc |= frt_runtime_builder_add_identity(b, "provider", "llama_cpp");
    rc |= frt_runtime_builder_add_identity(b, "model_family", "llm");
    rc |= frt_runtime_builder_add_identity(b, "model_path", config->model_path);
    if (config->struct_size >= sizeof(*config) && config->model_identity &&
        config->model_identity[0]) {
        rc |= frt_runtime_builder_add_identity(
            b, "weights_sha256", config->model_identity);
    }
    rc |= frt_runtime_builder_add_identity(b, "backend", config->backend);
    rc |= frt_runtime_builder_add_identity(
        b, "n_ctx", std::to_string(config->n_ctx).c_str());
    rc |= frt_runtime_builder_add_identity(
        b, "n_threads", std::to_string(config->n_threads).c_str());
    rc |= frt_runtime_builder_add_identity(
        b, "temp", std::to_string(config->temp).c_str());
    rc |= frt_runtime_builder_add_identity(
        b, "top_k", std::to_string(config->top_k).c_str());
    rc |= frt_runtime_builder_add_identity(
        b, "top_p", std::to_string(config->top_p).c_str());
    rc |= frt_runtime_builder_add_identity(
        b, "seed", std::to_string(config->seed).c_str());
    rc |= frt_runtime_builder_add_identity(
        b, "max_tokens", std::to_string(config->max_tokens).c_str());
    rc |= frt_runtime_builder_add_identity(
        b, "stage_plan", owner->staged_decode ? "staged_decode" : "full");
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

extern "C" int frt_llama_cpp_llm_runtime_open_with_engine_factory(
        const char* config_json,
        const frt_llama_cpp_engine_factory_v1* factory,
        frt_model_runtime_v1** out) {
    flashrt::providers::llama_cpp::clear_runtime_open_error();
    flashrt::providers::llama_cpp::set_runtime_open_error(
        "invalid LLM runtime open arguments or JSON config");
    if (!out) return -1;
    *out = nullptr;
    if (!factory ||
        factory->struct_size < sizeof(frt_llama_cpp_engine_factory_v1) ||
        !factory->create_llm || !factory->last_error) {
        return -1;
    }
    if (!config_json) {
        return -1;
    }

    frt_llama_cpp_llm_config config{};
    config.struct_size = sizeof(config);
    std::string model_family;
    std::string model_path;
    std::string backend;
    std::string model_identity;
    bool seen_model_family = false;
    bool seen_model_path = false;
    bool seen_backend = false;
    bool seen_n_ctx = false;
    bool seen_n_threads = false;
    bool seen_temp = false;
    bool seen_top_k = false;
    bool seen_top_p = false;
    bool seen_seed = false;
    bool seen_max_tokens = false;
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
        *out = value;
        return true;
    };
    auto parse_i32 = [&p](int32_t* out) {
        bool neg = false;
        if (*p == '-') { neg = true; ++p; }
        int32_t value = 0;
        if (*p < '0' || *p > '9') return false;
        if (*p == '0') {
            ++p;
        } else {
            while (*p >= '0' && *p <= '9') {
                const int32_t digit = static_cast<int32_t>(*p - '0');
                if (value > (INT32_MAX - digit) / 10) return false;
                value = value * 10 + digit;
                ++p;
            }
        }
        *out = neg ? -value : value;
        return true;
    };
    auto parse_f32 = [&p](float* out) {
        // Accepts an optional sign, integer part, optional fraction, optional
        // decimal exponent (e.g. "0.8", "-0.5", "1e-3", ".25"). Matches what
        // Python's json.dumps can emit for float fields. Does NOT accept
        // inf/nan (rejected as parse failure).
        bool neg = false;
        if (*p == '-' || *p == '+') { neg = (*p == '-'); ++p; }
        float mant = 0.0f;
        bool saw_digit = false;
        while (*p >= '0' && *p <= '9') {
            mant = mant * 10.0f + static_cast<float>(*p - '0');
            ++p;
            saw_digit = true;
        }
        if (*p == '.') {
            ++p;
            float scale = 0.1f;
            while (*p >= '0' && *p <= '9') {
                mant += static_cast<float>(*p - '0') * scale;
                scale *= 0.1f;
                ++p;
                saw_digit = true;
            }
        }
        if (!saw_digit) return false;
        if (*p == 'e' || *p == 'E') {
            ++p;
            bool exp_neg = false;
            if (*p == '-' || *p == '+') { exp_neg = (*p == '-'); ++p; }
            int exp = 0;
            bool saw_exp_digit = false;
            while (*p >= '0' && *p <= '9') {
                exp = exp * 10 + (*p - '0');
                ++p;
                saw_exp_digit = true;
            }
            if (!saw_exp_digit) return false;
            float scale = 1.0f;
            for (int i = 0; i < exp; ++i) scale *= 10.0f;
            if (exp_neg) mant /= scale; else mant *= scale;
        }
        *out = neg ? -mant : mant;
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
        } else if (key == "backend") {
            if (seen_backend || !parse_string(&backend) ||
                backend.empty()) {
                return -1;
            }
            seen_backend = true;
        } else if (key == "n_ctx") {
            if (seen_n_ctx || !parse_u32(&config.n_ctx)) return -1;
            seen_n_ctx = true;
        } else if (key == "n_threads") {
            if (seen_n_threads || !parse_i32(&config.n_threads)) return -1;
            seen_n_threads = true;
        } else if (key == "temp") {
            if (seen_temp || !parse_f32(&config.temp)) return -1;
            seen_temp = true;
        } else if (key == "top_k") {
            if (seen_top_k || !parse_i32(&config.top_k)) return -1;
            seen_top_k = true;
        } else if (key == "top_p") {
            if (seen_top_p || !parse_f32(&config.top_p)) return -1;
            seen_top_p = true;
        } else if (key == "seed") {
            if (seen_seed || !parse_u32(&config.seed)) return -1;
            seen_seed = true;
        } else if (key == "max_tokens") {
            if (seen_max_tokens || !parse_u32(&config.max_tokens)) return -1;
            seen_max_tokens = true;
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
    if (!seen_model_family || model_family != "llm" || !seen_model_path ||
        !seen_backend || !seen_n_ctx || !seen_n_threads || !seen_temp ||
        !seen_top_k || !seen_top_p || !seen_seed || !seen_max_tokens) {
        return -1;
    }
    config.model_path = model_path.c_str();
    config.backend = backend.c_str();
    std::string identity_error;
    if (!flashrt::providers::llama_cpp::checkpoint_identity(
            config.model_path, &model_identity, &identity_error)) {
        flashrt::providers::llama_cpp::set_runtime_open_error(identity_error);
        return -1;
    }
    config.model_identity = model_identity.c_str();

    frt_llama_cpp_engine_v1 engine{};
    const int rc = factory->create_llm(factory->self, &config, &engine);
    if (rc != 0) {
        const char* error = factory->last_error(factory->self);
        flashrt::providers::llama_cpp::set_runtime_open_error(
            error ? error : "LLM engine factory failed without an error");
        return rc;
    }
    if (engine.struct_size < FRT_LLAMA_CPP_ENGINE_V1_BASE_SIZE ||
        !engine.self || !engine.retain || !engine.release ||
        !engine.set_input || !engine.run_infer || !engine.get_output ||
        !engine.last_error) {
        flashrt::providers::llama_cpp::set_runtime_open_error(
            "factory returned an invalid LLM engine");
        return -1;
    }
    frt_model_runtime_v1* model = nullptr;
    const int create_rc =
        frt_llama_cpp_llm_runtime_create_with_engine(&config, &engine, &model);
    engine.release(engine.self);
    if (create_rc != 0) {
        flashrt::providers::llama_cpp::set_runtime_open_error(
            "failed to create LLM model runtime");
        return create_rc;
    }
    *out = model;
    flashrt::providers::llama_cpp::clear_runtime_open_error();
    return 0;
}
