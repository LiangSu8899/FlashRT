// Jetson-PI Pi0 engine: a thin mapping from the FlashRT frt_llama_cpp_engine_v1
// vtable onto the Jetson-PI jetson_pi_pi0 policy C API. The Pi0 infer glue
// (marker injection, tokenize, encode+decode, KV reset, state padding) lives
// inside jetson_pi_pi0; this file only translates frt_image_view/prompt/state
// into jetson_pi_pi0 inputs and the action chunk back out.
//
// Built only when FLASHRT_CPP_WITH_JETSON_PI is on. No GGML types are exposed
// through the FlashRT public header (c_api.h / jetson_pi_engine.h stay clean).

#include "flashrt/providers/llama_cpp/jetson_pi_engine.h"

#if defined(FLASHRT_CPP_WITH_JETSON_PI)

#include "flashrt/model_runtime.h"
#include "jetson_pi_pi0.h"
#include "jetson_pi_llm.h"
#include "jetson_pi_mllm.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Engine {
    jetson_pi_pi0 * pi0 = nullptr;

    // Config snapshot (owned strings).
    std::string model_path;
    std::string mmproj_path;
    std::string backend;
    uint32_t n_views       = 0;
    uint32_t image_height  = 0;
    uint32_t image_width   = 0;
    uint32_t action_steps  = 0;
    uint32_t action_dim    = 0;

    // Per-tick transient state, fed by set_input and consumed by run_infer.
    std::vector<uint8_t> rgb_scratch;        // packed RGB per view, concatenated
    std::vector<const uint8_t*> image_ptrs;  // n_views pointers into rgb_scratch
    std::string prompt;
    std::vector<float> state;                // input proprioception (any width)
    std::vector<float> actions_buf;          // output action chunk, size == action_steps*action_dim
    bool images_set = false;
    bool prompt_set = false;
    bool state_set  = false;

    std::string last_error;
    std::atomic<long> refs{1};

    void set_error(const std::string & m) { last_error = m; }
    void clear_error() { last_error.clear(); }
};

// Swizzle one frt_image_view into packed RGB (mtmd_bitmap's format). Honors
// stride_bytes and the declared pixel_format. Returns false on a bad view.
bool view_to_rgb(const frt_image_view & v, uint32_t expected_w,
                 uint32_t expected_h, std::vector<uint8_t> & out) {
    if (v.struct_size < sizeof(frt_image_view) || !v.data) return false;
    if (v.width <= 0 || v.height <= 0) return false;
    if (static_cast<uint32_t>(v.width) != expected_w ||
        static_cast<uint32_t>(v.height) != expected_h) {
        return false;
    }
    const int w = v.width;
    const int h = v.height;
    int ch_src;
    switch (v.pixel_format) {
        case FRT_RT_PIXEL_RGB8:
        case FRT_RT_PIXEL_BGR8:
            ch_src = 3; break;
        case FRT_RT_PIXEL_RGBA8:
        case FRT_RT_PIXEL_BGRA8:
            ch_src = 4; break;
        case FRT_RT_PIXEL_GRAY8:
            ch_src = 1; break;
        default:
            return false;  // reject unknown pixel formats (no silent fallback)
    }
    if (v.stride_bytes < 0) return false;
    const uint64_t row_bytes = static_cast<uint64_t>(w) * ch_src;
    const uint64_t stride = v.stride_bytes == 0
        ? row_bytes : static_cast<uint64_t>(v.stride_bytes);
    if (stride < row_bytes) return false;
    const uint64_t preceding_rows = static_cast<uint64_t>(h - 1);
    if (preceding_rows != 0 &&
        stride > (std::numeric_limits<uint64_t>::max() - row_bytes) /
                     preceding_rows) {
        return false;
    }
    const uint64_t required_bytes = preceding_rows * stride + row_bytes;
    const uint64_t rgb_bytes = static_cast<uint64_t>(w) * h * 3;
    if (v.bytes < required_bytes ||
        required_bytes > static_cast<uint64_t>(
            std::numeric_limits<ptrdiff_t>::max()) ||
        rgb_bytes > std::numeric_limits<size_t>::max()) {
        return false;
    }

    out.resize(static_cast<size_t>(rgb_bytes));
    const uint8_t * src = static_cast<const uint8_t*>(v.data);
    for (int y = 0; y < h; ++y) {
        const uint8_t * row = src + static_cast<ptrdiff_t>(y) * stride;
        for (int x = 0; x < w; ++x) {
            const uint8_t * px = row + x * ch_src;
            uint8_t * dst = &out[(static_cast<size_t>(y) * w + x) * 3];
            switch (v.pixel_format) {
                case FRT_RT_PIXEL_RGB8:
                    dst[0] = px[0]; dst[1] = px[1]; dst[2] = px[2];
                    break;
                case FRT_RT_PIXEL_BGR8:
                    dst[0] = px[2]; dst[1] = px[1]; dst[2] = px[0];
                    break;
                case FRT_RT_PIXEL_RGBA8:
                    dst[0] = px[0]; dst[1] = px[1]; dst[2] = px[2];
                    break;
                case FRT_RT_PIXEL_BGRA8:
                    dst[0] = px[2]; dst[1] = px[1]; dst[2] = px[0];
                    break;
                case FRT_RT_PIXEL_GRAY8:
                    dst[0] = dst[1] = dst[2] = px[0];
                    break;
                default:
                    return false;  // unreachable; ch_src gate above
            }
        }
    }
    return true;
}

// Open errors are reported through the factory's thread_local sink because
// the contract is the returned engine is zeroed on failure (no handle to
// query last_error from). This covers both jetson_pi_pi0_open failures and
// the engine's own create_pi0 validation (config/shape mismatch, OOM).
static thread_local std::string g_create_error;
static void set_create_error(const std::string & m) { g_create_error = m; }

int32_t pi0_status_to_engine(int32_t s) {
    switch (s) {
        case JETSON_PI_PI0_OK:               return 0;
        case JETSON_PI_PI0_ACTION_NOT_READY: return -7;
        case JETSON_PI_PI0_BUFFER_TOO_SMALL: return -5;
        case JETSON_PI_PI0_INVALID:          return -2;
        default:                             return -8;
    }
}

// ---- engine vtable ----------------------------------------------------------

void engine_retain(void * self) {
    static_cast<Engine*>(self)->refs.fetch_add(1, std::memory_order_relaxed);
}

void engine_release(void * self) {
    Engine * e = static_cast<Engine*>(self);
    if (e->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (e->pi0) jetson_pi_pi0_close(e->pi0);
        delete e;
    }
}

int engine_set_input(void * self, uint32_t port, const void * data,
                     uint64_t bytes, int /*stream*/) {
    Engine * e = static_cast<Engine*>(self);
    if (!e) return -1;
    e->clear_error();
    int32_t discard_status = jetson_pi_pi0_discard_context(e->pi0);
    if (discard_status != JETSON_PI_PI0_OK) {
        e->set_error(std::string("jetson_pi_pi0_discard_context failed: ") +
                     jetson_pi_pi0_last_error(e->pi0));
        return pi0_status_to_engine(discard_status);
    }
    // Any new input invalidates the previous tick's actions so get_output
    // cannot return stale data without a fresh run_infer.
    e->actions_buf.clear();
    switch (port) {
        case FRT_LLAMA_CPP_PI0_PORT_IMAGES: {
            e->images_set = false;
            e->image_ptrs.clear();
            e->rgb_scratch.clear();
            if (!data || bytes % sizeof(frt_image_view) != 0) {
                e->set_error("images payload must be frt_image_view[]");
                return -1;
            }
            const uint64_t n = bytes / sizeof(frt_image_view);
            if (n != e->n_views) {
                e->set_error("images view count != config.n_views");
                return -1;
            }
            const auto * views = static_cast<const frt_image_view*>(data);
            std::vector<uint8_t> packed;
            packed.reserve(static_cast<size_t>(e->n_views) * e->image_width *
                           e->image_height * 3);
            for (uint32_t i = 0; i < e->n_views; ++i) {
                std::vector<uint8_t> rgb;
                if (!view_to_rgb(views[i], e->image_width, e->image_height,
                                 rgb)) {
                    e->set_error("invalid frt_image_view at index " +
                                 std::to_string(i));
                    return -1;
                }
                packed.insert(packed.end(), rgb.begin(), rgb.end());
            }
            // Now that packed is fully grown (no more reallocs), compute the
            // per-view pointers into it. Storing them earlier would risk
            // dangling pointers if a later insert reallocated packed.
            e->rgb_scratch = std::move(packed);
            const size_t per_view =
                static_cast<size_t>(e->image_width) * e->image_height * 3;
            e->image_ptrs.resize(e->n_views);
            for (uint32_t i = 0; i < e->n_views; ++i) {
                e->image_ptrs[i] = e->rgb_scratch.data() + i * per_view;
            }
            e->images_set = true;
            return 0;
        }
        case FRT_LLAMA_CPP_PI0_PORT_PROMPT: {
            e->prompt_set = false;
            e->prompt.clear();
            if (!data || bytes == 0) {
                e->set_error("empty prompt");
                return -1;
            }
            e->prompt.assign(static_cast<const char*>(data), bytes);
            e->prompt_set = true;
            return 0;
        }
        case FRT_LLAMA_CPP_PI0_PORT_STATE: {
            e->state_set = false;
            e->state.clear();
            if (!data) {
                e->set_error("null state");
                return -1;
            }
            if (bytes == 0 || bytes % sizeof(float) != 0) {
                e->set_error("state bytes is not a nonzero multiple of sizeof(float)");
                return -1;
            }
            // State width is policy-specific and independent of action_dim:
            // PI0.5 discretizes up to 8 proprioception values into the prompt,
            // while legacy Pi0 consumes an action_dim tensor. Accept any
            // float count here; the backend enforces the per-model bound.
            const size_t n = bytes / sizeof(float);
            e->state.assign(static_cast<const float*>(data),
                            static_cast<const float*>(data) + n);
            e->state_set = true;
            return 0;
        }
        case FRT_LLAMA_CPP_PI0_PORT_ACTIONS:
        default:
            e->set_error("unknown/invalid pi0 input port");
            return -1;
    }
}

int engine_run_infer(void * self) {
    Engine * e = static_cast<Engine*>(self);
    if (!e) return -1;
    e->clear_error();
    if (!e->images_set || !e->prompt_set || !e->state_set) {
        e->set_error("infer requires images, prompt, and state to be set");
        return -1;
    }
    // jetson_pi_pi0_infer writes the action chunk into a buffer it sizes from
    // the model; allocate action_steps*action_dim floats.
    std::vector<float> actions(
        static_cast<size_t>(e->action_steps) * e->action_dim);
    size_t written = 0;
    int32_t s = jetson_pi_pi0_infer(e->pi0,
                                    e->image_ptrs.data(), e->image_ptrs.size(),
                                    e->prompt.data(), e->prompt.size(),
                                    e->state.data(), e->state.size(),
                                    actions.data(), actions.size(),
                                    &written);
    if (s != JETSON_PI_PI0_OK) {
        e->set_error(std::string("jetson_pi_pi0_infer failed: ") +
                     jetson_pi_pi0_last_error(e->pi0));
        return pi0_status_to_engine(s);
    }
    e->actions_buf.assign(actions.begin(), actions.end());
    return 0;
}

int engine_run_stage(void * self, uint32_t stage) {
    Engine * e = static_cast<Engine*>(self);
    if (!e) return -1;
    if (stage == FRT_LLAMA_CPP_PI0_STAGE_INDEX_INFER) {
        return engine_run_infer(self);
    }
    e->clear_error();
    if (!e->images_set || !e->prompt_set || !e->state_set) {
        e->set_error("Pi0 stage requires images, prompt, and state to be set");
        return -1;
    }
    if (stage == FRT_LLAMA_CPP_PI0_STAGE_INDEX_CONTEXT) {
        e->actions_buf.clear();
        int32_t s = jetson_pi_pi0_context(
            e->pi0, e->image_ptrs.data(), e->image_ptrs.size(),
            e->prompt.data(), e->prompt.size(), e->state.data(),
            e->state.size());
        if (s != JETSON_PI_PI0_OK) {
            e->set_error(std::string("jetson_pi_pi0_context failed: ") +
                         jetson_pi_pi0_last_error(e->pi0));
            return pi0_status_to_engine(s);
        }
        return 0;
    }
    if (stage == FRT_LLAMA_CPP_PI0_STAGE_INDEX_ACTION) {
        std::vector<float> actions(
            static_cast<size_t>(e->action_steps) * e->action_dim);
        size_t written = 0;
        int32_t s = jetson_pi_pi0_action(
            e->pi0, actions.data(), actions.size(), &written);
        if (s != JETSON_PI_PI0_OK) {
            e->set_error(std::string("jetson_pi_pi0_action failed: ") +
                         jetson_pi_pi0_last_error(e->pi0));
            return pi0_status_to_engine(s);
        }
        e->actions_buf.assign(actions.begin(), actions.end());
        return 0;
    }
    e->set_error("unknown Pi0 engine stage");
    return -1;
}

const char * engine_last_error(void * self) {
    Engine * e = static_cast<Engine*>(self);
    if (!e) return "null jetson_pi engine";
    if (e->last_error.empty()) return "ok";
    return e->last_error.c_str();
}

int engine_get_output(void * self, uint32_t port, void * out,
                      uint64_t capacity, uint64_t * written, int /*stream*/) {
    Engine * e = static_cast<Engine*>(self);
    if (!e) return -1;
    e->clear_error();
    if (port != FRT_LLAMA_CPP_PI0_PORT_ACTIONS) {
        e->set_error("pi0 only exports the actions port");
        return -1;
    }
    if (!out || !written) {
        e->set_error("get_output requires out and written");
        return -1;
    }
    const size_t need_elems =
        static_cast<size_t>(e->action_steps) * e->action_dim;
    const uint64_t need_bytes = need_elems * sizeof(float);
    *written = need_bytes;
    if (capacity < need_bytes) {
        e->set_error("action output buffer too small");
        return -5;
    }
    if (e->actions_buf.size() != need_elems) {
        e->set_error("actions not ready; run_infer did not complete");
        return -7;
    }
    std::memcpy(out, e->actions_buf.data(), need_bytes);
    return 0;
}

} // namespace

// ---------------------------------------------------------------------------
// LLM engine (generic GGUF text completion). Distinct IO shape from Pi0
// (1 prompt in -> 1 text out), so it has its own Engine struct + verbs, but
// reuses the frt_llama_cpp_engine_v1 vtable shape and the default factory.
// ---------------------------------------------------------------------------

namespace {

struct LlmEngine {
    jetson_pi_llm * llm = nullptr;

    std::string prompt;            // set_input(PROMPT) stash
    std::vector<int32_t> tokens;
    std::string text_buf;          // run_infer output, get_output(TEXT) source
    std::vector<float> logits_buf;
    int32_t next_token = 0;
    int32_t is_eog = 0;
    bool next_token_ready = false;
    bool prompt_set = false;

    std::string last_error;
    std::atomic<long> refs{1};

    void set_error(const std::string & m) { last_error = m; }
    void clear_error() { last_error.clear(); }
};

int llm_engine_run_infer(void * self);

void llm_engine_retain(void * self) {
    static_cast<LlmEngine*>(self)->refs.fetch_add(1, std::memory_order_relaxed);
}

void llm_engine_release(void * self) {
    LlmEngine * e = static_cast<LlmEngine*>(self);
    if (e->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (e->llm) jetson_pi_llm_close(e->llm);
        delete e;
    }
}

int llm_engine_set_input(void * self, uint32_t port, const void * data,
                         uint64_t bytes, int /*stream*/) {
    LlmEngine * e = static_cast<LlmEngine*>(self);
    if (!e) return -1;
    e->clear_error();
    if (port != FRT_LLAMA_CPP_LLM_PORT_PROMPT &&
        port != FRT_LLAMA_CPP_LLM_PORT_TOKENS) {
        e->set_error("unknown llm input port");
        return -1;
    }
    e->prompt_set = false;
    e->prompt.clear();
    e->tokens.clear();
    e->text_buf.clear();
    e->logits_buf.clear();
    e->next_token_ready = false;
    if (port == FRT_LLAMA_CPP_LLM_PORT_PROMPT) {
        if (!data || bytes == 0) {
            e->set_error("empty prompt");
            return -1;
        }
        e->prompt.assign(static_cast<const char*>(data), bytes);
        e->prompt_set = true;
    } else {
        if (!data || bytes == 0 || bytes % sizeof(int32_t) != 0) {
            e->set_error("tokens payload must be a non-empty int32 array");
            return -1;
        }
        const auto * begin = static_cast<const int32_t*>(data);
        e->tokens.assign(begin, begin + bytes / sizeof(int32_t));
        e->prompt_set = true;
    }
    return 0;
}

int llm_engine_refresh_logits(LlmEngine * e) {
    size_t n_logits = 0;
    int32_t s = jetson_pi_llm_get_logits(e->llm, nullptr, 0, &n_logits);
    if (s != JETSON_PI_LLM_BUFFER_TOO_SMALL || n_logits == 0) {
        e->set_error(std::string("jetson_pi_llm_get_logits size query failed: ") +
                     jetson_pi_llm_last_error(e->llm));
        return -8;
    }
    e->logits_buf.resize(n_logits);
    s = jetson_pi_llm_get_logits(e->llm, e->logits_buf.data(),
                                 e->logits_buf.size(), &n_logits);
    if (s != JETSON_PI_LLM_OK) {
        e->logits_buf.clear();
        e->set_error(std::string("jetson_pi_llm_get_logits failed: ") +
                     jetson_pi_llm_last_error(e->llm));
        return -8;
    }
    return 0;
}

int llm_engine_run_stage(void * self, uint32_t stage) {
    LlmEngine * e = static_cast<LlmEngine*>(self);
    if (!e) return -1;
    e->clear_error();
    if (stage == FRT_LLAMA_CPP_LLM_STAGE_INDEX_INFER) {
        return llm_engine_run_infer(self);
    }
    if (stage == FRT_LLAMA_CPP_LLM_STAGE_INDEX_RESET) {
        const int32_t s = jetson_pi_llm_reset(e->llm);
        if (s != JETSON_PI_LLM_OK) {
            e->set_error(std::string("jetson_pi_llm_reset failed: ") +
                         jetson_pi_llm_last_error(e->llm));
            return -8;
        }
        e->text_buf.clear();
        e->logits_buf.clear();
        e->next_token_ready = false;
        return 0;
    }
    if (stage == FRT_LLAMA_CPP_LLM_STAGE_INDEX_PREFILL) {
        if (!e->prompt_set) {
            e->set_error("prefill requires prompt to be set");
            return -1;
        }
        const int32_t s = e->tokens.empty()
            ? jetson_pi_llm_prefill(e->llm, e->prompt.data(), e->prompt.size())
            : jetson_pi_llm_prefill_tokens(e->llm, e->tokens.data(),
                                           e->tokens.size());
        if (s != JETSON_PI_LLM_OK) {
            e->set_error(std::string("jetson_pi_llm_prefill failed: ") +
                         jetson_pi_llm_last_error(e->llm));
            return -8;
        }
        e->text_buf.clear();
        e->next_token_ready = false;
        return llm_engine_refresh_logits(e);
    }
    if (stage == FRT_LLAMA_CPP_LLM_STAGE_INDEX_DECODE) {
        e->next_token_ready = false;
        e->logits_buf.clear();
        int32_t is_eog = 0;
        const int32_t s = jetson_pi_llm_decode_step(
            e->llm, &e->next_token, &is_eog);
        if (s != JETSON_PI_LLM_OK) {
            e->set_error(std::string("jetson_pi_llm_decode_step failed: ") +
                         jetson_pi_llm_last_error(e->llm));
            return -8;
        }
        e->next_token_ready = true;
        e->is_eog = is_eog;
        e->logits_buf.clear();
        if (!is_eog) {
            size_t piece_size = 0;
            int32_t piece_status = jetson_pi_llm_token_to_piece(
                e->llm, e->next_token, nullptr, 0, &piece_size);
            if (piece_status != JETSON_PI_LLM_BUFFER_TOO_SMALL) {
                e->set_error(std::string("token piece size query failed: ") +
                             jetson_pi_llm_last_error(e->llm));
                return -8;
            }
            if (piece_size == 0) return llm_engine_refresh_logits(e);
            const size_t old_size = e->text_buf.size();
            e->text_buf.resize(old_size + piece_size);
            piece_status = jetson_pi_llm_token_to_piece(
                e->llm, e->next_token, e->text_buf.data() + old_size,
                piece_size, &piece_size);
            if (piece_status != JETSON_PI_LLM_OK) {
                e->text_buf.resize(old_size);
                e->set_error(std::string("token_to_piece failed: ") +
                             jetson_pi_llm_last_error(e->llm));
                return -8;
            }
            return llm_engine_refresh_logits(e);
        }
        return 0;
    }
    e->set_error("unknown llm engine stage");
    return -1;
}

int llm_engine_run_infer(void * self) {
    LlmEngine * e = static_cast<LlmEngine*>(self);
    if (!e) return -1;
    e->clear_error();
    if (!e->prompt_set) {
        e->set_error("infer requires prompt to be set");
        return -1;
    }
    size_t cap = 0;
    int32_t s = jetson_pi_llm_generate(e->llm, e->prompt.data(),
                                       e->prompt.size(), nullptr, 0, &cap);
    if (s != JETSON_PI_LLM_BUFFER_TOO_SMALL || cap == 0) {
        e->set_error(std::string("jetson_pi_llm_generate size query failed: ") +
                     jetson_pi_llm_last_error(e->llm));
        return -8;
    }
    e->text_buf.assign(cap, '\0');
    size_t written = 0;
    s = jetson_pi_llm_generate(e->llm, e->prompt.data(), e->prompt.size(),
                               e->text_buf.data(), cap, &written);
    if (s != JETSON_PI_LLM_OK) {
        e->set_error(std::string("jetson_pi_llm_generate failed: ") +
                     jetson_pi_llm_last_error(e->llm));
        e->text_buf.clear();
        return -8;
    }
    e->text_buf.resize(written);
    e->logits_buf.clear();
    e->next_token_ready = false;
    return 0;
}

int llm_engine_get_output(void * self, uint32_t port, void * out,
                          uint64_t capacity, uint64_t * written,
                          int /*stream*/) {
    LlmEngine * e = static_cast<LlmEngine*>(self);
    if (!e) return -1;
    e->clear_error();
    if (!written) {
        e->set_error("get_output requires written");
        return -1;
    }
    if (port == FRT_LLAMA_CPP_LLM_PORT_NEXT_TOKEN) {
        *written = sizeof(e->next_token);
        if (!e->next_token_ready) {
            e->set_error("next token not ready");
            return -7;
        }
        if (!out || capacity < sizeof(e->next_token)) {
            e->set_error("next token output buffer too small");
            return -5;
        }
        std::memcpy(out, &e->next_token, sizeof(e->next_token));
        return 0;
    }
    if (port == FRT_LLAMA_CPP_LLM_PORT_IS_EOG) {
        *written = sizeof(e->is_eog);
        if (!e->next_token_ready) {
            e->set_error("is_eog not ready");
            return -7;
        }
        if (!out || capacity < sizeof(e->is_eog)) {
            e->set_error("is_eog output buffer too small");
            return -5;
        }
        std::memcpy(out, &e->is_eog, sizeof(e->is_eog));
        return 0;
    }
    if (port == FRT_LLAMA_CPP_LLM_PORT_LOGITS) {
        const uint64_t need = e->logits_buf.size() * sizeof(float);
        *written = need;
        if (need == 0) {
            e->set_error("logits not ready");
            return -7;
        }
        if (!out || capacity < need) {
            e->set_error("logits output buffer too small");
            return -5;
        }
        std::memcpy(out, e->logits_buf.data(), need);
        return 0;
    }
    if (port != FRT_LLAMA_CPP_LLM_PORT_TEXT) {
        e->set_error("unknown llm output port");
        return -1;
    }
    const uint64_t need = e->text_buf.size();
    *written = need;
    // Size-query (out=NULL) or too-small buffer: report need, return nonzero.
    if (!out || capacity < need) {
        if (need == 0) {
            e->set_error("text not ready; run_infer did not complete");
            return -7;
        }
        e->set_error("text output buffer too small");
        return -5;
    }
    std::memcpy(out, e->text_buf.data(), need);
    return 0;
}

const char * llm_engine_last_error(void * self) {
    LlmEngine * e = static_cast<LlmEngine*>(self);
    if (!e) return "null jetson_pi llm engine";
    if (e->last_error.empty()) return "ok";
    return e->last_error.c_str();
}

// ---------------------------------------------------------------------------
// MLLM engine (multimodal LLM: images + prompt -> text). IO shape = Pi0's
// image input + LLM's text output. Reuses view_to_rgb for images and the
// LLM generate contract for text.
// ---------------------------------------------------------------------------

struct MllmEngine {
    jetson_pi_mllm * mllm = nullptr;

    // Per-tick transient state.
    std::vector<uint8_t> rgb_scratch;
    std::vector<const uint8_t*> image_ptrs;
    uint32_t n_images = 0;
    uint32_t image_height = 0;
    uint32_t image_width = 0;
    std::string prompt;
    std::string text_buf;
    std::vector<float> logits_buf;
    int32_t next_token = 0;
    int32_t is_eog = 0;
    bool next_token_ready = false;
    bool images_set = false;
    bool prompt_set = false;

    std::string last_error;
    std::atomic<long> refs{1};

    void set_error(const std::string & m) { last_error = m; }
    void clear_error() { last_error.clear(); }
};

int mllm_engine_run_infer(void * self);

void mllm_engine_retain(void * self) {
    static_cast<MllmEngine*>(self)->refs.fetch_add(1, std::memory_order_relaxed);
}

void mllm_engine_release(void * self) {
    MllmEngine * e = static_cast<MllmEngine*>(self);
    if (e->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (e->mllm) jetson_pi_mllm_close(e->mllm);
        delete e;
    }
}

int mllm_engine_set_input(void * self, uint32_t port, const void * data,
                          uint64_t bytes, int /*stream*/) {
    MllmEngine * e = static_cast<MllmEngine*>(self);
    if (!e) return -1;
    e->clear_error();
    switch (port) {
        case FRT_LLAMA_CPP_MLLM_PORT_IMAGES: {
            e->images_set = false;
            e->image_ptrs.clear();
            e->rgb_scratch.clear();
            e->n_images = 0;
            e->image_height = 0;
            e->image_width = 0;
            e->text_buf.clear();
            e->logits_buf.clear();
            e->next_token_ready = false;
            if (bytes % sizeof(frt_image_view) != 0 ||
                (bytes != 0 && !data)) {
                e->set_error("images payload must be frt_image_view[]");
                return -1;
            }
            const uint64_t n = bytes / sizeof(frt_image_view);
            if (n == 0) {
                // n_images == 0 means text-only; allow but record zero.
                e->images_set = true;
                return 0;
            }
            if (n > std::numeric_limits<uint32_t>::max()) {
                e->set_error("images view count exceeds uint32_t");
                return -1;
            }
            const auto * views = static_cast<const frt_image_view*>(data);
            // All images must share the first view's H/W (jetson_pi_mllm
            // takes a single image_height/image_width for all images).
            if (views[0].width <= 0 || views[0].height <= 0) {
                e->set_error("invalid frt_image_view at index 0");
                return -1;
            }
            const uint32_t w = static_cast<uint32_t>(views[0].width);
            const uint32_t h = static_cast<uint32_t>(views[0].height);
            std::vector<uint8_t> packed;
            for (uint64_t i = 0; i < n; ++i) {
                std::vector<uint8_t> rgb;
                if (!view_to_rgb(views[i], w, h, rgb)) {
                    e->set_error("invalid frt_image_view at index " +
                                 std::to_string(i));
                    return -1;
                }
                packed.insert(packed.end(), rgb.begin(), rgb.end());
            }
            e->rgb_scratch = std::move(packed);
            const size_t per_view = static_cast<size_t>(w) * h * 3;
            e->image_ptrs.resize(n);
            for (uint64_t i = 0; i < n; ++i) {
                e->image_ptrs[i] = e->rgb_scratch.data() + i * per_view;
            }
            e->n_images = static_cast<uint32_t>(n);
            e->image_height = h;
            e->image_width = w;
            e->images_set = true;
            return 0;
        }
        case FRT_LLAMA_CPP_MLLM_PORT_PROMPT: {
            e->prompt_set = false;
            e->prompt.clear();
            e->text_buf.clear();
            e->logits_buf.clear();
            e->next_token_ready = false;
            if (!data || bytes == 0) {
                e->set_error("empty prompt");
                return -1;
            }
            e->prompt.assign(static_cast<const char*>(data), bytes);
            e->prompt_set = true;
            return 0;
        }
        case FRT_LLAMA_CPP_MLLM_PORT_TEXT:
        default:
            e->set_error("unknown/invalid mllm input port");
            return -1;
    }
}

int mllm_engine_refresh_logits(MllmEngine * e) {
    size_t n_logits = 0;
    int32_t s = jetson_pi_mllm_get_logits(e->mllm, nullptr, 0, &n_logits);
    if (s != JETSON_PI_MLLM_BUFFER_TOO_SMALL || n_logits == 0) {
        e->set_error(std::string("jetson_pi_mllm_get_logits size query failed: ") +
                     jetson_pi_mllm_last_error(e->mllm));
        return -8;
    }
    e->logits_buf.resize(n_logits);
    s = jetson_pi_mllm_get_logits(e->mllm, e->logits_buf.data(),
                                  e->logits_buf.size(), &n_logits);
    if (s != JETSON_PI_MLLM_OK) {
        e->logits_buf.clear();
        e->set_error(std::string("jetson_pi_mllm_get_logits failed: ") +
                     jetson_pi_mllm_last_error(e->mllm));
        return -8;
    }
    return 0;
}

int mllm_engine_run_stage(void * self, uint32_t stage) {
    MllmEngine * e = static_cast<MllmEngine*>(self);
    if (!e) return -1;
    e->clear_error();
    if (stage == FRT_LLAMA_CPP_MLLM_STAGE_INDEX_INFER) {
        return mllm_engine_run_infer(self);
    }
    if (stage == FRT_LLAMA_CPP_MLLM_STAGE_INDEX_RESET) {
        const int32_t s = jetson_pi_mllm_reset(e->mllm);
        if (s != JETSON_PI_MLLM_OK) {
            e->set_error(std::string("jetson_pi_mllm_reset failed: ") +
                         jetson_pi_mllm_last_error(e->mllm));
            return -8;
        }
        e->text_buf.clear();
        e->logits_buf.clear();
        e->next_token_ready = false;
        return 0;
    }
    if (stage == FRT_LLAMA_CPP_MLLM_STAGE_INDEX_PREFILL) {
        if (!e->images_set || !e->prompt_set) {
            e->set_error("prefill requires images and prompt to be set");
            return -1;
        }
        const int32_t s = jetson_pi_mllm_prefill(
            e->mllm, e->image_ptrs.data(), e->n_images,
            e->image_height, e->image_width,
            e->prompt.data(), e->prompt.size());
        if (s != JETSON_PI_MLLM_OK) {
            e->set_error(std::string("jetson_pi_mllm_prefill failed: ") +
                         jetson_pi_mllm_last_error(e->mllm));
            return -8;
        }
        e->text_buf.clear();
        e->next_token_ready = false;
        return mllm_engine_refresh_logits(e);
    }
    if (stage == FRT_LLAMA_CPP_MLLM_STAGE_INDEX_DECODE) {
        e->next_token_ready = false;
        e->logits_buf.clear();
        int32_t is_eog = 0;
        const int32_t s = jetson_pi_mllm_decode_step(
            e->mllm, &e->next_token, &is_eog);
        if (s != JETSON_PI_MLLM_OK) {
            e->set_error(std::string("jetson_pi_mllm_decode_step failed: ") +
                         jetson_pi_mllm_last_error(e->mllm));
            return -8;
        }
        e->next_token_ready = true;
        e->is_eog = is_eog;
        e->logits_buf.clear();
        if (is_eog) return 0;
        size_t piece_size = 0;
        int32_t piece_status = jetson_pi_mllm_token_to_piece(
            e->mllm, e->next_token, nullptr, 0, &piece_size);
        if (piece_status != JETSON_PI_MLLM_BUFFER_TOO_SMALL) {
            e->set_error(std::string("token piece size query failed: ") +
                         jetson_pi_mllm_last_error(e->mllm));
            return -8;
        }
        if (piece_size > 0) {
            const size_t old_size = e->text_buf.size();
            e->text_buf.resize(old_size + piece_size);
            piece_status = jetson_pi_mllm_token_to_piece(
                e->mllm, e->next_token, e->text_buf.data() + old_size,
                piece_size, &piece_size);
            if (piece_status != JETSON_PI_MLLM_OK) {
                e->text_buf.resize(old_size);
                e->set_error(std::string("token_to_piece failed: ") +
                             jetson_pi_mllm_last_error(e->mllm));
                return -8;
            }
        }
        return mllm_engine_refresh_logits(e);
    }
    e->set_error("unknown mllm engine stage");
    return -1;
}

int mllm_engine_run_infer(void * self) {
    MllmEngine * e = static_cast<MllmEngine*>(self);
    if (!e) return -1;
    e->clear_error();
    if (!e->images_set || !e->prompt_set) {
        e->set_error("infer requires images and prompt to be set");
        return -1;
    }
    size_t cap = 0;
    int32_t s = jetson_pi_mllm_infer(
        e->mllm, e->image_ptrs.data(), e->n_images,
        e->image_height, e->image_width,
        e->prompt.data(), e->prompt.size(), nullptr, 0, &cap);
    if (s != JETSON_PI_MLLM_BUFFER_TOO_SMALL || cap == 0) {
        e->set_error(std::string("jetson_pi_mllm_infer size query failed: ") +
                     jetson_pi_mllm_last_error(e->mllm));
        return -8;
    }
    e->text_buf.assign(cap, '\0');
    size_t written = 0;
    s = jetson_pi_mllm_infer(e->mllm,
                             e->image_ptrs.data(), e->n_images,
                             e->image_height, e->image_width,
                             e->prompt.data(), e->prompt.size(),
                             e->text_buf.data(), cap, &written);
    if (s != JETSON_PI_MLLM_OK) {
        e->set_error(std::string("jetson_pi_mllm_infer failed: ") +
                     jetson_pi_mllm_last_error(e->mllm));
        e->text_buf.clear();
        return -8;
    }
    e->text_buf.resize(written);
    e->logits_buf.clear();
    e->next_token_ready = false;
    return 0;
}

int mllm_engine_get_output(void * self, uint32_t port, void * out,
                           uint64_t capacity, uint64_t * written,
                           int /*stream*/) {
    MllmEngine * e = static_cast<MllmEngine*>(self);
    if (!e) return -1;
    e->clear_error();
    if (!written) {
        e->set_error("get_output requires written");
        return -1;
    }
    if (port == FRT_LLAMA_CPP_MLLM_PORT_NEXT_TOKEN ||
        port == FRT_LLAMA_CPP_MLLM_PORT_IS_EOG) {
        if (!e->next_token_ready) {
            e->set_error("decode output not ready");
            return -7;
        }
        const uint64_t need = sizeof(int32_t);
        *written = need;
        if (!out || capacity < need) {
            e->set_error("decode scalar output buffer too small");
            return -5;
        }
        const int32_t value = port == FRT_LLAMA_CPP_MLLM_PORT_NEXT_TOKEN
            ? e->next_token : e->is_eog;
        std::memcpy(out, &value, need);
        return 0;
    }
    if (port == FRT_LLAMA_CPP_MLLM_PORT_LOGITS) {
        const uint64_t need = e->logits_buf.size() * sizeof(float);
        *written = need;
        if (need == 0) {
            e->set_error("logits not ready");
            return -7;
        }
        if (!out || capacity < need) {
            e->set_error("logits output buffer too small");
            return -5;
        }
        std::memcpy(out, e->logits_buf.data(), need);
        return 0;
    }
    if (port != FRT_LLAMA_CPP_MLLM_PORT_TEXT) {
        e->set_error("unknown mllm output port");
        return -1;
    }
    const uint64_t need = e->text_buf.size();
    *written = need;
    if (!out || capacity < need) {
        if (need == 0) {
            e->set_error("text not ready; run_infer did not complete");
            return -7;
        }
        e->set_error("text output buffer too small");
        return -5;
    }
    std::memcpy(out, e->text_buf.data(), need);
    return 0;
}

const char * mllm_engine_last_error(void * self) {
    MllmEngine * e = static_cast<MllmEngine*>(self);
    if (!e) return "null jetson_pi mllm engine";
    if (e->last_error.empty()) return "ok";
    return e->last_error.c_str();
}

} // namespace

extern "C" const frt_llama_cpp_engine_factory_v1*
frt_llama_cpp_default_engine_factory(void) {
    static const frt_llama_cpp_engine_factory_v1 factory = []{
        frt_llama_cpp_engine_factory_v1 f{};
        f.struct_size = sizeof(f);
        f.self = nullptr;
        f.create_pi0 = [](void * /*self*/,
                          const frt_llama_cpp_pi0_config * config,
                          frt_llama_cpp_engine_v1 * out) -> int {
            g_create_error.clear();
            if (!config || config->struct_size < FRT_LLAMA_CPP_PI0_CONFIG_BASE_SIZE || !out) {
                set_create_error("invalid create_pi0 arguments");
                return -1;
            }
            if (!config->model_path || !config->model_path[0] ||
                !config->mmproj_path || !config->mmproj_path[0] ||
                !config->backend || !config->backend[0] ||
                !config->n_views || config->n_views > 3 ||
                !config->image_height ||
                !config->image_width || !config->image_channels ||
                !config->action_steps || !config->action_dim) {
                set_create_error("create_pi0 config has invalid fields");
                return -1;
            }
#if defined(JETSON_PI_PI0_HAS_CAPABILITIES_API) && \
    JETSON_PI_PI0_HAS_CAPABILITIES_API
            const uint32_t pi0_capabilities = jetson_pi_pi0_capabilities();
            if ((pi0_capabilities &
                 JETSON_PI_PI0_CAP_REAL_CONTEXT_ACTION) == 0) {
                set_create_error(
                    "loaded Jetson-PI runtime does not provide real Pi0 context/action split");
                return -1;
            }
#endif
            jetson_pi_pi0_config jc{};
            jc.struct_size   = sizeof(jc);
            jc.model_path    = config->model_path;
            jc.mmproj_path   = config->mmproj_path;
            jc.backend       = config->backend;
            jc.n_views       = config->n_views;
            jc.image_height  = config->image_height;
            jc.image_width   = config->image_width;
            jc.n_threads     = 0;  // let jetson_pi_pi0 pick hardware_concurrency

            jetson_pi_pi0 * pi0 = nullptr;
            int32_t s = jetson_pi_pi0_open(&jc, &pi0);
            if (s != JETSON_PI_PI0_OK || !pi0) {
                set_create_error(std::string("jetson_pi_pi0_open failed: ") +
                                 jetson_pi_pi0_open_error());
                return -1;
            }

            // Validate the model's real action shape against the config.
            uint32_t steps = 0, dim = 0;
            if (jetson_pi_pi0_action_shape(pi0, &steps, &dim) !=
                JETSON_PI_PI0_OK) {
                set_create_error("jetson_pi_pi0_action_shape failed");
                jetson_pi_pi0_close(pi0);
                return -1;
            }
            if (steps != config->action_steps || dim != config->action_dim) {
                set_create_error(
                    "action shape mismatch: config=" +
                    std::to_string(config->action_steps) + "x" +
                    std::to_string(config->action_dim) + " model=" +
                    std::to_string(steps) + "x" + std::to_string(dim));
                jetson_pi_pi0_close(pi0);
                return -1;
            }

            Engine * e = new (std::nothrow) Engine();
            if (!e) {
                set_create_error("engine allocation failed");
                jetson_pi_pi0_close(pi0);
                return -5;
            }
            e->pi0 = pi0;
            e->model_path    = config->model_path;
            e->mmproj_path   = config->mmproj_path;
            e->backend       = config->backend;
            e->n_views       = config->n_views;
            e->image_height  = config->image_height;
            e->image_width   = config->image_width;
            e->action_steps  = config->action_steps;
            e->action_dim    = config->action_dim;

            out->struct_size = sizeof(*out);
            out->reserved    = 0;
#if defined(JETSON_PI_PI0_HAS_REAL_CONTEXT_ACTION) && \
    JETSON_PI_PI0_HAS_REAL_CONTEXT_ACTION && \
    defined(JETSON_PI_PI0_HAS_CAPABILITIES_API) && \
    JETSON_PI_PI0_HAS_CAPABILITIES_API
            out->reserved |=
                FRT_LLAMA_CPP_ENGINE_CAP_PI0_REAL_CONTEXT_ACTION;
#endif
            out->self        = e;
            out->retain      = engine_retain;
            out->release     = engine_release;
            out->set_input   = engine_set_input;
            out->run_infer   = engine_run_infer;
            out->get_output  = engine_get_output;
            out->last_error  = engine_last_error;
            out->run_stage   = engine_run_stage;
            return 0;
        };
        f.create_llm = [](void * /*self*/,
                          const frt_llama_cpp_llm_config * config,
                          frt_llama_cpp_engine_v1 * out) -> int {
            g_create_error.clear();
            if (!config || config->struct_size < FRT_LLAMA_CPP_LLM_CONFIG_BASE_SIZE || !out) {
                set_create_error("invalid create_llm arguments");
                return -1;
            }
            if (!config->model_path || !config->model_path[0] ||
                !config->backend || !config->backend[0]) {
                set_create_error("create_llm config has empty/zero fields");
                return -1;
            }
            jetson_pi_llm_config jc{};
            jc.struct_size = sizeof(jc);
            jc.model_path  = config->model_path;
            jc.backend     = config->backend;
            jc.n_ctx       = config->n_ctx;
            jc.n_threads   = config->n_threads;
            jc.temp        = config->temp;
            jc.top_k       = config->top_k;
            jc.top_p       = config->top_p;
            jc.seed        = config->seed;
            jc.max_tokens  = config->max_tokens;

            jetson_pi_llm * llm = nullptr;
            int32_t s = jetson_pi_llm_open(&jc, &llm);
            if (s != JETSON_PI_LLM_OK || !llm) {
                set_create_error(std::string("jetson_pi_llm_open failed: ") +
                                 jetson_pi_llm_open_error());
                return -1;
            }

            LlmEngine * e = new (std::nothrow) LlmEngine();
            if (!e) {
                set_create_error("llm engine allocation failed");
                jetson_pi_llm_close(llm);
                return -5;
            }
            e->llm = llm;

            out->struct_size = sizeof(*out);
            out->reserved    = 0;
            out->self        = e;
            out->retain      = llm_engine_retain;
            out->release     = llm_engine_release;
            out->set_input   = llm_engine_set_input;
            out->run_infer   = llm_engine_run_infer;
            out->get_output  = llm_engine_get_output;
            out->last_error  = llm_engine_last_error;
            out->run_stage   = llm_engine_run_stage;
            return 0;
        };
        f.create_mllm = [](void * /*self*/,
                           const frt_llama_cpp_mllm_config * config,
                           frt_llama_cpp_engine_v1 * out) -> int {
            g_create_error.clear();
            if (!config || config->struct_size < FRT_LLAMA_CPP_MLLM_CONFIG_BASE_SIZE || !out) {
                set_create_error("invalid create_mllm arguments");
                return -1;
            }
            if (!config->model_path || !config->model_path[0] ||
                !config->mmproj_path || !config->mmproj_path[0] ||
                !config->backend || !config->backend[0]) {
                set_create_error("create_mllm config has empty/zero fields");
                return -1;
            }
            jetson_pi_mllm_config jc{};
            jc.struct_size  = sizeof(jc);
            jc.model_path   = config->model_path;
            jc.mmproj_path  = config->mmproj_path;
            jc.backend      = config->backend;
            jc.n_ctx        = config->n_ctx;
            jc.n_threads    = config->n_threads;
            jc.temp         = config->temp;
            jc.top_k        = config->top_k;
            jc.top_p        = config->top_p;
            jc.seed         = config->seed;
            jc.max_tokens   = config->max_tokens;

            jetson_pi_mllm * mllm = nullptr;
            int32_t s = jetson_pi_mllm_open(&jc, &mllm);
            if (s != JETSON_PI_MLLM_OK || !mllm) {
                set_create_error(std::string("jetson_pi_mllm_open failed: ") +
                                 jetson_pi_mllm_open_error());
                return -1;
            }

            MllmEngine * e = new (std::nothrow) MllmEngine();
            if (!e) {
                set_create_error("mllm engine allocation failed");
                jetson_pi_mllm_close(mllm);
                return -5;
            }
            e->mllm = mllm;

            out->struct_size = sizeof(*out);
            out->reserved    = 0;
            out->self        = e;
            out->retain      = mllm_engine_retain;
            out->release     = mllm_engine_release;
            out->set_input   = mllm_engine_set_input;
            out->run_infer   = mllm_engine_run_infer;
            out->get_output  = mllm_engine_get_output;
            out->last_error  = mllm_engine_last_error;
            out->run_stage   = mllm_engine_run_stage;
            return 0;
        };
        f.last_error = [](void * /*self*/) -> const char * {
            return g_create_error.c_str();
        };
        return f;
    }();
    return &factory;
}

#else  /* !FLASHRT_CPP_WITH_JETSON_PI */

extern "C" const frt_llama_cpp_engine_factory_v1*
frt_llama_cpp_default_engine_factory(void) {
    return nullptr;
}

#endif  /* FLASHRT_CPP_WITH_JETSON_PI */
