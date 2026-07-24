// Numerical-parity test: FlashRT's Jetson-PI LLM provider (via
// frt_model_runtime_v1) vs a DIRECT jetson_pi_llm call, same prompt / sampling
// / backend. Proves the FlashRT port/stage/verb plumbing does not perturb the
// generated text relative to the native narrow C API.
//
// Closes jetsonpi迁移.txt §14 (parity vs native). Skips (returns 0) when
// FLASHRT_LLM_MODEL is unset.
//
// Determinism foundation: greedy sampling (temp<=0) builds a
// llama_sampler_init_greedy()-only chain (argmax, no RNG) — fully
// deterministic regardless of seed. KV is cleared per generate. So both paths
// must produce the EXACT same token sequence / text. (temp>0 uses an RNG
// sampler and is not deterministic across handles; parity uses greedy only.)
//
// Env:
//   FLASHRT_LLM_MODEL    path to a GGUF LLM
//   FLASHRT_LLM_BACKEND  (optional) "cpu" (default) or "cuda".
//
// Note: both paths load the model independently, so a CPU run pays two full
// loads (several minutes for a 0.6B model); CUDA is the practical default
// when a GPU is available.

#include "flashrt/providers/llama_cpp/c_api.h"
#include "llama_cpp_generic_plan.h"
#include "flashrt/providers/llama_cpp/jetson_pi_engine.h"
#include "jetson_pi_llm.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", (msg)); g_fail = 1; } \
    else { std::printf("ok  : %s\n", (msg)); } \
} while (0)

int main() {
    const char * model_env = std::getenv("FLASHRT_LLM_MODEL");
    bool model_ok = false;
    if (model_env && *model_env) {
        std::ifstream f(model_env, std::ios::binary);
        model_ok = f.good();
    }
    if (!model_ok) {
        std::printf("SKIP - FLASHRT_LLM_MODEL not set or file missing\n");
        return 0;
    }
    const char * be_env = std::getenv("FLASHRT_LLM_BACKEND");
    const std::string backend = (be_env && be_env[0]) ? std::string(be_env)
                                                      : std::string("cpu");

    // Shared prompt + sampling. Greedy (temp=0) -> deterministic argmax.
    const char * prompt = "What is 2 plus 2? The answer is";
    const uint32_t n_ctx = 2048;
    const int32_t  n_threads = 0;
    const float    temp = 0.0f;
    const int32_t  top_k = 0;
    const float    top_p = 0.0f;
    const uint32_t seed = 1;
    const uint32_t max_tokens = 64;

    // ---- PATH A: FlashRT frt_model_runtime_v1 wrapper ----------------------
    std::string text_flashrt;
    std::vector<float> logits_flashrt;
    int32_t token_flashrt = 0;
    {
        const frt_llama_cpp_engine_factory_v1 * factory =
            frt_llama_cpp_default_engine_factory();
        CHECK(factory != nullptr && factory->create_llm != nullptr,
              "default engine factory exposes create_llm");

        std::string json =
            std::string("{") +
            "\"model_family\":\"llm\","
            "\"model_path\":\"" + model_env + "\","
            "\"backend\":\"" + backend + "\","
            "\"n_ctx\":" + std::to_string(n_ctx) + ","
            "\"n_threads\":" + std::to_string(n_threads) + ","
            "\"temp\":" + std::to_string(temp) + ","
            "\"top_k\":" + std::to_string(top_k) + ","
            "\"top_p\":" + std::to_string(top_p) + ","
            "\"seed\":" + std::to_string(seed) + ","
            "\"max_tokens\":" + std::to_string(max_tokens) + "}";
        frt_model_runtime_v1 * model = nullptr;
        int rc = frt_llama_cpp_llm_runtime_open_with_engine_factory(
            json.c_str(), factory, &model);
        if (rc != 0 || !model) {
            std::printf("FAIL: open FlashRT LLM runtime (rc=%d): %s\n", rc,
                        factory->last_error(factory->self));
            g_fail = 1;
        } else {
            CHECK(true, "open FlashRT LLM runtime from JSON");
            CHECK(model->verbs.set_input(model->self,
                                            FRT_LLAMA_CPP_LLM_PORT_PROMPT,
                                            prompt, std::strlen(prompt), -1) == 0,
                  "FlashRT set_input prompt");
            CHECK(model->verbs.step(model->self) == 0,
                  "FlashRT run_stage infer");
            uint64_t written = 0;
            rc = model->verbs.get_output(
                model->self, FRT_LLAMA_CPP_LLM_PORT_TEXT,
                nullptr, 0, &written, -1);
            CHECK(rc == -5 && written > 0,
                  "FlashRT query generated text size");
            text_flashrt.assign(written, '\0');
            written = 0;
            rc = model->verbs.get_output(
                model->self, FRT_LLAMA_CPP_LLM_PORT_TEXT, &text_flashrt[0],
                text_flashrt.size(), &written, -1);
            CHECK(rc == 0 && written > 0, "FlashRT get_output text non-empty");
            text_flashrt.resize(written);
            std::printf("    FlashRT text: %s\n", text_flashrt.c_str());
            CHECK(llama_cpp_run_generic_stage(model, "reset") == 0 &&
                  llama_cpp_run_generic_stage(model, "prefill") == 0,
                  "FlashRT staged prefill");
            uint64_t logits_bytes = 0;
            CHECK(model->verbs.get_output(
                      model->self, FRT_LLAMA_CPP_LLM_PORT_LOGITS,
                      nullptr, 0, &logits_bytes, -1) != 0 && logits_bytes > 0,
                  "FlashRT query logits size");
            logits_flashrt.resize(logits_bytes / sizeof(float));
            CHECK(model->verbs.get_output(
                      model->self, FRT_LLAMA_CPP_LLM_PORT_LOGITS,
                      logits_flashrt.data(), logits_bytes, &logits_bytes,
                      -1) == 0 &&
                  llama_cpp_run_generic_stage(model, "decode") == 0,
                  "FlashRT read logits and decode first token");
            uint64_t token_bytes = 0;
            CHECK(model->verbs.get_output(
                      model->self, FRT_LLAMA_CPP_LLM_PORT_NEXT_TOKEN,
                      &token_flashrt, sizeof(token_flashrt), &token_bytes,
                      -1) == 0,
                  "FlashRT read first token");
            model->release(model->owner);
        }
    }

    // ---- PATH B: direct jetson_pi_llm call ---------------------------------
    std::string text_native;
    std::vector<float> logits_native;
    int32_t token_native = 0;
    {
        jetson_pi_llm_config jc{};
        jc.struct_size = sizeof(jc);
        jc.model_path  = model_env;
        jc.backend     = backend.c_str();
        jc.n_ctx       = n_ctx;
        jc.n_threads   = n_threads;
        jc.temp        = temp;
        jc.top_k       = top_k;
        jc.top_p       = top_p;
        jc.seed        = seed;
        jc.max_tokens  = max_tokens;

        jetson_pi_llm * llm = nullptr;
        int32_t s = jetson_pi_llm_open(&jc, &llm);
        if (s != JETSON_PI_LLM_OK || !llm) {
            std::printf("FAIL: jetson_pi_llm_open (rc=%d): %s\n", s,
                        jetson_pi_llm_open_error());
            g_fail = 1;
        } else {
            CHECK(true, "direct jetson_pi_llm_open");
            size_t written = 0;
            s = jetson_pi_llm_generate(llm, prompt, std::strlen(prompt),
                                       nullptr, 0, &written);
            CHECK(s == JETSON_PI_LLM_BUFFER_TOO_SMALL && written > 0,
                  "direct query generated text bound");
            text_native.assign(written, '\0');
            written = 0;
            s = jetson_pi_llm_generate(llm, prompt, std::strlen(prompt),
                                       text_native.data(), text_native.size(),
                                       &written);
            CHECK(s == JETSON_PI_LLM_OK && written > 0,
                  "direct jetson_pi_llm_generate produced text");
            if (s != JETSON_PI_LLM_OK) {
                std::printf("    generate error: %s\n", jetson_pi_llm_last_error(llm));
            }
            text_native.resize(written);
            std::printf("    native  text: %s\n", text_native.c_str());
            CHECK(jetson_pi_llm_reset(llm) == JETSON_PI_LLM_OK &&
                  jetson_pi_llm_prefill(llm, prompt, std::strlen(prompt)) ==
                      JETSON_PI_LLM_OK,
                  "direct staged prefill");
            size_t logits_count = 0;
            CHECK(jetson_pi_llm_get_logits(
                      llm, nullptr, 0, &logits_count) ==
                      JETSON_PI_LLM_BUFFER_TOO_SMALL && logits_count > 0,
                  "direct query logits size");
            logits_native.resize(logits_count);
            CHECK(jetson_pi_llm_get_logits(
                      llm, logits_native.data(), logits_native.size(),
                      &logits_count) == JETSON_PI_LLM_OK,
                  "direct read logits");
            int32_t is_eog = 0;
            CHECK(jetson_pi_llm_decode_step(
                      llm, &token_native, &is_eog) == JETSON_PI_LLM_OK,
                  "direct decode first token");
            jetson_pi_llm_close(llm);
        }
    }

    if (g_fail) {
        std::printf("\n== LLM PARITY FAILED ==\n");
        return g_fail;
    }

    // Greedy = argmax, deterministic -> EXACT equality (no tolerance).
    if (text_flashrt == text_native) {
        CHECK(true, "FlashRT text == direct jetson_pi_llm text (exact)");
    } else {
        std::printf("FAIL: text mismatch (FlashRT %zu bytes, native %zu bytes)\n",
                    text_flashrt.size(), text_native.size());
        size_t cmn = 0;
        while (cmn < text_flashrt.size() && cmn < text_native.size() &&
               text_flashrt[cmn] == text_native[cmn]) ++cmn;
        std::printf("    first divergence at byte %zu: FlashRT=0x%02x native=0x%02x\n",
                    cmn,
                    cmn < text_flashrt.size() ? (unsigned char)text_flashrt[cmn] : 0,
                    cmn < text_native.size()  ? (unsigned char)text_native[cmn]  : 0);
        g_fail = 1;
    }
    CHECK(token_flashrt == token_native,
          "FlashRT first token matches direct narrow API");
    bool logits_match = logits_flashrt.size() == logits_native.size();
    float logits_max_diff = 0.0f;
    for (size_t i = 0; logits_match && i < logits_flashrt.size(); ++i) {
        const float diff = std::fabs(logits_flashrt[i] - logits_native[i]);
        if (diff > logits_max_diff) logits_max_diff = diff;
        if (diff > 1e-6f) logits_match = false;
    }
    std::printf("    logits max abs diff = %.9g\n", logits_max_diff);
    CHECK(logits_match, "FlashRT prefill logits match direct narrow API");

    std::printf(g_fail ? "\n== LLM PARITY FAILED ==\n"
                       : "\n== LLM PARITY PASSED ==\n");
    return g_fail;
}
