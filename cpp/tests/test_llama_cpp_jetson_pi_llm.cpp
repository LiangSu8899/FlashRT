// End-to-end LLM engine test: drives the real Jetson-PI LLM engine factory
// with a GGUF LLM (when available) and asserts a sane text completion.
// Skips (returns 0) when FLASHRT_LLM_MODEL is unset.

#include "flashrt/providers/llama_cpp/c_api.h"
#include "llama_cpp_generic_plan.h"
#include "flashrt/providers/llama_cpp/jetson_pi_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", (msg)); g_fail = 1; } \
    else { std::printf("ok  : %s\n", (msg)); } \
} while (0)

int main() {
    const char * model_env = std::getenv("FLASHRT_LLM_MODEL");
    if (!model_env || !*model_env || !std::fopen(model_env, "rb")) {
        std::printf("SKIP - FLASHRT_LLM_MODEL not set or file missing\n");
        return 0;
    }
    const char * backend_env = std::getenv("FLASHRT_LLM_BACKEND");
    const std::string backend = backend_env && *backend_env ? backend_env : "cpu";

    const frt_llama_cpp_engine_factory_v1 * factory =
        frt_llama_cpp_default_engine_factory();
    CHECK(factory != nullptr && factory->create_llm != nullptr,
          "default engine factory exposes create_llm");

    // Bogus model path must fail cleanly.
    const char * bogus_json =
        "{\"model_family\":\"llm\","
        "\"model_path\":\"/nonexistent/bogus.gguf\","
        "\"backend\":\"cpu\","
        "\"n_ctx\":2048,\"n_threads\":0,"
        "\"temp\":0.8,\"top_k\":40,\"top_p\":0.9,\"seed\":1,\"max_tokens\":64}";
    frt_model_runtime_v1 * bogus = nullptr;
    CHECK(frt_llama_cpp_llm_runtime_open_with_engine_factory(
              bogus_json, factory, &bogus) != 0 && bogus == nullptr,
          "open LLM with bogus model path fails without crashing");
    CHECK(std::strstr(frt_llama_cpp_runtime_open_error(),
                      "failed to open checkpoint for identity"),
          "bogus model path reports checkpoint identity error");

    // Real model.
    const char * prompt = "What is 2 plus 2? The answer is";
    std::string json =
        std::string("{") +
        "\"model_family\":\"llm\","
        "\"model_path\":\"" + model_env + "\","
        "\"backend\":\"" + backend + "\","
        "\"n_ctx\":2048,\"n_threads\":0,"
        "\"temp\":0.0,\"top_k\":0,\"top_p\":0.0,\"seed\":1,\"max_tokens\":16}";
    frt_model_runtime_v1 * model = nullptr;
    int rc = frt_llama_cpp_llm_runtime_open_with_engine_factory(
        json.c_str(), factory, &model);
    if (rc != 0 || !model) {
        std::printf("FAIL: open LLM runtime (rc=%d): %s\n", rc,
                    factory->last_error(factory->self));
        return 1;
    }
    CHECK(rc == 0 && model != nullptr, "open LLM runtime from JSON");
    const auto* plan = llama_cpp_generic_plan(model);
    std::printf("    runtime stages (%llu):",
                static_cast<unsigned long long>(plan ? plan->n_stages : 0));
    for (uint64_t i = 0; plan && i < plan->n_stages; ++i) {
        std::printf(" %s", plan->stages[i].name);
    }
    std::printf("\n");
    CHECK(model->n_stages == 0 && plan && plan->n_stages == 3,
          "LLM runtime exposes selected reset/prefill/decode plan");
    CHECK(model->n_ports == 6,
          "LLM runtime exposes prompt/text/token/logits/eog/tokens ports");
    CHECK(model->exp && std::strstr(model->exp->identity,
                                   "weights_sha256="),
          "deployment identity includes checkpoint content digest");
    CHECK(model->verbs.get_output(
              model->self, FRT_LLAMA_CPP_LLM_PORT_LOGITS,
              nullptr, 0, nullptr, -1) == -1,
          "get_output rejects null written without crashing");

    rc = model->verbs.set_input(model->self,
                                   FRT_LLAMA_CPP_LLM_PORT_PROMPT,
                                   prompt, std::strlen(prompt), -1);
    CHECK(rc == 0, "set_input prompt");

    rc = model->verbs.step(model->self);
    if (rc != 0) {
        std::printf("FAIL: run_stage infer (rc=%d): %s\n", rc,
                    model->verbs.last_error(model->self));
        g_fail = 1;
    } else {
        std::printf("ok  : run_stage infer\n");
    }

    uint64_t written = 0;
    rc = model->verbs.get_output(
        model->self, FRT_LLAMA_CPP_LLM_PORT_TEXT, nullptr, 0, &written, -1);
    CHECK(rc == -5 && written > 0, "query one-shot text size");
    std::string text(written, '\0');
    written = 0;
    rc = model->verbs.get_output(
        model->self, FRT_LLAMA_CPP_LLM_PORT_TEXT, &text[0], text.size(),
        &written, -1);
    CHECK(rc == 0 && written > 0, "get_output text non-empty");
    if (rc == 0) {
        text.resize(written);
        std::printf("    generated: %s\n", text.c_str());
        // Sanity: contains at least one printable character.
        bool printable = false;
        for (char c : text) {
            if (c >= 0x20 && c < 0x7f) { printable = true; break; }
        }
        CHECK(printable, "generated text contains printable chars");
    }
    CHECK(model->verbs.set_input(
              model->self, FRT_LLAMA_CPP_LLM_PORT_PROMPT,
              nullptr, 0, -1) != 0,
          "invalid prompt replacement is rejected");
    CHECK(model->verbs.step(model->self) != 0,
          "failed prompt replacement invalidates the previous prompt");
    CHECK(model->verbs.set_input(
              model->self, FRT_LLAMA_CPP_LLM_PORT_PROMPT,
              prompt, std::strlen(prompt), -1) == 0,
          "restore valid prompt after failed replacement");

    CHECK(llama_cpp_run_generic_stage(model, "reset") == 0,
          "run_stage reset");
    const auto prefill_start = std::chrono::steady_clock::now();
    rc = llama_cpp_run_generic_stage(model, "prefill");
    const auto prefill_end = std::chrono::steady_clock::now();
    CHECK(rc == 0, "run_stage prefill");
    std::printf("    staged prefill latency: %.3f ms\n",
                std::chrono::duration<double, std::milli>(
                    prefill_end - prefill_start).count());
    uint64_t logits_bytes = 0;
    rc = model->verbs.get_output(
        model->self, FRT_LLAMA_CPP_LLM_PORT_LOGITS, nullptr, 0,
        &logits_bytes, -1);
    CHECK(rc != 0 && logits_bytes > 1000 * sizeof(float),
          "prefill exposes logits size");
    std::vector<float> logits(logits_bytes / sizeof(float));
    rc = model->verbs.get_output(
        model->self, FRT_LLAMA_CPP_LLM_PORT_LOGITS, logits.data(),
        logits_bytes, &logits_bytes, -1);
    CHECK(rc == 0, "prefill get_output logits");
    bool finite_logits = true;
    for (float value : logits) finite_logits &= std::isfinite(value);
    CHECK(finite_logits, "prefill logits contain no NaN/Inf");

    std::string staged_text;
    std::vector<int32_t> baseline_tokens;
    std::vector<int32_t> baseline_eog;
    double decode_ms = 0.0;
    for (uint32_t i = 0; i < 16; ++i) {
        const auto decode_start = std::chrono::steady_clock::now();
        rc = llama_cpp_run_generic_stage(model, "decode");
        const auto decode_end = std::chrono::steady_clock::now();
        decode_ms += std::chrono::duration<double, std::milli>(
            decode_end - decode_start).count();
        CHECK(rc == 0, "run_stage decode");
        int32_t token = 0;
        int32_t is_eog = 0;
        uint64_t scalar_bytes = 0;
        CHECK(model->verbs.get_output(
                  model->self, FRT_LLAMA_CPP_LLM_PORT_NEXT_TOKEN, &token,
                  sizeof(token), &scalar_bytes, -1) == 0 &&
                  scalar_bytes == sizeof(token),
              "decode exposes next_token");
        CHECK(model->verbs.get_output(
                  model->self, FRT_LLAMA_CPP_LLM_PORT_IS_EOG, &is_eog,
                  sizeof(is_eog), &scalar_bytes, -1) == 0 &&
                  scalar_bytes == sizeof(is_eog),
              "decode exposes is_eog");
        baseline_tokens.push_back(token);
        baseline_eog.push_back(is_eog);
        if (is_eog) break;
    }
    std::printf("    staged decode throughput: %.2f token/s "
                "(%zu tokens in %.3f ms)\n",
                baseline_tokens.size() * 1000.0 / decode_ms,
                baseline_tokens.size(), decode_ms);
    written = 0;
    rc = model->verbs.get_output(
        model->self, FRT_LLAMA_CPP_LLM_PORT_TEXT, nullptr, 0, &written, -1);
    CHECK(rc == -5 && written > 0, "query staged text size");
    staged_text.assign(written, '\0');
    written = 0;
    rc = model->verbs.get_output(
        model->self, FRT_LLAMA_CPP_LLM_PORT_TEXT, staged_text.data(),
        staged_text.size(), &written, -1);
    CHECK(rc == 0 && written > 0, "staged decode exposes accumulated text");
    staged_text.resize(written);
    CHECK(staged_text == text, "staged decode text matches one-shot infer");
    rc = model->verbs.step(model->self);
    CHECK(rc == 0, "one-shot infer after staged decode");
    uint64_t stale_written = 0;
    rc = model->verbs.get_output(
        model->self, FRT_LLAMA_CPP_LLM_PORT_NEXT_TOKEN, nullptr, 0,
        &stale_written, -1);
    CHECK(rc == -7, "one-shot infer invalidates staged token output");

    frt_model_runtime_v1* peer = nullptr;
    rc = frt_llama_cpp_llm_runtime_open_with_engine_factory(
        json.c_str(), factory, &peer);
    CHECK(rc == 0 && peer, "open independent peer LLM session");
    if (peer) {
        const char* peer_prompt =
            "Complete this sequence with one number: 3, 6, 9,";
        CHECK(peer->verbs.set_input(
                  peer->self, FRT_LLAMA_CPP_LLM_PORT_PROMPT,
                  peer_prompt, std::strlen(peer_prompt), -1) == 0,
              "peer set_input distinct prompt");
        CHECK(llama_cpp_run_generic_stage(peer, "reset") == 0 &&
                  llama_cpp_run_generic_stage(peer, "prefill") == 0,
              "prepare peer standalone baseline");
        std::vector<int32_t> peer_baseline_tokens;
        std::vector<int32_t> peer_baseline_eog;
        for (uint32_t i = 0; i < 16; ++i) {
            CHECK(llama_cpp_run_generic_stage(peer, "decode") == 0,
                  "decode peer standalone baseline");
            int32_t token = 0;
            int32_t is_eog = 0;
            uint64_t scalar_bytes = 0;
            CHECK(peer->verbs.get_output(
                      peer->self, FRT_LLAMA_CPP_LLM_PORT_NEXT_TOKEN, &token,
                      sizeof(token), &scalar_bytes, -1) == 0 &&
                      scalar_bytes == sizeof(token) &&
                      peer->verbs.get_output(
                      peer->self, FRT_LLAMA_CPP_LLM_PORT_IS_EOG, &is_eog,
                      sizeof(is_eog), &scalar_bytes, -1) == 0,
                  "capture peer standalone token sequence");
            peer_baseline_tokens.push_back(token);
            peer_baseline_eog.push_back(is_eog);
            if (is_eog) break;
        }
        uint64_t peer_baseline_written = 0;
        CHECK(peer->verbs.get_output(
                  peer->self, FRT_LLAMA_CPP_LLM_PORT_TEXT,
                  nullptr, 0, &peer_baseline_written, -1) == -5 &&
                  peer_baseline_written > 0,
              "query peer standalone baseline text size");
        std::string peer_baseline_text(peer_baseline_written, '\0');
        peer_baseline_written = 0;
        CHECK(peer->verbs.get_output(
                  peer->self, FRT_LLAMA_CPP_LLM_PORT_TEXT,
                  peer_baseline_text.data(), peer_baseline_text.size(),
                  &peer_baseline_written, -1) == 0,
              "read peer standalone baseline text");
        peer_baseline_text.resize(peer_baseline_written);
        CHECK(llama_cpp_run_generic_stage(model, "reset") == 0 &&
                  llama_cpp_run_generic_stage(peer, "reset") == 0,
              "reset two independent sessions");
        CHECK(llama_cpp_run_generic_stage(model, "prefill") == 0 &&
                  llama_cpp_run_generic_stage(peer, "prefill") == 0,
              "prefill two independent sessions");
        const size_t interleaved_steps =
            std::max(baseline_tokens.size(), peer_baseline_tokens.size());
        for (size_t i = 0; i < interleaved_steps; ++i) {
            uint64_t scalar_bytes = 0;
            if (i < baseline_tokens.size()) {
                int32_t token = 0;
                int32_t is_eog = 0;
                CHECK(llama_cpp_run_generic_stage(model, "decode") == 0 &&
                          model->verbs.get_output(
                          model->self, FRT_LLAMA_CPP_LLM_PORT_NEXT_TOKEN,
                          &token, sizeof(token), &scalar_bytes, -1) == 0 &&
                          model->verbs.get_output(
                          model->self, FRT_LLAMA_CPP_LLM_PORT_IS_EOG,
                          &is_eog, sizeof(is_eog), &scalar_bytes, -1) == 0 &&
                          token == baseline_tokens[i] &&
                          is_eog == baseline_eog[i],
                      "interleaved primary token matches standalone baseline");
            }
            if (i < peer_baseline_tokens.size()) {
                int32_t token = 0;
                int32_t is_eog = 0;
                CHECK(llama_cpp_run_generic_stage(peer, "decode") == 0 &&
                          peer->verbs.get_output(
                          peer->self, FRT_LLAMA_CPP_LLM_PORT_NEXT_TOKEN,
                          &token, sizeof(token), &scalar_bytes, -1) == 0 &&
                          peer->verbs.get_output(
                          peer->self, FRT_LLAMA_CPP_LLM_PORT_IS_EOG,
                          &is_eog, sizeof(is_eog), &scalar_bytes, -1) == 0 &&
                          token == peer_baseline_tokens[i] &&
                          is_eog == peer_baseline_eog[i],
                      "interleaved peer token matches standalone baseline");
            }
        }
        uint64_t model_written = 0;
        uint64_t peer_written = 0;
        CHECK(model->verbs.get_output(
                  model->self, FRT_LLAMA_CPP_LLM_PORT_TEXT,
                  nullptr, 0, &model_written, -1) == -5 &&
                  model_written > 0 &&
                  peer->verbs.get_output(
                  peer->self, FRT_LLAMA_CPP_LLM_PORT_TEXT,
                  nullptr, 0, &peer_written, -1) == -5 &&
                  peer_written > 0,
              "query interleaved session output sizes");
        std::string model_text(model_written, '\0');
        std::string peer_text(peer_written, '\0');
        model_written = 0;
        peer_written = 0;
        CHECK(model->verbs.get_output(
                  model->self, FRT_LLAMA_CPP_LLM_PORT_TEXT, model_text.data(),
                  model_text.size(), &model_written, -1) == 0 &&
                  peer->verbs.get_output(
                  peer->self, FRT_LLAMA_CPP_LLM_PORT_TEXT, peer_text.data(),
                  peer_text.size(), &peer_written, -1) == 0,
              "read interleaved session outputs");
        model_text.resize(model_written);
        peer_text.resize(peer_written);
        CHECK(model_text == staged_text && peer_text == peer_baseline_text,
              "distinct sessions reproduce baselines without KV leakage");
        peer->release(peer->owner);
    }

    model->release(model->owner);

    std::string budget_json =
        std::string("{") +
        "\"model_family\":\"llm\"," +
        "\"model_path\":\"" + model_env + "\"," +
        "\"backend\":\"" + backend + "\"," +
        "\"n_ctx\":2048,\"n_threads\":0,"
        "\"temp\":0.0,\"top_k\":0,\"top_p\":0.0,\"seed\":1,"
        "\"max_tokens\":1}";
    frt_model_runtime_v1 * budget_model = nullptr;
    rc = frt_llama_cpp_llm_runtime_open_with_engine_factory(
        budget_json.c_str(), factory, &budget_model);
    CHECK(rc == 0 && budget_model, "open max_tokens=1 LLM runtime");
    if (budget_model) {
        CHECK(budget_model->verbs.set_input(
                  budget_model->self, FRT_LLAMA_CPP_LLM_PORT_PROMPT,
                  prompt, std::strlen(prompt), -1) == 0 &&
                  llama_cpp_run_generic_stage(budget_model, "prefill") == 0 &&
                  llama_cpp_run_generic_stage(budget_model, "decode") == 0,
              "max_tokens=1 session permits exactly one decode");
        int32_t first_is_eog = 1;
        uint64_t scalar_written = 0;
        CHECK(budget_model->verbs.get_output(
                  budget_model->self, FRT_LLAMA_CPP_LLM_PORT_IS_EOG,
                  &first_is_eog, sizeof(first_is_eog), &scalar_written, -1) == 0 &&
                  first_is_eog == 0,
              "budget test prompt first token is not EOG");
        CHECK(llama_cpp_run_generic_stage(budget_model, "decode") != 0 &&
                  std::strstr(budget_model->verbs.last_error(
                                  budget_model->self),
                              "max_tokens"),
              "staged decode rejects calls beyond max_tokens");
        int32_t stale_scalar = 0;
        CHECK(budget_model->verbs.get_output(
                  budget_model->self, FRT_LLAMA_CPP_LLM_PORT_NEXT_TOKEN,
                  &stale_scalar, sizeof(stale_scalar), &scalar_written, -1) == -7 &&
                  budget_model->verbs.get_output(
                  budget_model->self, FRT_LLAMA_CPP_LLM_PORT_IS_EOG,
                  &stale_scalar, sizeof(stale_scalar), &scalar_written, -1) == -7,
              "failed decode invalidates staged scalar outputs");
        CHECK(llama_cpp_run_generic_stage(budget_model, "prefill") == 0 &&
                  llama_cpp_run_generic_stage(budget_model, "decode") == 0,
              "prefill restores staged decode budget");
        budget_model->release(budget_model->owner);
    }

    std::printf(g_fail ? "\n== JETSON_PI LLM FAILED ==\n"
                       : "\n== JETSON_PI LLM PASSED ==\n");
    return g_fail;
}
