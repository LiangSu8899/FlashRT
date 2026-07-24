// End-to-end MLLM engine test: drives the real Jetson-PI MLLM engine factory
// with a GGUF VLM (when available) + a programmatically generated test image,
// and asserts a sane text completion. Skips (returns 0) when env vars unset.
//
// Env:
//   FLASHRT_MLLM_MODEL     path to VLM GGUF (e.g. Qwen2.5-VL-3B-Instruct-q4_0.gguf)
//   FLASHRT_MLLM_MMPROJ    path to VIT mmproj GGUF

#include "flashrt/providers/llama_cpp/c_api.h"
#include "llama_cpp_generic_plan.h"
#include "flashrt/providers/llama_cpp/jetson_pi_engine.h"

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
    const char * model_env = std::getenv("FLASHRT_MLLM_MODEL");
    const char * mmproj_env = std::getenv("FLASHRT_MLLM_MMPROJ");
    const char * backend_env = std::getenv("FLASHRT_MLLM_BACKEND");
    const char * backend = (backend_env && *backend_env) ? backend_env : "cpu";
    auto file_exists = [](const char * path) {
        if (!path || !*path) return false;
        FILE * f = std::fopen(path, "rb");
        if (!f) return false;
        std::fclose(f);
        return true;
    };
    if (!file_exists(model_env) || !file_exists(mmproj_env)) {
        std::printf("SKIP - FLASHRT_MLLM_MODEL / FLASHRT_MLLM_MMPROJ not set or missing\n");
        return 0;
    }

    const frt_llama_cpp_engine_factory_v1 * factory =
        frt_llama_cpp_default_engine_factory();
    CHECK(factory != nullptr && factory->create_mllm != nullptr,
          "default engine factory exposes create_mllm");

    // Bogus model path must fail cleanly.
    const char * bogus_json =
        "{\"model_family\":\"mllm\","
        "\"model_path\":\"/nonexistent/bogus.gguf\","
        "\"mmproj_path\":\"/nonexistent/bogus-mmproj.gguf\","
        "\"backend\":\"cpu\","
        "\"n_ctx\":2048,\"n_threads\":0,"
        "\"temp\":0.0,\"top_k\":0,\"top_p\":0.0,\"seed\":1,\"max_tokens\":32}";
    frt_model_runtime_v1 * bogus = nullptr;
    CHECK(frt_llama_cpp_mllm_runtime_open_with_engine_factory(
              bogus_json, factory, &bogus) != 0 && bogus == nullptr,
          "open MLLM with bogus model path fails without crashing");

    // Real VLM.
    std::string json =
        std::string("{") +
        "\"model_family\":\"mllm\","
        "\"model_path\":\"" + model_env + "\","
        "\"mmproj_path\":\"" + mmproj_env + "\","
        "\"backend\":\"" + std::string(backend) + "\","
        "\"n_ctx\":2048,\"n_threads\":0,"
        "\"temp\":0.0,\"top_k\":0,\"top_p\":0.0,\"seed\":1,\"max_tokens\":16}";
    frt_model_runtime_v1 * model = nullptr;
    int rc = frt_llama_cpp_mllm_runtime_open_with_engine_factory(
        json.c_str(), factory, &model);
    if (rc != 0 || !model) {
        std::printf("FAIL: open MLLM runtime (rc=%d): %s\n", rc,
                    factory->last_error(factory->self));
        return 1;
    }
    CHECK(rc == 0 && model != nullptr, "open MLLM runtime from JSON");
    const auto* plan = llama_cpp_generic_plan(model);
    CHECK(model->n_stages == 0 && plan && plan->n_stages == 3,
          "MLLM runtime exposes selected reset/prefill/decode plan");
    CHECK(model->n_ports == 6,
          "MLLM runtime exposes images/prompt/text/token/logits/eog ports");
    CHECK(model->verbs.get_output(
              model->self, FRT_LLAMA_CPP_MLLM_PORT_LOGITS,
              nullptr, 0, nullptr, -1) == -1,
          "get_output rejects null written without crashing");

    // Generate a 224x224 solid-red RGB image inline (no stb_image_write dep).
    const int W = 224, H = 224;
    std::vector<uint8_t> rgb(static_cast<size_t>(W) * H * 3);
    for (size_t i = 0; i < rgb.size(); i += 3) {
        rgb[i] = 255; rgb[i + 1] = 0; rgb[i + 2] = 0;
    }

    frt_image_view view;
    view.struct_size = sizeof(frt_image_view);
    view.pixel_format = FRT_RT_PIXEL_RGB8;
    view.data = rgb.data();
    view.bytes = rgb.size();
    view.width = W; view.height = H; view.stride_bytes = 0;
    view.reserved = 0; view.timestamp_ns = 0;

    rc = model->verbs.set_input(model->self,
                                   FRT_LLAMA_CPP_MLLM_PORT_IMAGES,
                                   &view, sizeof(view), -1);
    CHECK(rc == 0, "set_input images");

    const char * prompt = "Describe this image in one sentence.";
    rc = model->verbs.set_input(model->self,
                                   FRT_LLAMA_CPP_MLLM_PORT_PROMPT,
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
        model->self, FRT_LLAMA_CPP_MLLM_PORT_TEXT, nullptr, 0, &written, -1);
    CHECK(rc == -5 && written > 0, "query one-shot text size");
    std::string text(written, '\0');
    written = 0;
    rc = model->verbs.get_output(
        model->self, FRT_LLAMA_CPP_MLLM_PORT_TEXT, &text[0], text.size(),
        &written, -1);
    CHECK(rc == 0 && written > 0, "get_output text non-empty");
    if (rc == 0) {
        text.resize(written);
        std::printf("    generated: %s\n", text.c_str());
        bool printable = false;
        for (char c : text) {
            if (c >= 0x20 && c < 0x7f) { printable = true; break; }
        }
        CHECK(printable, "generated text contains printable chars");
    }
    frt_image_view bad_view = view;
    bad_view.stride_bytes = 1;
    CHECK(model->verbs.set_input(
              model->self, FRT_LLAMA_CPP_MLLM_PORT_IMAGES,
              &bad_view, sizeof(bad_view), -1) != 0,
          "undersized image stride is rejected");
    CHECK(model->verbs.step(model->self) != 0,
          "failed image replacement invalidates previous images");
    CHECK(model->verbs.set_input(
              model->self, FRT_LLAMA_CPP_MLLM_PORT_IMAGES,
              &view, sizeof(view), -1) == 0,
          "restore valid images after failed replacement");
    CHECK(model->verbs.set_input(
              model->self, FRT_LLAMA_CPP_MLLM_PORT_PROMPT,
              nullptr, 0, -1) != 0,
          "invalid prompt replacement is rejected");
    CHECK(model->verbs.step(model->self) != 0,
          "failed prompt replacement invalidates previous prompt");
    CHECK(model->verbs.set_input(
              model->self, FRT_LLAMA_CPP_MLLM_PORT_PROMPT,
              prompt, std::strlen(prompt), -1) == 0,
          "restore valid prompt after failed replacement");

    rc = llama_cpp_run_generic_stage(model, "reset");
    CHECK(rc == 0, "run_stage reset");
    const auto prefill_start = std::chrono::steady_clock::now();
    rc = llama_cpp_run_generic_stage(model, "prefill");
    const auto prefill_end = std::chrono::steady_clock::now();
    CHECK(rc == 0, "run_stage prefill");
    std::printf("    staged MLLM prefill latency: %.3f ms\n",
                std::chrono::duration<double, std::milli>(
                    prefill_end - prefill_start).count());

    uint64_t logits_bytes = 0;
    rc = model->verbs.get_output(
        model->self, FRT_LLAMA_CPP_MLLM_PORT_LOGITS, nullptr, 0,
        &logits_bytes, -1);
    CHECK(rc == -5 && logits_bytes > 0 && logits_bytes % sizeof(float) == 0,
          "prefill exposes logits size");
    std::vector<float> logits(logits_bytes / sizeof(float));
    uint64_t logits_written = 0;
    rc = model->verbs.get_output(
        model->self, FRT_LLAMA_CPP_MLLM_PORT_LOGITS, logits.data(),
        logits_bytes, &logits_written, -1);
    CHECK(rc == 0 && logits_written == logits_bytes,
          "prefill get_output logits");
    bool finite = true;
    for (float value : logits) finite = finite && std::isfinite(value);
    CHECK(finite, "prefill logits contain no NaN/Inf");

    double decode_ms = 0.0;
    uint32_t decoded_tokens = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        const auto decode_start = std::chrono::steady_clock::now();
        rc = llama_cpp_run_generic_stage(model, "decode");
        const auto decode_end = std::chrono::steady_clock::now();
        decode_ms += std::chrono::duration<double, std::milli>(
            decode_end - decode_start).count();
        ++decoded_tokens;
        CHECK(rc == 0, "run_stage decode");
        int32_t token = 0;
        int32_t is_eog = 0;
        uint64_t scalar_written = 0;
        rc = model->verbs.get_output(
            model->self, FRT_LLAMA_CPP_MLLM_PORT_NEXT_TOKEN, &token,
            sizeof(token), &scalar_written, -1);
        CHECK(rc == 0 && scalar_written == sizeof(token),
              "decode exposes next_token");
        rc = model->verbs.get_output(
            model->self, FRT_LLAMA_CPP_MLLM_PORT_IS_EOG, &is_eog,
            sizeof(is_eog), &scalar_written, -1);
        CHECK(rc == 0 && scalar_written == sizeof(is_eog),
              "decode exposes is_eog");
        if (is_eog) break;
    }
    std::printf("    staged MLLM decode throughput: %.2f token/s "
                "(%u tokens in %.3f ms)\n",
                decoded_tokens * 1000.0 / decode_ms,
                decoded_tokens, decode_ms);
    written = 0;
    rc = model->verbs.get_output(
        model->self, FRT_LLAMA_CPP_MLLM_PORT_TEXT, nullptr, 0, &written, -1);
    CHECK(rc == -5 && written > 0, "query staged text size");
    std::string staged(written, '\0');
    written = 0;
    rc = model->verbs.get_output(
        model->self, FRT_LLAMA_CPP_MLLM_PORT_TEXT, staged.data(), staged.size(),
        &written, -1);
    CHECK(rc == 0 && written > 0, "staged decode exposes accumulated text");
    if (rc == 0) staged.resize(written);
    CHECK(staged == text, "staged decode text matches one-shot infer");
    rc = model->verbs.step(model->self);
    CHECK(rc == 0, "one-shot infer after staged decode");
    uint64_t stale_written = 0;
    rc = model->verbs.get_output(
        model->self, FRT_LLAMA_CPP_MLLM_PORT_NEXT_TOKEN, nullptr, 0,
        &stale_written, -1);
    CHECK(rc == -7, "one-shot infer invalidates staged token output");

    model->release(model->owner);

    std::string budget_json =
        std::string("{") +
        "\"model_family\":\"mllm\"," +
        "\"model_path\":\"" + model_env + "\"," +
        "\"mmproj_path\":\"" + mmproj_env + "\"," +
        "\"backend\":\"" + std::string(backend) + "\"," +
        "\"n_ctx\":2048,\"n_threads\":0,"
        "\"temp\":0.0,\"top_k\":0,\"top_p\":0.0,\"seed\":1,"
        "\"max_tokens\":1}";
    frt_model_runtime_v1 * budget_model = nullptr;
    rc = frt_llama_cpp_mllm_runtime_open_with_engine_factory(
        budget_json.c_str(), factory, &budget_model);
    CHECK(rc == 0 && budget_model, "open max_tokens=1 MLLM runtime");
    if (budget_model) {
        CHECK(budget_model->verbs.set_input(
                  budget_model->self, FRT_LLAMA_CPP_MLLM_PORT_IMAGES,
                  &view, sizeof(view), -1) == 0 &&
                  budget_model->verbs.set_input(
                  budget_model->self, FRT_LLAMA_CPP_MLLM_PORT_PROMPT,
                  prompt, std::strlen(prompt), -1) == 0 &&
                  llama_cpp_run_generic_stage(budget_model, "prefill") == 0 &&
                  llama_cpp_run_generic_stage(budget_model, "decode") == 0,
              "max_tokens=1 MLLM session permits exactly one decode");
        int32_t first_is_eog = 1;
        uint64_t scalar_written = 0;
        CHECK(budget_model->verbs.get_output(
                  budget_model->self, FRT_LLAMA_CPP_MLLM_PORT_IS_EOG,
                  &first_is_eog, sizeof(first_is_eog), &scalar_written, -1) == 0 &&
                  first_is_eog == 0,
              "budget test image prompt first token is not EOG");
        CHECK(llama_cpp_run_generic_stage(budget_model, "decode") != 0 &&
                  std::strstr(budget_model->verbs.last_error(
                                  budget_model->self),
                              "max_tokens"),
              "staged decode rejects calls beyond max_tokens");
        int32_t stale_scalar = 0;
        CHECK(budget_model->verbs.get_output(
                  budget_model->self, FRT_LLAMA_CPP_MLLM_PORT_NEXT_TOKEN,
                  &stale_scalar, sizeof(stale_scalar), &scalar_written, -1) == -7 &&
                  budget_model->verbs.get_output(
                  budget_model->self, FRT_LLAMA_CPP_MLLM_PORT_IS_EOG,
                  &stale_scalar, sizeof(stale_scalar), &scalar_written, -1) == -7,
              "failed decode invalidates staged scalar outputs");
        CHECK(llama_cpp_run_generic_stage(budget_model, "prefill") == 0 &&
                  llama_cpp_run_generic_stage(budget_model, "decode") == 0,
              "prefill restores staged decode budget");
        budget_model->release(budget_model->owner);
    }

    std::printf(g_fail ? "\n== JETSON_PI MLLM FAILED ==\n"
                       : "\n== JETSON_PI MLLM PASSED ==\n");
    return g_fail;
}
