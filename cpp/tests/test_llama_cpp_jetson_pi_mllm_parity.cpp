// Numerical parity: FlashRT MLLM provider versus direct jetson_pi_mllm.

#include "flashrt/providers/llama_cpp/c_api.h"
#include "llama_cpp_generic_plan.h"
#include "flashrt/providers/llama_cpp/jetson_pi_engine.h"
#include "jetson_pi_mllm.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", (msg)); g_fail = 1; } \
    else { std::printf("ok  : %s\n", (msg)); } \
} while (0)

int main() {
    const char* model_path = std::getenv("FLASHRT_MLLM_MODEL");
    const char* mmproj_path = std::getenv("FLASHRT_MLLM_MMPROJ");
    const char* backend_env = std::getenv("FLASHRT_MLLM_BACKEND");
    const std::string backend =
        backend_env && *backend_env ? backend_env : "cpu";
    if (!model_path || !mmproj_path ||
        !std::ifstream(model_path, std::ios::binary).good() ||
        !std::ifstream(mmproj_path, std::ios::binary).good()) {
        std::printf("SKIP - MLLM model/mmproj missing\n");
        return 0;
    }

    const uint32_t width = 224;
    const uint32_t height = 224;
    const uint32_t max_tokens = 16;
    const char* prompt = "Describe this image in one sentence.";
    std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3);
    for (size_t i = 0; i < rgb.size(); i += 3) rgb[i] = 255;

    std::string flashrt_text;
    {
        const auto* factory = frt_llama_cpp_default_engine_factory();
        std::string json =
            std::string("{") +
            "\"model_family\":\"mllm\",\"model_path\":\"" + model_path +
            "\",\"mmproj_path\":\"" + mmproj_path +
            "\",\"backend\":\"" + backend +
            "\",\"n_ctx\":2048,\"n_threads\":0,\"temp\":0.0,"
            "\"top_k\":0,\"top_p\":0.0,\"seed\":1,\"max_tokens\":16}";
        frt_model_runtime_v1* model = nullptr;
        int rc = frt_llama_cpp_mllm_runtime_open_with_engine_factory(
            json.c_str(), factory, &model);
        CHECK(rc == 0 && model, "open FlashRT MLLM runtime");
        if (model) {
            frt_image_view image{};
            image.struct_size = sizeof(image);
            image.pixel_format = FRT_RT_PIXEL_RGB8;
            image.data = rgb.data();
            image.bytes = rgb.size();
            image.width = width;
            image.height = height;
            CHECK(model->verbs.set_input(
                      model->self, FRT_LLAMA_CPP_MLLM_PORT_IMAGES,
                      &image, sizeof(image), -1) == 0 &&
                  model->verbs.set_input(
                      model->self, FRT_LLAMA_CPP_MLLM_PORT_PROMPT,
                      prompt, std::strlen(prompt), -1) == 0 &&
                  model->verbs.step(model->self) == 0,
                  "run FlashRT MLLM image+text infer");
            uint64_t written = 0;
            rc = model->verbs.get_output(
                model->self, FRT_LLAMA_CPP_MLLM_PORT_TEXT,
                nullptr, 0, &written, -1);
            CHECK(rc == -5 && written > 0, "query FlashRT MLLM text size");
            flashrt_text.assign(written, '\0');
            written = 0;
            rc = model->verbs.get_output(
                model->self, FRT_LLAMA_CPP_MLLM_PORT_TEXT,
                flashrt_text.data(), flashrt_text.size(), &written, -1);
            CHECK(rc == 0 && written > 0, "read FlashRT MLLM text");
            flashrt_text.resize(written);
            model->release(model->owner);
        }
    }

    std::string direct_text;
    {
        jetson_pi_mllm_config config{};
        config.struct_size = sizeof(config);
        config.model_path = model_path;
        config.mmproj_path = mmproj_path;
        config.backend = backend.c_str();
        config.n_ctx = 2048;
        config.temp = 0.0f;
        config.seed = 1;
        config.max_tokens = max_tokens;
        jetson_pi_mllm* handle = nullptr;
        int32_t rc = jetson_pi_mllm_open(&config, &handle);
        CHECK(rc == JETSON_PI_MLLM_OK && handle,
              "open direct jetson_pi_mllm");
        if (handle) {
            const uint8_t* images[] = {rgb.data()};
            size_t written = 0;
            rc = jetson_pi_mllm_infer(
                handle, images, 1, height, width, prompt,
                std::strlen(prompt), nullptr, 0, &written);
            CHECK(rc == JETSON_PI_MLLM_BUFFER_TOO_SMALL && written > 0,
                  "query direct MLLM text bound");
            direct_text.assign(written, '\0');
            written = 0;
            rc = jetson_pi_mllm_infer(
                handle, images, 1, height, width, prompt,
                std::strlen(prompt), direct_text.data(), direct_text.size(),
                &written);
            CHECK(rc == JETSON_PI_MLLM_OK && written > 0,
                  "run direct MLLM image+text infer");
            direct_text.resize(written);
            jetson_pi_mllm_close(handle);
        }
    }

    CHECK(flashrt_text == direct_text,
          "FlashRT MLLM text matches direct narrow API exactly");
    std::printf(g_fail ? "\n== MLLM PARITY FAILED ==\n"
                       : "\n== MLLM PARITY PASSED ==\n");
    return g_fail;
}
