// End-to-end Pi0 engine test: drives the real Jetson-PI engine factory with
// Pi0 weights (when available) and asserts a sane action chunk. Skips (returns
// 0) when the weights/fixture env vars are unset, so CI without weights still
// passes.
//
// Coverage:
//   A. bogus model path fails cleanly (no-weights contract)
//   B. end-to-end single Pi0 tick: set_input images/prompt/state -> run_stage
//      infer -> get_output actions; asserts shape, no NaN/Inf, non-zero.
//   C. config vs model action_dim mismatch: open must reject (engine freed).
//   D. multi-tick: a second infer with fresh inputs must (1) return -7 from
//      get_output after set_input but before run_infer (staleness guard),
//      and (2) reproduce the first tick's actions (KV reset, no leak).
//
// Env:
//   FLASHRT_PI0_MODEL        path to Pi0 policy GGUF
//   FLASHRT_PI0_MMPROJ       path to VIT mmproj GGUF
//   FLASHRT_PI0_FIXTURE_DIR  dir containing image.png, wrist_image.png,
//                            state.bin (action_dim float32), prompt.txt
//   FLASHRT_PI0_ACTION_STEPS required model-specific action horizon.
//   FLASHRT_PI0_ACTION_DIM   required model-specific action width.

#include "flashrt/providers/llama_cpp/c_api.h"
#include "llama_cpp_generic_plan.h"
#include "flashrt/providers/llama_cpp/jetson_pi_engine.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb/stb_image.h"

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", (msg)); g_fail = 1; } \
    else { std::printf("ok  : %s\n", (msg)); } \
} while (0)

namespace {

bool file_exists(const char * p) {
    std::ifstream f(p);
    return f.good();
}

std::string read_file(const std::string & path, bool * ok) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { if (ok) *ok = false; return {}; }
    std::string s((std::istreambuf_iterator<char>(f)),
                  std::istreambuf_iterator<char>());
    if (ok) *ok = true;
    return s;
}

} // namespace

int main() {
    const char * model_env = std::getenv("FLASHRT_PI0_MODEL");
    const char * mmproj_env = std::getenv("FLASHRT_PI0_MMPROJ");
    const char * fixture_env = std::getenv("FLASHRT_PI0_FIXTURE_DIR");
    if (!model_env || !mmproj_env || !fixture_env ||
        !file_exists(model_env) || !file_exists(mmproj_env)) {
        std::printf("SKIP - FLASHRT_PI0_MODEL / FLASHRT_PI0_MMPROJ / "
                    "FLASHRT_PI0_FIXTURE_DIR not set or files missing\n");
        return 0;
    }

    const std::string fixture_dir = fixture_env;
    const std::string img_path = fixture_dir + "/image.png";
    const std::string wrist_path = fixture_dir + "/wrist_image.png";
    const std::string state_path = fixture_dir + "/state.bin";
    const std::string prompt_path = fixture_dir + "/prompt.txt";
    if (!file_exists(img_path.c_str()) ||
        !file_exists(wrist_path.c_str()) ||
        !file_exists(state_path.c_str()) ||
        !file_exists(prompt_path.c_str())) {
        std::printf("SKIP - fixture files missing in %s\n", fixture_dir.c_str());
        return 0;
    }

    // ---- sub-test A: bogus model path fails (no-weights contract) ----------
    const frt_llama_cpp_engine_factory_v1 * factory =
        frt_llama_cpp_default_engine_factory();
    CHECK(factory != nullptr, "default engine factory is non-null");
    CHECK(factory->create_pi0 != nullptr && factory->last_error != nullptr,
          "factory vtable complete");

    const char * bogus_json =
        "{\"model_family\":\"pi0\","
        "\"model_path\":\"/nonexistent/bogus.gguf\","
        "\"mmproj_path\":\"/nonexistent/bogus-mmproj.gguf\","
        "\"backend\":\"cpu\",\"n_views\":2,\"image_height\":224,"
        "\"image_width\":224,\"image_channels\":3,"
        "\"action_steps\":10,\"action_dim\":32}";
    frt_model_runtime_v1 * bogus = nullptr;
    int rc = frt_llama_cpp_pi0_runtime_open_with_engine_factory(
        bogus_json, factory, &bogus);
    CHECK(rc != 0 && bogus == nullptr,
          "open with bogus model path fails without crashing");

    // ---- sub-test B: end-to-end Pi0 tick -----------------------------------
    // Action dims come from env (verified pi0_base is 50x32; other checkpoints
    // may differ and must provide their model-specific shape explicitly).
    const char * steps_env = std::getenv("FLASHRT_PI0_ACTION_STEPS");
    const char * dim_env   = std::getenv("FLASHRT_PI0_ACTION_DIM");
    if (!steps_env || !dim_env) {
        std::printf("SKIP - FLASHRT_PI0_ACTION_STEPS/ACTION_DIM not set\n");
        return 0;
    }
    // Backend is "cpu" by default (byte-identical to the original test). Set
    // FLASHRT_PI0_BACKEND=cuda to run the real forward pass on the GPU (the
    // Jetson-PI engine maps backend=="cuda" to n_gpu_layers=9999 and use_gpu
    // for the mmproj). CUDA_VISIBLE_DEVICES selects the physical card.
    // Applied to the real-Pi0 JSON only (sub-test B); the bogus-path and
    // action_dim-mismatch sub-tests keep "cpu" hardcoded so they stay cheap
    // and deterministic.
    const char * be_env = std::getenv("FLASHRT_PI0_BACKEND");
    const std::string backend = (be_env && be_env[0]) ? std::string(be_env)
                                                     : std::string("cpu");
    long action_steps = std::atol(steps_env);
    long action_dim   = std::atol(dim_env);
    if (action_steps <= 0 || action_dim <= 0 || action_steps > 10000 ||
        action_dim > 10000) {
        std::printf("SKIP - bad FLASHRT_PI0_ACTION_STEPS/DIM\n");
        return 0;
    }

    // ---- sub-test C: config vs model action_dim mismatch fails cleanly ----
    // The selected model uses the configured shape; claim a wrong action_dim
    // and expect create_pi0 to reject without leaking the opened engine.
    // Backend is hardcoded "cpu" here on purpose: this path only exercises the
    // action_dim rejection, so paying for a full GPU model load + 37-layer
    // offload (which FLASHRT_PI0_BACKEND=cuda would trigger) is wasteful and
    // nondeterministic. The env-driven backend below is for the real forward
    // pass only.
    {
        std::string mismatch_json =
            std::string("{") +
            "\"model_family\":\"pi0\","
            "\"model_path\":\"" + model_env + "\","
            "\"mmproj_path\":\"" + mmproj_env + "\","
            "\"backend\":\"cpu\","
            "\"n_views\":2,\"image_height\":224,\"image_width\":224,"
            "\"image_channels\":3,\"action_steps\":" +
            std::to_string(action_steps) +
            ",\"action_dim\":" + std::to_string(action_dim + 1) + "}";
        frt_model_runtime_v1 * m = nullptr;
        int mrc = frt_llama_cpp_pi0_runtime_open_with_engine_factory(
            mismatch_json.c_str(), factory, &m);
        CHECK(mrc != 0 && m == nullptr,
              "open rejects action_dim mismatch with model");
    }

    std::string json =
        std::string("{") +
        "\"model_family\":\"pi0\","
        "\"model_path\":\"" + model_env + "\","
        "\"mmproj_path\":\"" + mmproj_env + "\","
        "\"backend\":\"" + backend + "\","
        "\"n_views\":2,\"image_height\":224,\"image_width\":224,"
        "\"image_channels\":3,\"action_steps\":" + std::to_string(action_steps) +
        ",\"action_dim\":" + std::to_string(action_dim) + "}";
    frt_model_runtime_v1 * model = nullptr;
    rc = frt_llama_cpp_pi0_runtime_open_with_engine_factory(
        json.c_str(), factory, &model);
    if (rc != 0 || !model) {
        std::printf("FAIL: open real Pi0 runtime (rc=%d): %s\n", rc,
                    factory->last_error(factory->self));
        return 1;
    }
    CHECK(rc == 0 && model != nullptr, "open real Pi0 runtime from JSON");

    // Load fixture.
    int iw = 0, ih = 0, ic = 0;
    unsigned char * img = stbi_load(img_path.c_str(), &iw, &ih, &ic, 3);
    CHECK(img != nullptr && iw == 224 && ih == 224, "load image.png 224x224");
    int ww = 0, wh = 0, wc = 0;
    unsigned char * wrist = stbi_load(wrist_path.c_str(), &ww, &wh, &wc, 3);
    CHECK(wrist != nullptr && ww == 224 && wh == 224, "load wrist_image.png 224x224");
    bool state_ok = false;
    std::string state_bytes = read_file(state_path, &state_ok);
    CHECK(state_ok && state_bytes.size() ==
                      static_cast<size_t>(action_dim) * sizeof(float),
          "state.bin matches action_dim float32");
    bool prompt_ok = false;
    std::string prompt = read_file(prompt_path, &prompt_ok);
    CHECK(prompt_ok && !prompt.empty(), "prompt.txt non-empty");
    if (!prompt.empty() && prompt.back() == '\n') prompt.pop_back();

    if (img && wrist && state_ok && prompt_ok) {
        frt_image_view views[2];
        views[0].struct_size = sizeof(frt_image_view);
        views[0].pixel_format = FRT_RT_PIXEL_RGB8;
        views[0].data = img;
        views[0].bytes = static_cast<uint64_t>(224) * 224 * 3;
        views[0].width = 224; views[0].height = 224; views[0].stride_bytes = 0;
        views[0].reserved = 0; views[0].timestamp_ns = 0;
        views[1] = views[0];
        views[1].data = wrist;

        CHECK(model->verbs.set_input(model->self,
                                        FRT_LLAMA_CPP_PI0_PORT_IMAGES,
                                        &views[0], sizeof(views), -1) == 0,
              "set_input images");
        CHECK(model->verbs.set_input(model->self,
                                        FRT_LLAMA_CPP_PI0_PORT_PROMPT,
                                        prompt.data(), prompt.size(), -1) == 0,
              "set_input prompt");
        CHECK(model->verbs.set_input(model->self,
                                        FRT_LLAMA_CPP_PI0_PORT_STATE,
                                        state_bytes.data(),
                                        state_bytes.size(), -1) == 0,
              "set_input state");

        rc = model->verbs.step(model->self);
        if (rc != 0) {
            std::printf("FAIL: run_stage infer (rc=%d): %s\n", rc,
                        model->verbs.last_error(model->self));
            g_fail = 1;
        } else {
            std::printf("ok  : run_stage infer\n");
        }

        std::vector<float> actions(
            static_cast<size_t>(action_steps) * action_dim);
        uint64_t written = 0;
        rc = model->verbs.get_output(
            model->self, FRT_LLAMA_CPP_PI0_PORT_ACTIONS,
            actions.data(), actions.size() * sizeof(float), &written, -1);
        CHECK(rc == 0 &&
                  written == static_cast<uint64_t>(action_steps) * action_dim *
                                 sizeof(float),
              "get_output actions shape matches config");
        if (rc == 0) {
            bool nan_inf = false, all_zero = true;
            for (float v : actions) {
                if (std::isnan(v) || std::isinf(v)) { nan_inf = true; break; }
                if (v != 0.0f) all_zero = false;
            }
            CHECK(!nan_inf, "actions contain no NaN/Inf");
            CHECK(!all_zero, "actions are not all zero");
        }

        // ---- multi-tick: a second infer with fresh inputs must not leak the
        // first tick's KV (KV reset) nor return the first tick's actions
        // (staleness guard). We reuse the same inputs; the action should
        // reproduce within tolerance, and a get_output before run_infer (after
        // set_input) must return -7 (actions not ready). ----
        CHECK(model->verbs.set_input(model->self,
                                        FRT_LLAMA_CPP_PI0_PORT_IMAGES,
                                        &views[0], sizeof(views), -1) == 0,
              "tick2 set_input images");
        CHECK(model->verbs.set_input(model->self,
                                        FRT_LLAMA_CPP_PI0_PORT_PROMPT,
                                        prompt.data(), prompt.size(), -1) == 0,
              "tick2 set_input prompt");
        CHECK(model->verbs.set_input(model->self,
                                        FRT_LLAMA_CPP_PI0_PORT_STATE,
                                        state_bytes.data(),
                                        state_bytes.size(), -1) == 0,
              "tick2 set_input state");
        // set_input must have invalidated actions_buf: get_output now -7.
        {
            uint64_t w2 = 0;
            int rc2 = model->verbs.get_output(
                model->self, FRT_LLAMA_CPP_PI0_PORT_ACTIONS,
                actions.data(), actions.size() * sizeof(float), &w2, -1);
            CHECK(rc2 != 0,
                  "get_output after set_input (before run_infer) fails");
        }
        CHECK(model->verbs.step(model->self) == 0,
              "tick2 run_stage infer");
        std::vector<float> actions2(
            static_cast<size_t>(action_steps) * action_dim);
        written = 0;
        rc = model->verbs.get_output(
            model->self, FRT_LLAMA_CPP_PI0_PORT_ACTIONS,
            actions2.data(), actions2.size() * sizeof(float), &written, -1);
        CHECK(rc == 0 &&
                  written == static_cast<uint64_t>(action_steps) * action_dim *
                                 sizeof(float),
              "tick2 get_output actions shape matches config");
        if (rc == 0) {
            // Same inputs -> deterministically reproducible action chunk
            // (Pi0 with fixed seed noise). Verify the two ticks match.
            bool match = actions.size() == actions2.size();
            for (size_t i = 0; match && i < actions.size(); ++i) {
                if (std::fabs(actions[i] - actions2[i]) > 1e-5f) match = false;
            }
            CHECK(match, "tick2 actions reproduce tick1 (KV reset, no leak)");
        }
    }

    model->release(model->owner);
    if (img)   stbi_image_free(img);
    if (wrist) stbi_image_free(wrist);

    std::printf(g_fail ? "\n== JETSON_PI ENGINE FAILED ==\n"
                       : "\n== JETSON_PI ENGINE PASSED ==\n");
    return g_fail;
}
