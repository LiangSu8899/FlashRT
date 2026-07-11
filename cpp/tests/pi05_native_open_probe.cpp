#include "flashrt/model_runtime.h"
#include "flashrt/cpp/modalities/types.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

extern "C" int frt_model_runtime_open_v1(const char* config_json,
                                          frt_model_runtime_v1** out);
extern "C" const char* frt_pi05_native_open_last_error();

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: pi05_native_open_probe CHECKPOINT TOKENIZER\n";
        return 2;
    }
    int replay_count = 1;
    const char* replay_env = std::getenv("FLASHRT_PROFILE_REPLAYS");
    if (replay_env) {
        char* end = nullptr;
        const long parsed = std::strtol(replay_env, &end, 10);
        if (!end || *end != '\0' || parsed <= 0 || parsed > 100000) {
            std::cerr << "FLASHRT_PROFILE_REPLAYS must be in [1, 100000]\n";
            return 2;
        }
        replay_count = static_cast<int>(parsed);
    }
    const bool profile_service_loop =
        std::getenv("FLASHRT_PROFILE_SERVICE_LOOP") != nullptr;
    if (profile_service_loop && !replay_env) {
        std::cerr << "FLASHRT_PROFILE_SERVICE_LOOP requires "
                     "FLASHRT_PROFILE_REPLAYS\n";
        return 2;
    }
    int hot_state_updates = 0;
    const char* hot_updates_env = std::getenv("FLASHRT_HOT_STATE_UPDATES");
    if (hot_updates_env) {
        char* end = nullptr;
        const long parsed = std::strtol(hot_updates_env, &end, 10);
        if (!end || *end != '\0' || parsed <= 0 || parsed > 100000) {
            std::cerr << "FLASHRT_HOT_STATE_UPDATES must be in [1, 100000]\n";
            return 2;
        }
        hot_state_updates = static_cast<int>(parsed);
    }
    double hot_state_p99_limit_us = 0.0;
    const char* hot_limit_env = std::getenv("FLASHRT_HOT_STATE_P99_US");
    if (hot_limit_env) {
        char* end = nullptr;
        hot_state_p99_limit_us = std::strtod(hot_limit_env, &end);
        if (!end || *end != '\0' || !std::isfinite(hot_state_p99_limit_us) ||
            hot_state_p99_limit_us <= 0.0) {
            std::cerr << "FLASHRT_HOT_STATE_P99_US must be positive\n";
            return 2;
        }
    }
    std::ostringstream json;
    json << "{\"io\":\"native_v2\",\"checkpoint_path\":\""
         << argv[1] << "\",\"tokenizer_model_path\":\"" << argv[2]
         << "\",\"state_prompt_mode\":\"fixed\","
         << "\"max_prompt_tokens\":200,\"state_dim\":8,"
         << "\"num_views\":2,\"chunk\":10,\"num_steps\":10,"
         << "\"vision_pool_factor\":1}";
    frt_model_runtime_v1* model = nullptr;
    const int open_rc = frt_model_runtime_open_v1(json.str().c_str(), &model);
    if (open_rc != 0 || !model) {
        std::cerr << "native open failed: rc=" << open_rc << " error="
                  << frt_pi05_native_open_last_error() << '\n';
        return 1;
    }
    const char* port_names[] = {
        "prompt", "state", "images", "noise", "actions", "actions_raw"};
    const frt_runtime_export_v1* exp = model->exp;
    int active_device = 0;
    cudaDeviceProp active_properties{};
    const bool device_identity_ok =
        cudaGetDevice(&active_device) == cudaSuccess &&
        cudaGetDeviceProperties(&active_properties, active_device) ==
            cudaSuccess;
    const std::string hardware_id = device_identity_ok
        ? "sm" + std::to_string(active_properties.major * 10 +
                                active_properties.minor)
        : std::string();
    const std::string hardware_identity = "hardware=" + hardware_id;
    const std::string hardware_manifest =
        "\"hardware\":\"" + hardware_id + "\"";
    bool ok = model->abi_version == FRT_MODEL_RUNTIME_ABI_VERSION &&
              model->struct_size == sizeof(frt_model_runtime_v1) && exp &&
              exp->abi_version == FRT_RUNTIME_ABI_VERSION &&
              exp->struct_size == sizeof(frt_runtime_export_v1) &&
              model->n_ports == 6 && model->n_stages == 1 &&
              exp->n_graphs == 1 && exp->n_streams == 1 &&
              exp->n_capsule_regions == 1 && exp->n_buffers == 7 &&
              exp->fingerprint != 0 && exp->identity &&
              std::strstr(exp->identity, "producer=native") &&
              device_identity_ok &&
              std::strstr(exp->identity, hardware_identity.c_str()) &&
              exp->manifest_json &&
              std::strstr(exp->manifest_json, hardware_manifest.c_str()) &&
              std::strstr(exp->identity, "weights_sha256=") &&
              std::strstr(exp->identity, "tokenizer_sha256=") &&
              model->stages[0].graph == 0 &&
              frt_graph_variant_count(exp->graphs[0].handle) == 1;
    for (std::uint64_t i = 0; i < model->n_ports; ++i) {
        ok = ok && std::strcmp(model->ports[i].name, port_names[i]) == 0;
    }
    ok = ok &&
         std::strcmp(exp->capsule_regions[0].name, "rollout_boundary") == 0 &&
         model->ports[0].modality == FRT_RT_MOD_TEXT &&
         model->ports[0].update == FRT_RT_PORT_STAGED &&
         model->ports[1].modality == FRT_RT_MOD_STATE &&
         model->ports[2].modality == FRT_RT_MOD_IMAGE &&
         model->ports[3].update == FRT_RT_PORT_SWAP &&
         model->ports[4].direction == FRT_RT_PORT_OUT &&
         model->ports[4].dtype == FRT_RT_DTYPE_F32 &&
         model->ports[4].update == FRT_RT_PORT_STAGED &&
         model->ports[4].buffer == nullptr &&
         model->ports[4].bytes == 10 * 7 * sizeof(float) &&
         model->ports[5].dtype == FRT_RT_DTYPE_BF16 &&
         model->ports[5].update == FRT_RT_PORT_SWAP &&
         model->ports[5].buffer == model->ports[3].buffer &&
         model->ports[5].offset == model->ports[3].offset &&
         model->ports[5].bytes == model->ports[3].bytes;
    if (!ok) {
        std::cerr << "native schema validation failed\n";
        model->release(model->owner);
        return 1;
    }
    const char* schema_output = std::getenv("FLASHRT_SCHEMA_OUTPUT");
    if (schema_output && schema_output[0] != '\0') {
        std::ofstream output(schema_output);
        std::istringstream identity(exp->identity ? exp->identity : "");
        std::string line;
        while (std::getline(identity, line)) {
            if (line.compare(0, 7, "region:") == 0 ||
                line.compare(0, 5, "port:") == 0 ||
                line.compare(0, 6, "stage:") == 0) {
                output << line << '\n';
            }
        }
        if (!output) {
            std::cerr << "native schema output failed\n";
            model->release(model->owner);
            return 1;
        }
        if (std::getenv("FLASHRT_SCHEMA_ONLY")) {
            model->release(model->owner);
            std::cout << "PASS native schema export\n";
            return 0;
        }
    }
    if (model->verbs.prepare(model->self, 0, 0) != 0 ||
        model->verbs.prepare(model->self, 0, 1) != -2) {
        std::cerr << "native prepare validation failed\n";
        model->release(model->owner);
        return 1;
    }

    const std::string prompt = "pick up the black bowl";
    float state[8] = {0.1f, -0.2f, 0.3f, -0.4f,
                      0.5f, -0.6f, 0.7f, -0.8f};
    if (model->verbs.set_input(model->self, 0, prompt.data(), prompt.size(),
                               -1) != 0 ||
        model->verbs.set_input(model->self, 1, state, sizeof(state), -1) != 0) {
        std::cerr << "native prompt/state staging failed: "
                  << model->verbs.last_error(model->self) << '\n';
        model->release(model->owner);
        return 1;
    }
    if (hot_state_updates) {
        constexpr int kWarmUpdates = 20;
        std::vector<double> hot_state_latencies;
        hot_state_latencies.reserve(hot_state_updates);
        for (int update = -kWarmUpdates; update < hot_state_updates; ++update) {
            for (int dim = 0; dim < 8; ++dim) {
                state[dim] = std::sin(
                    static_cast<float>((update + kWarmUpdates) * 8 + dim) *
                    0.017f);
            }
            const auto begin = std::chrono::steady_clock::now();
            const int rc = model->verbs.set_input(
                model->self, 1, state, sizeof(state), -1);
            const auto end = std::chrono::steady_clock::now();
            if (rc != 0) {
                std::cerr << "native hot state update failed: "
                          << model->verbs.last_error(model->self) << '\n';
                model->release(model->owner);
                return 1;
            }
            if (update >= 0) {
                hot_state_latencies.push_back(
                    std::chrono::duration<double, std::micro>(end - begin)
                        .count());
            }
        }
        std::sort(hot_state_latencies.begin(), hot_state_latencies.end());
        const std::size_t p99_index =
            (hot_state_latencies.size() * 99 + 99) / 100 - 1;
        const double p50 = hot_state_latencies[hot_state_latencies.size() / 2];
        const double p99 = hot_state_latencies[p99_index];
        const double maximum = hot_state_latencies.back();
        std::cout << "hot state updates: n=" << hot_state_latencies.size()
                  << " p50_us=" << p50 << " p99_us=" << p99
                  << " max_us=" << maximum << '\n';
        if ((hot_state_p99_limit_us > 0.0 &&
             p99 > hot_state_p99_limit_us) ||
            frt_graph_variant_count(exp->graphs[0].handle) != 1) {
            std::cerr << "native hot state update gate failed\n";
            model->release(model->owner);
            return 1;
        }
    }
    std::vector<std::uint8_t> rgb0(224 * 224 * 3);
    std::vector<std::uint8_t> rgb1(rgb0.size());
    for (std::size_t i = 0; i < rgb0.size(); ++i) {
        rgb0[i] = static_cast<std::uint8_t>(i % 251);
        rgb1[i] = static_cast<std::uint8_t>((3 * i + 17) % 251);
    }
    frt_image_view views[2]{};
    for (int i = 0; i < 2; ++i) {
        views[i].struct_size = sizeof(frt_image_view);
        views[i].pixel_format = FRT_RT_PIXEL_RGB8;
        views[i].data = i ? static_cast<const void*>(rgb1.data())
                          : static_cast<const void*>(rgb0.data());
        views[i].bytes = rgb0.size();
        views[i].width = 224;
        views[i].height = 224;
        views[i].stride_bytes = 224 * 3;
    }
    if (model->verbs.set_input(model->self, 2, views, sizeof(views), -1) != 0) {
        std::cerr << "native image staging failed: "
                  << model->verbs.last_error(model->self) << '\n';
        model->release(model->owner);
        return 1;
    }
    frt_buffer noise = model->ports[3].buffer;
    std::vector<std::uint16_t> host_noise(10 * 32);
    for (std::size_t i = 0; i < host_noise.size(); ++i) {
        host_noise[i] = flashrt::modalities::float_to_bfloat16(
            static_cast<float>(static_cast<int>(i % 23) - 11) / 12.0f);
    }
    const bool profile_range = replay_env ||
        std::getenv("FLASHRT_PROFILE_RANGE") != nullptr;
    float actions[10 * 7]{};
    std::uint64_t written = 0;
    if (!noise || model->verbs.set_input(model->self, 3, host_noise.data(),
                                         host_noise.size() * 2, -1) != -3 ||
        cudaMemcpy(frt_buffer_dptr(noise), host_noise.data(),
                   host_noise.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        (profile_range && cudaProfilerStart() != cudaSuccess)) {
        std::cerr << "native step failed: "
                  << model->verbs.last_error(model->self) << '\n';
        model->release(model->owner);
        return 1;
    }
    int step_rc = 0;
    cudaError_t upload_rc = cudaSuccess;
    for (int replay = 0; replay < replay_count; ++replay) {
        if (profile_service_loop) {
            for (int dim = 0; dim < 8; ++dim) {
                state[dim] = std::sin(
                    static_cast<float>(replay * 8 + dim) * 0.017f);
            }
            const char* live_prompt = replay % 2 == 0
                ? "pick up the black bowl"
                : "move the black bowl to the plate";
            if (model->verbs.set_input(
                    model->self, 0, live_prompt, std::strlen(live_prompt),
                    -1) != 0 ||
                model->verbs.set_input(
                    model->self, 1, state, sizeof(state), -1) != 0 ||
                model->verbs.set_input(
                    model->self, 2, views, sizeof(views), -1) != 0) {
                step_rc = -1;
                break;
            }
        }
        if (replay != 0 || profile_service_loop) {
            upload_rc = cudaMemcpy(
                frt_buffer_dptr(noise), host_noise.data(),
                host_noise.size() * sizeof(std::uint16_t),
                cudaMemcpyHostToDevice);
            if (upload_rc != cudaSuccess) break;
        }
        step_rc = model->verbs.step(model->self);
        if (step_rc != 0) break;
        if (profile_service_loop &&
            (model->verbs.get_output(
                 model->self, 4, actions, sizeof(actions), &written, -1) != 0 ||
             written != sizeof(actions))) {
            step_rc = -1;
            break;
        }
    }
    const cudaError_t sync_rc = cudaDeviceSynchronize();
    const cudaError_t profiler_rc =
        profile_range ? cudaProfilerStop() : cudaSuccess;
    if (step_rc != 0 || upload_rc != cudaSuccess || sync_rc != cudaSuccess ||
        profiler_rc != cudaSuccess ||
        frt_graph_variant_count(exp->graphs[0].handle) != 1) {
        std::cerr << "native step failed: "
                  << model->verbs.last_error(model->self) << '\n';
        model->release(model->owner);
        return 1;
    }
    if (model->verbs.get_output(model->self, 4, actions, sizeof(actions),
                                &written, -1) != 0 ||
        written != sizeof(actions)) {
        std::cerr << "native action output failed: "
                  << model->verbs.last_error(model->self) << '\n';
        model->release(model->owner);
        return 1;
    }
    for (float value : actions) {
        if (!std::isfinite(value)) {
            std::cerr << "native action output is not finite\n";
            model->release(model->owner);
            return 1;
        }
    }
    model->retain(model->owner);
    model->release(model->owner);
    model->release(model->owner);
    std::cout << "PASS native open_v1 full lifecycle\n";
    return 0;
}
