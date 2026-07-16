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

namespace {

constexpr int kLatencyWarmupReplays = 10;

bool all_graph_variants_stable(const frt_runtime_export_v1* exp) {
    for (std::uint64_t graph = 0; graph < exp->n_graphs; ++graph) {
        if (frt_graph_variant_count(exp->graphs[graph].handle) != 1) {
            return false;
        }
    }
    return true;
}

const frt_runtime_buffer_desc* find_buffer(
    const frt_runtime_export_v1* exp,
    const char* name) {
    if (!exp || !name) return nullptr;
    for (std::uint64_t index = 0; index < exp->n_buffers; ++index) {
        if (std::strcmp(exp->buffers[index].name, name) == 0) {
            return &exp->buffers[index];
        }
    }
    return nullptr;
}

bool write_diagnostics(
    const frt_runtime_export_v1* exp,
    const char* path) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) return false;
    for (const char* name : {
             "observation_images_normalized",
             "prompt_embedding",
             "encoder_x"}) {
        const frt_runtime_buffer_desc* buffer = find_buffer(exp, name);
        if (!buffer || !buffer->handle || !buffer->bytes) return false;
        std::vector<unsigned char> host(
            static_cast<std::size_t>(buffer->bytes));
        if (cudaMemcpy(host.data(), frt_buffer_dptr(buffer->handle),
                       host.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
        output.write(
            reinterpret_cast<const char*>(host.data()),
            static_cast<std::streamsize>(host.size()));
        if (!output) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: pi05_native_open_probe CHECKPOINT TOKENIZER "
                     "[CALIBRATION]\n";
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
    bool profile_service_loop =
        std::getenv("FLASHRT_PROFILE_SERVICE_LOOP") != nullptr;
    int latency_replays = 0;
    const char* latency_env =
        std::getenv("FLASHRT_E2E_LATENCY_REPLAYS");
    if (latency_env) {
        char* end = nullptr;
        const long parsed = std::strtol(latency_env, &end, 10);
        if (!end || *end != '\0' || parsed <= 0 || parsed > 100000) {
            std::cerr << "FLASHRT_E2E_LATENCY_REPLAYS must be in "
                         "[1, 100000]\n";
            return 2;
        }
        if (replay_env) {
            std::cerr << "FLASHRT_E2E_LATENCY_REPLAYS cannot be combined "
                         "with FLASHRT_PROFILE_REPLAYS\n";
            return 2;
        }
        latency_replays = static_cast<int>(parsed);
        replay_count = kLatencyWarmupReplays + latency_replays;
        profile_service_loop = true;
    }
    if (profile_service_loop && !replay_env && !latency_env) {
        std::cerr << "FLASHRT_PROFILE_SERVICE_LOOP requires "
                     "FLASHRT_PROFILE_REPLAYS\n";
        return 2;
    }
    double latency_p99_limit_ms = 0.0;
    const char* latency_limit_env = std::getenv("FLASHRT_E2E_P99_MS");
    if (latency_limit_env) {
        char* end = nullptr;
        latency_p99_limit_ms = std::strtod(latency_limit_env, &end);
        if (!end || *end != '\0' || !std::isfinite(latency_p99_limit_ms) ||
            latency_p99_limit_ms <= 0.0 || latency_replays == 0) {
            std::cerr << "FLASHRT_E2E_P99_MS must be positive and requires "
                         "FLASHRT_E2E_LATENCY_REPLAYS\n";
            return 2;
        }
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
    const char* stage_plan_env = std::getenv("FLASHRT_STAGE_PLAN");
    const std::string stage_plan =
        stage_plan_env ? stage_plan_env : "full";
    if (stage_plan != "full" && stage_plan != "context_action") {
        std::cerr << "FLASHRT_STAGE_PLAN must be full or context_action\n";
        return 2;
    }
    const bool split_stage_plan = stage_plan == "context_action";
    int num_views = 2;
    const char* num_views_env = std::getenv("FLASHRT_NUM_VIEWS");
    if (num_views_env) {
        char* end = nullptr;
        const long parsed = std::strtol(num_views_env, &end, 10);
        if (!end || *end != '\0' || parsed < 1 || parsed > 3) {
            std::cerr << "FLASHRT_NUM_VIEWS must be in [1, 3]\n";
            return 2;
        }
        num_views = static_cast<int>(parsed);
    }
    int max_prompt_tokens = 200;
    const char* max_prompt_env =
        std::getenv("FLASHRT_MAX_PROMPT_TOKENS");
    if (max_prompt_env) {
        char* end = nullptr;
        const long parsed = std::strtol(max_prompt_env, &end, 10);
        if (!end || *end != '\0' || parsed < 1 || parsed > 100000) {
            std::cerr << "FLASHRT_MAX_PROMPT_TOKENS must be in "
                         "[1, 100000]\n";
            return 2;
        }
        max_prompt_tokens = static_cast<int>(parsed);
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
         << "\"max_prompt_tokens\":" << max_prompt_tokens
         << ",\"state_dim\":8,"
         << "\"num_views\":" << num_views
         << ",\"chunk\":10,\"num_steps\":10,"
         << "\"vision_pool_factor\":1,\"stage_plan\":\""
         << stage_plan << '"';
    if (argc == 4) {
        json << ",\"calibration_path\":\"" << argv[3] << '"';
    }
    json << '}';
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
    const uint32_t expected_io_dtype = hardware_id == "sm110"
        ? FRT_RT_DTYPE_F16
        : FRT_RT_DTYPE_BF16;
    bool ok = model->abi_version == FRT_MODEL_RUNTIME_ABI_VERSION &&
              model->struct_size == sizeof(frt_model_runtime_v1) && exp &&
              exp->abi_version == FRT_RUNTIME_ABI_VERSION &&
              exp->struct_size == sizeof(frt_runtime_export_v1) &&
              model->n_ports == 6 &&
              model->n_stages == (split_stage_plan ? 2u : 1u) &&
              exp->n_graphs == 3 && exp->n_streams == 1 &&
              exp->n_capsule_regions == 1 && exp->n_buffers == 7 &&
              exp->fingerprint != 0 && exp->identity &&
              std::strstr(exp->identity, "producer=native") &&
              device_identity_ok &&
              std::strstr(exp->identity, hardware_identity.c_str()) &&
              exp->manifest_json &&
              std::strstr(exp->manifest_json, hardware_manifest.c_str()) &&
              std::strstr(exp->identity, "weights_sha256=") &&
              std::strstr(exp->identity, "tokenizer_sha256=") &&
              model->stages[0].graph == (split_stage_plan ? 2u : 0u) &&
              std::strcmp(exp->graphs[0].name, "infer") == 0 &&
              std::strcmp(exp->graphs[1].name, "decode_only") == 0 &&
              std::strcmp(exp->graphs[2].name, "context") == 0;
    ok = ok && all_graph_variants_stable(exp);
    if (split_stage_plan) {
        ok = ok && model->stages[0].n_after == 0 &&
             model->stages[1].graph == 1 &&
             model->stages[1].n_after == 1 &&
             model->stages[1].after[0] == 0;
    }
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
         model->ports[2].dtype == expected_io_dtype &&
         model->ports[3].dtype == expected_io_dtype &&
         model->ports[5].dtype == expected_io_dtype &&
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
        model->verbs.prepare(model->self, 1, 0) != 0 ||
        model->verbs.prepare(model->self, 2, 0) != 0 ||
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
            !all_graph_variants_stable(exp)) {
            std::cerr << "native hot state update gate failed\n";
            model->release(model->owner);
            return 1;
        }
    }
    std::vector<std::vector<std::uint8_t>> rgb(
        num_views, std::vector<std::uint8_t>(224 * 224 * 3));
    for (int view = 0; view < num_views; ++view) {
        for (std::size_t i = 0; i < rgb[view].size(); ++i) {
            rgb[view][i] = static_cast<std::uint8_t>(
                ((2 * view + 1) * i + 17 * view) % 251);
        }
    }
    std::vector<frt_image_view> views(num_views);
    for (int i = 0; i < num_views; ++i) {
        views[i].struct_size = sizeof(frt_image_view);
        views[i].pixel_format = FRT_RT_PIXEL_RGB8;
        views[i].data = rgb[i].data();
        views[i].bytes = rgb[i].size();
        views[i].width = 224;
        views[i].height = 224;
        views[i].stride_bytes = 224 * 3;
    }
    if (model->verbs.set_input(
            model->self, 2, views.data(),
            views.size() * sizeof(frt_image_view), -1) != 0) {
        std::cerr << "native image staging failed: "
                  << model->verbs.last_error(model->self) << '\n';
        model->release(model->owner);
        return 1;
    }
    frt_buffer noise = model->ports[3].buffer;
    std::vector<std::uint16_t> host_noise(10 * 32);
    for (std::size_t i = 0; i < host_noise.size(); ++i) {
        const float value =
            static_cast<float>(static_cast<int>(i % 23) - 11) / 12.0f;
        host_noise[i] = expected_io_dtype == FRT_RT_DTYPE_F16
            ? flashrt::modalities::float_to_float16(value)
            : flashrt::modalities::float_to_bfloat16(value);
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
    cudaError_t replay_sync_rc = cudaSuccess;
    std::vector<double> latency_samples_ms;
    latency_samples_ms.reserve(latency_replays);
    for (int replay = 0; replay < replay_count; ++replay) {
        const auto replay_begin = std::chrono::steady_clock::now();
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
                    model->self, 2, views.data(),
                    views.size() * sizeof(frt_image_view), -1) != 0) {
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
        if (latency_replays != 0) {
            replay_sync_rc = cudaDeviceSynchronize();
            if (replay_sync_rc != cudaSuccess) break;
            if (replay >= kLatencyWarmupReplays) {
                const auto replay_end = std::chrono::steady_clock::now();
                latency_samples_ms.push_back(
                    std::chrono::duration<double, std::milli>(
                        replay_end - replay_begin)
                        .count());
            }
        }
    }
    const cudaError_t sync_rc = cudaDeviceSynchronize();
    const cudaError_t profiler_rc =
        profile_range ? cudaProfilerStop() : cudaSuccess;
    if (step_rc != 0 || upload_rc != cudaSuccess ||
        replay_sync_rc != cudaSuccess || sync_rc != cudaSuccess ||
        profiler_rc != cudaSuccess || !all_graph_variants_stable(exp)) {
        std::cerr << "native step failed: "
                  << model->verbs.last_error(model->self) << '\n';
        model->release(model->owner);
        return 1;
    }
    if (latency_replays != 0) {
        std::sort(latency_samples_ms.begin(), latency_samples_ms.end());
        const std::size_t p99_index =
            (latency_samples_ms.size() * 99 + 99) / 100 - 1;
        const double p50 = latency_samples_ms[latency_samples_ms.size() / 2];
        const double p99 = latency_samples_ms[p99_index];
        const double maximum = latency_samples_ms.back();
        std::cout << "native service latency: n=" << latency_samples_ms.size()
                  << " p50_ms=" << p50 << " p99_ms=" << p99
                  << " max_ms=" << maximum << '\n';
        if (latency_p99_limit_ms > 0.0 && p99 > latency_p99_limit_ms) {
            std::cerr << "native service latency gate failed\n";
            model->release(model->owner);
            return 1;
        }
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
    const char* raw_output = std::getenv("FLASHRT_RAW_ACTION_OUTPUT");
    if (raw_output && raw_output[0] != '\0') {
        std::vector<std::uint16_t> raw(host_noise.size());
        if (!model->ports[5].buffer ||
            cudaMemcpy(raw.data(), frt_buffer_dptr(model->ports[5].buffer),
                       raw.size() * sizeof(raw[0]),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::cerr << "native raw action download failed\n";
            model->release(model->owner);
            return 1;
        }
        std::ofstream output(raw_output, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(raw.data()),
                     static_cast<std::streamsize>(
                         raw.size() * sizeof(raw[0])));
        if (!output) {
            std::cerr << "native raw action output failed\n";
            model->release(model->owner);
            return 1;
        }
    }
    const char* diagnostic_output =
        std::getenv("FLASHRT_DIAGNOSTIC_OUTPUT");
    if (diagnostic_output && diagnostic_output[0] != '\0' &&
        !write_diagnostics(exp, diagnostic_output)) {
        std::cerr << "native diagnostic output failed\n";
        model->release(model->owner);
        return 1;
    }
    model->retain(model->owner);
    model->release(model->owner);
    model->release(model->owner);
    std::cout << "PASS native open_v1 full lifecycle\n";
    return 0;
}
