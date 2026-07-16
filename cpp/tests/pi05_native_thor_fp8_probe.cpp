#include "flashrt/model_runtime.h"
#include "flashrt/cpp/modalities/types.h"
#include "flashrt/cpp/models/pi05/c_api.h"
#include "flashrt/cpp/models/pi05/native_calibration.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

extern "C" int frt_model_runtime_open_v1(const char* config_json,
                                          frt_model_runtime_v1** out);
extern "C" const char* frt_pi05_native_open_last_error();

namespace {

std::string json_string(const std::string& value) {
    std::string output = "\"";
    for (char c : value) {
        if (c == '\\' || c == '"') output.push_back('\\');
        output.push_back(c);
    }
    output.push_back('"');
    return output;
}

std::string config_json(const std::string& checkpoint,
                        const std::string& tokenizer,
                        int num_views,
                        int max_prompt_tokens,
                        const std::string& calibration = "") {
    std::ostringstream out;
    out << "{\"io\":\"native_v2\",\"checkpoint_path\":"
        << json_string(checkpoint) << ",\"tokenizer_model_path\":"
        << json_string(tokenizer)
        << ",\"state_prompt_mode\":\"fixed\","
           "\"precision\":\"fp8_e4m3fn\","
           "\"max_prompt_tokens\":"
        << max_prompt_tokens << ",\"state_dim\":8,\"num_views\":"
        << num_views << ",\"chunk\":10,\"num_steps\":10,"
           "\"vision_pool_factor\":1";
    if (!calibration.empty()) {
        out << ",\"calibration_path\":" << json_string(calibration);
    }
    out << '}';
    return out.str();
}

int calibration_error(frt_pi05_calibration_session* session,
                      const char* operation,
                      int rc) {
    std::cerr << operation << " failed: rc=" << rc << " error="
              << (session ? frt_pi05_calibration_last_error_v1(session)
                          : frt_pi05_calibration_create_last_error_v1())
              << '\n';
    if (session) frt_pi05_calibration_destroy_v1(session);
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) {
        std::cerr << "usage: pi05_native_thor_fp8_probe CHECKPOINT TOKENIZER "
                     "ARTIFACT SAMPLES VIEWS [RAW_ACTION_OUTPUT]\n";
        return 2;
    }
    const int samples = std::atoi(argv[4]);
    if (samples < 1 || samples > 256) {
        std::cerr << "SAMPLES must be in [1, 256]\n";
        return 2;
    }
    const int num_views = std::atoi(argv[5]);
    if (num_views < 1 || num_views > 3) {
        std::cerr << "VIEWS must be in [1, 3]\n";
        return 2;
    }
    int max_prompt_tokens = 200;
    if (const char* value = std::getenv("FLASHRT_MAX_PROMPT_TOKENS")) {
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (!end || *end != '\0' || parsed < 1 || parsed > 100000) {
            std::cerr << "FLASHRT_MAX_PROMPT_TOKENS must be in "
                         "[1, 100000]\n";
            return 2;
        }
        max_prompt_tokens = static_cast<int>(parsed);
    }
    const std::string calibration_path = argv[3];
    const std::string single_path = calibration_path + ".single";
    const std::string calibration_config =
        config_json(argv[1], argv[2], num_views, max_prompt_tokens);
    frt_pi05_calibration_session* session = nullptr;
    int rc = frt_pi05_calibration_create_v1(
        calibration_config.c_str(), 99.9, &session);
    if (rc != 0 || !session) {
        return calibration_error(nullptr, "calibration create", rc);
    }

    std::vector<std::uint8_t> image(224 * 224 * 3);
    std::vector<std::uint8_t> wrist(image.size());
    std::vector<std::uint8_t> right_wrist(image.size());
    float state[8]{};
    std::vector<float> noise(10 * 32);
    std::vector<frt_pi05_vision_frame> frames(num_views);
    const char* names[] = {
        "image", "wrist_image", "wrist_image_right"};
    std::vector<std::uint8_t>* pixels[] = {
        &image, &wrist, &right_wrist};
    for (int view = 0; view < num_views; ++view) {
        frames[view].struct_size = sizeof(frt_pi05_vision_frame);
        frames[view].name = names[view];
        frames[view].data = pixels[view]->data();
        frames[view].bytes = pixels[view]->size();
        frames[view].width = 224;
        frames[view].height = 224;
        frames[view].stride_bytes = 224 * 3;
        frames[view].pixel_format = FRT_PI05_PIXEL_RGB8;
    }
    if (num_views > 1) std::reverse(frames.begin(), frames.end());
    for (int sample_index = 0; sample_index < samples; ++sample_index) {
        for (std::size_t i = 0; i < image.size(); ++i) {
            image[i] = static_cast<std::uint8_t>(
                (i * 3 + sample_index * 17) % 251);
            wrist[i] = static_cast<std::uint8_t>(
                (i * 7 + sample_index * 29 + 11) % 253);
            right_wrist[i] = static_cast<std::uint8_t>(
                (i * 11 + sample_index * 37 + 19) % 247);
        }
        for (int i = 0; i < 8; ++i) {
            state[i] = static_cast<float>(
                (sample_index * 8 + i) % 17 - 8) / 8.0f;
        }
        for (std::size_t i = 0; i < noise.size(); ++i) {
            const int centered = static_cast<int>(
                (static_cast<std::size_t>(sample_index) * noise.size() + i) %
                31) - 15;
            noise[i] = static_cast<float>(centered) / 16.0f;
        }
        frt_pi05_calibration_sample_v1 sample{};
        sample.struct_size = sizeof(sample);
        sample.prompt = sample_index & 1
                            ? "move the black bowl to the plate"
                            : "pick up the black bowl";
        sample.state = state;
        sample.n_state = 8;
        sample.frames = frames.data();
        sample.n_frames = frames.size();
        sample.noise = noise.data();
        sample.n_noise = noise.size();
        sample.noise_seed = 1234;
        if (sample_index == 0) {
            frt_pi05_calibration_sample_v1 incomplete = sample;
            incomplete.n_frames = static_cast<std::uint64_t>(num_views - 1);
            rc = frt_pi05_calibration_observe_v1(session, &incomplete);
            if (rc == 0 ||
                frt_pi05_calibration_sample_count_v1(session) != 0) {
                std::cerr << "incomplete camera set was accepted\n";
                frt_pi05_calibration_destroy_v1(session);
                return 1;
            }
            if (num_views > 1) {
                std::vector<frt_pi05_vision_frame> duplicate = frames;
                duplicate[1].name = duplicate[0].name;
                frt_pi05_calibration_sample_v1 duplicate_names = sample;
                duplicate_names.frames = duplicate.data();
                rc = frt_pi05_calibration_observe_v1(
                    session, &duplicate_names);
                if (rc == 0 ||
                    frt_pi05_calibration_sample_count_v1(session) != 0) {
                    std::cerr <<
                        "duplicate calibration camera name was accepted\n";
                    frt_pi05_calibration_destroy_v1(session);
                    return 1;
                }
            }
            std::vector<frt_pi05_vision_frame> unknown = frames;
            unknown[0].name = "unknown_camera";
            frt_pi05_calibration_sample_v1 unknown_name = sample;
            unknown_name.frames = unknown.data();
            rc = frt_pi05_calibration_observe_v1(session, &unknown_name);
            if (rc == 0 ||
                frt_pi05_calibration_sample_count_v1(session) != 0) {
                std::cerr << "unknown calibration camera name was accepted\n";
                frt_pi05_calibration_destroy_v1(session);
                return 1;
            }
            std::vector<frt_pi05_vision_frame> bgr = frames;
            bgr[0].pixel_format = FRT_PI05_PIXEL_BGR8;
            frt_pi05_calibration_sample_v1 wrong_format = sample;
            wrong_format.frames = bgr.data();
            rc = frt_pi05_calibration_observe_v1(session, &wrong_format);
            if (rc == 0 ||
                frt_pi05_calibration_sample_count_v1(session) != 0) {
                std::cerr << "non-RGB calibration frame was accepted\n";
                frt_pi05_calibration_destroy_v1(session);
                return 1;
            }
            frt_pi05_calibration_sample_v1 malformed_noise = sample;
            malformed_noise.n_noise--;
            rc = frt_pi05_calibration_observe_v1(session, &malformed_noise);
            if (rc == 0 ||
                frt_pi05_calibration_sample_count_v1(session) != 0) {
                std::cerr << "malformed calibration noise was accepted\n";
                frt_pi05_calibration_destroy_v1(session);
                return 1;
            }
        }
        rc = frt_pi05_calibration_observe_v1(session, &sample);
        if (rc != 0) {
            return calibration_error(session, "calibration observe", rc);
        }
        if (sample_index == 0) {
            rc = frt_pi05_calibration_finalize_v1(
                session, single_path.c_str());
            if (rc != 0) {
                return calibration_error(session, "single finalize", rc);
            }
        }
    }
    if (frt_pi05_calibration_sample_count_v1(session) !=
        static_cast<std::uint64_t>(samples)) {
        return calibration_error(session, "sample count", -1);
    }
    rc = frt_pi05_calibration_finalize_v1(
        session, calibration_path.c_str());
    if (rc != 0) {
        return calibration_error(session, "dataset finalize", rc);
    }
    frt_pi05_calibration_sample_v1 generated_noise{};
    generated_noise.struct_size = sizeof(generated_noise);
    generated_noise.prompt = "pick up the black bowl";
    generated_noise.state = state;
    generated_noise.n_state = 8;
    generated_noise.frames = frames.data();
    generated_noise.n_frames = frames.size();
    generated_noise.noise_seed = 4321;
    rc = frt_pi05_calibration_observe_v1(session, &generated_noise);
    if (rc != 0 || frt_pi05_calibration_sample_count_v1(session) !=
                       static_cast<std::uint64_t>(samples + 1)) {
        return calibration_error(session, "generated-noise observe", rc);
    }
    frt_pi05_calibration_destroy_v1(session);

    flashrt::models::pi05::NativeCalibrationArtifact artifact;
    auto status = flashrt::models::pi05::load_native_calibration_artifact(
        single_path, &artifact);
    if (!status.ok_status() || artifact.sample_count != 1) {
        std::cerr << "single calibration artifact validation failed\n";
        return 1;
    }
    status = flashrt::models::pi05::load_native_calibration_artifact(
        calibration_path, &artifact);
    if (!status.ok_status() || artifact.sample_count !=
                                   static_cast<std::uint64_t>(samples) ||
        artifact.num_views != num_views ||
        artifact.max_prompt_tokens != max_prompt_tokens) {
        std::cerr << "dataset calibration artifact validation failed\n";
        return 1;
    }

    const std::string runtime_config =
        config_json(argv[1], argv[2], num_views, max_prompt_tokens,
                    calibration_path);
    frt_model_runtime_v1* model = nullptr;
    rc = frt_model_runtime_open_v1(runtime_config.c_str(), &model);
    if (rc != 0 || !model) {
        std::cerr << "native FP8 open failed: rc=" << rc << " error="
                  << frt_pi05_native_open_last_error() << '\n';
        return 1;
    }
    const frt_runtime_export_v1* exp = model->exp;
    bool schema_ok = model->n_ports == 6 && model->n_stages == 1 &&
                     exp && exp->n_graphs == 3 &&
                     std::strcmp(exp->graphs[0].name, "infer") == 0 &&
                     std::strcmp(exp->graphs[1].name, "decode_only") == 0 &&
                     std::strcmp(exp->graphs[2].name, "context") == 0 &&
                     exp->identity &&
                     std::strstr(exp->identity,
                                 "precision=fp8_e4m3fn") &&
                     std::strstr(exp->identity, "hardware=sm110") &&
                     std::strstr(exp->identity, "calibration_sha256=") &&
                     model->ports[2].dtype == FRT_RT_DTYPE_F16 &&
                     model->ports[3].dtype == FRT_RT_DTYPE_F16 &&
                     model->ports[5].dtype == FRT_RT_DTYPE_F16;
    for (std::uint64_t i = 0; schema_ok && i < exp->n_graphs; ++i) {
        schema_ok =
            frt_graph_variant_count(exp->graphs[i].handle) == 1;
    }
    if (!schema_ok) {
        std::cerr << "native FP8 schema validation failed\n";
        model->release(model->owner);
        return 1;
    }

    const std::string prompt = "pick up the black bowl";
    if (model->verbs.set_input(model->self, 0, prompt.data(), prompt.size(),
                               -1) != 0 ||
        model->verbs.set_input(model->self, 1, state, sizeof(state), -1) != 0) {
        std::cerr << "native FP8 prompt/state staging failed\n";
        model->release(model->owner);
        return 1;
    }
    std::vector<frt_image_view> views(num_views);
    for (int view = 0; view < num_views; ++view) {
        views[view].struct_size = sizeof(frt_image_view);
        views[view].pixel_format = FRT_RT_PIXEL_RGB8;
        views[view].data = pixels[view]->data();
        views[view].bytes = pixels[view]->size();
        views[view].width = 224;
        views[view].height = 224;
        views[view].stride_bytes = 224 * 3;
    }
    if (model->verbs.set_input(
            model->self, 2, views.data(),
            views.size() * sizeof(frt_image_view), -1) != 0) {
        std::cerr << "native FP8 image staging failed\n";
        model->release(model->owner);
        return 1;
    }
    std::vector<std::uint16_t> noise_f16(noise.size());
    for (std::size_t i = 0; i < noise.size(); ++i) {
        noise_f16[i] = flashrt::modalities::float_to_float16(noise[i]);
    }
    if (!model->ports[3].buffer ||
        cudaMemcpy(frt_buffer_dptr(model->ports[3].buffer), noise_f16.data(),
                   noise_f16.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        model->verbs.step(model->self) != 0 ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "native FP8 replay failed: "
                  << model->verbs.last_error(model->self) << '\n';
        model->release(model->owner);
        return 1;
    }
    std::vector<std::uint16_t> raw(noise.size());
    if (!model->ports[5].buffer ||
        cudaMemcpy(raw.data(), frt_buffer_dptr(model->ports[5].buffer),
                   raw.size() * sizeof(raw[0]),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(frt_buffer_dptr(model->ports[3].buffer), noise_f16.data(),
                   noise_f16.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        frt_graph_replay(exp->graphs[2].handle, 0,
                         exp->graphs[2].stream_id) != FRT_OK ||
        frt_graph_replay(exp->graphs[1].handle, 0,
                         exp->graphs[1].stream_id) != FRT_OK ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "native FP8 split replay failed\n";
        model->release(model->owner);
        return 1;
    }
    std::vector<std::uint16_t> split_raw(noise.size());
    if (cudaMemcpy(split_raw.data(),
                   frt_buffer_dptr(model->ports[5].buffer),
                   split_raw.size() * sizeof(split_raw[0]),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        split_raw != raw) {
        std::cerr << "native FP8 full/split raw actions differ\n";
        model->release(model->owner);
        return 1;
    }
    if (argc == 7) {
        std::ofstream output(argv[6], std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(raw.data()),
                     static_cast<std::streamsize>(
                         raw.size() * sizeof(raw[0])));
        if (!output) {
            std::cerr << "native FP8 raw action write failed\n";
            model->release(model->owner);
            return 1;
        }
    }
    std::vector<float> actions(10 * 7);
    std::uint64_t written = 0;
    if (model->verbs.get_output(model->self, 4, actions.data(),
                                actions.size() * sizeof(float), &written,
                                -1) != 0 ||
        written != actions.size() * sizeof(float)) {
        std::cerr << "native FP8 action read failed\n";
        model->release(model->owner);
        return 1;
    }
    for (float value : actions) {
        if (!std::isfinite(value)) {
            std::cerr << "native FP8 action is non-finite\n";
            model->release(model->owner);
            return 1;
        }
    }
    model->release(model->owner);
    std::cout << "PASS native Thor FP8 calibration and runtime lifecycle\n";
    return 0;
}
