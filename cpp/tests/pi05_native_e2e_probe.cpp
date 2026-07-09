#include "flashrt/model_runtime.h"

#include <cuda_runtime_api.h>

#include <cstdint>
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

bool read_file(const std::string& path, std::vector<std::uint8_t>* out) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0) return false;
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    if (size && !input.read(reinterpret_cast<char*>(data.data()), size)) {
        return false;
    }
    *out = std::move(data);
    return true;
}

bool write_file(const std::string& path, const void* data, std::size_t bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) return false;
    output.write(static_cast<const char*>(data),
                 static_cast<std::streamsize>(bytes));
    return output.good();
}

std::string json_string(const std::string& value) {
    std::string output = "\"";
    for (char c : value) {
        if (c == '\\' || c == '"') output.push_back('\\');
        output.push_back(c);
    }
    output.push_back('"');
    return output;
}

int model_error(frt_model_runtime_v1* model, const char* message) {
    std::cerr << message;
    if (model && model->verbs.last_error) {
        const char* detail = model->verbs.last_error(model->self);
        if (detail && *detail) std::cerr << ": " << detail;
    }
    std::cerr << '\n';
    if (model) model->release(model->owner);
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: pi05_native_e2e_probe CHECKPOINT TOKENIZER "
                     "FIXTURE_DIR OUTPUT_DIR\n";
        return 2;
    }
    const std::string checkpoint = argv[1];
    const std::string tokenizer = argv[2];
    const std::string fixture = argv[3];
    const std::string output = argv[4];

    std::ostringstream json;
    json << "{\"io\":\"native_v2\",\"checkpoint_path\":"
         << json_string(checkpoint) << ",\"tokenizer_model_path\":"
         << json_string(tokenizer)
         << ",\"state_prompt_mode\":\"fixed\","
            "\"max_prompt_tokens\":200,\"state_dim\":8,"
            "\"num_views\":2,\"chunk\":10,\"num_steps\":10,"
            "\"vision_pool_factor\":1}";
    frt_model_runtime_v1* model = nullptr;
    const int open_rc = frt_model_runtime_open_v1(json.str().c_str(), &model);
    if (open_rc != 0 || !model) {
        std::cerr << "native open failed: rc=" << open_rc << " error="
                  << frt_pi05_native_open_last_error() << '\n';
        return 1;
    }
    const char* names[] = {
        "prompt", "state", "images", "noise", "actions", "actions_raw"};
    if (model->n_ports != 6) return model_error(model, "unexpected port count");
    for (std::uint64_t i = 0; i < model->n_ports; ++i) {
        if (!model->ports[i].name ||
            std::strcmp(model->ports[i].name, names[i]) != 0) {
            return model_error(model, "unexpected port schema");
        }
    }

    std::vector<std::uint8_t> prompt;
    std::vector<std::uint8_t> state;
    std::vector<std::uint8_t> image0;
    std::vector<std::uint8_t> image1;
    std::vector<std::uint8_t> noise;
    if (!read_file(fixture + "/prompt.txt", &prompt) ||
        !read_file(fixture + "/state.f32", &state) ||
        !read_file(fixture + "/image_0.rgb", &image0) ||
        !read_file(fixture + "/image_1.rgb", &image1) ||
        !read_file(fixture + "/noise.bf16", &noise) ||
        state.size() != 8 * sizeof(float) ||
        image0.size() != 224 * 224 * 3 || image1.size() != image0.size() ||
        noise.size() != 10 * 32 * sizeof(std::uint16_t)) {
        return model_error(model, "invalid E2E fixture");
    }
    if (model->verbs.set_input(model->self, 0, prompt.data(), prompt.size(),
                               -1) != 0 ||
        model->verbs.set_input(model->self, 1, state.data(), state.size(),
                               -1) != 0) {
        return model_error(model, "prompt/state staging failed");
    }

    frt_image_view views[2]{};
    const std::vector<std::uint8_t>* images[] = {&image0, &image1};
    for (int i = 0; i < 2; ++i) {
        views[i].struct_size = sizeof(frt_image_view);
        views[i].pixel_format = FRT_RT_PIXEL_RGB8;
        views[i].data = images[i]->data();
        views[i].bytes = images[i]->size();
        views[i].width = 224;
        views[i].height = 224;
        views[i].stride_bytes = 224 * 3;
    }
    if (model->verbs.set_input(model->self, 2, views, sizeof(views), -1) != 0) {
        return model_error(model, "image staging failed");
    }
    frt_buffer noise_buffer = model->ports[3].buffer;
    if (!noise_buffer || frt_buffer_bytes(noise_buffer) != noise.size() ||
        cudaMemcpy(frt_buffer_dptr(noise_buffer), noise.data(), noise.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return model_error(model, "noise upload failed");
    }
    if (model->verbs.step(model->self) != 0) {
        return model_error(model, "native infer failed");
    }

    std::vector<float> actions(10 * 7);
    std::uint64_t written = 0;
    if (model->verbs.get_output(model->self, 4, actions.data(),
                                actions.size() * sizeof(float), &written,
                                -1) != 0 ||
        written != actions.size() * sizeof(float)) {
        return model_error(model, "action output failed");
    }
    std::vector<std::uint8_t> raw(noise.size());
    if (cudaMemcpy(raw.data(), frt_buffer_dptr(model->ports[5].buffer),
                   raw.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return model_error(model, "raw action download failed");
    }
    if (!write_file(output + "/native_raw.bf16", raw.data(), raw.size()) ||
        !write_file(output + "/native_actions.f32", actions.data(),
                    actions.size() * sizeof(float))) {
        return model_error(model, "native output write failed");
    }
    model->release(model->owner);
    std::cout << "PASS native real-episode fixture\n";
    return 0;
}
