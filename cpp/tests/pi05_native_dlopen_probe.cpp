#include "flashrt/model_runtime.h"

#include <dlfcn.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

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

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: pi05_native_dlopen_probe SO CHECKPOINT "
                     "TOKENIZER CYCLES\n";
        return 2;
    }
    const int cycles = std::atoi(argv[4]);
    if (cycles <= 0) return 2;
    std::ostringstream config;
    config << "{\"io\":\"native_v2\",\"checkpoint_path\":"
           << json_string(argv[2]) << ",\"tokenizer_model_path\":"
           << json_string(argv[3])
           << ",\"state_prompt_mode\":\"fixed\","
              "\"max_prompt_tokens\":200,\"state_dim\":8,"
              "\"num_views\":2,\"chunk\":10,\"num_steps\":10,"
              "\"vision_pool_factor\":1}";
    for (int cycle = 0; cycle < cycles; ++cycle) {
        void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!library) {
            std::cerr << "dlopen failed: " << dlerror() << '\n';
            return 1;
        }
        auto open = reinterpret_cast<frt_model_runtime_open_v1_fn>(
            dlsym(library, FRT_MODEL_RUNTIME_OPEN_V1_SYMBOL));
        auto last_error = reinterpret_cast<const char* (*)()>(
            dlsym(library, "frt_pi05_native_open_last_error"));
        if (!open || !last_error) {
            std::cerr << "native factory symbols are missing\n";
            dlclose(library);
            return 1;
        }
        frt_model_runtime_v1* model = nullptr;
        const int rc = open(config.str().c_str(), &model);
        if (rc != 0 || !model) {
            std::cerr << "native open failed: rc=" << rc << " error="
                      << last_error() << '\n';
            dlclose(library);
            return 1;
        }
        if (model->abi_version != FRT_MODEL_RUNTIME_ABI_VERSION ||
            model->struct_size < sizeof(*model) || !model->retain ||
            !model->release) {
            std::cerr << "native model ABI is invalid\n";
            if (model->release) model->release(model->owner);
            dlclose(library);
            return 1;
        }
        model->retain(model->owner);
        model->release(model->owner);
        model->release(model->owner);
        if (dlclose(library) != 0) {
            std::cerr << "dlclose failed: " << dlerror() << '\n';
            return 1;
        }
        std::cout << "cycle " << (cycle + 1) << " released\n";
    }
    std::cout << "PASS native model dlopen/release/dlclose lifecycle\n";
    return 0;
}
