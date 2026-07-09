#include "flashrt/model_runtime.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

extern "C" int frt_model_runtime_open_v1(const char* config_json,
                                          frt_model_runtime_v1** out);
extern "C" const char* frt_pi05_native_open_last_error();

namespace {

std::string make_temp_dir() {
    char tmpl[] = "/tmp/frt_pi05_native_open_XXXXXX";
    char* path = ::mkdtemp(tmpl);
    assert(path);
    return path;
}

void write_file(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    f << "x";
    assert(f.good());
}

void write_raw_safetensors(const std::string& path,
                           const std::string& header,
                           uint64_t payload_bytes) {
    std::ofstream f(path, std::ios::binary);
    uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) {
        const char b = static_cast<char>((n >> (8 * i)) & 0xffu);
        f.write(&b, 1);
    }
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    std::string payload(static_cast<size_t>(payload_bytes), '\0');
    f.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    assert(f.good());
}

void write_safetensors(const std::string& path) {
    write_raw_safetensors(
        path,
        "{\"paligemma_with_expert.paligemma.lm_head.weight\":{"
        "\"dtype\":\"BF16\",\"shape\":[257216,2048],"
        "\"data_offsets\":[0,1024]},"
        "\"__metadata__\":{\"format\":\"pt\"}}",
        1024);
}

void write_lerobot_policy_stats(const std::string& root) {
    write_raw_safetensors(
        root + "/policy_preprocessor_step_2_normalizer_processor.safetensors",
        "{\"observation.state.q01\":{\"dtype\":\"F32\",\"shape\":[8],"
        "\"data_offsets\":[0,32]},"
        "\"observation.state.q99\":{\"dtype\":\"F32\",\"shape\":[8],"
        "\"data_offsets\":[32,64]}}",
        64);
    write_raw_safetensors(
        root + "/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "{\"action.q01\":{\"dtype\":\"F32\",\"shape\":[7],"
        "\"data_offsets\":[0,28]},"
        "\"action.q99\":{\"dtype\":\"F32\",\"shape\":[7],"
        "\"data_offsets\":[28,56]}}",
        56);
}

void write_norm_stats(const std::string& path) {
    std::ofstream f(path);
    f << "{"
      << "\"norm_stats\":{"
      << "\"state\":{\"q01\":[0,0,0,0,0,0,0,0],"
      << "\"q99\":[1,1,1,1,1,1,1,1]},"
      << "\"actions\":{\"q01\":[0,0,0,0,0,0,0],"
      << "\"q99\":[1,1,1,1,1,1,1]}"
      << "}}";
    assert(f.good());
}

std::string config(const std::string& ckpt,
                   const std::string& tokenizer,
                   const char* extra = "") {
    return std::string("{") +
           "\"io\":\"native_v2\"," +
           "\"checkpoint_path\":\"" + ckpt + "\"," +
           "\"tokenizer_model_path\":\"" + tokenizer + "\"," +
           "\"state_prompt_mode\":\"fixed\"," +
           "\"max_prompt_tokens\":200," +
           "\"state_dim\":8," +
           "\"num_views\":2," +
           "\"chunk\":10" +
           extra + "}";
}

}  // namespace

int main() {
    frt_model_runtime_v1* out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    int rc = frt_model_runtime_open_v1(nullptr, &out);
    assert(rc == -1);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(), "null"));

    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1("{", &out);
    assert(rc == -1);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(), "JSON"));

    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(
        "{\"io\":\"native\",\"checkpoint_path\":\"/tmp\","
        "\"tokenizer_model_path\":\"/tmp/x\","
        "\"state_prompt_mode\":\"fixed\","
        "\"max_prompt_tokens\":200,\"state_dim\":1}",
        &out);
    assert(rc == -1);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(), "native_v2"));

    const std::string root = make_temp_dir();
    const std::string tokenizer = root + "/tokenizer.model";
    write_file(tokenizer);

    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(config(root, tokenizer).c_str(), &out);
    assert(rc == -2);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(),
                       "model.safetensors"));

    write_safetensors(root + "/model.safetensors");
    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(config(root, tokenizer).c_str(), &out);
    assert(rc == -2);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(), "norm_stats"));

    write_norm_stats(root + "/norm_stats.json");
    const std::string good = config(root, tokenizer);
    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(good.c_str(), &out);
    assert(rc == -3);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(), "validated"));

    const std::string short_prompt =
        std::string("{") +
        "\"io\":\"native_v2\"," +
        "\"checkpoint_path\":\"" + root + "\"," +
        "\"tokenizer_model_path\":\"" + tokenizer + "\"," +
        "\"state_prompt_mode\":\"fixed\"," +
        "\"max_prompt_tokens\":199," +
        "\"state_dim\":8}";
    rc = frt_model_runtime_open_v1(short_prompt.c_str(), &out);
    assert(rc == -1);
    assert(std::strstr(frt_pi05_native_open_last_error(),
                       "max_prompt_tokens"));

    const std::string lerobot_root = make_temp_dir();
    write_safetensors(lerobot_root + "/model.safetensors");
    write_lerobot_policy_stats(lerobot_root);
    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(
        config(lerobot_root, tokenizer).c_str(), &out);
    assert(rc == -3);
    assert(out == nullptr);

    ::unlink((lerobot_root + "/model.safetensors").c_str());
    ::unlink((lerobot_root +
              "/policy_preprocessor_step_2_normalizer_processor.safetensors")
                 .c_str());
    ::unlink((lerobot_root +
              "/policy_postprocessor_step_0_unnormalizer_processor.safetensors")
                 .c_str());
    ::rmdir(lerobot_root.c_str());

    ::unlink(tokenizer.c_str());
    ::unlink((root + "/model.safetensors").c_str());
    ::unlink((root + "/norm_stats.json").c_str());
    ::rmdir(root.c_str());
    std::printf("PASS - Pi05 native open scaffold\n");
    return 0;
}
