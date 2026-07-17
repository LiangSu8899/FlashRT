#include "flashrt/model_runtime.h"
#include "flashrt/cpp/models/pi05/support/native_weights.h"

#include <algorithm>
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
                           const std::string& payload) {
    std::ofstream f(path, std::ios::binary);
    uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) {
        const char b = static_cast<char>((n >> (8 * i)) & 0xffu);
        f.write(&b, 1);
    }
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    f.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    assert(f.good());
}

void write_raw_safetensors(const std::string& path,
                           const std::string& header,
                           uint64_t payload_bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) {
        const char b = static_cast<char>((n >> (8 * i)) & 0xffu);
        f.write(&b, 1);
    }
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (payload_bytes) {
        f.seekp(static_cast<std::streamoff>(payload_bytes - 1), std::ios::cur);
        const char zero = 0;
        f.write(&zero, 1);
    }
    assert(f.good());
}

void append_f32(std::string* out, float value) {
    char bytes[sizeof(float)];
    std::memcpy(bytes, &value, sizeof(value));
    out->append(bytes, sizeof(bytes));
}

void write_safetensors(const std::string& path,
                       const std::string& omit = "") {
    const auto& entries =
        flashrt::models::pi05::native_tensor_requirements();
    std::string header = "{";
    uint64_t offset = 0;
    bool first = true;
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        if (e.key == omit) continue;
        if (!first) header += ",";
        first = false;
        header += "\"";
        header += e.key;
        header += "\":{\"dtype\":\"F32\",\"shape\":[";
        uint64_t elements = 1;
        for (size_t dim = 0; dim < e.shape.size(); ++dim) {
            if (dim) header += ",";
            header += std::to_string(e.shape[dim]);
            elements *= e.shape[dim];
        }
        header += "]";
        header += ",\"data_offsets\":[";
        header += std::to_string(offset);
        header += ",";
        offset += elements * sizeof(float);
        header += std::to_string(offset);
        header += "]}";
    }
    header += ",\"__metadata__\":{\"format\":\"pt\"}}";
    write_raw_safetensors(path, header, offset);
}

void write_bad_safetensors(const std::string& path) {
    const uint64_t bytes = 1001ull * 2048ull * 2ull;
    write_raw_safetensors(
        path,
        "{\"paligemma_with_expert.paligemma.lm_head.weight\":{"
        "\"dtype\":\"BF16\",\"shape\":[1001,2048],"
        "\"data_offsets\":[0," + std::to_string(bytes) + "]}}",
        1024);
}

void write_lerobot_policy_stats(const std::string& root, bool valid = true) {
    std::string state_payload;
    for (int i = 0; i < 8; ++i) append_f32(&state_payload, 0.0f);
    for (int i = 0; i < 8; ++i) append_f32(&state_payload, valid ? 1.0f : 0.0f);
    write_raw_safetensors(
        root + "/policy_preprocessor_step_2_normalizer_processor.safetensors",
        "{\"observation.state.q01\":{\"dtype\":\"F32\",\"shape\":[8],"
        "\"data_offsets\":[0,32]},"
        "\"observation.state.q99\":{\"dtype\":\"F32\",\"shape\":[8],"
        "\"data_offsets\":[32,64]}}",
        state_payload);
    std::string action_payload;
    for (int i = 0; i < 7; ++i) append_f32(&action_payload, 0.0f);
    for (int i = 0; i < 7; ++i) append_f32(&action_payload, valid ? 1.0f : 0.0f);
    write_raw_safetensors(
        root + "/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "{\"action.q01\":{\"dtype\":\"F32\",\"shape\":[7],"
        "\"data_offsets\":[0,28]},"
        "\"action.q99\":{\"dtype\":\"F32\",\"shape\":[7],"
        "\"data_offsets\":[28,56]}}",
        action_payload);
}

void write_norm_stats(const std::string& path, bool valid = true) {
    std::ofstream f(path);
    f << "{"
      << "\"norm_stats\":{"
      << "\"state\":{\"q01\":[0,0,0,0,0,0,0,0],"
      << (valid ? "\"q99\":[1,1,1,1,1,1,1,1]},"
                : "\"q99\":[0,0,0,0,0,0,0,0]},")
      << "\"actions\":{\"q01\":[0,0,0,0,0,0,0],"
      << (valid ? "\"q99\":[1,1,1,1,1,1,1]}"
                : "\"q99\":[0,0,0,0,0,0,0]}")
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
    const auto& inventory =
        flashrt::models::pi05::native_tensor_requirements();
    assert(inventory.size() == 812);
    const auto has_key = [&inventory](const char* key) {
        return std::any_of(inventory.begin(), inventory.end(),
                           [key](const auto& item) { return item.key == key; });
    };
    assert(has_key(
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
        "encoder.layers.26.mlp.fc2.weight"));
    assert(has_key(
        "paligemma_with_expert.paligemma.model.language_model.layers.17."
        "mlp.down_proj.weight"));
    assert(has_key(
        "paligemma_with_expert.gemma_expert.model.layers.17."
        "post_attention_layernorm.dense.bias"));

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
    rc = frt_model_runtime_open_v1(
        config(root, tokenizer, ",\"precision\":\"nvfp4\"").c_str(),
        &out);
    assert(rc == -1);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(), "precision"));

    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(
        config(root, tokenizer, ",\"stage_plan\":\"async_magic\"").c_str(),
        &out);
    assert(rc == -1);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(), "stage_plan"));

    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(
        config(root, tokenizer,
               ",\"calibration_path\":\"/missing/calibration.safetensors\"")
            .c_str(),
        &out);
    assert(rc == -2);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(),
                       "calibration_path"));

    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(config(root, tokenizer).c_str(), &out);
    assert(rc == -2);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(),
                       "model.safetensors"));

    write_bad_safetensors(root + "/model.safetensors");
    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(config(root, tokenizer).c_str(), &out);
    assert(rc == -2);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(), "byte range"));

    const std::string missing_key =
        flashrt::models::pi05::native_tensor_requirements().back().key;
    write_safetensors(root + "/model.safetensors", missing_key);
    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(config(root, tokenizer).c_str(), &out);
    assert(rc == -2);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(),
                       missing_key.c_str()));

    write_safetensors(root + "/model.safetensors");
    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(
        config(root, tokenizer, ",\"num_views\":0").c_str(), &out);
    assert(rc == -1);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(), "num_views"));

    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(
        config(root, tokenizer, ",\"chunk\":0").c_str(), &out);
    assert(rc == -1);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(), "chunk"));

    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(
        config(root, tokenizer, ",\"max_frame_width\":0").c_str(), &out);
    assert(rc == -1);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(), "max_frame"));

    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(config(root, tokenizer).c_str(), &out);
    assert(rc == -2);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(), "norm_stats"));

    write_norm_stats(root + "/norm_stats.json", false);
    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(config(root, tokenizer).c_str(), &out);
    assert(rc == -2);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(), "q01/q99"));

    write_norm_stats(root + "/norm_stats.json");
#ifndef FLASHRT_CPP_PI05_NATIVE_OPEN_ENABLED
    const std::string good = config(root, tokenizer);
    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(good.c_str(), &out);
    assert(rc == -3);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(), "validated"));
#endif

    const std::string invalid_prompt_capacity =
        std::string("{") +
        "\"io\":\"native_v2\"," +
        "\"checkpoint_path\":\"" + root + "\"," +
        "\"tokenizer_model_path\":\"" + tokenizer + "\"," +
        "\"state_prompt_mode\":\"fixed\"," +
        "\"max_prompt_tokens\":0," +
        "\"state_dim\":8}";
    rc = frt_model_runtime_open_v1(
        invalid_prompt_capacity.c_str(), &out);
    assert(rc == -1);
    assert(std::strstr(frt_pi05_native_open_last_error(),
                       "max_prompt_tokens"));

    const std::string lerobot_root = make_temp_dir();
    write_safetensors(lerobot_root + "/model.safetensors");
    write_lerobot_policy_stats(lerobot_root, false);
    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(
        config(lerobot_root, tokenizer).c_str(), &out);
    assert(rc == -2);
    assert(out == nullptr);

    write_lerobot_policy_stats(lerobot_root);
#ifndef FLASHRT_CPP_PI05_NATIVE_OPEN_ENABLED
    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(
        config(lerobot_root, tokenizer).c_str(), &out);
    assert(rc == -3);
    assert(out == nullptr);
#endif

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
