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
    write_raw_safetensors(path, header,
                          std::string(static_cast<size_t>(payload_bytes),
                                      '\0'));
}

void append_f32(std::string* out, float value) {
    char bytes[sizeof(float)];
    std::memcpy(bytes, &value, sizeof(value));
    out->append(bytes, sizeof(bytes));
}

void write_safetensors(const std::string& path) {
    struct Entry {
        const char* key;
        const char* dtype;
        const char* shape;
        uint64_t bytes;
    };
    const Entry entries[] = {
        {"paligemma_with_expert.paligemma.lm_head.weight", "BF16",
         "[1001,2048]", 1001ull * 2048ull * 2ull},
        {"paligemma_with_expert.paligemma.model.vision_tower.vision_model"
         ".embeddings.patch_embedding.weight",
         "F32", "[1152,3,14,14]", 1152ull * 3ull * 14ull * 14ull * 4ull},
        {"paligemma_with_expert.paligemma.model.vision_tower.vision_model"
         ".embeddings.patch_embedding.bias",
         "F32", "[1152]", 1152ull * 4ull},
        {"paligemma_with_expert.paligemma.model.vision_tower.vision_model"
         ".embeddings.position_embedding.weight",
         "F32", "[256,1152]", 256ull * 1152ull * 4ull},
        {"paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
         ".weight",
         "F32", "[2048,1152]", 2048ull * 1152ull * 4ull},
        {"paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
         ".bias",
         "F32", "[2048]", 2048ull * 4ull},
        {"action_in_proj.weight", "F32", "[1024,32]", 1024ull * 32ull * 4ull},
        {"action_in_proj.bias", "F32", "[1024]", 1024ull * 4ull},
        {"action_out_proj.weight", "F32", "[32,1024]", 32ull * 1024ull * 4ull},
        {"action_out_proj.bias", "F32", "[32]", 32ull * 4ull},
        {"time_mlp_in.weight", "F32", "[1024,1024]", 1024ull * 1024ull * 4ull},
        {"time_mlp_in.bias", "F32", "[1024]", 1024ull * 4ull},
        {"time_mlp_out.weight", "F32", "[1024,1024]", 1024ull * 1024ull * 4ull},
        {"time_mlp_out.bias", "F32", "[1024]", 1024ull * 4ull},
        {"paligemma_with_expert.paligemma.model.language_model.layers.0"
         ".self_attn.q_proj.weight",
         "F32", "[2048,2048]", 2048ull * 2048ull * 4ull},
        {"paligemma_with_expert.gemma_expert.model.layers.0.self_attn"
         ".q_proj.weight",
         "F32", "[2048,1024]", 2048ull * 1024ull * 4ull},
    };
    std::string header = "{";
    uint64_t offset = 0;
    for (size_t i = 0; i < sizeof(entries) / sizeof(entries[0]); ++i) {
        const Entry& e = entries[i];
        if (i) header += ",";
        header += "\"";
        header += e.key;
        header += "\":{\"dtype\":\"";
        header += e.dtype;
        header += "\",\"shape\":";
        header += e.shape;
        header += ",\"data_offsets\":[";
        header += std::to_string(offset);
        header += ",";
        offset += e.bytes;
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

    write_bad_safetensors(root + "/model.safetensors");
    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(config(root, tokenizer).c_str(), &out);
    assert(rc == -2);
    assert(out == nullptr);
    assert(std::strstr(frt_pi05_native_open_last_error(), "byte range"));

    write_safetensors(root + "/model.safetensors");
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
    write_lerobot_policy_stats(lerobot_root, false);
    out = reinterpret_cast<frt_model_runtime_v1*>(0x1);
    rc = frt_model_runtime_open_v1(
        config(lerobot_root, tokenizer).c_str(), &out);
    assert(rc == -2);
    assert(out == nullptr);

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
