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

    ::unlink(tokenizer.c_str());
    ::rmdir(root.c_str());
    std::printf("PASS - Pi05 native open scaffold\n");
    return 0;
}
