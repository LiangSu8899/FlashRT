#include "flashrt/cpp/models/pi05/model_runtime.h"
#include "flashrt/cpp/loader/safetensors.h"
#include "flashrt/cpp/models/pi05/model/execution_plan.h"
#include "flashrt/cpp/models/pi05/support/native_weights.h"
#include "flashrt/cpp/native/config_object.h"
#include "native_open_internal.h"

#include <cerrno>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <cmath>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

thread_local std::string g_last_error;

bool path_exists(const std::string& path) {
    struct stat st {};
    return !path.empty() && ::stat(path.c_str(), &st) == 0;
}

bool regular_file_exists(const std::string& path) {
    struct stat st {};
    return !path.empty() && ::stat(path.c_str(), &st) == 0 &&
           S_ISREG(st.st_mode);
}

std::string join_path(const std::string& dir, const char* leaf) {
    if (dir.empty() || dir.back() == '/') return dir + leaf;
    return dir + "/" + leaf;
}

std::string quoted_key(const std::string& key) {
    std::string out = "\"";
    for (char c : key) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

bool object_for_key(const std::string& json,
                    const std::string& key,
                    std::string* object) {
    const std::string q = quoted_key(key);
    size_t pos = json.find(q);
    while (pos != std::string::npos) {
        size_t p = pos + q.size();
        while (p < json.size() &&
               std::isspace(static_cast<unsigned char>(json[p]))) {
            ++p;
        }
        if (p < json.size() && json[p] == ':') {
            ++p;
            while (p < json.size() &&
                   std::isspace(static_cast<unsigned char>(json[p]))) {
                ++p;
            }
            if (p < json.size() && json[p] == '{') {
                int depth = 0;
                bool in_string = false;
                bool escaped = false;
                for (size_t i = p; i < json.size(); ++i) {
                    const char c = json[i];
                    if (in_string) {
                        if (escaped) {
                            escaped = false;
                        } else if (c == '\\') {
                            escaped = true;
                        } else if (c == '"') {
                            in_string = false;
                        }
                        continue;
                    }
                    if (c == '"') {
                        in_string = true;
                    } else if (c == '{') {
                        ++depth;
                    } else if (c == '}') {
                        --depth;
                        if (depth == 0) {
                            if (object) *object = json.substr(p, i - p + 1);
                            return true;
                        }
                    }
                }
            }
        }
        pos = json.find(q, pos + 1);
    }
    return false;
}

bool parse_f64_array_property(const std::string& object,
                              const char* name,
                              std::vector<double>* out) {
    const std::string q = quoted_key(name);
    size_t p = object.find(q);
    if (p == std::string::npos) return false;
    p += q.size();
    while (p < object.size() &&
           std::isspace(static_cast<unsigned char>(object[p]))) ++p;
    if (p >= object.size() || object[p++] != ':') return false;
    while (p < object.size() &&
           std::isspace(static_cast<unsigned char>(object[p]))) ++p;
    if (p >= object.size() || object[p++] != '[') return false;
    std::vector<double> values;
    while (p < object.size()) {
        while (p < object.size() &&
               std::isspace(static_cast<unsigned char>(object[p]))) ++p;
        if (p < object.size() && object[p] == ']') {
            ++p;
            if (out) *out = std::move(values);
            return true;
        }
        errno = 0;
        char* end = nullptr;
        const double value = std::strtod(object.c_str() + p, &end);
        if (errno || end == object.c_str() + p) return false;
        values.push_back(value);
        p = static_cast<size_t>(end - object.c_str());
        while (p < object.size() &&
               std::isspace(static_cast<unsigned char>(object[p]))) ++p;
        if (p < object.size() && object[p] == ',') {
            ++p;
            continue;
        }
        if (p < object.size() && object[p] == ']') continue;
        return false;
    }
    return false;
}

bool read_safetensors_f32_vector(const std::string& path,
                                 const char* key,
                                 std::vector<float>* out) {
    if (!out) return false;
    flashrt::loader::SafetensorsFile file;
    if (!file.open(path)) {
        g_last_error = file.error();
        return false;
    }
    const auto* tensor = file.find(key);
    if (!tensor || tensor->dtype != "F32" || tensor->shape.size() != 1) {
        g_last_error = "safetensors F32 vector metadata is invalid";
        return false;
    }
    const uint64_t n = tensor->shape[0];
    if (n > (1ull << 20)) {
        g_last_error = "safetensors vector is too large";
        return false;
    }
    std::vector<float> values(static_cast<size_t>(n));
    std::memcpy(values.data(), file.data(*tensor),
                static_cast<size_t>(tensor->bytes));
    *out = std::move(values);
    return true;
}

bool sane_quantile_pair(const std::vector<double>& q01,
                        const std::vector<double>& q99) {
    if (q01.empty() || q01.size() != q99.size()) return false;
    for (size_t i = 0; i < q01.size(); ++i) {
        if (!std::isfinite(q01[i]) || !std::isfinite(q99[i]) ||
            q99[i] <= q01[i]) {
            return false;
        }
    }
    return true;
}

bool sane_quantile_pair(const std::vector<float>& q01,
                        const std::vector<float>& q99) {
    if (q01.empty() || q01.size() != q99.size()) return false;
    for (size_t i = 0; i < q01.size(); ++i) {
        if (!std::isfinite(q01[i]) || !std::isfinite(q99[i]) ||
            q99[i] <= q01[i]) {
            return false;
        }
    }
    return true;
}

bool read_text_file(const std::string& path, std::string* out) {
    if (!out) return false;
    std::ifstream f(path);
    if (!f) return false;
    out->assign((std::istreambuf_iterator<char>(f)),
                std::istreambuf_iterator<char>());
    return f.good() || f.eof();
}

std::string dirname(const std::string& path) {
    const size_t p = path.find_last_of('/');
    if (p == std::string::npos) return ".";
    if (p == 0) return "/";
    return path.substr(0, p);
}

bool norm_block_values(const std::string& json,
                       const char* block_name,
                       std::vector<float>* q01_out,
                       std::vector<float>* q99_out) {
    std::string block;
    if (!object_for_key(json, block_name, &block)) return false;
    std::vector<double> q01;
    std::vector<double> q99;
    if (!parse_f64_array_property(block, "q01", &q01) ||
        !parse_f64_array_property(block, "q99", &q99) ||
        !sane_quantile_pair(q01, q99)) {
        return false;
    }
    if (q01_out) q01_out->assign(q01.begin(), q01.end());
    if (q99_out) q99_out->assign(q99.begin(), q99.end());
    return true;
}

bool validate_norm_stats_file(const std::string& path,
                              int64_t state_dim,
                              flashrt::models::pi05::NativeOpenConfig* config) {
    std::string json;
    if (!read_text_file(path, &json)) return false;
    std::vector<float> action_q01;
    std::vector<float> action_q99;
    std::vector<float> state_q01;
    std::vector<float> state_q99;
    if (!norm_block_values(json, "actions", &action_q01, &action_q99) ||
        !norm_block_values(json, "state", &state_q01, &state_q99)) {
        g_last_error = "norm_stats.json is missing actions/state q01/q99";
        return false;
    }
    if (action_q01.empty() || action_q01.size() > 32) {
        g_last_error = "norm_stats action dimension is invalid";
        return false;
    }
    if (state_q01.size() != static_cast<size_t>(state_dim)) {
        g_last_error = "norm_stats state dimension does not match config";
        return false;
    }
    if (config) {
        config->state_q01 = std::move(state_q01);
        config->state_q99 = std::move(state_q99);
        config->action_q01 = std::move(action_q01);
        config->action_q99 = std::move(action_q99);
    }
    g_last_error.clear();
    return true;
}

bool has_prefix(const std::string& s, const char* prefix) {
    const size_t n = std::strlen(prefix);
    return s.size() >= n && s.compare(0, n, prefix) == 0;
}

bool has_suffix(const std::string& s, const char* suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

std::string find_child(const std::string& dir,
                       const char* prefix,
                       const char* suffix) {
    DIR* d = ::opendir(dir.c_str());
    if (!d) return "";
    std::string found;
    while (dirent* ent = ::readdir(d)) {
        const std::string name = ent->d_name;
        if (has_prefix(name, prefix) && has_suffix(name, suffix)) {
            found = join_path(dir, name.c_str());
            break;
        }
    }
    ::closedir(d);
    return found;
}

bool validate_lerobot_policy_norm_stats(const std::string& checkpoint_path,
                                        int64_t state_dim,
                                        flashrt::models::pi05::NativeOpenConfig*
                                            config) {
    const std::string pre = find_child(
        checkpoint_path, "policy_preprocessor_step_",
        "_normalizer_processor.safetensors");
    const std::string post = find_child(
        checkpoint_path, "policy_postprocessor_step_",
        "_unnormalizer_processor.safetensors");
    if (pre.empty() || post.empty()) return false;

    std::vector<float> state_q01;
    std::vector<float> state_q99;
    std::vector<float> action_q01;
    std::vector<float> action_q99;
    if (!read_safetensors_f32_vector(pre, "observation.state.q01",
                                     &state_q01) ||
        !read_safetensors_f32_vector(pre, "observation.state.q99",
                                     &state_q99) ||
        !read_safetensors_f32_vector(post, "action.q01", &action_q01) ||
        !read_safetensors_f32_vector(post, "action.q99", &action_q99)) {
        g_last_error =
            "lerobot policy stats are missing action/state q01/q99";
        return false;
    }
    if (state_q01.size() != static_cast<size_t>(state_dim) ||
        !sane_quantile_pair(state_q01, state_q99)) {
        g_last_error =
            "lerobot policy state dimension does not match config";
        return false;
    }
    if (action_q01.size() > 32 ||
        !sane_quantile_pair(action_q01, action_q99)) {
        g_last_error = "lerobot policy action dimension is invalid";
        return false;
    }
    if (config) {
        config->state_q01 = std::move(state_q01);
        config->state_q99 = std::move(state_q99);
        config->action_q01 = std::move(action_q01);
        config->action_q99 = std::move(action_q99);
    }
    g_last_error.clear();
    return true;
}

bool validate_norm_stats(const std::string& checkpoint_path,
                         int64_t state_dim,
                         flashrt::models::pi05::NativeOpenConfig* config) {
    const std::string parent = dirname(checkpoint_path);
    const std::string candidates[] = {
        join_path(checkpoint_path,
                  "assets/physical-intelligence/libero/norm_stats.json"),
        join_path(checkpoint_path, "assets/droid/norm_stats.json"),
        join_path(checkpoint_path, "norm_stats.json"),
        join_path(parent,
                  "pi05_libero/assets/physical-intelligence/libero/"
                  "norm_stats.json"),
        join_path(parent, "pi05_droid/assets/droid/norm_stats.json"),
        join_path(parent, "pi05_droid_pytorch/assets/droid/norm_stats.json"),
    };
    bool saw_malformed = false;
    std::string malformed_error;
    for (const std::string& path : candidates) {
        if (!regular_file_exists(path)) continue;
        if (validate_norm_stats_file(path, state_dim, config)) return true;
        saw_malformed = true;
        malformed_error = g_last_error;
    }
    if (validate_lerobot_policy_norm_stats(checkpoint_path, state_dim,
                                           config)) {
        return true;
    }
    g_last_error = saw_malformed
                       ? malformed_error
                       : "norm_stats.json not found for Pi0.5 native_v2";
    return false;
}

bool validate_pi05_safetensors(const std::string& checkpoint_path) {
    const std::string path = join_path(checkpoint_path, "model.safetensors");
    if (!regular_file_exists(path)) {
        g_last_error = "checkpoint_path must contain model.safetensors";
        return false;
    }
    flashrt::loader::SafetensorsFile file;
    if (!file.open(path)) {
        g_last_error = file.error();
        return false;
    }

    for (const auto& req :
         flashrt::models::pi05::native_tensor_requirements()) {
        std::string key = req.key;
        const flashrt::loader::SafetensorInfo* meta = file.find(key);
        if (!meta) {
            key = std::string("model.") + req.key;
            meta = file.find(key);
            if (!meta) {
                g_last_error = std::string("model.safetensors is missing ") +
                               req.key;
                return false;
            }
        }
        if (meta->dtype != "BF16" && meta->dtype != "F16" &&
            meta->dtype != "F32") {
            g_last_error = std::string("Pi0.5 tensor dtype is unsupported: ") +
                           req.key;
            return false;
        }
        if (meta->shape != req.shape) {
            g_last_error = std::string("Pi0.5 tensor shape mismatch: ") +
                           req.key;
            return false;
        }
    }
    g_last_error.clear();
    return true;
}

int validate_config(
    const char* config_json,
    flashrt::models::pi05::NativeOpenConfig* parsed) {
    if (!config_json) {
        g_last_error = "config_json is null";
        return -1;
    }
    flashrt::native::ConfigObject config_object;
    if (!config_object.parse(config_json)) {
        g_last_error = config_object.error();
        return -1;
    }

    std::string io;
    std::string checkpoint_path;
    std::string tokenizer_model_path;
    std::string state_prompt_mode;
    std::string precision;
    std::string calibration_path;
    std::string stage_plan;
    if (!config_object.string_field("io", &io, true) ||
        !config_object.string_field("checkpoint_path", &checkpoint_path,
                                    true) ||
        !config_object.string_field("tokenizer_model_path",
                                    &tokenizer_model_path, true) ||
        !config_object.string_field("state_prompt_mode", &state_prompt_mode,
                                    true) ||
        !config_object.string_field("precision", &precision, false) ||
        !config_object.string_field("calibration_path", &calibration_path,
                                    false) ||
        !config_object.string_field("stage_plan", &stage_plan, false)) {
        g_last_error = config_object.error();
        return -1;
    }
    if (io != "native_v2") {
        g_last_error = "Pi0.5 native open requires io='native_v2'";
        return -1;
    }
    if (state_prompt_mode != "fixed") {
        g_last_error =
            "Pi0.5 native_v2 requires state_prompt_mode='fixed'";
        return -1;
    }
    if (precision.empty()) precision = "auto";
    if (precision != "auto" && precision != "bf16" &&
        precision != "fp8_e4m3fn") {
        g_last_error =
            "precision must be 'auto', 'bf16', or 'fp8_e4m3fn'";
        return -1;
    }
    if (stage_plan.empty()) stage_plan = "full";
    const auto* execution_plan =
        flashrt::models::pi05::pi05_execution_plan(stage_plan.c_str());
    if (!execution_plan) {
        g_last_error =
            "stage_plan must be 'full' or 'context_action'";
        return -1;
    }
    stage_plan = execution_plan->name;
    if (!path_exists(checkpoint_path)) {
        g_last_error = "checkpoint_path does not exist";
        return -2;
    }
    if (!regular_file_exists(tokenizer_model_path)) {
        g_last_error = "tokenizer_model_path does not name a file";
        return -2;
    }
    if (!calibration_path.empty() &&
        !regular_file_exists(calibration_path)) {
        g_last_error = "calibration_path does not name a file";
        return -2;
    }
    if (!validate_pi05_safetensors(checkpoint_path)) {
        return -2;
    }

    int64_t max_prompt_tokens = 0;
    int64_t state_dim = 0;
    int64_t num_views = 2;
    int64_t chunk = 10;
    int64_t num_steps = 10;
    int64_t vision_pool_factor = 1;
    int64_t max_frame_width = 1280;
    int64_t max_frame_height = 720;
    if (!config_object.integer_field("max_prompt_tokens",
                                     &max_prompt_tokens) ||
        !config_object.integer_field("state_dim", &state_dim) ||
        !config_object.integer_field("num_views", &num_views) ||
        !config_object.integer_field("chunk", &chunk) ||
        !config_object.integer_field("num_steps", &num_steps) ||
        !config_object.integer_field("vision_pool_factor",
                                     &vision_pool_factor) ||
        !config_object.integer_field("max_frame_width", &max_frame_width) ||
        !config_object.integer_field("max_frame_height", &max_frame_height)) {
        g_last_error = config_object.error();
        return -1;
    }
    if (max_prompt_tokens <= 0 || max_prompt_tokens > INT_MAX) {
        g_last_error = "max_prompt_tokens must be in [1, INT_MAX]";
        return -1;
    }
    if (state_dim <= 0 || state_dim > INT_MAX) {
        g_last_error = "state_dim must be in [1, INT_MAX]";
        return -1;
    }
    if (num_views < 1 || num_views > 3) {
        g_last_error = "num_views must be in [1, 3]";
        return -1;
    }
    if (chunk <= 0 || chunk > INT_MAX) {
        g_last_error = "chunk must be in [1, INT_MAX]";
        return -1;
    }
    if (num_steps <= 0 || num_steps > INT_MAX) {
        g_last_error = "num_steps must be in [1, INT_MAX]";
        return -1;
    }
    if (vision_pool_factor != 1 && vision_pool_factor != 2 &&
        vision_pool_factor != 4) {
        g_last_error = "vision_pool_factor must be one of 1, 2, or 4";
        return -1;
    }
    if (max_frame_width <= 0 || max_frame_width > INT_MAX ||
        max_frame_height <= 0 || max_frame_height > INT_MAX) {
        g_last_error = "max_frame_width/height must be in [1, INT_MAX]";
        return -1;
    }
    flashrt::models::pi05::NativeOpenConfig config;
    config.checkpoint_path = checkpoint_path;
    config.tokenizer_model_path = tokenizer_model_path;
    config.precision = precision;
    config.calibration_path = calibration_path;
    config.stage_plan = stage_plan;
    config.max_prompt_tokens = static_cast<int>(max_prompt_tokens);
    config.state_dim = static_cast<int>(state_dim);
    config.num_views = static_cast<int>(num_views);
    config.chunk = static_cast<int>(chunk);
    config.num_steps = static_cast<int>(num_steps);
    config.vision_pool_factor = static_cast<int>(vision_pool_factor);
    config.max_frame_width = static_cast<int>(max_frame_width);
    config.max_frame_height = static_cast<int>(max_frame_height);
    if (!validate_norm_stats(checkpoint_path, state_dim, &config)) {
        return -2;
    }

    if (parsed) *parsed = std::move(config);
    g_last_error.clear();
    return 0;
}

}  // namespace

namespace flashrt {
namespace models {
namespace pi05 {

int parse_native_open_config(const char* config_json,
                             NativeOpenConfig* out,
                             std::string* error) {
    const int rc = validate_config(config_json, out);
    if (error) *error = g_last_error;
    return rc;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

extern "C" int frt_model_runtime_open_v1(const char* config_json,
                                          frt_model_runtime_v1** out) try {
    if (!out) {
        g_last_error = "out is null";
        return -1;
    }
    *out = nullptr;
    flashrt::models::pi05::NativeOpenConfig config;
    const int rc = flashrt::models::pi05::parse_native_open_config(
        config_json, &config, &g_last_error);
    if (rc != 0) return rc;
#if defined(FLASHRT_CPP_HAS_SENTENCEPIECE) && \
    (defined(FLASHRT_CPP_WITH_PI05_SM120_TARGET) || \
     defined(FLASHRT_CPP_WITH_PI05_SM110_TARGET))
    return flashrt::models::pi05::build_native_model_runtime(
        config, out, &g_last_error);
#else
    g_last_error =
        "Pi0.5 native open validated config; this build requires "
        "SentencePiece and a native graph backend";
    return -3;
#endif
} catch (const std::exception& error) {
    if (out) *out = nullptr;
    g_last_error = error.what();
    return -6;
} catch (...) {
    if (out) *out = nullptr;
    g_last_error = "Pi0.5 native open failed";
    return -6;
}

extern "C" const char* frt_pi05_native_open_last_error() {
    return g_last_error.c_str();
}
