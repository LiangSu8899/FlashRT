#include "flashrt/model_runtime.h"
#include "flashrt/cpp/loader/safetensors.h"
#include "flashrt/cpp/models/pi05/native_weights.h"

#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <cmath>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

thread_local std::string g_last_error;

struct JsonValue {
    enum class Type { kString, kInteger, kBool, kNull };
    Type type = Type::kNull;
    std::string text;
    int64_t integer = 0;
    bool boolean = false;
};

class JsonParser {
public:
    explicit JsonParser(const char* src) : cur_(src ? src : "") {}

    bool parse_object(std::map<std::string, JsonValue>* out) {
        skip_ws();
        if (!consume('{')) return fail("config_json must be a JSON object");
        skip_ws();
        if (consume('}')) return finish(out);
        while (true) {
            std::string key;
            if (!parse_string(&key)) return false;
            skip_ws();
            if (!consume(':')) return fail("expected ':' after JSON key");
            skip_ws();
            JsonValue value;
            if (!parse_value(&value)) return false;
            values_[key] = value;
            skip_ws();
            if (consume('}')) return finish(out);
            if (!consume(',')) return fail("expected ',' or '}' in object");
            skip_ws();
        }
    }

    const std::string& error() const { return error_; }

private:
    bool finish(std::map<std::string, JsonValue>* out) {
        skip_ws();
        if (*cur_) return fail("unexpected trailing data after JSON object");
        if (out) *out = std::move(values_);
        return true;
    }

    void skip_ws() {
        while (*cur_ && std::isspace(static_cast<unsigned char>(*cur_))) ++cur_;
    }

    bool consume(char c) {
        if (*cur_ != c) return false;
        ++cur_;
        return true;
    }

    bool parse_value(JsonValue* value) {
        if (!value) return fail("internal parser error");
        if (*cur_ == '"') {
            value->type = JsonValue::Type::kString;
            return parse_string(&value->text);
        }
        if (*cur_ == '-' || std::isdigit(static_cast<unsigned char>(*cur_))) {
            value->type = JsonValue::Type::kInteger;
            return parse_integer(&value->integer);
        }
        if (match_literal("true")) {
            value->type = JsonValue::Type::kBool;
            value->boolean = true;
            return true;
        }
        if (match_literal("false")) {
            value->type = JsonValue::Type::kBool;
            value->boolean = false;
            return true;
        }
        if (match_literal("null")) {
            value->type = JsonValue::Type::kNull;
            return true;
        }
        return fail("unsupported JSON value");
    }

    bool parse_string(std::string* out) {
        if (!consume('"')) return fail("expected JSON string");
        std::string s;
        while (*cur_ && *cur_ != '"') {
            unsigned char c = static_cast<unsigned char>(*cur_++);
            if (c < 0x20) return fail("control character in JSON string");
            if (c != '\\') {
                s.push_back(static_cast<char>(c));
                continue;
            }
            char esc = *cur_++;
            switch (esc) {
                case '"': s.push_back('"'); break;
                case '\\': s.push_back('\\'); break;
                case '/': s.push_back('/'); break;
                case 'b': s.push_back('\b'); break;
                case 'f': s.push_back('\f'); break;
                case 'n': s.push_back('\n'); break;
                case 'r': s.push_back('\r'); break;
                case 't': s.push_back('\t'); break;
                default:
                    return fail("unsupported JSON string escape");
            }
        }
        if (!consume('"')) return fail("unterminated JSON string");
        if (out) *out = std::move(s);
        return true;
    }

    bool parse_integer(int64_t* out) {
        const char* begin = cur_;
        if (*cur_ == '-') ++cur_;
        if (!std::isdigit(static_cast<unsigned char>(*cur_))) {
            return fail("expected JSON integer");
        }
        if (*cur_ == '0') {
            ++cur_;
        } else {
            while (std::isdigit(static_cast<unsigned char>(*cur_))) ++cur_;
        }
        if (*cur_ == '.' || *cur_ == 'e' || *cur_ == 'E') {
            return fail("JSON number must be an integer");
        }
        errno = 0;
        char* end = nullptr;
        const long long value = std::strtoll(begin, &end, 10);
        if (errno || end != cur_) return fail("integer value is out of range");
        if (out) *out = static_cast<int64_t>(value);
        return true;
    }

    bool match_literal(const char* text) {
        const std::size_t n = std::strlen(text);
        if (std::strncmp(cur_, text, n) != 0) return false;
        cur_ += n;
        return true;
    }

    bool fail(const char* msg) {
        error_ = msg;
        return false;
    }

    const char* cur_;
    std::string error_;
    std::map<std::string, JsonValue> values_;
};

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

bool norm_block_dims(const std::string& json,
                     const char* block_name,
                     size_t* dims) {
    std::string block;
    if (!object_for_key(json, block_name, &block)) return false;
    std::vector<double> q01;
    std::vector<double> q99;
    if (!parse_f64_array_property(block, "q01", &q01) ||
        !parse_f64_array_property(block, "q99", &q99) ||
        !sane_quantile_pair(q01, q99)) {
        return false;
    }
    if (dims) *dims = q01.size();
    return true;
}

bool validate_norm_stats_file(const std::string& path,
                              int64_t state_dim) {
    std::string json;
    if (!read_text_file(path, &json)) return false;
    size_t action_dims = 0;
    size_t state_dims = 0;
    if (!norm_block_dims(json, "actions", &action_dims) ||
        !norm_block_dims(json, "state", &state_dims)) {
        g_last_error = "norm_stats.json is missing actions/state q01/q99";
        return false;
    }
    if (action_dims == 0 || action_dims > 32) {
        g_last_error = "norm_stats action dimension is invalid";
        return false;
    }
    if (state_dims != static_cast<size_t>(state_dim)) {
        g_last_error = "norm_stats state dimension does not match config";
        return false;
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
                                        int64_t state_dim) {
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
    g_last_error.clear();
    return true;
}

bool validate_norm_stats(const std::string& checkpoint_path,
                         int64_t state_dim) {
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
        if (validate_norm_stats_file(path, state_dim)) return true;
        saw_malformed = true;
        malformed_error = g_last_error;
    }
    if (validate_lerobot_policy_norm_stats(checkpoint_path, state_dim)) {
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

bool string_field(const std::map<std::string, JsonValue>& obj,
                  const char* key,
                  std::string* out,
                  bool required) {
    auto it = obj.find(key);
    if (it == obj.end()) {
        if (!required) return true;
        g_last_error = std::string("missing required field: ") + key;
        return false;
    }
    if (it->second.type != JsonValue::Type::kString ||
        it->second.text.empty()) {
        g_last_error = std::string("field must be a non-empty string: ") + key;
        return false;
    }
    if (out) *out = it->second.text;
    return true;
}

bool integer_field(const std::map<std::string, JsonValue>& obj,
                   const char* key,
                   int64_t* out) {
    auto it = obj.find(key);
    if (it == obj.end()) return true;
    if (it->second.type != JsonValue::Type::kInteger) {
        g_last_error = std::string("field must be an integer: ") + key;
        return false;
    }
    if (out) *out = it->second.integer;
    return true;
}

int validate_config(const char* config_json) {
    if (!config_json) {
        g_last_error = "config_json is null";
        return -1;
    }
    std::map<std::string, JsonValue> obj;
    JsonParser parser(config_json);
    if (!parser.parse_object(&obj)) {
        g_last_error = parser.error();
        return -1;
    }

    std::string io;
    std::string checkpoint_path;
    std::string tokenizer_model_path;
    std::string state_prompt_mode;
    if (!string_field(obj, "io", &io, true) ||
        !string_field(obj, "checkpoint_path", &checkpoint_path, true) ||
        !string_field(obj, "tokenizer_model_path", &tokenizer_model_path,
                      true) ||
        !string_field(obj, "state_prompt_mode", &state_prompt_mode, true)) {
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
    if (!path_exists(checkpoint_path)) {
        g_last_error = "checkpoint_path does not exist";
        return -2;
    }
    if (!regular_file_exists(tokenizer_model_path)) {
        g_last_error = "tokenizer_model_path does not name a file";
        return -2;
    }
    if (!validate_pi05_safetensors(checkpoint_path)) {
        return -2;
    }

    int64_t max_prompt_tokens = 0;
    int64_t state_dim = 0;
    int64_t num_views = 0;
    int64_t chunk = 0;
    if (!integer_field(obj, "max_prompt_tokens", &max_prompt_tokens) ||
        !integer_field(obj, "state_dim", &state_dim) ||
        !integer_field(obj, "num_views", &num_views) ||
        !integer_field(obj, "chunk", &chunk)) {
        return -1;
    }
    if (max_prompt_tokens < 200) {
        g_last_error = "max_prompt_tokens must be at least 200";
        return -1;
    }
    if (state_dim <= 0) {
        g_last_error = "state_dim must be positive";
        return -1;
    }
    if (num_views && (num_views < 1 || num_views > 3)) {
        g_last_error = "num_views must be in [1, 3]";
        return -1;
    }
    if (chunk && chunk <= 0) {
        g_last_error = "chunk must be positive";
        return -1;
    }
    if (!validate_norm_stats(checkpoint_path, state_dim)) {
        return -2;
    }

    g_last_error =
        "Pi0.5 native open validated config; native graph capture is not "
        "implemented yet";
    return -3;
}

}  // namespace

extern "C" int frt_model_runtime_open_v1(const char* config_json,
                                          frt_model_runtime_v1** out) {
    if (!out) {
        g_last_error = "out is null";
        return -1;
    }
    *out = nullptr;
    return validate_config(config_json);
}

extern "C" const char* frt_pi05_native_open_last_error() {
    return g_last_error.c_str();
}
