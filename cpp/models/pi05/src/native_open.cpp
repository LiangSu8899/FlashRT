#include "flashrt/model_runtime.h"

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

struct TensorMeta {
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t data_begin = 0;
    uint64_t data_end = 0;
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

uint64_t file_size(const std::string& path) {
    struct stat st {};
    if (path.empty() || ::stat(path.c_str(), &st) != 0 || st.st_size < 0) {
        return 0;
    }
    return static_cast<uint64_t>(st.st_size);
}

std::string join_path(const std::string& dir, const char* leaf) {
    if (dir.empty() || dir.back() == '/') return dir + leaf;
    return dir + "/" + leaf;
}

bool read_safetensors_header(const std::string& path, std::string* header,
                             uint64_t* data_start = nullptr,
                             uint64_t* total_bytes = nullptr) {
    if (!header) return false;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        g_last_error = "unable to open safetensors file";
        return false;
    }
    unsigned char len_bytes[8] = {};
    f.read(reinterpret_cast<char*>(len_bytes), sizeof(len_bytes));
    if (f.gcount() != static_cast<std::streamsize>(sizeof(len_bytes))) {
        g_last_error = "safetensors file is too small";
        return false;
    }
    uint64_t header_len = 0;
    for (int i = 7; i >= 0; --i) {
        header_len = (header_len << 8) | len_bytes[i];
    }
    if (header_len == 0 || header_len > (128ull << 20)) {
        g_last_error = "safetensors header length is invalid";
        return false;
    }
    const uint64_t start = 8ull + header_len;
    const uint64_t size = file_size(path);
    if (size < start) {
        g_last_error = "safetensors header exceeds file size";
        return false;
    }
    header->assign(static_cast<size_t>(header_len), '\0');
    f.read(&(*header)[0], static_cast<std::streamsize>(header_len));
    if (f.gcount() != static_cast<std::streamsize>(header_len)) {
        g_last_error = "safetensors header is truncated";
        return false;
    }
    if (data_start) *data_start = start;
    if (total_bytes) *total_bytes = size;
    return true;
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

bool parse_string_property(const std::string& object,
                           const char* name,
                           std::string* out) {
    const std::string q = quoted_key(name);
    size_t p = object.find(q);
    if (p == std::string::npos) return false;
    p += q.size();
    while (p < object.size() &&
           std::isspace(static_cast<unsigned char>(object[p]))) ++p;
    if (p >= object.size() || object[p++] != ':') return false;
    while (p < object.size() &&
           std::isspace(static_cast<unsigned char>(object[p]))) ++p;
    if (p >= object.size() || object[p++] != '"') return false;
    std::string value;
    while (p < object.size() && object[p] != '"') {
        if (object[p] == '\\') return false;
        value.push_back(object[p++]);
    }
    if (p >= object.size()) return false;
    if (out) *out = value;
    return true;
}

bool parse_u64_array_property(const std::string& object,
                              const char* name,
                              std::vector<uint64_t>* out) {
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
    std::vector<uint64_t> values;
    while (p < object.size()) {
        while (p < object.size() &&
               std::isspace(static_cast<unsigned char>(object[p]))) ++p;
        if (p < object.size() && object[p] == ']') {
            ++p;
            if (out) *out = std::move(values);
            return true;
        }
        if (p >= object.size() ||
            !std::isdigit(static_cast<unsigned char>(object[p]))) {
            return false;
        }
        uint64_t value = 0;
        while (p < object.size() &&
               std::isdigit(static_cast<unsigned char>(object[p]))) {
            const uint64_t digit = static_cast<uint64_t>(object[p] - '0');
            if (value > (UINT64_MAX - digit) / 10ull) return false;
            value = value * 10ull + digit;
            ++p;
        }
        values.push_back(value);
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

bool tensor_meta(const std::string& header,
                 const std::string& key,
                 TensorMeta* meta) {
    std::string object;
    if (!object_for_key(header, key, &object)) return false;
    std::string dtype;
    std::vector<uint64_t> shape;
    std::vector<uint64_t> offsets;
    if (!parse_string_property(object, "dtype", &dtype) ||
        !parse_u64_array_property(object, "shape", &shape) ||
        !parse_u64_array_property(object, "data_offsets", &offsets) ||
        offsets.size() != 2 || offsets[1] < offsets[0]) {
        g_last_error = "safetensors tensor metadata is malformed";
        return false;
    }
    if (meta) {
        meta->dtype = std::move(dtype);
        meta->shape = std::move(shape);
        meta->data_begin = offsets[0];
        meta->data_end = offsets[1];
    }
    return true;
}

uint64_t dtype_bytes(const std::string& dtype) {
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
    if (dtype == "BF16" || dtype == "F16" || dtype == "I16" ||
        dtype == "U16") {
        return 2;
    }
    if (dtype == "I64" || dtype == "U64" || dtype == "F64") return 8;
    if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") return 1;
    return 0;
}

bool tensor_nbytes(const TensorMeta& meta, uint64_t* out) {
    const uint64_t elem = dtype_bytes(meta.dtype);
    if (!elem) return false;
    uint64_t n = elem;
    for (uint64_t dim : meta.shape) {
        if (dim == 0 || n > UINT64_MAX / dim) return false;
        n *= dim;
    }
    if (out) *out = n;
    return true;
}

bool tensor_payload_valid(const TensorMeta& meta,
                          uint64_t data_start,
                          uint64_t total_bytes) {
    uint64_t expected = 0;
    if (!tensor_nbytes(meta, &expected)) {
        g_last_error = "safetensors tensor dtype/shape is unsupported";
        return false;
    }
    if (meta.data_end < meta.data_begin ||
        meta.data_end - meta.data_begin != expected ||
        data_start > total_bytes ||
        meta.data_end > total_bytes - data_start) {
        g_last_error = "safetensors tensor byte range is invalid";
        return false;
    }
    return true;
}

bool read_safetensors_f32_vector(const std::string& path,
                                 const char* key,
                                 std::vector<float>* out) {
    if (!out) return false;
    std::string header;
    uint64_t data_start = 0;
    uint64_t total_bytes = 0;
    if (!read_safetensors_header(path, &header, &data_start, &total_bytes)) {
        return false;
    }
    TensorMeta meta;
    if (!tensor_meta(header, key, &meta)) return false;
    if (meta.dtype != "F32" || meta.shape.size() != 1 ||
        !tensor_payload_valid(meta, data_start, total_bytes)) {
        g_last_error = "safetensors F32 vector metadata is invalid";
        return false;
    }
    const uint64_t n = meta.shape[0];
    if (n > (1ull << 20)) {
        g_last_error = "safetensors vector is too large";
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        g_last_error = "unable to open safetensors file";
        return false;
    }
    f.seekg(static_cast<std::streamoff>(data_start + meta.data_begin),
            std::ios::beg);
    std::vector<float> values(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(values.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
    if (f.gcount() != static_cast<std::streamsize>(n * sizeof(float))) {
        g_last_error = "safetensors vector payload is truncated";
        return false;
    }
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

    std::string pre_header;
    std::string post_header;
    if (!read_safetensors_header(pre, &pre_header) ||
        !read_safetensors_header(post, &post_header)) {
        return false;
    }
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
    std::string header;
    uint64_t data_start = 0;
    uint64_t total_bytes = 0;
    if (!read_safetensors_header(path, &header, &data_start, &total_bytes)) {
        return false;
    }

    const char* embedding_keys[] = {
        "paligemma_with_expert.paligemma.lm_head.weight",
        "model.paligemma_with_expert.paligemma.lm_head.weight",
    };
    TensorMeta embedding;
    bool found = false;
    for (const char* key : embedding_keys) {
        if (tensor_meta(header, key, &embedding)) {
            found = true;
            break;
        }
    }
    if (!found) {
        if (g_last_error.empty()) {
            g_last_error = "model.safetensors is missing Pi0.5 embedding";
        }
        return false;
    }
    if (embedding.dtype != "BF16" && embedding.dtype != "F16" &&
        embedding.dtype != "F32") {
        g_last_error = "Pi0.5 embedding dtype is unsupported";
        return false;
    }
    if (embedding.shape.size() != 2 || embedding.shape[1] != 2048 ||
        embedding.shape[0] < 1000) {
        g_last_error = "Pi0.5 embedding shape is invalid";
        return false;
    }
    if (!tensor_payload_valid(embedding, data_start, total_bytes)) {
        return false;
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
