#include "flashrt/model_runtime.h"

#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <sys/stat.h>

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
