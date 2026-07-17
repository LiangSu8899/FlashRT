#include "flashrt/cpp/native/config_object.h"

#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace flashrt::native {

class ConfigObjectParser {
public:
    explicit ConfigObjectParser(const char* source)
        : current_(source ? source : "") {}

    bool parse(std::map<std::string, ConfigObject::Value>* values,
               std::string* error) {
        values_ = values;
        error_ = error;
        skip_whitespace();
        if (!consume('{')) return fail("config_json must be a JSON object");
        skip_whitespace();
        if (consume('}')) return finish();

        while (true) {
            std::string key;
            if (!parse_string(&key)) return false;
            skip_whitespace();
            if (!consume(':')) return fail("expected ':' after JSON key");
            skip_whitespace();

            ConfigObject::Value value;
            if (!parse_value(&value)) return false;
            (*values_)[key] = std::move(value);

            skip_whitespace();
            if (consume('}')) return finish();
            if (!consume(',')) return fail("expected ',' or '}' in object");
            skip_whitespace();
        }
    }

private:
    bool finish() {
        skip_whitespace();
        if (*current_) {
            return fail("unexpected trailing data after JSON object");
        }
        return true;
    }

    void skip_whitespace() {
        while (*current_ &&
               std::isspace(static_cast<unsigned char>(*current_))) {
            ++current_;
        }
    }

    bool consume(char value) {
        if (*current_ != value) return false;
        ++current_;
        return true;
    }

    bool parse_value(ConfigObject::Value* value) {
        if (!value) return fail("internal parser error");
        if (*current_ == '"') {
            value->type = ConfigObject::ValueType::kString;
            return parse_string(&value->text);
        }
        if (*current_ == '-' ||
            std::isdigit(static_cast<unsigned char>(*current_))) {
            value->type = ConfigObject::ValueType::kInteger;
            return parse_integer(&value->integer);
        }
        if (match_literal("true")) {
            value->type = ConfigObject::ValueType::kBool;
            return true;
        }
        if (match_literal("false")) {
            value->type = ConfigObject::ValueType::kBool;
            return true;
        }
        if (match_literal("null")) {
            value->type = ConfigObject::ValueType::kNull;
            return true;
        }
        return fail("unsupported JSON value");
    }

    bool parse_string(std::string* out) {
        if (!consume('"')) return fail("expected JSON string");
        std::string result;
        while (*current_ && *current_ != '"') {
            const unsigned char value =
                static_cast<unsigned char>(*current_++);
            if (value < 0x20) {
                return fail("control character in JSON string");
            }
            if (value != '\\') {
                result.push_back(static_cast<char>(value));
                continue;
            }

            const char escape = *current_++;
            switch (escape) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                default: return fail("unsupported JSON string escape");
            }
        }
        if (!consume('"')) return fail("unterminated JSON string");
        if (out) *out = std::move(result);
        return true;
    }

    bool parse_integer(int64_t* out) {
        const char* begin = current_;
        if (*current_ == '-') ++current_;
        if (!std::isdigit(static_cast<unsigned char>(*current_))) {
            return fail("expected JSON integer");
        }
        if (*current_ == '0') {
            ++current_;
        } else {
            while (std::isdigit(static_cast<unsigned char>(*current_))) {
                ++current_;
            }
        }
        if (*current_ == '.' || *current_ == 'e' || *current_ == 'E') {
            return fail("JSON number must be an integer");
        }

        errno = 0;
        char* end = nullptr;
        const long long value = std::strtoll(begin, &end, 10);
        if (errno || end != current_) {
            return fail("integer value is out of range");
        }
        if (out) *out = static_cast<int64_t>(value);
        return true;
    }

    bool match_literal(const char* text) {
        const std::size_t size = std::strlen(text);
        if (std::strncmp(current_, text, size) != 0) return false;
        current_ += size;
        return true;
    }

    bool fail(const char* message) {
        if (error_) *error_ = message;
        return false;
    }

    const char* current_;
    std::map<std::string, ConfigObject::Value>* values_ = nullptr;
    std::string* error_ = nullptr;
};

bool ConfigObject::parse(const char* json) {
    values_.clear();
    error_.clear();
    ConfigObjectParser parser(json);
    return parser.parse(&values_, &error_);
}

bool ConfigObject::string_field(const char* key,
                                std::string* out,
                                bool required) const {
    const auto it = values_.find(key);
    if (it == values_.end()) {
        if (!required) return true;
        return fail(std::string("missing required field: ") + key);
    }
    if (it->second.type != ValueType::kString || it->second.text.empty()) {
        return fail(std::string("field must be a non-empty string: ") + key);
    }
    if (out) *out = it->second.text;
    return true;
}

bool ConfigObject::integer_field(const char* key, int64_t* out) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return true;
    if (it->second.type != ValueType::kInteger) {
        return fail(std::string("field must be an integer: ") + key);
    }
    if (out) *out = it->second.integer;
    return true;
}

bool ConfigObject::fail(std::string message) const {
    error_ = std::move(message);
    return false;
}

}  // namespace flashrt::native
