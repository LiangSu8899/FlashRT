#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace flashrt::native {

class ConfigObject {
public:
    bool parse(const char* json);

    bool string_field(const char* key,
                      std::string* out,
                      bool required) const;
    bool integer_field(const char* key, int64_t* out) const;

    const std::string& error() const { return error_; }

private:
    friend class ConfigObjectParser;

    enum class ValueType { kString, kInteger, kBool, kNull };

    struct Value {
        ValueType type = ValueType::kNull;
        std::string text;
        int64_t integer = 0;
    };

    bool fail(std::string message) const;

    std::map<std::string, Value> values_;
    mutable std::string error_;
};

}  // namespace flashrt::native
