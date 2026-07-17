#include "flashrt/cpp/native/config_object.h"

#include <cassert>
#include <cstdint>
#include <string>

int main() {
    flashrt::native::ConfigObject config;
    assert(config.parse(
        R"({"io":"native","count":3,"enabled":true,"unused":null})"));

    std::string io;
    int64_t count = 0;
    assert(config.string_field("io", &io, true));
    assert(io == "native");
    assert(config.integer_field("count", &count));
    assert(count == 3);
    assert(config.string_field("optional", nullptr, false));
    assert(config.integer_field("optional", nullptr));

    assert(!config.string_field("missing", nullptr, true));
    assert(config.error() == "missing required field: missing");
    assert(!config.string_field("count", nullptr, true));
    assert(config.error() == "field must be a non-empty string: count");
    assert(!config.integer_field("io", nullptr));
    assert(config.error() == "field must be an integer: io");

    assert(config.parse(R"({"value":1,"value":2})"));
    int64_t value = 0;
    assert(config.integer_field("value", &value));
    assert(value == 2);

    assert(!config.parse(R"({"value":1.5})"));
    assert(config.error() == "JSON number must be an integer");
    assert(!config.parse(R"({"value":[]})"));
    assert(config.error() == "unsupported JSON value");
    assert(!config.parse(R"({"value":1} trailing)"));
    assert(config.error() == "unexpected trailing data after JSON object");
    return 0;
}
