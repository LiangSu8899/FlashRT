#include "flashrt/cpp/loader/safetensors.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace flashrt {
namespace loader {
namespace {

constexpr std::uint64_t kMaxHeaderBytes = 128ull << 20;

std::uint64_t dtype_bytes(const std::string& dtype) {
    if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" ||
        dtype == "U16") {
        return 2;
    }
    if (dtype == "I8" || dtype == "U8" || dtype == "BOOL" ||
        dtype == "F8_E4M3FN" || dtype == "F8_E5M2") {
        return 1;
    }
    return 0;
}

bool tensor_bytes(const SafetensorInfo& tensor, std::uint64_t* out) {
    std::uint64_t bytes = dtype_bytes(tensor.dtype);
    if (!bytes) return false;
    for (std::uint64_t dim : tensor.shape) {
        if (dim && bytes > std::numeric_limits<std::uint64_t>::max() / dim) {
            return false;
        }
        bytes *= dim;
    }
    if (out) *out = bytes;
    return true;
}

class HeaderParser {
public:
    HeaderParser(const char* begin, const char* end)
        : begin_(begin), cur_(begin), end_(end) {}

    bool parse(std::map<std::string, SafetensorInfo>* tensors,
               std::map<std::string, std::string>* metadata) {
        skip_ws();
        if (!consume('{')) return fail("safetensors header must be an object");
        skip_ws();
        if (consume('}')) return finish(tensors, metadata);
        while (cur_ < end_) {
            std::string key;
            if (!parse_string(&key)) return false;
            skip_ws();
            if (!consume(':')) return fail("expected ':' after tensor name");
            skip_ws();
            if (key == "__metadata__") {
                if (metadata_seen_) {
                    return fail("duplicate safetensors metadata object");
                }
                metadata_seen_ = true;
                if (!parse_metadata()) return false;
            } else {
                SafetensorInfo tensor;
                if (!parse_tensor(&tensor)) return false;
                if (!parsed_.emplace(std::move(key), std::move(tensor)).second) {
                    return fail("duplicate tensor name in safetensors header");
                }
            }
            skip_ws();
            if (consume('}')) return finish(tensors, metadata);
            if (!consume(',')) return fail("expected ',' or '}' in header");
            skip_ws();
        }
        return fail("unterminated safetensors header");
    }

    const std::string& error() const { return error_; }

private:
    bool finish(std::map<std::string, SafetensorInfo>* tensors,
                std::map<std::string, std::string>* metadata) {
        skip_ws();
        if (cur_ != end_) return fail("trailing data in safetensors header");
        if (parsed_.empty()) return fail("safetensors file contains no tensors");
        if (tensors) *tensors = std::move(parsed_);
        if (metadata) *metadata = std::move(metadata_);
        return true;
    }

    bool parse_metadata() {
        if (!consume('{')) return fail("safetensors metadata must be an object");
        skip_ws();
        if (consume('}')) return true;
        while (cur_ < end_) {
            std::string key;
            if (!parse_string(&key)) return false;
            skip_ws();
            if (!consume(':')) return fail("expected ':' in safetensors metadata");
            skip_ws();
            std::string value;
            if (cur_ >= end_ || *cur_ != '"') {
                return fail("safetensors metadata values must be strings");
            }
            if (!parse_string(&value)) return false;
            if (!metadata_.emplace(std::move(key), std::move(value)).second) {
                return fail("duplicate safetensors metadata key");
            }
            skip_ws();
            if (consume('}')) return true;
            if (!consume(',')) {
                return fail("expected ',' in safetensors metadata");
            }
            skip_ws();
        }
        return fail("unterminated safetensors metadata");
    }

    bool parse_tensor(SafetensorInfo* tensor) {
        if (!consume('{')) return fail("tensor metadata must be an object");
        bool have_dtype = false;
        bool have_shape = false;
        bool have_offsets = false;
        bool closed = false;
        std::vector<std::uint64_t> offsets;
        skip_ws();
        if (consume('}')) return fail("tensor metadata is empty");
        while (cur_ < end_) {
            std::string key;
            if (!parse_string(&key)) return false;
            skip_ws();
            if (!consume(':')) return fail("expected ':' in tensor metadata");
            skip_ws();
            if (key == "dtype") {
                if (have_dtype || !parse_string(&tensor->dtype)) {
                    return fail("invalid tensor dtype");
                }
                have_dtype = true;
            } else if (key == "shape") {
                if (have_shape || !parse_u64_array(&tensor->shape)) {
                    return fail("invalid tensor shape");
                }
                have_shape = true;
            } else if (key == "data_offsets") {
                if (have_offsets || !parse_u64_array(&offsets)) {
                    return fail("invalid tensor data_offsets");
                }
                have_offsets = true;
            } else if (!skip_value()) {
                return false;
            }
            skip_ws();
            if (consume('}')) {
                closed = true;
                break;
            }
            if (!consume(',')) return fail("expected ',' in tensor metadata");
            skip_ws();
        }
        if (!closed) return fail("unterminated tensor metadata");
        if (!have_dtype || !have_shape || !have_offsets || offsets.size() != 2 ||
            offsets[1] < offsets[0]) {
            return fail("incomplete tensor metadata");
        }
        tensor->data_offset = offsets[0];
        tensor->bytes = offsets[1] - offsets[0];
        std::uint64_t expected = 0;
        if (!tensor_bytes(*tensor, &expected)) {
            return fail("unsupported tensor dtype or overflowing shape");
        }
        if (expected != tensor->bytes) {
            return fail("tensor byte range does not match dtype and shape");
        }
        return true;
    }

    bool parse_u64_array(std::vector<std::uint64_t>* values) {
        if (!consume('[')) return false;
        std::vector<std::uint64_t> parsed;
        skip_ws();
        if (consume(']')) {
            if (values) *values = std::move(parsed);
            return true;
        }
        while (cur_ < end_) {
            std::uint64_t value = 0;
            if (!parse_u64(&value)) return false;
            parsed.push_back(value);
            skip_ws();
            if (consume(']')) {
                if (values) *values = std::move(parsed);
                return true;
            }
            if (!consume(',')) return false;
            skip_ws();
        }
        return false;
    }

    bool parse_u64(std::uint64_t* out) {
        if (cur_ >= end_ || !std::isdigit(static_cast<unsigned char>(*cur_))) {
            return false;
        }
        std::uint64_t value = 0;
        while (cur_ < end_ &&
               std::isdigit(static_cast<unsigned char>(*cur_))) {
            const std::uint64_t digit = static_cast<std::uint64_t>(*cur_ - '0');
            if (value > (std::numeric_limits<std::uint64_t>::max() - digit) /
                            10ull) {
                return false;
            }
            value = value * 10ull + digit;
            ++cur_;
        }
        if (out) *out = value;
        return true;
    }

    bool parse_string(std::string* out) {
        if (!consume('"')) return fail("expected JSON string");
        std::string value;
        while (cur_ < end_ && *cur_ != '"') {
            unsigned char c = static_cast<unsigned char>(*cur_++);
            if (c < 0x20) return fail("control character in JSON string");
            if (c != '\\') {
                value.push_back(static_cast<char>(c));
                continue;
            }
            if (cur_ >= end_) return fail("unterminated JSON escape");
            switch (*cur_++) {
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case '/': value.push_back('/'); break;
                case 'b': value.push_back('\b'); break;
                case 'f': value.push_back('\f'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                default: return fail("unsupported JSON string escape");
            }
        }
        if (!consume('"')) return fail("unterminated JSON string");
        if (out) *out = std::move(value);
        return true;
    }

    bool skip_value() {
        skip_ws();
        if (cur_ >= end_) return fail("missing JSON value");
        if (*cur_ == '"') return parse_string(nullptr);
        if (*cur_ == '{') return skip_object();
        if (*cur_ == '[') return skip_array();
        const char* literals[] = {"true", "false", "null"};
        for (const char* literal : literals) {
            const std::size_t n = std::strlen(literal);
            if (static_cast<std::size_t>(end_ - cur_) >= n &&
                std::strncmp(cur_, literal, n) == 0) {
                cur_ += n;
                return true;
            }
        }
        return skip_number();
    }

    bool skip_object() {
        if (!consume('{')) return false;
        skip_ws();
        if (consume('}')) return true;
        while (cur_ < end_) {
            if (!parse_string(nullptr)) return false;
            skip_ws();
            if (!consume(':')) return fail("expected ':' in JSON object");
            if (!skip_value()) return false;
            skip_ws();
            if (consume('}')) return true;
            if (!consume(',')) return fail("expected ',' in JSON object");
            skip_ws();
        }
        return fail("unterminated JSON object");
    }

    bool skip_array() {
        if (!consume('[')) return false;
        skip_ws();
        if (consume(']')) return true;
        while (cur_ < end_) {
            if (!skip_value()) return false;
            skip_ws();
            if (consume(']')) return true;
            if (!consume(',')) return fail("expected ',' in JSON array");
            skip_ws();
        }
        return fail("unterminated JSON array");
    }

    bool skip_number() {
        const char* start = cur_;
        if (cur_ < end_ && *cur_ == '-') ++cur_;
        if (cur_ >= end_ ||
            !std::isdigit(static_cast<unsigned char>(*cur_))) {
            cur_ = start;
            return fail("unsupported JSON value");
        }
        if (*cur_ == '0') {
            ++cur_;
        } else {
            while (cur_ < end_ &&
                   std::isdigit(static_cast<unsigned char>(*cur_))) ++cur_;
        }
        if (cur_ < end_ && *cur_ == '.') {
            ++cur_;
            const char* fractional = cur_;
            while (cur_ < end_ &&
                   std::isdigit(static_cast<unsigned char>(*cur_))) ++cur_;
            if (cur_ == fractional) return fail("invalid JSON number");
        }
        if (cur_ < end_ && (*cur_ == 'e' || *cur_ == 'E')) {
            ++cur_;
            if (cur_ < end_ && (*cur_ == '+' || *cur_ == '-')) ++cur_;
            const char* exponent = cur_;
            while (cur_ < end_ &&
                   std::isdigit(static_cast<unsigned char>(*cur_))) ++cur_;
            if (cur_ == exponent) return fail("invalid JSON number");
        }
        return true;
    }

    void skip_ws() {
        while (cur_ < end_ &&
               std::isspace(static_cast<unsigned char>(*cur_))) ++cur_;
    }

    bool consume(char c) {
        if (cur_ >= end_ || *cur_ != c) return false;
        ++cur_;
        return true;
    }

    bool fail(const char* message) {
        error_ = message;
        error_ += " at header byte ";
        error_ += std::to_string(static_cast<std::uint64_t>(cur_ - begin_));
        return false;
    }

    const char* begin_;
    const char* cur_;
    const char* end_;
    std::string error_;
    std::map<std::string, SafetensorInfo> parsed_;
    std::map<std::string, std::string> metadata_;
    bool metadata_seen_ = false;
};

}  // namespace

SafetensorsFile::~SafetensorsFile() { close(); }

SafetensorsFile::SafetensorsFile(SafetensorsFile&& other) noexcept {
    move_from(std::move(other));
}

SafetensorsFile& SafetensorsFile::operator=(SafetensorsFile&& other) noexcept {
    if (this != &other) {
        close();
        move_from(std::move(other));
    }
    return *this;
}

void SafetensorsFile::move_from(SafetensorsFile&& other) noexcept {
    fd_ = other.fd_;
    mapping_ = other.mapping_;
    mapping_bytes_ = other.mapping_bytes_;
    data_offset_ = other.data_offset_;
    path_ = std::move(other.path_);
    error_ = std::move(other.error_);
    tensors_ = std::move(other.tensors_);
    metadata_ = std::move(other.metadata_);
    other.fd_ = -1;
    other.mapping_ = nullptr;
    other.mapping_bytes_ = 0;
    other.data_offset_ = 0;
}

bool SafetensorsFile::open(const std::string& path) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
        error_ = "unable to open safetensors file: ";
        error_ += std::strerror(errno);
        return false;
    }
    struct stat st {};
    if (::fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 9) {
        error_ = "safetensors path is not a non-empty regular file";
        close();
        return false;
    }
    mapping_bytes_ = static_cast<std::uint64_t>(st.st_size);
    mapping_ = ::mmap(nullptr, static_cast<std::size_t>(mapping_bytes_),
                      PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapping_ == MAP_FAILED) {
        mapping_ = nullptr;
        error_ = "unable to mmap safetensors file: ";
        error_ += std::strerror(errno);
        close();
        return false;
    }

    const auto* bytes = static_cast<const unsigned char*>(mapping_);
    std::uint64_t header_bytes = 0;
    for (int i = 7; i >= 0; --i) {
        header_bytes = (header_bytes << 8) | bytes[i];
    }
    if (!header_bytes || header_bytes > kMaxHeaderBytes ||
        header_bytes > mapping_bytes_ - 8) {
        error_ = "safetensors header length is invalid";
        close();
        return false;
    }
    data_offset_ = 8 + header_bytes;
    const char* header = static_cast<const char*>(mapping_) + 8;
    HeaderParser parser(header, header + header_bytes);
    if (!parser.parse(&tensors_, &metadata_)) {
        error_ = parser.error();
        close();
        return false;
    }

    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans;
    spans.reserve(tensors_.size());
    const std::uint64_t payload_bytes = mapping_bytes_ - data_offset_;
    for (const auto& entry : tensors_) {
        const SafetensorInfo& tensor = entry.second;
        if (tensor.data_offset > payload_bytes ||
            tensor.bytes > payload_bytes - tensor.data_offset) {
            error_ = "tensor byte range exceeds safetensors payload: ";
            error_ += entry.first;
            close();
            return false;
        }
        spans.emplace_back(tensor.data_offset,
                           tensor.data_offset + tensor.bytes);
    }
    std::sort(spans.begin(), spans.end());
    for (std::size_t i = 1; i < spans.size(); ++i) {
        if (spans[i].first < spans[i - 1].second) {
            error_ = "overlapping tensor byte ranges in safetensors payload";
            close();
            return false;
        }
    }
    path_ = path;
    error_.clear();
    return true;
}

void SafetensorsFile::close() {
    if (mapping_) {
        ::munmap(mapping_, static_cast<std::size_t>(mapping_bytes_));
    }
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    mapping_ = nullptr;
    mapping_bytes_ = 0;
    data_offset_ = 0;
    path_.clear();
    tensors_.clear();
    metadata_.clear();
}

const SafetensorInfo* SafetensorsFile::find(const std::string& name) const {
    const auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const void* SafetensorsFile::data(const SafetensorInfo& tensor) const {
    if (!mapping_ || tensor.data_offset > mapping_bytes_ - data_offset_ ||
        tensor.bytes > mapping_bytes_ - data_offset_ - tensor.data_offset) {
        return nullptr;
    }
    return static_cast<const unsigned char*>(mapping_) + data_offset_ +
           tensor.data_offset;
}

}  // namespace loader
}  // namespace flashrt
