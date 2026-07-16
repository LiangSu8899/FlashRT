#include "flashrt/cpp/models/pi05/native_calibration.h"

#include "flashrt/cpp/loader/safetensors.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

constexpr char kSchema[] = "flashrt.pi05.fp8_calibration.v1";
constexpr char kModel[] = "pi05";
constexpr char kPrecision[] = "fp8_e4m3fn";
constexpr char kTensorDtype[] = "float16";
constexpr char kReducer[] = "linear_percentile";
constexpr std::size_t kEncoderScaleCount = 18 * 4;

modalities::Status invalid(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status not_found(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kNotFound,
                                     message);
}

modalities::Status backend(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

bool valid_digest(const std::string& value) {
    if (value.size() != 64) return false;
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

bool valid_scales(const std::vector<float>& scales) {
    return std::all_of(scales.begin(), scales.end(), [](float value) {
        return std::isfinite(value) && value > 0.0f;
    });
}

std::string json_string(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : value) {
        switch (c) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (c < 0x20) {
                    escaped += "\\u00";
                    escaped.push_back(hex[c >> 4]);
                    escaped.push_back(hex[c & 0xf]);
                } else {
                    escaped.push_back(static_cast<char>(c));
                }
        }
    }
    escaped.push_back('"');
    return escaped;
}

std::string decimal(double value) {
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

std::map<std::string, std::string> artifact_metadata(
    const NativeCalibrationArtifact& artifact) {
    return {
        {"schema", kSchema},
        {"model", kModel},
        {"precision", kPrecision},
        {"tensor_dtype", kTensorDtype},
        {"reducer", kReducer},
        {"hardware", artifact.hardware},
        {"weights_sha256", artifact.weights_sha256},
        {"tokenizer_sha256", artifact.tokenizer_sha256},
        {"num_views", std::to_string(artifact.num_views)},
        {"max_prompt_tokens", std::to_string(artifact.max_prompt_tokens)},
        {"state_dim", std::to_string(artifact.state_dim)},
        {"chunk_size", std::to_string(artifact.chunk_size)},
        {"num_steps", std::to_string(artifact.num_steps)},
        {"vision_pool_factor", std::to_string(artifact.vision_pool_factor)},
        {"sample_count", std::to_string(artifact.sample_count)},
        {"percentile", decimal(artifact.percentile)},
    };
}

std::string make_header(const NativeCalibrationArtifact& artifact) {
    const auto metadata = artifact_metadata(artifact);
    const std::uint64_t encoder_bytes =
        artifact.encoder_scales.size() * sizeof(float);
    const std::uint64_t decoder_bytes =
        artifact.decoder_scales.size() * sizeof(float);
    std::ostringstream out;
    out << "{\"__metadata__\":{";
    bool first = true;
    for (const auto& entry : metadata) {
        if (!first) out << ',';
        first = false;
        out << json_string(entry.first) << ':' << json_string(entry.second);
    }
    out << "},\"encoder_scales\":{\"dtype\":\"F32\",\"shape\":["
        << artifact.encoder_scales.size()
        << "],\"data_offsets\":[0," << encoder_bytes
        << "]},\"decoder_scales\":{\"dtype\":\"F32\",\"shape\":["
        << artifact.decoder_scales.size() << "],\"data_offsets\":["
        << encoder_bytes << ',' << encoder_bytes + decoder_bytes << "]}}";
    std::string header = out.str();
    while (header.size() % 8) header.push_back(' ');
    return header;
}

bool write_all(int fd, const void* data, std::size_t bytes) {
    const auto* cursor = static_cast<const unsigned char*>(data);
    while (bytes) {
        const ssize_t written = ::write(fd, cursor, bytes);
        if (written < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (!written) return false;
        cursor += written;
        bytes -= static_cast<std::size_t>(written);
    }
    return true;
}

std::string parent_directory(const std::string& path) {
    const std::size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) return ".";
    if (slash == 0) return "/";
    return path.substr(0, slash);
}

bool parse_int(const std::map<std::string, std::string>& metadata,
               const char* key,
               int* value) {
    const auto it = metadata.find(key);
    if (it == metadata.end() || it->second.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(it->second.c_str(), &end, 10);
    if (errno || end != it->second.c_str() + it->second.size() ||
        parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    if (value) *value = static_cast<int>(parsed);
    return true;
}

bool parse_u64(const std::map<std::string, std::string>& metadata,
               const char* key,
               std::uint64_t* value) {
    const auto it = metadata.find(key);
    if (it == metadata.end() || it->second.empty() || it->second[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed =
        std::strtoull(it->second.c_str(), &end, 10);
    if (errno || end != it->second.c_str() + it->second.size()) return false;
    if (value) *value = static_cast<std::uint64_t>(parsed);
    return true;
}

bool parse_double(const std::map<std::string, std::string>& metadata,
                  const char* key,
                  double* value) {
    const auto it = metadata.find(key);
    if (it == metadata.end() || it->second.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(it->second.c_str(), &end);
    if (errno || end != it->second.c_str() + it->second.size() ||
        !std::isfinite(parsed)) {
        return false;
    }
    if (value) *value = parsed;
    return true;
}

bool metadata_is(const std::map<std::string, std::string>& metadata,
                 const char* key,
                 const char* expected) {
    const auto it = metadata.find(key);
    return it != metadata.end() && it->second == expected;
}

bool copy_f32_tensor(const loader::SafetensorsFile& file,
                     const char* name,
                     std::vector<float>* values) {
    if (!values) return false;
    const loader::SafetensorInfo* tensor = file.find(name);
    if (!tensor || tensor->dtype != "F32" || tensor->shape.size() != 1 ||
        tensor->shape[0] >
            std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        tensor->bytes != tensor->shape[0] * sizeof(float)) {
        return false;
    }
    std::vector<float> copied(static_cast<std::size_t>(tensor->shape[0]));
    if (!copied.empty()) {
        const void* data = file.data(*tensor);
        if (!data) return false;
        std::memcpy(copied.data(), data, tensor->bytes);
    }
    *values = std::move(copied);
    return true;
}

}  // namespace

modalities::Status validate_native_calibration_artifact(
    const NativeCalibrationArtifact& artifact) {
    if (artifact.hardware.empty() || !valid_digest(artifact.weights_sha256) ||
        !valid_digest(artifact.tokenizer_sha256) || artifact.num_views < 1 ||
        artifact.num_views > 3 || artifact.max_prompt_tokens < 1 ||
        artifact.state_dim < 1 || artifact.chunk_size < 1 ||
        artifact.num_steps < 1 ||
        (artifact.vision_pool_factor != 1 &&
         artifact.vision_pool_factor != 2 &&
         artifact.vision_pool_factor != 4) ||
        !artifact.sample_count || !std::isfinite(artifact.percentile) ||
        artifact.percentile < 0.0 || artifact.percentile > 100.0) {
        return invalid("native calibration metadata is invalid");
    }
    if (artifact.encoder_scales.size() != kEncoderScaleCount ||
        artifact.decoder_scales.size() !=
            static_cast<std::size_t>(artifact.num_steps) * 18 * 4 ||
        !valid_scales(artifact.encoder_scales) ||
        !valid_scales(artifact.decoder_scales)) {
        return invalid("native calibration scale payload is invalid");
    }
    return modalities::Status::ok();
}

modalities::Status save_native_calibration_artifact(
    const std::string& path,
    const NativeCalibrationArtifact& artifact) {
    if (path.empty()) return invalid("native calibration path is empty");
    modalities::Status st = validate_native_calibration_artifact(artifact);
    if (!st.ok_status()) return st;

    const std::uint16_t endian_probe = 1;
    if (*reinterpret_cast<const std::uint8_t*>(&endian_probe) != 1) {
        return backend("safetensors writing requires a little-endian host");
    }
    const std::string header = make_header(artifact);
    std::string temporary = path + ".tmp.XXXXXX";
    std::vector<char> writable(temporary.begin(), temporary.end());
    writable.push_back('\0');
    const int fd = ::mkstemp(writable.data());
    if (fd < 0) {
        return backend(std::string("unable to create calibration artifact: ") +
                       std::strerror(errno));
    }
    temporary = writable.data();
    bool ok = true;
    unsigned char header_size[8];
    std::uint64_t size = header.size();
    for (int i = 0; i < 8; ++i) {
        header_size[i] = static_cast<unsigned char>((size >> (8 * i)) & 0xffu);
    }
    ok = write_all(fd, header_size, sizeof(header_size)) &&
         write_all(fd, header.data(), header.size()) &&
         write_all(fd, artifact.encoder_scales.data(),
                   artifact.encoder_scales.size() * sizeof(float)) &&
         write_all(fd, artifact.decoder_scales.data(),
                   artifact.decoder_scales.size() * sizeof(float)) &&
         ::fsync(fd) == 0;
    const int close_rc = ::close(fd);
    if (!ok || close_rc != 0 || ::rename(temporary.c_str(), path.c_str()) != 0) {
        const int saved_errno = errno;
        ::unlink(temporary.c_str());
        return backend(std::string("unable to publish calibration artifact: ") +
                       std::strerror(saved_errno));
    }
    const std::string parent = parent_directory(path);
    const int dir_fd = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dir_fd >= 0) {
        ::fsync(dir_fd);
        ::close(dir_fd);
    }
    return modalities::Status::ok();
}

modalities::Status load_native_calibration_artifact(
    const std::string& path,
    NativeCalibrationArtifact* artifact) {
    if (path.empty() || !artifact) {
        return invalid("native calibration load arguments are invalid");
    }
    loader::SafetensorsFile file;
    if (!file.open(path)) return not_found(file.error());
    const auto& metadata = file.metadata();
    if (!metadata_is(metadata, "schema", kSchema) ||
        !metadata_is(metadata, "model", kModel) ||
        !metadata_is(metadata, "precision", kPrecision) ||
        !metadata_is(metadata, "tensor_dtype", kTensorDtype) ||
        !metadata_is(metadata, "reducer", kReducer)) {
        return invalid("native calibration artifact schema is incompatible");
    }
    NativeCalibrationArtifact loaded;
    const auto hardware = metadata.find("hardware");
    const auto weights = metadata.find("weights_sha256");
    const auto tokenizer = metadata.find("tokenizer_sha256");
    if (hardware == metadata.end() || weights == metadata.end() ||
        tokenizer == metadata.end() ||
        !parse_int(metadata, "num_views", &loaded.num_views) ||
        !parse_int(metadata, "max_prompt_tokens", &loaded.max_prompt_tokens) ||
        !parse_int(metadata, "state_dim", &loaded.state_dim) ||
        !parse_int(metadata, "chunk_size", &loaded.chunk_size) ||
        !parse_int(metadata, "num_steps", &loaded.num_steps) ||
        !parse_int(metadata, "vision_pool_factor", &loaded.vision_pool_factor) ||
        !parse_u64(metadata, "sample_count", &loaded.sample_count) ||
        !parse_double(metadata, "percentile", &loaded.percentile) ||
        !copy_f32_tensor(file, "encoder_scales", &loaded.encoder_scales) ||
        !copy_f32_tensor(file, "decoder_scales", &loaded.decoder_scales)) {
        return invalid("native calibration artifact metadata is incomplete");
    }
    loaded.hardware = hardware->second;
    loaded.weights_sha256 = weights->second;
    loaded.tokenizer_sha256 = tokenizer->second;
    modalities::Status st = validate_native_calibration_artifact(loaded);
    if (!st.ok_status()) return st;
    *artifact = std::move(loaded);
    return modalities::Status::ok();
}

modalities::Status reduce_native_calibration_samples(
    const std::vector<std::vector<float>>& samples,
    double percentile,
    std::vector<float>* reduced) {
    if (!reduced || samples.empty() || !std::isfinite(percentile) ||
        percentile < 0.0 || percentile > 100.0 || samples.front().empty()) {
        return invalid("native calibration reduction input is invalid");
    }
    const std::size_t points = samples.front().size();
    for (const auto& sample : samples) {
        if (sample.size() != points || !valid_scales(sample)) {
            return invalid("native calibration samples are incompatible");
        }
    }
    std::vector<float> result(points);
    std::vector<double> column(samples.size());
    const double rank =
        percentile * static_cast<double>(samples.size() - 1) / 100.0;
    const std::size_t lower = static_cast<std::size_t>(std::floor(rank));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(rank));
    const double fraction = rank - static_cast<double>(lower);
    for (std::size_t point = 0; point < points; ++point) {
        for (std::size_t sample = 0; sample < samples.size(); ++sample) {
            column[sample] = static_cast<double>(samples[sample][point]);
        }
        std::sort(column.begin(), column.end());
        const double value = column[lower] +
                             (column[upper] - column[lower]) * fraction;
        result[point] = static_cast<float>(value);
    }
    *reduced = std::move(result);
    return modalities::Status::ok();
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt
