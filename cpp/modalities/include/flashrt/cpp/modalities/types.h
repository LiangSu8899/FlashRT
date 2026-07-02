#ifndef FLASHRT_MODALITIES_TYPES_H
#define FLASHRT_MODALITIES_TYPES_H

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

namespace flashrt {
namespace modalities {

enum class DType {
    kUInt8,
    kFloat32,
    kFloat16,
    kBFloat16,
};

enum class MemoryPlace {
    kHost,
    kHostPinned,
    kDevice,
    kExternal,
};

enum class Layout {
    kFlat,
    kHWC,
    kNHWC,
    kCHW,
    kNCHW,
};

enum class PixelFormat {
    kRGB8,
    kBGR8,
    kRGBA8,
    kBGRA8,
    kGRAY8,
};

enum class StatusCode {
    kOk = 0,
    kInvalidArgument,
    kNotFound,
    kUnsupported,
    kShapeMismatch,
    kInsufficientStorage,
    kBackend,
};

struct Status {
    StatusCode code = StatusCode::kOk;
    std::string message;

    static Status ok() { return {}; }
    static Status error(StatusCode c, std::string msg) {
        Status s;
        s.code = c;
        s.message = std::move(msg);
        return s;
    }
    bool ok_status() const { return code == StatusCode::kOk; }
};

struct Shape {
    static constexpr std::size_t kMaxRank = 8;

    std::uint64_t dims[kMaxRank] = {};
    std::uint32_t rank = 0;

    Shape() = default;
    Shape(std::initializer_list<std::uint64_t> values);

    std::uint64_t elements() const;
};

struct TensorView {
    void* data = nullptr;
    std::uint64_t bytes = 0;
    DType dtype = DType::kUInt8;
    MemoryPlace place = MemoryPlace::kHost;
    Layout layout = Layout::kFlat;
    Shape shape;
};

std::size_t dtype_size(DType dtype);
const char* dtype_name(DType dtype);
const char* layout_name(Layout layout);

std::uint16_t float_to_bfloat16(float value);
float bfloat16_to_float(std::uint16_t value);
std::uint16_t float_to_float16(float value);
float float16_to_float(std::uint16_t value);

Status validate_host_tensor(const TensorView& tensor, const char* name);

}  // namespace modalities
}  // namespace flashrt

#endif  // FLASHRT_MODALITIES_TYPES_H
