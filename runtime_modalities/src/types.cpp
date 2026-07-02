#include "flashrt/modalities/types.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace flashrt {
namespace modalities {

Shape::Shape(std::initializer_list<std::uint64_t> values) {
    rank = static_cast<std::uint32_t>(
        std::min(values.size(), static_cast<std::size_t>(kMaxRank)));
    std::size_t i = 0;
    for (std::uint64_t v : values) {
        if (i >= kMaxRank) break;
        dims[i++] = v;
    }
}

std::uint64_t Shape::elements() const {
    if (rank == 0) return 0;
    std::uint64_t n = 1;
    for (std::uint32_t i = 0; i < rank; ++i) n *= dims[i];
    return n;
}

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::kUInt8: return 1;
        case DType::kFloat32: return 4;
        case DType::kFloat16: return 2;
        case DType::kBFloat16: return 2;
    }
    return 0;
}

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::kUInt8: return "uint8";
        case DType::kFloat32: return "float32";
        case DType::kFloat16: return "float16";
        case DType::kBFloat16: return "bfloat16";
    }
    return "unknown";
}

const char* layout_name(Layout layout) {
    switch (layout) {
        case Layout::kFlat: return "flat";
        case Layout::kHWC: return "hwc";
        case Layout::kNHWC: return "nhwc";
        case Layout::kCHW: return "chw";
        case Layout::kNCHW: return "nchw";
    }
    return "unknown";
}

std::uint16_t float_to_bfloat16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(bits >> 16);
}

float bfloat16_to_float(std::uint16_t value) {
    std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::uint16_t float_to_float16(float value) {
    std::uint32_t x = 0;
    std::memcpy(&x, &value, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mant = x & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        mant |= 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
        std::uint32_t half = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half += 1;
        return static_cast<std::uint16_t>(sign | half);
    }
    if (exp >= 31) {
        if (mant == 0) return static_cast<std::uint16_t>(sign | 0x7c00u);
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mant >> 13) | 1u);
    }

    std::uint32_t half = (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13);
    if (mant & 0x1000u) half += 1;
    return static_cast<std::uint16_t>(sign | half);
}

float float16_to_float(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
    std::uint32_t exp = (value >> 10) & 0x1fu;
    std::uint32_t mant = value & 0x03ffu;
    std::uint32_t out = 0;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f = 0.0f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

Status validate_host_tensor(const TensorView& tensor, const char* name) {
    if (!tensor.data) {
        return Status::error(StatusCode::kInvalidArgument,
                             std::string(name) + " has null data");
    }
    if (tensor.place != MemoryPlace::kHost &&
        tensor.place != MemoryPlace::kHostPinned) {
        return Status::error(StatusCode::kUnsupported,
                             std::string(name) + " is not host memory");
    }
    const std::uint64_t elem = tensor.shape.elements();
    const std::uint64_t need = elem * dtype_size(tensor.dtype);
    if (elem != 0 && tensor.bytes < need) {
        return Status::error(StatusCode::kInsufficientStorage,
                             std::string(name) + " storage is too small");
    }
    return Status::ok();
}

}  // namespace modalities
}  // namespace flashrt
