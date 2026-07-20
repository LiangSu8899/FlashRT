#include "flashrt/cpp/models/pi05/support/native_float16.h"

#include <cstring>

namespace flashrt {
namespace models {
namespace pi05 {

std::uint16_t float_to_float16_rne(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exponent_bits = (bits >> 23) & 0xffu;
    std::uint32_t mantissa = bits & 0x7fffffu;

    if (exponent_bits == 0xffu) {
        if (!mantissa) return static_cast<std::uint16_t>(sign | 0x7c00u);
        return static_cast<std::uint16_t>(sign | 0x7e00u);
    }
    const std::int32_t exponent =
        static_cast<std::int32_t>(exponent_bits) - 127;
    if (exponent > 15) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (exponent >= -14) {
        std::uint32_t half_exponent =
            static_cast<std::uint32_t>(exponent + 15);
        std::uint32_t half_mantissa = mantissa >> 13;
        const std::uint32_t remainder = mantissa & 0x1fffu;
        if (remainder > 0x1000u ||
            (remainder == 0x1000u && (half_mantissa & 1u))) {
            if (++half_mantissa == 0x400u) {
                half_mantissa = 0;
                if (++half_exponent == 31u) {
                    return static_cast<std::uint16_t>(sign | 0x7c00u);
                }
            }
        }
        return static_cast<std::uint16_t>(
            sign | (half_exponent << 10) | half_mantissa);
    }
    if (exponent < -25) return static_cast<std::uint16_t>(sign);

    mantissa |= 0x800000u;
    const std::uint32_t shift = static_cast<std::uint32_t>(-exponent - 1);
    std::uint32_t half_mantissa = mantissa >> shift;
    const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway ||
        (remainder == halfway && (half_mantissa & 1u))) {
        ++half_mantissa;
    }
    return static_cast<std::uint16_t>(sign | half_mantissa);
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt
