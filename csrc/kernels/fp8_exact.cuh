#pragma once

#include <cuda_fp8.h>
#include <cuda_runtime.h>

// CUDA selects a native float-to-FP8 instruction for SM89+ and a software
// conversion below SM89. Keep persisted and replay-visible E4M3 values
// independent of the compilation target while retaining a float-only path.
__device__ __forceinline__ __nv_fp8_storage_t
flashrt_fp8_e4m3_storage_rn(float value) {
    const unsigned int bits = __float_as_uint(value);
    const unsigned int magnitude_bits = bits & 0x7fffffffu;
    const unsigned int sign = (bits >> 24u) & 0x80u;

    if (magnitude_bits > 0x7f800000u) return 0x7fu;
    if (magnitude_bits >= 0x43e00000u) {
        return static_cast<__nv_fp8_storage_t>(sign | 0x7eu);
    }
    if (magnitude_bits <= 0x3a800000u) {
        return static_cast<__nv_fp8_storage_t>(sign);
    }
    if (magnitude_bits < 0x3c800000u) {
        const float magnitude = __uint_as_float(magnitude_bits);
        const unsigned int rounded =
            static_cast<unsigned int>(__float2int_rn(magnitude * 512.0f));
        return static_cast<__nv_fp8_storage_t>(sign | rounded);
    }

    const unsigned int mantissa = magnitude_bits & 0x7fffffu;
    const unsigned int remainder = mantissa & 0xfffffu;
    if (remainder != 0x80000u) {
        return __nv_cvt_float_to_fp8(
            value, __NV_SATFINITE, __NV_E4M3);
    }
    const unsigned int exponent = magnitude_bits >> 23u;
    unsigned int result = ((exponent - 120u) << 3u) | (mantissa >> 20u);
    if (result & 1u) ++result;
    if (result > 0x7eu) result = 0x7eu;
    return static_cast<__nv_fp8_storage_t>(sign | result);
}

__device__ __forceinline__ __nv_fp8_e4m3
flashrt_fp8_e4m3_rn(float value) {
    __nv_fp8_e4m3 result;
    result.__x = flashrt_fp8_e4m3_storage_rn(value);
    return result;
}
