#include "flashrt/cpp/models/pi05/native_quantization.h"

#include <cassert>
#include <cmath>
#include <cstdio>

int main() {
    using namespace flashrt::models::pi05;

    NativeFloatTensor fp8_input{
        {1, 5}, {-448.0f, -1.0f, 0.0f, 1.0f, 448.0f}};
    NativeFp8Tensor fp8;
    assert(native_quantize_fp8_e4m3(fp8_input, false, &fp8).ok_status());
    assert(fp8.shape == fp8_input.shape);
    assert(fp8.scale == 1.0f);
    assert(fp8.values == std::vector<std::uint8_t>(
                              {0xfe, 0xb8, 0x00, 0x38, 0x7e}));

    NativeF16Tensor fp16_input{{2, 3}, {}};
    for (float value : {-448.0f, -1.0f, 0.0f, 1.0f, 224.0f, 448.0f}) {
        fp16_input.values.push_back(
            flashrt::modalities::float_to_float16(value));
    }
    assert(native_quantize_fp8_e4m3(fp16_input, true, &fp8).ok_status());
    assert(fp8.shape == std::vector<std::uint64_t>({3, 2}));
    assert(fp8.scale == 1.0f);
    assert(fp8.values == std::vector<std::uint8_t>(
                              {0xfe, 0x38, 0xb8, 0x76, 0x00, 0x7e}));

    NativeF16Tensor reciprocal_input{{1, 2}, {}};
    for (float value : {-0.10406494140625f, 3.0078125f}) {
        reciprocal_input.values.push_back(
            flashrt::modalities::float_to_float16(value));
    }
    assert(native_quantize_fp8_e4m3(
               reciprocal_input, false, &fp8).ok_status());
    assert(fp8.scale == 0.0067138671875f);
    assert(fp8.values == std::vector<std::uint8_t>({0xd7, 0x7e}));

    NativeFloatTensor int8_input{{2, 3}, {1, 2, 3, 4, 5, 6}};
    NativeInt8Tensor int8;
    assert(native_quantize_int8_per_output(int8_input, &int8).ok_status());
    assert(int8.shape == std::vector<std::uint64_t>({3, 2}));
    assert(int8.values ==
           std::vector<std::int8_t>({32, 127, 51, 127, 64, 127}));
    assert(int8.scales.size() == 3);
    assert(std::fabs(int8.scales[0] - 4.0f / 127.0f) < 1e-9f);
    assert(std::fabs(int8.scales[1] - 5.0f / 127.0f) < 1e-9f);
    assert(std::fabs(int8.scales[2] - 6.0f / 127.0f) < 1e-9f);

    NativeFloatTensor invalid{{2}, {1, 2}};
    assert(!native_quantize_fp8_e4m3(invalid, false, &fp8).ok_status());
    assert(!native_quantize_int8_per_output(invalid, &int8).ok_status());
    std::printf("PASS - Pi0.5 native weight quantization\n");
    return 0;
}
