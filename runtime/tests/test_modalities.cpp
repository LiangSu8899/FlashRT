#include "flashrt/modalities/action.h"
#include "flashrt/modalities/vision.h"
#include "flashrt/models/pi05/cpp_runtime_io.h"
#include "flashrt/models/pi05/cpp_runtime_spec.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using flashrt::modalities::DType;
using flashrt::modalities::Layout;
using flashrt::modalities::MemoryPlace;
using flashrt::modalities::PixelFormat;
using flashrt::modalities::Shape;
using flashrt::modalities::StatusCode;
using flashrt::modalities::TensorView;
using flashrt::modalities::VisionFrame;
using flashrt::modalities::bfloat16_to_float;
using flashrt::modalities::float_to_bfloat16;
using flashrt::modalities::postprocess_action_cpu;
using flashrt::modalities::preprocess_vision_cpu;
using flashrt::modalities::required_vision_output_bytes;

namespace {

void test_pi05_vision_spec_and_preprocess() {
    const auto spec = flashrt::models::pi05::vision_preprocess_spec(2);
    assert(spec.view_order.size() == 2);
    assert(spec.view_order[0] == "image");
    assert(spec.view_order[1] == "wrist_image");
    assert(spec.target_width == 224);
    assert(spec.output_dtype == DType::kBFloat16);

    const std::uint8_t image_rgb[] = {
        0, 127, 255, 255, 127, 0,
        10, 20, 30, 40, 50, 60,
    };
    const std::uint8_t wrist_bgr[] = {
        30, 20, 10, 60, 50, 40,
        90, 80, 70, 120, 110, 100,
    };
    VisionFrame image;
    image.name = "image";
    image.image = {const_cast<std::uint8_t*>(image_rgb), sizeof(image_rgb),
                   DType::kUInt8, MemoryPlace::kHost, Layout::kHWC, Shape{2, 2, 3}};
    image.format = PixelFormat::kRGB8;
    image.width = 2;
    image.height = 2;

    VisionFrame wrist;
    wrist.name = "wrist_image";
    wrist.image = {const_cast<std::uint8_t*>(wrist_bgr), sizeof(wrist_bgr),
                   DType::kUInt8, MemoryPlace::kHost, Layout::kHWC, Shape{2, 2, 3}};
    wrist.format = PixelFormat::kBGR8;
    wrist.width = 2;
    wrist.height = 2;

    std::vector<std::uint16_t> out(required_vision_output_bytes(spec) / 2);
    TensorView dst{out.data(), static_cast<std::uint64_t>(out.size() * 2),
                   DType::kBFloat16, MemoryPlace::kHost, Layout::kNHWC,
                   Shape{2, 224, 224, 3}};
    auto st = preprocess_vision_cpu(spec, {image, wrist}, dst);
    assert(st.ok_status());

    const float first_r = bfloat16_to_float(out[0]);
    const float first_g = bfloat16_to_float(out[1]);
    const float first_b = bfloat16_to_float(out[2]);
    assert(std::fabs(first_r - (-1.0f)) < 0.01f);
    assert(std::fabs(first_g - (127.0f / 127.5f - 1.0f)) < 0.01f);
    assert(std::fabs(first_b - 1.0f) < 0.01f);
}

void test_view_order_guard() {
    auto spec = flashrt::models::pi05::vision_preprocess_spec(2);
    std::uint8_t image_rgb[12] = {};
    VisionFrame image;
    image.name = "image";
    image.image = {image_rgb, sizeof(image_rgb), DType::kUInt8,
                   MemoryPlace::kHost, Layout::kHWC, Shape{2, 2, 3}};
    image.format = PixelFormat::kRGB8;
    image.width = 2;
    image.height = 2;

    std::vector<std::uint16_t> out(required_vision_output_bytes(spec) / 2);
    TensorView dst{out.data(), static_cast<std::uint64_t>(out.size() * 2),
                   DType::kBFloat16, MemoryPlace::kHost, Layout::kNHWC,
                   Shape{2, 224, 224, 3}};
    auto st = preprocess_vision_cpu(spec, {image}, dst);
    assert(!st.ok_status());
    assert(st.code == StatusCode::kShapeMismatch);
}

void test_action_postprocess() {
    std::vector<std::uint16_t> model(2 * 4);
    model[0] = float_to_bfloat16(1.0f);
    model[1] = float_to_bfloat16(-2.0f);
    model[2] = float_to_bfloat16(3.0f);
    model[3] = float_to_bfloat16(99.0f);
    model[4] = float_to_bfloat16(0.5f);
    model[5] = float_to_bfloat16(1.5f);
    model[6] = float_to_bfloat16(-1.0f);
    model[7] = float_to_bfloat16(88.0f);

    TensorView src{model.data(), static_cast<std::uint64_t>(model.size() * 2),
                   DType::kBFloat16, MemoryPlace::kHost, Layout::kFlat,
                   Shape{2, 4}};
    auto spec = flashrt::models::pi05::action_postprocess_spec(
        {10.0f, 20.0f, 30.0f}, {2.0f, 3.0f, 4.0f},
        /*chunk=*/2, /*model_dim=*/4, /*robot_dim=*/3);
    std::vector<float> out;
    auto st = postprocess_action_cpu(spec, src, &out);
    assert(st.ok_status());
    assert(out.size() == 6);
    assert(std::fabs(out[0] - 12.0f) < 0.01f);
    assert(std::fabs(out[1] - 14.0f) < 0.01f);
    assert(std::fabs(out[2] - 42.0f) < 0.01f);
    assert(std::fabs(out[3] - 11.0f) < 0.01f);
    assert(std::fabs(out[4] - 24.5f) < 0.01f);
    assert(std::fabs(out[5] - 26.0f) < 0.01f);
}

void test_pi05_runtime_io_adapter() {
    const auto vision_spec = flashrt::models::pi05::vision_preprocess_spec(1);
    std::vector<std::uint16_t> image_out(required_vision_output_bytes(vision_spec) / 2);
    TensorView image_dst{image_out.data(),
                         static_cast<std::uint64_t>(image_out.size() * 2),
                         DType::kBFloat16, MemoryPlace::kHost, Layout::kNHWC,
                         Shape{1, 224, 224, 3}};

    std::vector<float> action_model(1 * 4);
    action_model[0] = 2.0f;
    action_model[1] = 3.0f;
    action_model[2] = 4.0f;
    action_model[3] = 100.0f;
    TensorView action_src{action_model.data(),
                          static_cast<std::uint64_t>(action_model.size() * 4),
                          DType::kFloat32, MemoryPlace::kHost, Layout::kFlat,
                          Shape{1, 4}};

    flashrt::models::pi05::RuntimeIo io(
        1, image_dst, action_src,
        {1.0f, 2.0f, 3.0f}, {10.0f, 20.0f, 30.0f},
        /*chunk=*/1, /*model_action_dim=*/4, /*robot_action_dim=*/3);

    const std::uint8_t image_rgb[] = {
        127, 127, 127, 127, 127, 127,
        127, 127, 127, 127, 127, 127,
    };
    VisionFrame image;
    image.name = "image";
    image.image = {const_cast<std::uint8_t*>(image_rgb), sizeof(image_rgb),
                   DType::kUInt8, MemoryPlace::kHost, Layout::kHWC, Shape{2, 2, 3}};
    image.format = PixelFormat::kRGB8;
    image.width = 2;
    image.height = 2;

    auto st = io.prepare_vision({image});
    assert(st.ok_status());
    std::vector<float> actions;
    st = io.read_actions(&actions);
    assert(st.ok_status());
    assert(actions.size() == 3);
    assert(std::fabs(actions[0] - 21.0f) < 0.01f);
    assert(std::fabs(actions[1] - 62.0f) < 0.01f);
    assert(std::fabs(actions[2] - 123.0f) < 0.01f);
}

}  // namespace

int main() {
    test_pi05_vision_spec_and_preprocess();
    test_view_order_guard();
    test_action_postprocess();
    test_pi05_runtime_io_adapter();
    std::cout << "PASS - runtime modality contracts\n";
    return 0;
}
