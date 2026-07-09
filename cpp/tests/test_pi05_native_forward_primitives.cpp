#include "flashrt/cpp/models/pi05/native_kernel_driver.h"
#include "flashrt/cpp/modalities/types.h"
#include "flashrt/exec.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using flashrt::models::pi05::NativeKernelDriver;

frt_buffer allocate(frt_ctx ctx, const char* name, std::size_t elements) {
    frt_buffer buffer =
        frt_buffer_alloc(ctx, name, elements * sizeof(std::uint16_t));
    assert(buffer);
    return buffer;
}

std::vector<std::uint16_t> bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        result[i] = flashrt::modalities::float_to_bfloat16(values[i]);
    }
    return result;
}

void upload(frt_buffer buffer, const std::vector<std::uint16_t>& values) {
    assert(cudaMemcpy(frt_buffer_dptr(buffer), values.data(),
                      values.size() * sizeof(std::uint16_t),
                      cudaMemcpyHostToDevice) == cudaSuccess);
}

std::vector<float> download(frt_buffer buffer, std::size_t elements) {
    std::vector<std::uint16_t> bits(elements);
    assert(cudaMemcpy(bits.data(), frt_buffer_dptr(buffer),
                      bits.size() * sizeof(std::uint16_t),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    std::vector<float> result(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        result[i] = flashrt::modalities::bfloat16_to_float(bits[i]);
    }
    return result;
}

void expect_close(const std::vector<float>& actual,
                  const std::vector<float>& expected, float tolerance) {
    assert(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        assert(std::fabs(actual[i] - expected[i]) <= tolerance);
    }
}

struct CaptureArgs {
    const NativeKernelDriver* driver = nullptr;
    void* values = nullptr;
    const void* weight = nullptr;
    void* norm_output = nullptr;
    const void* qkv = nullptr;
    const void* rope = nullptr;
    void* query = nullptr;
    void* key = nullptr;
    void* value = nullptr;
    const void* devpos = nullptr;
    bool recorded = false;
};

void record_primitives(void* user, void* stream) {
    auto* args = static_cast<CaptureArgs*>(user);
    const std::uintptr_t native_stream =
        reinterpret_cast<std::uintptr_t>(stream);
    args->recorded =
        args->driver
            ->rms_norm_bf16(args->values, args->weight, args->norm_output,
                            2, 4, 1e-6f, native_stream)
            .ok_status() &&
        args->driver
            ->qkv_split_rope_devpos_bf16(
                args->qkv, args->rope, args->query, args->key, args->value,
                args->devpos, 2, 4, 2, 2, 2, native_stream)
            .ok_status();
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || !device_count) {
        cudaGetLastError();
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }

    NativeKernelDriver driver;
    assert(driver.status().ok_status());
    frt_ctx ctx = frt_ctx_create();
    assert(ctx);
    cudaStream_t stream = nullptr;
    assert(cudaStreamCreate(&stream) == cudaSuccess);
    const std::uintptr_t native_stream =
        reinterpret_cast<std::uintptr_t>(stream);

    const std::vector<float> host_x = {1, 2, 3, 4, -1, 0, 1, 2};
    const std::vector<float> host_weight = {1, 1.5f, 0.5f, 2};
    const std::vector<float> host_bias = {0.1f, -0.2f, 0.3f, -0.4f};
    frt_buffer x = allocate(ctx, "primitive_x", 8);
    frt_buffer weight = allocate(ctx, "primitive_weight", 4);
    frt_buffer bias = allocate(ctx, "primitive_bias", 4);
    frt_buffer output = allocate(ctx, "primitive_output", 8);
    frt_buffer gate = allocate(ctx, "primitive_gate", 8);
    upload(x, bf16(host_x));
    upload(weight, bf16(host_weight));
    upload(bias, bf16(host_bias));

    assert(driver.rms_norm_bf16(
                     frt_buffer_dptr(x), frt_buffer_dptr(weight),
                     frt_buffer_dptr(output), 2, 4, 1e-6f, native_stream)
               .ok_status());
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<float> rms_expected(8);
    for (int row = 0; row < 2; ++row) {
        float sum = 0;
        for (int col = 0; col < 4; ++col) {
            const float value = host_x[row * 4 + col];
            sum += value * value;
        }
        const float scale = 1.0f / std::sqrt(sum / 4 + 1e-6f);
        for (int col = 0; col < 4; ++col) {
            rms_expected[row * 4 + col] =
                host_x[row * 4 + col] * scale * host_weight[col];
        }
    }
    expect_close(download(output, 8), rms_expected, 0.025f);

    assert(driver.layer_norm_bf16(
                     frt_buffer_dptr(x), frt_buffer_dptr(weight),
                     frt_buffer_dptr(bias), frt_buffer_dptr(output), 2, 4,
                     1e-5f, native_stream)
               .ok_status());
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<float> layer_expected(8);
    for (int row = 0; row < 2; ++row) {
        float mean = 0;
        for (int col = 0; col < 4; ++col) mean += host_x[row * 4 + col];
        mean /= 4;
        float variance = 0;
        for (int col = 0; col < 4; ++col) {
            const float delta = host_x[row * 4 + col] - mean;
            variance += delta * delta;
        }
        const float scale = 1.0f / std::sqrt(variance / 4 + 1e-5f);
        for (int col = 0; col < 4; ++col) {
            layer_expected[row * 4 + col] =
                (host_x[row * 4 + col] - mean) * scale * host_weight[col] +
                host_bias[col];
        }
    }
    expect_close(download(output, 8), layer_expected, 0.025f);

    std::vector<float> style(24, 0.0f);
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 4; ++col) {
            style[row * 12 + col] = 0.25f;
            style[row * 12 + 4 + col] = -0.5f;
            style[row * 12 + 8 + col] = 0.75f;
        }
    }
    frt_buffer style_buffer = allocate(ctx, "primitive_style", 24);
    upload(style_buffer, bf16(style));
    assert(driver.ada_rms_norm_style_bf16(
                     frt_buffer_dptr(x), frt_buffer_dptr(weight),
                     frt_buffer_dptr(style_buffer), frt_buffer_dptr(output),
                     frt_buffer_dptr(gate), 2, 4, 1e-6f, native_stream)
               .ok_status());
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<float> ada_expected(8);
    for (std::size_t i = 0; i < ada_expected.size(); ++i) {
        ada_expected[i] = rms_expected[i] * 1.25f - 0.5f;
    }
    expect_close(download(output, 8), ada_expected, 0.035f);
    expect_close(download(gate, 8), std::vector<float>(8, 0.75f), 0.0f);

    frt_buffer residual = allocate(ctx, "primitive_residual", 8);
    upload(residual, bf16(std::vector<float>(8, 1.0f)));
    upload(output, bf16(std::vector<float>(8, 2.0f)));
    upload(gate, bf16(std::vector<float>(8, 0.5f)));
    assert(driver.gate_mul_residual_bf16(
                     frt_buffer_dptr(residual), frt_buffer_dptr(output),
                     frt_buffer_dptr(gate), 8, native_stream)
               .ok_status());
    assert(driver.bias_residual_bf16(
                     frt_buffer_dptr(residual), frt_buffer_dptr(output),
                     frt_buffer_dptr(bias), 2, 4, native_stream)
               .ok_status());
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<float> residual_expected(8);
    for (int i = 0; i < 8; ++i) {
        residual_expected[i] = 4.0f + host_bias[i % 4];
    }
    expect_close(download(residual, 8), residual_expected, 0.025f);

    upload(residual, bf16(std::vector<float>(8, 1.0f)));
    upload(output, bf16(std::vector<float>(8, 2.0f)));
    assert(driver.residual_add_bf16(
                     frt_buffer_dptr(residual), frt_buffer_dptr(output), 8,
                     native_stream)
               .ok_status());
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    expect_close(download(residual, 8), std::vector<float>(8, 3.0f), 0.0f);

    const std::vector<float> activation_input =
        {-3, -2, -1, 0, 0.5f, 1, 2, 3};
    upload(gate, bf16(activation_input));
    upload(output, bf16(std::vector<float>(8, 1.5f)));
    assert(driver.gate_gelu_bf16(
                     frt_buffer_dptr(gate), frt_buffer_dptr(output),
                     frt_buffer_dptr(residual), 8, native_stream)
               .ok_status());
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<float> gated_expected(8);
    for (int i = 0; i < 8; ++i) {
        const float value = activation_input[i];
        const float gelu = value /
            (1.0f + std::exp(-1.5957691216057308f * value *
                             (1.0f + 0.044715f * value * value)));
        gated_expected[i] = gelu * 1.5f;
    }
    expect_close(download(residual, 8), gated_expected, 0.025f);
    upload(output, bf16(activation_input));
    assert(driver.gelu_bf16(frt_buffer_dptr(output), 8, native_stream)
               .ok_status());
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<float> gelu_expected(8);
    for (int i = 0; i < 8; ++i) {
        const float value = activation_input[i];
        gelu_expected[i] = value * 0.5f *
            (1.0f + std::tanh(0.7978845608f *
                              (value + 0.044715f * value * value * value)));
    }
    expect_close(download(output, 8), gelu_expected, 0.025f);

    std::vector<float> pool_input(4 * 4 * 2);
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            pool_input[(row * 4 + col) * 2] = float(row * 4 + col);
            pool_input[(row * 4 + col) * 2 + 1] = float(row * 4 + col + 1);
        }
    }
    frt_buffer pool_values = allocate(ctx, "primitive_pool_values", 32);
    frt_buffer pool_output = allocate(ctx, "primitive_pool_output", 8);
    upload(pool_values, bf16(pool_input));
    assert(driver.avg_pool_vision_bf16(
                     frt_buffer_dptr(pool_values), frt_buffer_dptr(pool_output),
                     1, 4, 4, 2, 2, native_stream)
               .ok_status());
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    expect_close(download(pool_output, 8),
                 {2.5f, 3.5f, 4.5f, 5.5f, 10.5f, 11.5f, 12.5f, 13.5f},
                 0.0f);

    const std::vector<float> host_qkv = {
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16};
    frt_buffer qkv = allocate(ctx, "primitive_qkv", 16);
    frt_buffer rope = allocate(ctx, "primitive_rope", 4);
    frt_buffer query = allocate(ctx, "primitive_query", 8);
    frt_buffer key = allocate(ctx, "primitive_key", 8);
    frt_buffer value = allocate(ctx, "primitive_value", 8);
    frt_buffer devpos = frt_buffer_alloc(ctx, "primitive_devpos", sizeof(int));
    assert(devpos);
    upload(qkv, bf16(host_qkv));
    upload(rope, bf16({1, 0, 0, 1}));
    const int position = 1;
    assert(cudaMemcpy(frt_buffer_dptr(devpos), &position, sizeof(position),
                      cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemset(frt_buffer_dptr(key), 0, 8 * sizeof(std::uint16_t)) ==
           cudaSuccess);
    assert(cudaMemset(frt_buffer_dptr(value), 0, 8 * sizeof(std::uint16_t)) ==
           cudaSuccess);
    assert(driver.qkv_split_rope_devpos_bf16(
                     frt_buffer_dptr(qkv), frt_buffer_dptr(rope),
                     frt_buffer_dptr(query), frt_buffer_dptr(key),
                     frt_buffer_dptr(value), frt_buffer_dptr(devpos), 2, 4, 2,
                     2, 2, native_stream)
               .ok_status());
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    expect_close(download(query, 8), {1, 2, 3, 4, -10, 9, -12, 11}, 0.0f);
    expect_close(download(key, 8), {0, 0, 5, 6, -14, 13, 0, 0}, 0.0f);
    expect_close(download(value, 8), {0, 0, 7, 8, 15, 16, 0, 0}, 0.0f);

    const std::size_t image_elements = 224 * 224 * 3;
    frt_buffer image = allocate(ctx, "primitive_image", image_elements);
    frt_buffer patches = allocate(ctx, "primitive_patches", image_elements);
    std::vector<std::uint16_t> image_bits(image_elements);
    for (std::size_t i = 0; i < image_bits.size(); ++i) {
        image_bits[i] = static_cast<std::uint16_t>(i);
    }
    upload(image, image_bits);
    assert(driver.patch_im2col_16bit(
                     frt_buffer_dptr(image), frt_buffer_dptr(patches), 1,
                     native_stream)
               .ok_status());
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<std::uint16_t> patch_bits(image_elements);
    assert(cudaMemcpy(patch_bits.data(), frt_buffer_dptr(patches),
                      patch_bits.size() * sizeof(std::uint16_t),
                      cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int patch = 0; patch < 256; ++patch) {
        const int patch_row = patch / 16;
        const int patch_col = patch % 16;
        for (int feature = 0; feature < 588; ++feature) {
            const int pixel_row = feature / 42;
            const int pixel_col = (feature % 42) / 3;
            const int channel = feature % 3;
            const std::size_t source =
                static_cast<std::size_t>(patch_row * 14 + pixel_row) * 224 * 3 +
                (patch_col * 14 + pixel_col) * 3 + channel;
            assert(patch_bits[patch * 588 + feature] == image_bits[source]);
        }
    }

    frt_graph graph = frt_graph_create(ctx, "native_forward_primitives", 7);
    assert(graph);
    CaptureArgs capture{
        &driver, frt_buffer_dptr(x), frt_buffer_dptr(weight),
        frt_buffer_dptr(output), frt_buffer_dptr(qkv), frt_buffer_dptr(rope),
        frt_buffer_dptr(query), frt_buffer_dptr(key), frt_buffer_dptr(value),
        frt_buffer_dptr(devpos), false};
    assert(frt_graph_capture(graph, 7, record_primitives, &capture) == FRT_OK);
    assert(capture.recorded);
    const int stream_id = frt_ctx_wrap_stream(ctx, stream);
    assert(stream_id >= 0);
    for (int i = 0; i < 100; ++i) {
        assert(frt_graph_replay(graph, 7, stream_id) == FRT_OK);
    }
    assert(frt_graph_variant_count(graph) == 1);
    assert(cudaStreamSynchronize(stream) == cudaSuccess);

    frt_graph_destroy(graph);
    assert(cudaStreamDestroy(stream) == cudaSuccess);
    frt_ctx_destroy(ctx);
    std::printf("PASS - Pi0.5 native forward primitives\n");
    return 0;
}
