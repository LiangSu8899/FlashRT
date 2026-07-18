#include "flashrt/cpp/models/pi05/targets/sm120/fp8_linear.h"

#include "flashrt/cpp/models/pi05/targets/sm120/fp8_weight_packer.h"
#include "pi05_resolved_fixture.h"

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace pi05 = flashrt::models::pi05;
namespace sm120 = flashrt::models::pi05::targets::sm120;
namespace fixture = flashrt::tests::pi05_fixture;

namespace {

constexpr std::size_t kMinimumSm120Fp8CublasVersion = 130100;

[[noreturn]] void fail(const char* expression, int line) {
    std::cerr << "FAIL line " << line << ": " << expression << '\n';
    std::abort();
}

#define CHECK(expression)                                      \
    do {                                                       \
        if (!(expression)) fail(#expression, __LINE__);        \
    } while (false)

void check_status(const flashrt::modalities::Status& status, int line) {
    if (status.ok_status()) return;
    std::cerr << "FAIL line " << line << ": " << status.message << '\n';
    std::abort();
}

#define CHECK_STATUS(expression) check_status((expression), __LINE__)

bool has_sm120() {
    int device = 0;
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDevice(&device);
    if (status == cudaSuccess) {
        status = cudaGetDeviceProperties(&properties, device);
    }
    if (status != cudaSuccess) cudaGetLastError();
    return status == cudaSuccess && properties.major == 12 &&
           properties.minor == 0;
}

pi05::NativeCalibrationArtifact artifact_for(
    const pi05::Pi05ResolvedShape& shape) {
    pi05::Pi05LinearScaleLayout layout;
    CHECK(pi05::resolve_pi05_linear_scale_layout(shape, &layout).ok_status());
    pi05::NativeCalibrationArtifact artifact;
    artifact.activation_dtype = "bfloat16";
    artifact.hardware = "sm120";
    artifact.weights_sha256 = std::string(64, 'a');
    artifact.tokenizer_sha256 = std::string(64, 'b');
    artifact.num_views = shape.num_views;
    artifact.max_prompt_tokens = shape.max_prompt_tokens;
    artifact.state_dim = shape.state_dim;
    artifact.chunk_size = shape.chunk;
    artifact.num_steps = shape.num_steps;
    artifact.vision_pool_factor = shape.vision_pool_factor;
    artifact.sample_count = 1;
    artifact.percentile = 99.9;
    constexpr float kScale = 1.0f / 448.0f;
    artifact.vision_scales.assign(layout.vision, kScale);
    artifact.encoder_scales.assign(layout.encoder, 2.0f * kScale);
    artifact.decoder_scales.assign(layout.decoder, 3.0f * kScale);
    return artifact;
}

pi05::Pi05ResolvedWeight upload_bf16(
    frt_ctx context,
    const char* name,
    const std::vector<std::uint16_t>& values,
    int rows,
    int columns) {
    frt_buffer buffer = frt_buffer_alloc(
        context, name, values.size() * sizeof(std::uint16_t));
    CHECK(buffer != nullptr);
    CHECK(cudaMemcpy(
              frt_buffer_dptr(buffer), values.data(),
              values.size() * sizeof(std::uint16_t),
              cudaMemcpyHostToDevice) == cudaSuccess);
    pi05::Pi05ResolvedWeight weight;
    weight.device_data = frt_buffer_dptr(buffer);
    weight.bytes = values.size() * sizeof(std::uint16_t);
    weight.storage = pi05::Pi05WeightStorage::kBFloat16;
    weight.shape = flashrt::modalities::Shape({
        static_cast<std::uint64_t>(rows),
        static_cast<std::uint64_t>(columns)});
    return weight;
}

frt_buffer upload_buffer(
    frt_ctx context,
    const char* name,
    const std::vector<std::uint16_t>& values) {
    frt_buffer buffer = frt_buffer_alloc(
        context, name, values.size() * sizeof(std::uint16_t));
    CHECK(buffer != nullptr);
    CHECK(cudaMemcpy(
              frt_buffer_dptr(buffer), values.data(),
              values.size() * sizeof(std::uint16_t),
              cudaMemcpyHostToDevice) == cudaSuccess);
    return buffer;
}

pi05::Pi05ResolvedWeight pack_weight(
    frt_ctx context,
    const char* name,
    int input_width,
    int output_width,
    std::unique_ptr<sm120::Sm120Fp8WeightPacker>* owner) {
    std::vector<std::uint16_t> values(
        static_cast<std::size_t>(input_width) * output_width, 0x3f80);
    pi05::Pi05ResolvedWeight weight = upload_bf16(
        context, name, values, input_width, output_width);
    std::unique_ptr<sm120::Sm120Fp8WeightPacker> packer(
        new sm120::Sm120Fp8WeightPacker(context));
    CHECK(packer->record({
              {pi05::Pi05LinearDomain::kVision,
               pi05::Pi05LinearRole::kAttentionQkv, 0},
              &weight, nullptr, nullptr})
              .ok_status());
    CHECK(packer->finish().ok_status());
    *owner = std::move(packer);
    return weight;
}

void test_backing_contract() {
    frt_ctx context = frt_ctx_create();
    CHECK(context != nullptr);
    const pi05::Pi05ResolvedShape shape = fixture::canonical_shape();
    const pi05::NativeCalibrationArtifact artifact = artifact_for(shape);

    sm120::Sm120Fp8ActivationBacking backing(context);
    CHECK(backing.initialize_static(shape, artifact).ok_status());
    CHECK(backing.initialized());
    CHECK(backing.allocation_count() == 4);
    CHECK(backing.scratch_data() != nullptr && backing.scratch_bytes() > 0);
    CHECK(backing.scale_layout().vision == artifact.vision_scales.size());
    CHECK(backing.scale_layout().encoder == artifact.encoder_scales.size());
    CHECK(backing.scale_layout().decoder == artifact.decoder_scales.size());
    std::vector<float> vision;
    std::vector<float> encoder;
    std::vector<float> decoder;
    CHECK(backing.download_scales(&vision, &encoder, &decoder).ok_status());
    CHECK(vision == artifact.vision_scales);
    CHECK(encoder == artifact.encoder_scales);
    CHECK(decoder == artifact.decoder_scales);
    CHECK(!backing.observing());
    CHECK(!backing.reset_observer_scales(0).ok_status());
    CHECK(!backing.download_observer_scales(
                      &vision, &encoder, &decoder)
               .ok_status());
    CHECK(!backing.initialize_static(shape, artifact).ok_status());

    pi05::NativeCalibrationArtifact incompatible = artifact;
    incompatible.num_views = shape.num_views - 1;
    sm120::Sm120Fp8ActivationBacking rejected(context);
    CHECK(!rejected.initialize_static(shape, incompatible).ok_status());
    frt_ctx_destroy(context);
}

void test_observer_backing_contract() {
    frt_ctx context = frt_ctx_create();
    CHECK(context != nullptr);
    const pi05::Pi05ResolvedShape shape = fixture::canonical_shape();
    sm120::Sm120Fp8ActivationBacking backing(context);
    CHECK_STATUS(backing.initialize_observer(shape));
    CHECK(backing.initialized() && backing.observing());
    CHECK(backing.allocation_count() == 4);

    const pi05::Pi05LinearScaleLayout layout = backing.scale_layout();
    std::vector<float> vision(layout.vision);
    std::vector<float> encoder(layout.encoder);
    std::vector<float> decoder(layout.decoder);
    CHECK_STATUS(backing.download_observer_scales(
        &vision, &encoder, &decoder));
    const auto all_one = [](const std::vector<float>& values) {
        return std::all_of(values.begin(), values.end(),
                           [](float value) { return value == 1.0f; });
    };
    CHECK(all_one(vision) && all_one(encoder) && all_one(decoder));

    std::vector<float> wrong_size;
    CHECK(!backing.download_observer_scales(
                      &wrong_size, &encoder, &decoder)
               .ok_status());
    for (const pi05::Pi05LinearDomain domain : {
             pi05::Pi05LinearDomain::kVision,
             pi05::Pi05LinearDomain::kEncoder,
             pi05::Pi05LinearDomain::kDecoder}) {
        float* scale = backing.scale_data({domain, 0});
        CHECK(scale != nullptr);
        CHECK(cudaMemset(scale, 0, sizeof(float)) == cudaSuccess);
    }
    CHECK_STATUS(backing.reset_observer_scales(0));
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK_STATUS(backing.download_observer_scales(
        &vision, &encoder, &decoder));
    CHECK(all_one(vision) && all_one(encoder) && all_one(decoder));
    CHECK(!backing.initialize_observer(shape).ok_status());
    frt_ctx_destroy(context);
}

void test_unsupported_runtime_contract() {
    frt_ctx context = frt_ctx_create();
    CHECK(context != nullptr);
    const pi05::Pi05ResolvedShape shape = fixture::canonical_shape();
    const pi05::NativeCalibrationArtifact artifact = artifact_for(shape);
    sm120::Sm120Fp8ActivationBacking backing(context);
    CHECK(backing.initialize_static(shape, artifact).ok_status());
    sm120::Sm120Fp8Linear linear(&backing);
    const flashrt::modalities::Status status = linear.status();
    CHECK(!status.ok_status());
    CHECK(status.code == flashrt::modalities::StatusCode::kUnsupported);
    frt_ctx_destroy(context);
}

void test_linear_contract() {
    frt_ctx context = frt_ctx_create();
    CHECK(context != nullptr);
    const pi05::Pi05ResolvedShape shape = fixture::canonical_shape();
    const pi05::NativeCalibrationArtifact artifact = artifact_for(shape);
    sm120::Sm120Fp8ActivationBacking backing(context);
    CHECK(backing.initialize_static(shape, artifact).ok_status());
    sm120::Sm120Fp8Linear linear(&backing);
    CHECK(linear.status().ok_status());

    std::unique_ptr<sm120::Sm120Fp8WeightPacker> unaligned_owner;
    pi05::Pi05ResolvedWeight unaligned_weight = pack_weight(
        context, "fp8_linear_unaligned_weight", 3, 4, &unaligned_owner);
    frt_buffer unaligned_input = upload_buffer(
        context, "fp8_linear_unaligned_input",
        std::vector<std::uint16_t>(3, 0x3f80));
    frt_buffer unaligned_output = frt_buffer_alloc(
        context, "fp8_linear_unaligned_output",
        4 * sizeof(std::uint16_t));
    CHECK(unaligned_output != nullptr);
    CHECK(!linear.autotune(
               unaligned_weight,
               {pi05::Pi05LinearDomain::kVision, 0},
               frt_buffer_dptr(unaligned_input),
               frt_buffer_dptr(unaligned_output), 1, 3, 4)
               .ok_status());

    const int kRows = shape.vision_sequence;
    constexpr int kInputWidth = 1152;
    constexpr int kOutputWidth = 3456;
    std::unique_ptr<sm120::Sm120Fp8WeightPacker> packed_owner;
    pi05::Pi05ResolvedWeight weight = pack_weight(
        context, "fp8_linear_weight", kInputWidth, kOutputWidth,
        &packed_owner);
    std::vector<std::uint16_t> input(
        static_cast<std::size_t>(kRows) * kInputWidth, 0x3f80);
    const std::size_t output_elements =
        static_cast<std::size_t>(kRows) * kOutputWidth;
    frt_buffer input_buffer = upload_buffer(
        context, "fp8_linear_input", input);
    frt_buffer output_a = frt_buffer_alloc(
        context, "fp8_linear_output_a",
        output_elements * sizeof(std::uint16_t));
    frt_buffer output_b = frt_buffer_alloc(
        context, "fp8_linear_output_b",
        output_elements * sizeof(std::uint16_t));
    CHECK(output_a && output_b);
    pi05::Pi05LinearActivationSite site{
        pi05::Pi05LinearDomain::kVision, 0};
    CHECK(!linear.run(
               weight, site, frt_buffer_dptr(input_buffer),
               frt_buffer_dptr(output_a), kRows, kInputWidth,
               kOutputWidth, 0)
               .ok_status());
    CHECK(linear.autotuned_shape_count() == 0);
    CHECK_STATUS(linear.autotune(
        weight, site, frt_buffer_dptr(input_buffer),
        frt_buffer_dptr(output_a), kRows, kInputWidth, kOutputWidth));
    CHECK_STATUS(linear.autotune(
        weight, site, frt_buffer_dptr(input_buffer),
        frt_buffer_dptr(output_a), kRows, kInputWidth, kOutputWidth));
    CHECK(linear.autotuned_shape_count() == 1);
    CHECK(!linear.run(
               weight, site, frt_buffer_dptr(input_buffer),
               frt_buffer_dptr(output_a), kRows, kInputWidth,
               kOutputWidth, 0)
               .ok_status());
    CHECK(linear.freeze_autotune().ok_status());
    CHECK(linear.autotune_frozen());
    CHECK_STATUS(linear.run(
        weight, site, frt_buffer_dptr(input_buffer),
        frt_buffer_dptr(output_a), kRows, kInputWidth, kOutputWidth, 0));
    CHECK(linear.autotuned_shape_count() == 1);
    CHECK(linear.run_prequantized(
              weight, site, linear.scratch_data(),
              frt_buffer_dptr(output_b), kRows, kInputWidth, kOutputWidth, 0)
              .ok_status());
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<std::uint16_t> host_a(output_elements);
    std::vector<std::uint16_t> host_b(output_elements);
    CHECK(cudaMemcpy(
              host_a.data(), frt_buffer_dptr(output_a),
              host_a.size() * sizeof(std::uint16_t),
              cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(
              host_b.data(), frt_buffer_dptr(output_b),
              host_b.size() * sizeof(std::uint16_t),
              cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(host_a == host_b);
    CHECK(std::all_of(host_a.begin(), host_a.end(),
                      [](std::uint16_t value) {
                          return value == 0x4490;
                      }));
    CHECK(!linear.freeze_autotune().ok_status());
    CHECK(!linear.autotune(
               weight, site, frt_buffer_dptr(input_buffer),
               frt_buffer_dptr(output_a), kRows, kInputWidth,
               kOutputWidth)
               .ok_status());
    CHECK(!linear.run(
               weight, site, frt_buffer_dptr(input_buffer),
               frt_buffer_dptr(output_a), kRows - 1, kInputWidth,
               kOutputWidth, 0)
               .ok_status());

    sm120::Sm120Fp8ActivationBacking observer_backing(context);
    CHECK_STATUS(observer_backing.initialize_observer(shape));
    sm120::Sm120Fp8Linear observer_linear(
        &observer_backing, sm120::Sm120Fp8ExecutionMode::kObserve);
    CHECK_STATUS(observer_linear.status());
    CHECK(observer_linear.observing());
    CHECK_STATUS(observer_linear.autotune(
        weight, site, frt_buffer_dptr(input_buffer),
        frt_buffer_dptr(output_a), kRows, kInputWidth, kOutputWidth));
    CHECK_STATUS(observer_linear.freeze_autotune());
    CHECK(!observer_linear.run_prequantized(
                       weight, site, observer_linear.scratch_data(),
                       frt_buffer_dptr(output_b), kRows, kInputWidth,
                       kOutputWidth, 0)
               .ok_status());
    CHECK_STATUS(observer_linear.run(
        weight, site, frt_buffer_dptr(input_buffer),
        frt_buffer_dptr(output_a), kRows, kInputWidth, kOutputWidth, 0));
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<float> observed_vision(
        observer_backing.scale_layout().vision);
    std::vector<float> observed_encoder(
        observer_backing.scale_layout().encoder);
    std::vector<float> observed_decoder(
        observer_backing.scale_layout().decoder);
    CHECK_STATUS(observer_backing.download_observer_scales(
        &observed_vision, &observed_encoder, &observed_decoder));
    CHECK(observed_vision[site.index] == 1.0f / 448.0f);
    CHECK_STATUS(observer_backing.reset_observer_scales(0));
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK_STATUS(observer_backing.download_observer_scales(
        &observed_vision, &observed_encoder, &observed_decoder));
    CHECK(observed_vision[site.index] == 1.0f);

    sm120::Sm120Fp8Linear mismatched_static(
        &observer_backing, sm120::Sm120Fp8ExecutionMode::kStatic);
    CHECK(!mismatched_static.status().ok_status());
    sm120::Sm120Fp8Linear mismatched_observer(
        &backing, sm120::Sm120Fp8ExecutionMode::kObserve);
    CHECK(!mismatched_observer.status().ok_status());
    frt_ctx_destroy(context);
}

}  // namespace

int main() {
    if (!has_sm120()) {
        std::cout << "SKIP - SM120 is unavailable\n";
        return 0;
    }
    test_backing_contract();
    test_observer_backing_contract();
    if (cublasLtGetVersion() < kMinimumSm120Fp8CublasVersion) {
        test_unsupported_runtime_contract();
        std::cout << "SKIP - SM120 FP8 requires cuBLAS 13.1 or newer\n";
        return 0;
    }
    test_linear_contract();
    return 0;
}
