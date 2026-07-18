#include "flashrt/cpp/models/pi05/targets/sm120/fp8_linear.h"

#include "gemm_runner.h"
#include "quantize.cuh"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <set>
#include <tuple>
#include <utility>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {
namespace {

constexpr std::size_t kMinimumSm120Fp8CublasVersion = 130100;

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

cudaStream_t cuda_stream(Pi05Stream stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

bool multiply(std::size_t left, std::size_t right, std::size_t* out) {
    if (!out || (right && left > std::numeric_limits<std::size_t>::max() /
                                   right)) {
        return false;
    }
    *out = left * right;
    return true;
}

bool artifact_matches(
    const NativeCalibrationArtifact& artifact,
    const Pi05ResolvedShape& shape,
    const Pi05LinearScaleLayout& layout) {
    return artifact.activation_dtype == "bfloat16" &&
           artifact.hardware == "sm120" &&
           artifact.num_views == shape.num_views &&
           artifact.max_prompt_tokens == shape.max_prompt_tokens &&
           artifact.state_dim == shape.state_dim &&
           artifact.chunk_size == shape.chunk &&
           artifact.num_steps == shape.num_steps &&
           artifact.vision_pool_factor == shape.vision_pool_factor &&
           artifact.vision_scales.size() == layout.vision &&
           artifact.encoder_scales.size() == layout.encoder &&
           artifact.decoder_scales.size() == layout.decoder;
}

bool valid_weight(
    const Pi05ResolvedWeight& weight,
    int input_width,
    int output_width) {
    std::size_t elements = 0;
    return input_width > 0 && output_width > 0 && weight.device_data &&
           weight.scale_data &&
           weight.storage == Pi05WeightStorage::kFp8E4M3 &&
           weight.shape.rank == 2 &&
           weight.shape.dims[0] == static_cast<std::uint64_t>(input_width) &&
           weight.shape.dims[1] == static_cast<std::uint64_t>(output_width) &&
           multiply(static_cast<std::size_t>(input_width),
                    static_cast<std::size_t>(output_width), &elements) &&
           weight.bytes == elements;
}

}  // namespace

modalities::Status Sm120Fp8ActivationBacking::add(
    const char* name,
    std::size_t elements,
    modalities::DType dtype,
    Sm120DeviceBuffer* out) {
    if (!context_ || !name || !*name || !elements || !out) {
        return invalid("SM120 FP8 backing allocation is invalid");
    }
    const std::size_t width =
        dtype == modalities::DType::kFloat32 ? sizeof(float) : 1;
    std::size_t bytes = 0;
    if (!multiply(elements, width, &bytes)) {
        return invalid("SM120 FP8 backing size overflows");
    }
    frt_buffer buffer = frt_buffer_alloc(context_, name, bytes);
    if (!buffer || !frt_buffer_dptr(buffer) || frt_buffer_bytes(buffer) != bytes) {
        return backend("SM120 FP8 backing allocation failed");
    }
    out->buffer = buffer;
    out->dtype = dtype;
    out->shape = modalities::Shape({static_cast<std::uint64_t>(elements)});
    ++allocation_count_;
    allocated_bytes_ += bytes;
    return modalities::Status::ok();
}

modalities::Status Sm120Fp8ActivationBacking::upload(
    const Sm120DeviceBuffer& destination,
    const std::vector<float>& values,
    Pi05Stream stream) const {
    if (!destination.device_data() ||
        destination.dtype != modalities::DType::kFloat32 || values.empty() ||
        destination.bytes() != values.size() * sizeof(float)) {
        return invalid("SM120 FP8 scale upload is invalid");
    }
    const cudaError_t result = cudaMemcpyAsync(
        destination.device_data(), values.data(), destination.bytes(),
        cudaMemcpyHostToDevice, cuda_stream(stream));
    return result == cudaSuccess ? modalities::Status::ok()
                                 : backend(cudaGetErrorString(result));
}

modalities::Status Sm120Fp8ActivationBacking::initialize_static(
    const Pi05ResolvedShape& shape,
    const NativeCalibrationArtifact& artifact) {
    return initialize(shape, artifact);
}

modalities::Status Sm120Fp8ActivationBacking::initialize(
    const Pi05ResolvedShape& shape,
    const NativeCalibrationArtifact& artifact) {
    if (initialization_started_ || !context_) {
        return invalid("SM120 FP8 backing state is invalid");
    }
    initialization_started_ = true;
    Pi05LinearScaleLayout layout;
    modalities::Status status =
        resolve_pi05_linear_scale_layout(shape, &layout);
    if (!status.ok_status()) return status;
    status = validate_native_calibration_artifact(artifact);
    if (!status.ok_status() || !artifact_matches(artifact, shape, layout)) {
        return invalid("SM120 FP8 calibration artifact is incompatible");
    }

    std::size_t vision_elements = 0;
    std::size_t encoder_elements = 0;
    std::size_t decoder_elements = 0;
    if (!multiply(static_cast<std::size_t>(shape.vision_sequence),
                  static_cast<std::size_t>(kPi05ModelDims.vision_hidden),
                  &vision_elements) ||
        !multiply(static_cast<std::size_t>(shape.encoder_sequence),
                  static_cast<std::size_t>(kPi05ModelDims.encoder_hidden),
                  &encoder_elements) ||
        !multiply(static_cast<std::size_t>(shape.chunk),
                  static_cast<std::size_t>(kPi05ModelDims.decoder_hidden),
                  &decoder_elements)) {
        return invalid("SM120 FP8 scratch size overflows");
    }
    const std::size_t scratch_elements =
        std::max({vision_elements, encoder_elements, decoder_elements});
    status = add("pi05_sm120_fp8_activation", scratch_elements,
                 modalities::DType::kUInt8, &scratch_);
    if (!status.ok_status()) return status;
    status = add("pi05_sm120_fp8_vision_scales", layout.vision,
                 modalities::DType::kFloat32, &vision_scales_);
    if (!status.ok_status()) return status;
    status = add("pi05_sm120_fp8_encoder_scales", layout.encoder,
                 modalities::DType::kFloat32, &encoder_scales_);
    if (!status.ok_status()) return status;
    status = add("pi05_sm120_fp8_decoder_scales", layout.decoder,
                 modalities::DType::kFloat32, &decoder_scales_);
    if (!status.ok_status()) return status;

    shape_ = shape;
    layout_ = layout;
    status = upload(vision_scales_, artifact.vision_scales, 0);
    if (status.ok_status()) {
        status = upload(encoder_scales_, artifact.encoder_scales, 0);
    }
    if (status.ok_status()) {
        status = upload(decoder_scales_, artifact.decoder_scales, 0);
    }
    if (!status.ok_status()) return status;
    const cudaError_t synchronized = cudaStreamSynchronize(nullptr);
    if (synchronized != cudaSuccess) {
        return backend(cudaGetErrorString(synchronized));
    }
    initialized_ = true;
    return modalities::Status::ok();
}

const Sm120DeviceBuffer* Sm120Fp8ActivationBacking::scale_buffer(
    Pi05LinearDomain domain) const {
    switch (domain) {
        case Pi05LinearDomain::kVision: return &vision_scales_;
        case Pi05LinearDomain::kEncoder: return &encoder_scales_;
        case Pi05LinearDomain::kDecoder: return &decoder_scales_;
    }
    return nullptr;
}

float* Sm120Fp8ActivationBacking::scale_data(
    const Pi05LinearActivationSite& site) const {
    if (!initialized_) return nullptr;
    const Sm120DeviceBuffer* buffer = scale_buffer(site.domain);
    const std::size_t count = buffer ? buffer->bytes() / sizeof(float) : 0;
    return buffer && site.index < count
               ? static_cast<float*>(buffer->device_data()) + site.index
               : nullptr;
}

modalities::Status Sm120Fp8ActivationBacking::download_scales(
    std::vector<float>* vision,
    std::vector<float>* encoder,
    std::vector<float>* decoder) const {
    if (!initialized_ || !vision || !encoder || !decoder) {
        return invalid("SM120 FP8 scale download is invalid");
    }
    const struct Download {
        const Sm120DeviceBuffer* source;
        std::vector<float>* destination;
    } downloads[] = {
        {&vision_scales_, vision},
        {&encoder_scales_, encoder},
        {&decoder_scales_, decoder},
    };
    for (const Download& download : downloads) {
        download.destination->resize(download.source->bytes() / sizeof(float));
        const cudaError_t result = cudaMemcpy(
            download.destination->data(), download.source->device_data(),
            download.source->bytes(), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) return backend(cudaGetErrorString(result));
    }
    return modalities::Status::ok();
}

struct Sm120Fp8Linear::Impl final {
    using Shape = std::tuple<int, int, int>;

    GemmRunner gemm;
    std::set<Shape> autotuned_shapes;
    bool autotune_frozen = false;
};

Sm120Fp8Linear::Sm120Fp8Linear(
    Sm120Fp8ActivationBacking* activation) noexcept
    : activation_(activation) {
    if (cublasLtGetVersion() < kMinimumSm120Fp8CublasVersion) {
        error_code_ = modalities::StatusCode::kUnsupported;
        error_ = "SM120 FP8 requires cuBLAS 13.1 or newer";
        return;
    }
    try {
        impl_.reset(new Impl());
    } catch (const std::exception& error) {
        error_ = error.what();
    } catch (...) {
        error_ = "SM120 FP8 linear initialization failed";
    }
}

Sm120Fp8Linear::~Sm120Fp8Linear() = default;

modalities::Status Sm120Fp8Linear::status() const {
    if (!activation_ || !activation_->initialized()) {
        return invalid("SM120 FP8 linear backing is incomplete");
    }
    return impl_ ? modalities::Status::ok()
                 : modalities::Status::error(
                       error_code_, error_.empty()
                                        ? "SM120 FP8 linear is incomplete"
                                        : error_);
}

modalities::Status Sm120Fp8Linear::autotune(
    const Pi05ResolvedWeight& weight,
    const Pi05LinearActivationSite& site,
    const void* input,
    void* output,
    int rows,
    int input_width,
    int output_width) {
    modalities::Status ready = status();
    std::size_t input_elements = 0;
    float* activation_scale = scale_data(site);
    if (!ready.ok_status()) return ready;
    if (!input || !output || rows <= 0 || impl_->autotune_frozen ||
        !multiply(static_cast<std::size_t>(rows),
                  static_cast<std::size_t>(input_width), &input_elements) ||
        (input_elements & 3u) != 0 ||
        input_elements > scratch_bytes() ||
        input_elements > static_cast<std::size_t>(
                             std::numeric_limits<int>::max()) ||
        !valid_weight(weight, input_width, output_width) ||
        !activation_scale) {
        return invalid("SM120 FP8 autotune arguments are invalid");
    }
    const Impl::Shape shape(rows, output_width, input_width);
    if (impl_->autotuned_shapes.find(shape) !=
        impl_->autotuned_shapes.end()) {
        return modalities::Status::ok();
    }

    quantize_fp8_static(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_fp8_e4m3*>(scratch_data()), activation_scale,
        static_cast<int>(input_elements), 0);
    const cudaError_t launched = cudaGetLastError();
    if (launched != cudaSuccess) return backend(cudaGetErrorString(launched));
    try {
        impl_->gemm.autotune_fp8_nn_dev(
            scratch_data(), const_cast<void*>(weight.device_data), output,
            rows, output_width, input_width, activation_scale,
            const_cast<float*>(weight.scale_data));
        impl_->autotuned_shapes.insert(shape);
        return modalities::Status::ok();
    } catch (const std::exception& error) {
        return backend(error.what());
    } catch (...) {
        return backend("SM120 FP8 GEMM autotune failed");
    }
}

modalities::Status Sm120Fp8Linear::run(
    const Pi05ResolvedWeight& weight,
    const Pi05LinearActivationSite& site,
    const void* input,
    void* output,
    int rows,
    int input_width,
    int output_width,
    Pi05Stream stream) {
    return launch(weight, site, input, output, rows, input_width,
                  output_width, stream, false);
}

modalities::Status Sm120Fp8Linear::run_prequantized(
    const Pi05ResolvedWeight& weight,
    const Pi05LinearActivationSite& site,
    const void* input,
    void* output,
    int rows,
    int input_width,
    int output_width,
    Pi05Stream stream) {
    return launch(weight, site, input, output, rows, input_width,
                  output_width, stream, true);
}

modalities::Status Sm120Fp8Linear::launch(
    const Pi05ResolvedWeight& weight,
    const Pi05LinearActivationSite& site,
    const void* input,
    void* output,
    int rows,
    int input_width,
    int output_width,
    Pi05Stream stream,
    bool prequantized) {
    modalities::Status ready = status();
    std::size_t input_elements = 0;
    float* activation_scale = scale_data(site);
    if (!ready.ok_status()) return ready;
    if (!input || !output || rows <= 0 || !impl_->autotune_frozen ||
        !multiply(static_cast<std::size_t>(rows),
                  static_cast<std::size_t>(input_width), &input_elements) ||
        (input_elements & 3u) != 0 ||
        input_elements > scratch_bytes() ||
        input_elements > static_cast<std::size_t>(
                             std::numeric_limits<int>::max()) ||
        !valid_weight(weight, input_width, output_width) ||
        !activation_scale) {
        return invalid("SM120 FP8 linear arguments are invalid");
    }
    const Impl::Shape shape(rows, output_width, input_width);
    if (impl_->autotuned_shapes.find(shape) ==
        impl_->autotuned_shapes.end()) {
        return invalid("SM120 FP8 GEMM shape was not tuned during setup");
    }
    void* quantized = prequantized ? const_cast<void*>(input) : scratch_data();
    if (!prequantized) {
        quantize_fp8_static(
            static_cast<const __nv_bfloat16*>(input),
            static_cast<__nv_fp8_e4m3*>(quantized), activation_scale,
            static_cast<int>(input_elements), cuda_stream(stream));
        const cudaError_t launched = cudaGetLastError();
        if (launched != cudaSuccess) return backend(cudaGetErrorString(launched));
    }
    try {
        impl_->gemm.fp8_nn_dev(
            quantized, const_cast<void*>(weight.device_data), output,
            rows, output_width, input_width, activation_scale,
            const_cast<float*>(weight.scale_data), cuda_stream(stream));
        return modalities::Status::ok();
    } catch (const std::exception& error) {
        return backend(error.what());
    } catch (...) {
        return backend("SM120 FP8 linear launch failed");
    }
}

modalities::Status Sm120Fp8Linear::freeze_autotune() {
    if (!impl_ || impl_->autotune_frozen || impl_->autotuned_shapes.empty()) {
        return invalid("SM120 FP8 autotune state is invalid");
    }
    impl_->autotune_frozen = true;
    return modalities::Status::ok();
}

float* Sm120Fp8Linear::scale_data(
    const Pi05LinearActivationSite& site) const {
    return activation_ ? activation_->scale_data(site) : nullptr;
}

void* Sm120Fp8Linear::scratch_data() const {
    return activation_ ? activation_->scratch_data() : nullptr;
}

std::size_t Sm120Fp8Linear::scratch_bytes() const {
    return activation_ ? activation_->scratch_bytes() : 0;
}

std::size_t Sm120Fp8Linear::autotuned_shape_count() const {
    return impl_ ? impl_->autotuned_shapes.size() : 0;
}

bool Sm120Fp8Linear::autotune_frozen() const {
    return impl_ && impl_->autotune_frozen;
}

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt
