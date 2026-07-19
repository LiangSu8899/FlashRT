#include "flashrt/cpp/models/pi05/support/native_calibration_session.h"

#include "flashrt/cpp/loader/sha256.h"
#include "flashrt/cpp/models/pi05/io.h"
#include "flashrt/cpp/models/pi05/model/dims.h"
#include "flashrt/cpp/models/pi05/model/prompt_embed.h"
#include "native_session_factory.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <future>
#include <new>
#include <utility>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

modalities::TensorView device_view(const Pi05ResolvedBuffer& buffer,
                                   modalities::Layout layout) {
    modalities::TensorView view;
    view.data = frt_buffer_dptr(buffer.buffer);
    view.bytes = frt_buffer_bytes(buffer.buffer);
    view.dtype = buffer.physical_dtype;
    view.place = modalities::MemoryPlace::kDevice;
    view.layout = layout;
    view.shape = buffer.physical_shape;
    return view;
}

struct HashResult final {
    bool ok = false;
    std::string digest;
    std::string error;
};

}  // namespace

struct NativeCalibrationSession::Impl final {
    Impl(NativeCalibrationConfig value, double requested_percentile)
        : config(std::move(value)), percentile(requested_percentile) {}

    ~Impl() {
        io.reset();
        modalities::text_embedding_staging_destroy(&text_staging);
        modalities::vision_staging_destroy(&vision_staging);
    }

    modalities::Status initialize() {
        const std::string weights_path =
            config.checkpoint_path + "/model.safetensors";
        std::future<HashResult> weights_hash = std::async(
            std::launch::async, [weights_path] {
                HashResult result;
                result.ok = loader::sha256_file_cached(
                    weights_path, &result.digest, nullptr, &result.error);
                return result;
            });
        std::future<HashResult> tokenizer_hash = std::async(
            std::launch::async, [path = config.tokenizer_model_path] {
                HashResult result;
                result.ok = loader::sha256_file(
                    path, &result.digest, &result.error);
                return result;
            });

        Pi05ShapeConfig shape_config;
        shape_config.num_views = config.num_views;
        shape_config.max_prompt_tokens = config.max_prompt_tokens;
        shape_config.chunk = config.chunk_size;
        shape_config.num_steps = config.num_steps;
        shape_config.vision_pool_factor = config.vision_pool_factor;
        shape_config.state_dim = config.state_dim;
        shape_config.robot_action_dim = kPi05ModelDims.action_width;
        Pi05ResolvedShape shape;
        modalities::Status status = resolve_pi05_shape(shape_config, &shape);
        if (!status.ok_status()) return status;
        session = create_native_session(
            config.checkpoint_path, shape,
            NativeSessionPrecision::kFp8E4M3Fn, std::nullopt,
            Pi05SessionMode::kUncaptured, &device, &status);
        if (!session) return status;

        HashResult weights = weights_hash.get();
        if (!weights.ok) {
            return modalities::Status::error(
                modalities::StatusCode::kNotFound, weights.error);
        }
        weights_sha256 = std::move(weights.digest);
        HashResult tokenizer = tokenizer_hash.get();
        if (!tokenizer.ok) {
            return modalities::Status::error(
                modalities::StatusCode::kNotFound, tokenizer.error);
        }
        tokenizer_sha256 = std::move(tokenizer.digest);

        const Pi05ResolvedResources& resources = session->resources();
        image_output = device_view(resources.buffers.images,
                                   modalities::Layout::kNHWC);
        noise_output = device_view(resources.buffers.noise,
                                   modalities::Layout::kFlat);
        prompt_output = device_view(resources.buffers.prompt_embedding,
                                    modalities::Layout::kFlat);
        const Pi05ResolvedWeight& embedding = resources.weights.embedding_table;
        if (!image_output.data || !noise_output.data || !prompt_output.data ||
            !embedding.device_data || embedding.shape.rank != 2 ||
            embedding.shape.dims[1] !=
                static_cast<std::uint64_t>(kPi05ModelDims.encoder_width)) {
            return invalid("native calibration staging buffers are invalid");
        }
        embedding_table.data = const_cast<void*>(embedding.device_data);
        embedding_table.bytes = embedding.bytes;
        embedding_table.dtype = device.activation_dtype;
        embedding_table.place = modalities::MemoryPlace::kDevice;
        embedding_table.layout = modalities::Layout::kFlat;
        embedding_table.shape = embedding.shape;

        const std::uint64_t width =
            static_cast<std::uint64_t>(config.max_frame_width);
        const std::uint64_t height =
            static_cast<std::uint64_t>(config.max_frame_height);
        status = modalities::vision_staging_create(
            &vision_staging, static_cast<std::uint32_t>(config.num_views),
            width * height * 4);
        if (!status.ok_status()) return status;
        status = modalities::text_embedding_staging_create(
            &text_staging, config.max_prompt_tokens);
        if (!status.ok_status()) return status;
        status = prompt_tokenizer.load_model(config.tokenizer_model_path);
        if (!status.ok_status()) return status;
        if (prompt_tokenizer.vocab_size() != embedding.shape.dims[0]) {
            return invalid("native calibration tokenizer vocabulary mismatch");
        }
        prompt_tokenizer.reserve(config.max_prompt_tokens);
        prompt_spec.vocab_size = embedding.shape.dims[0];
        prompt_spec.hidden_dim = embedding.shape.dims[1];
        prompt_spec.max_tokens = config.max_prompt_tokens;
        prompt_spec.scale = std::sqrt(
            static_cast<float>(kPi05ModelDims.encoder_width));
        token_ids.reserve(
            static_cast<std::size_t>(config.max_prompt_tokens) + 1);
        formatted_prompt.reserve(
            static_cast<std::size_t>(config.max_prompt_tokens) * 8 +
            static_cast<std::size_t>(config.state_dim) * 5 + 32);
        noise_words.resize(
            static_cast<std::size_t>(config.chunk_size) *
            kPi05ModelDims.action_width);

        io.reset(new (std::nothrow) RuntimeIo(
            config.num_views, image_output, noise_output, {}, {},
            session->native_stream(), config.chunk_size,
            kPi05ModelDims.action_width, kPi05ModelDims.action_width,
            device.activation_dtype, &vision_staging, nullptr, true));
        return io ? modalities::Status::ok()
                  : backend("native calibration IO allocation failed");
    }

    modalities::Status observe(
        const std::string& prompt,
        const float* state,
        std::uint64_t n_state,
        const std::vector<modalities::VisionFrame>& frames,
        const float* noise,
        std::uint64_t n_noise,
        std::uint64_t noise_seed) {
        if (!session || !io) return invalid("calibration session is invalid");
        modalities::Status status = normalize_native_calibration_state(
            config, state, n_state, &normalized_state);
        if (!status.ok_status()) return status;
        std::uint64_t prompt_length = 0;
        status = embed_prompt(
            prompt_tokenizer, prompt_spec, prompt, normalized_state.data(),
            normalized_state.size(), embedding_table, prompt_output,
            &token_ids, &prompt_length, session->native_stream(),
            &text_staging, &formatted_prompt);
        if (!status.ok_status()) return status;
        status = session->set_prompt_length(static_cast<int>(prompt_length));
        if (!status.ok_status()) return status;
        status = io->prepare_vision(frames);
        if (!status.ok_status()) return status;
        status = prepare_native_calibration_noise(
            noise, n_noise,
            noise_seed + static_cast<std::uint64_t>(samples.size()),
            noise_words.size(), device.activation_dtype, &noise_words);
        if (!status.ok_status()) return status;
        const cudaError_t copy = cudaMemcpyAsync(
            noise_output.data, noise_words.data(),
            noise_words.size() * sizeof(std::uint16_t),
            cudaMemcpyHostToDevice,
            static_cast<cudaStream_t>(session->native_stream()));
        if (copy != cudaSuccess) return backend(cudaGetErrorString(copy));

        status = session->reset_observer();
        if (!status.ok_status()) return status;
        status = session->execute(Pi05GraphId::kInfer);
        if (!status.ok_status()) return status;
        Pi05ObservedScales observed;
        status = session->download_observer(&observed);
        if (!status.ok_status()) return status;
        status = session->synchronize();
        if (!status.ok_status()) return status;
        samples.push_back(std::move(observed));
        return modalities::Status::ok();
    }

    modalities::Status finalize(const std::string& artifact_path) const {
        if (samples.empty()) return invalid("calibration has no samples");
        NativeCalibrationArtifact artifact;
        artifact.activation_dtype =
            device.activation_dtype == modalities::DType::kFloat16
                ? "float16"
                : "bfloat16";
        artifact.hardware = device.hardware;
        artifact.weights_sha256 = weights_sha256;
        artifact.tokenizer_sha256 = tokenizer_sha256;
        artifact.num_views = config.num_views;
        artifact.max_prompt_tokens = config.max_prompt_tokens;
        artifact.state_dim = config.state_dim;
        artifact.chunk_size = config.chunk_size;
        artifact.num_steps = config.num_steps;
        artifact.vision_pool_factor = config.vision_pool_factor;
        artifact.sample_count = samples.size();
        artifact.percentile = percentile;

        std::vector<std::vector<float>> vision;
        std::vector<std::vector<float>> encoder;
        std::vector<std::vector<float>> decoder;
        vision.reserve(samples.size());
        encoder.reserve(samples.size());
        decoder.reserve(samples.size());
        for (const Pi05ObservedScales& sample : samples) {
            if (!sample.vision.empty()) vision.push_back(sample.vision);
            encoder.push_back(sample.encoder);
            decoder.push_back(sample.decoder);
        }
        modalities::Status status = modalities::Status::ok();
        if (!vision.empty()) {
            if (vision.size() != samples.size()) {
                return invalid("calibration vision samples are incomplete");
            }
            status = reduce_native_calibration_samples(
                vision, percentile, &artifact.vision_scales);
            if (!status.ok_status()) return status;
        }
        status = reduce_native_calibration_samples(
            encoder, percentile, &artifact.encoder_scales);
        if (!status.ok_status()) return status;
        status = reduce_native_calibration_samples(
            decoder, percentile, &artifact.decoder_scales);
        if (!status.ok_status()) return status;
        return save_native_calibration_artifact(artifact_path, artifact);
    }

    NativeCalibrationConfig config;
    double percentile = 99.9;
    NativeDeviceProfile device;
    std::string weights_sha256;
    std::string tokenizer_sha256;
    std::unique_ptr<Pi05NativeSession> session;
    modalities::VisionStaging vision_staging;
    modalities::TextEmbeddingStaging text_staging;
    modalities::SentencePieceTokenizer prompt_tokenizer;
    std::unique_ptr<RuntimeIo> io;
    modalities::TensorView image_output;
    modalities::TensorView noise_output;
    modalities::TensorView embedding_table;
    modalities::TensorView prompt_output;
    PromptEmbeddingSpec prompt_spec;
    std::vector<std::int32_t> token_ids;
    std::vector<float> normalized_state;
    std::string formatted_prompt;
    std::vector<std::uint16_t> noise_words;
    std::vector<Pi05ObservedScales> samples;
};

NativeCalibrationSession::NativeCalibrationSession(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

NativeCalibrationSession::~NativeCalibrationSession() = default;

std::unique_ptr<NativeCalibrationSession> NativeCalibrationSession::create(
    const NativeCalibrationConfig& config,
    double percentile,
    modalities::Status* status) {
    if (!valid_native_calibration_config(config) ||
        !std::isfinite(percentile) || percentile < 0.0 ||
        percentile > 100.0) {
        if (status) *status = invalid("native calibration config is invalid");
        return nullptr;
    }
    std::unique_ptr<Impl> impl(
        new (std::nothrow) Impl(config, percentile));
    if (!impl) {
        if (status) *status = backend("native calibration allocation failed");
        return nullptr;
    }
    modalities::Status result = impl->initialize();
    if (!result.ok_status()) {
        if (status) *status = std::move(result);
        return nullptr;
    }
    std::unique_ptr<NativeCalibrationSession> session(
        new (std::nothrow) NativeCalibrationSession(std::move(impl)));
    if (!session) {
        if (status) {
            *status = backend("native calibration session allocation failed");
        }
        return nullptr;
    }
    if (status) *status = modalities::Status::ok();
    return session;
}

modalities::Status NativeCalibrationSession::observe(
    const std::string& prompt,
    const float* state,
    std::uint64_t n_state,
    const std::vector<modalities::VisionFrame>& frames,
    const float* noise,
    std::uint64_t n_noise,
    std::uint64_t noise_seed) {
    return impl_ ? impl_->observe(prompt, state, n_state, frames, noise,
                                  n_noise, noise_seed)
                 : invalid("native calibration session is invalid");
}

modalities::Status NativeCalibrationSession::finalize(
    const std::string& artifact_path) const {
    return impl_ ? impl_->finalize(artifact_path)
                 : invalid("native calibration session is invalid");
}

std::uint64_t NativeCalibrationSession::sample_count() const {
    return impl_ ? impl_->samples.size() : 0;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt
