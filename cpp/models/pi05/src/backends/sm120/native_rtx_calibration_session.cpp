#include "flashrt/cpp/models/pi05/backends/sm120/native_rtx_calibration_session.h"

#include "flashrt/cpp/loader/sha256.h"
#include "flashrt/cpp/models/pi05/model/io.h"
#include "flashrt/cpp/models/pi05/support/native_calibration.h"
#include "flashrt/cpp/models/pi05/backends/sm120/native_graph_owner.h"
#include "flashrt/cpp/models/pi05/model/prompt_embed.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <future>
#include <new>
#include <utility>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

constexpr std::size_t kVisionScales = 27 * 4 + 1;
constexpr std::size_t kEncoderScales = 18 * 4;

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status backend(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kBackend,
                                     message);
}

struct HashResult {
    bool ok = false;
    std::string digest;
    std::string error;
};

modalities::TensorView device_view(const NativeWorkspaceBuffer* buffer,
                                   modalities::DType dtype,
                                   modalities::Layout layout,
                                   modalities::Shape shape) {
    modalities::TensorView view;
    if (!buffer) return view;
    view.data = frt_buffer_dptr(buffer->buffer);
    view.bytes = frt_buffer_bytes(buffer->buffer);
    view.dtype = dtype;
    view.place = modalities::MemoryPlace::kDevice;
    view.layout = layout;
    view.shape = shape;
    return view;
}

}  // namespace

struct NativeRtxCalibrationSession::Impl {
    Impl(NativeCalibrationConfig value, double requested_percentile)
        : config(std::move(value)), percentile(requested_percentile) {}

    ~Impl() {
        modalities::text_embedding_staging_destroy(&text_staging);
        modalities::vision_staging_destroy(&vision_staging);
    }

    modalities::Status initialize() {
        int device = 0;
        cudaDeviceProp properties{};
        cudaError_t rc = cudaGetDevice(&device);
        if (rc == cudaSuccess) {
            rc = cudaGetDeviceProperties(&properties, device);
        }
        if (rc != cudaSuccess) return backend(cudaGetErrorString(rc));
        if (properties.major != 12 || properties.minor != 0) {
            return modalities::Status::error(
                modalities::StatusCode::kUnsupported,
                "Pi0.5 RTX FP8 calibration requires SM120");
        }
        hardware =
            "sm" + std::to_string(properties.major * 10 + properties.minor);

        const std::string weights_path =
            config.checkpoint_path + "/model.safetensors";
        std::future<HashResult> weights_hash = std::async(
            std::launch::async, [weights_path] {
                HashResult result;
                result.ok = loader::sha256_file(
                    weights_path, &result.digest, &result.error);
                return result;
            });
        const std::string tokenizer_path = config.tokenizer_model_path;
        std::future<HashResult> tokenizer_hash = std::async(
            std::launch::async, [tokenizer_path] {
                HashResult result;
                result.ok = loader::sha256_file(
                    tokenizer_path, &result.digest, &result.error);
                return result;
            });

        NativeGraphConfig graph_config;
        graph_config.num_views = config.num_views;
        graph_config.max_prompt_tokens = config.max_prompt_tokens;
        graph_config.chunk_size = config.chunk_size;
        graph_config.num_steps = config.num_steps;
        graph_config.vision_pool_factor = config.vision_pool_factor;
        graph_config.precision = NativeGraphPrecision::kFp8E4M3;
        modalities::Status st;
        graph = NativeGraphOwner::create_calibration(
            config.checkpoint_path, graph_config, &st);
        if (!graph) return st;

        HashResult weights_digest = weights_hash.get();
        if (!weights_digest.ok) {
            return modalities::Status::error(
                modalities::StatusCode::kNotFound, weights_digest.error);
        }
        weights_sha256 = std::move(weights_digest.digest);
        HashResult tokenizer_digest = tokenizer_hash.get();
        if (!tokenizer_digest.ok) {
            return modalities::Status::error(
                modalities::StatusCode::kNotFound, tokenizer_digest.error);
        }
        tokenizer_sha256 = std::move(tokenizer_digest.digest);

        const std::uint64_t frame_capacity =
            static_cast<std::uint64_t>(config.max_frame_width) *
            static_cast<std::uint64_t>(config.max_frame_height) * 4;
        st = modalities::vision_staging_create(
            &vision_staging, static_cast<std::uint32_t>(config.num_views),
            frame_capacity);
        if (!st.ok_status()) return st;
        st = modalities::text_embedding_staging_create(
            &text_staging, config.max_prompt_tokens);
        if (!st.ok_status()) return st;
        st = tokenizer.load_model(config.tokenizer_model_path);
        if (!st.ok_status()) return st;
        tokenizer.reserve(config.max_prompt_tokens);

        const NativeWorkspaceBuffer* images =
            graph->workspace().find("observation_images_normalized");
        const NativeWorkspaceBuffer* noise =
            graph->workspace().find("diffusion_noise");
        image_output = device_view(
            images, modalities::DType::kBFloat16,
            modalities::Layout::kNHWC,
            {static_cast<std::uint64_t>(config.num_views), 224, 224, 3});
        action_output = device_view(
            noise, modalities::DType::kBFloat16,
            modalities::Layout::kFlat,
            {static_cast<std::uint64_t>(config.chunk_size), 32});
        if (!image_output.data || !action_output.data) {
            return invalid("RTX calibration IO buffers are incomplete");
        }
        io.reset(new (std::nothrow) RuntimeIo(
            config.num_views, image_output, action_output, {}, {},
            graph->native_stream(), config.chunk_size, 32, 32,
            modalities::DType::kBFloat16, &vision_staging, nullptr, true));
        if (!io) return backend("RTX calibration IO allocation failed");

        const NativeDeviceWeight* embedding =
            graph->weights().find("embedding_weight");
        const NativeWorkspaceBuffer* prompt =
            graph->workspace().find("prompt_embedding");
        if (!embedding || !prompt ||
            embedding->dtype != NativeWeightDType::kBf16 ||
            embedding->shape.size() != 2 || embedding->shape[1] != 2048) {
            return invalid("RTX calibration embedding buffers are invalid");
        }
        embedding_table.data = frt_buffer_dptr(embedding->buffer);
        embedding_table.bytes = frt_buffer_bytes(embedding->buffer);
        embedding_table.dtype = modalities::DType::kBFloat16;
        embedding_table.place = modalities::MemoryPlace::kDevice;
        embedding_table.layout = modalities::Layout::kFlat;
        embedding_table.shape = {embedding->shape[0], embedding->shape[1]};
        prompt_output = device_view(
            prompt, modalities::DType::kBFloat16,
            modalities::Layout::kFlat,
            {static_cast<std::uint64_t>(config.max_prompt_tokens), 2048});
        prompt_spec.vocab_size = embedding->shape[0];
        prompt_spec.hidden_dim = 2048;
        prompt_spec.max_tokens = config.max_prompt_tokens;
        prompt_spec.scale = std::sqrt(2048.0f);
        token_ids.reserve(static_cast<std::size_t>(config.max_prompt_tokens) + 1);
        const std::size_t max_prompt_bytes =
            static_cast<std::size_t>(config.max_prompt_tokens) * 8;
        formatted_prompt.reserve(
            max_prompt_bytes + static_cast<std::size_t>(config.state_dim) * 5 +
            32);
        vision_scale_ones.assign(kVisionScales, 1.0f);
        encoder_scale_ones.assign(kEncoderScales, 1.0f);
        decoder_scale_ones.assign(
            static_cast<std::size_t>(config.num_steps) * 18 * 4, 1.0f);
        return modalities::Status::ok();
    }

    const NativeWorkspaceBuffer* scale_buffer(
        const char* name,
        std::size_t elements) const {
        const NativeWorkspaceBuffer* buffer =
            graph ? graph->workspace().find(name) : nullptr;
        if (!buffer || buffer->dtype != modalities::DType::kFloat32 ||
            buffer->shape != std::vector<std::uint64_t>({elements}) ||
            frt_buffer_bytes(buffer->buffer) != elements * sizeof(float)) {
            return nullptr;
        }
        return buffer;
    }

    modalities::Status upload_scales(
        const char* name,
        const std::vector<float>& values) const {
        const NativeWorkspaceBuffer* buffer =
            scale_buffer(name, values.size());
        if (!buffer) return invalid("RTX calibration scale buffer is invalid");
        const cudaError_t rc = cudaMemcpyAsync(
            frt_buffer_dptr(buffer->buffer), values.data(),
            values.size() * sizeof(float), cudaMemcpyHostToDevice,
            static_cast<cudaStream_t>(graph->native_stream()));
        return rc == cudaSuccess
                   ? modalities::Status::ok()
                   : backend(cudaGetErrorString(rc));
    }

    modalities::Status download_scales(
        const char* name,
        std::size_t elements,
        std::vector<float>* output) const {
        if (!output) return invalid("RTX calibration scale output is null");
        const NativeWorkspaceBuffer* buffer = scale_buffer(name, elements);
        if (!buffer) return invalid("RTX calibration scale buffer is invalid");
        output->resize(elements);
        const cudaError_t rc = cudaMemcpyAsync(
            output->data(), frt_buffer_dptr(buffer->buffer),
            elements * sizeof(float), cudaMemcpyDeviceToHost,
            static_cast<cudaStream_t>(graph->native_stream()));
        return rc == cudaSuccess
                   ? modalities::Status::ok()
                   : backend(cudaGetErrorString(rc));
    }

    modalities::Status reset_scales() const {
        modalities::Status st = upload_scales(
            "rtx_fp8_vision_scales", vision_scale_ones);
        if (!st.ok_status()) return st;
        st = upload_scales(
            "rtx_fp8_encoder_scales", encoder_scale_ones);
        if (!st.ok_status()) return st;
        return upload_scales(
            "rtx_fp8_decoder_scales", decoder_scale_ones);
    }

    modalities::Status collect_scales(
        std::vector<float>* vision,
        std::vector<float>* encoder,
        std::vector<float>* decoder) const {
        if (!vision || !encoder || !decoder) {
            return invalid("RTX calibration scale outputs are null");
        }
        modalities::Status st = download_scales(
            "rtx_fp8_vision_scales", vision_scale_ones.size(), vision);
        if (!st.ok_status()) return st;
        st = download_scales(
            "rtx_fp8_encoder_scales", encoder_scale_ones.size(), encoder);
        if (!st.ok_status()) return st;
        return download_scales(
            "rtx_fp8_decoder_scales", decoder_scale_ones.size(), decoder);
    }

    modalities::Status observe(
        const std::string& prompt,
        const float* state,
        std::uint64_t n_state,
        const std::vector<modalities::VisionFrame>& frames,
        const float* noise,
        std::uint64_t n_noise,
        std::uint64_t noise_seed) {
        modalities::Status st = normalize_native_calibration_state(
            config, state, n_state, &normalized_state);
        if (!st.ok_status()) return st;
        st = prepare_native_calibration_noise(
            noise, n_noise,
            noise_seed + static_cast<std::uint64_t>(vision_samples.size()),
            static_cast<std::size_t>(config.chunk_size) * 32,
            modalities::DType::kBFloat16, &noise_bf16);
        if (!st.ok_status()) return st;

        std::uint64_t prompt_len = 0;
        st = embed_prompt(
            tokenizer, prompt_spec, prompt, normalized_state.data(),
            normalized_state.size(), embedding_table, prompt_output,
            &token_ids, &prompt_len, graph->native_stream(), &text_staging,
            &formatted_prompt);
        if (!st.ok_status()) return st;
        st = graph->set_prompt_length(static_cast<int>(prompt_len));
        if (!st.ok_status()) return st;
        st = io->prepare_vision(frames);
        if (!st.ok_status()) return st;
        st = reset_scales();
        if (!st.ok_status()) return st;

        cudaError_t rc = cudaMemcpyAsync(
            action_output.data, noise_bf16.data(),
            noise_bf16.size() * sizeof(std::uint16_t),
            cudaMemcpyHostToDevice,
            static_cast<cudaStream_t>(graph->native_stream()));
        if (rc != cudaSuccess) return backend(cudaGetErrorString(rc));
        if (graph->replay() != FRT_OK) {
            return backend("RTX calibration graph replay failed");
        }

        std::vector<float> vision_scale;
        std::vector<float> encoder_scale;
        std::vector<float> decoder_scale;
        st = collect_scales(
            &vision_scale, &encoder_scale, &decoder_scale);
        if (!st.ok_status()) return st;
        st = graph->synchronize();
        if (!st.ok_status()) return st;
        vision_samples.push_back(std::move(vision_scale));
        encoder_samples.push_back(std::move(encoder_scale));
        decoder_samples.push_back(std::move(decoder_scale));
        return modalities::Status::ok();
    }

    modalities::Status finalize(const std::string& artifact_path) const {
        if (vision_samples.empty() ||
            vision_samples.size() != encoder_samples.size() ||
            vision_samples.size() != decoder_samples.size()) {
            return invalid("RTX calibration has no complete samples");
        }
        NativeCalibrationArtifact artifact;
        artifact.activation_dtype = "bfloat16";
        artifact.hardware = hardware;
        artifact.weights_sha256 = weights_sha256;
        artifact.tokenizer_sha256 = tokenizer_sha256;
        artifact.num_views = config.num_views;
        artifact.max_prompt_tokens = config.max_prompt_tokens;
        artifact.state_dim = config.state_dim;
        artifact.chunk_size = config.chunk_size;
        artifact.num_steps = config.num_steps;
        artifact.vision_pool_factor = config.vision_pool_factor;
        artifact.sample_count = vision_samples.size();
        artifact.percentile = percentile;
        modalities::Status st = reduce_native_calibration_samples(
            vision_samples, percentile, &artifact.vision_scales);
        if (!st.ok_status()) return st;
        st = reduce_native_calibration_samples(
            encoder_samples, percentile, &artifact.encoder_scales);
        if (!st.ok_status()) return st;
        st = reduce_native_calibration_samples(
            decoder_samples, percentile, &artifact.decoder_scales);
        if (!st.ok_status()) return st;
        return save_native_calibration_artifact(artifact_path, artifact);
    }

    NativeCalibrationConfig config;
    double percentile = 99.9;
    std::string hardware;
    std::string weights_sha256;
    std::string tokenizer_sha256;
    std::unique_ptr<NativeGraphOwner> graph;
    modalities::VisionStaging vision_staging;
    modalities::TextEmbeddingStaging text_staging;
    modalities::SentencePieceTokenizer tokenizer;
    std::unique_ptr<RuntimeIo> io;
    modalities::TensorView image_output;
    modalities::TensorView action_output;
    modalities::TensorView embedding_table;
    modalities::TensorView prompt_output;
    PromptEmbeddingSpec prompt_spec;
    std::vector<std::int32_t> token_ids;
    std::vector<float> normalized_state;
    std::string formatted_prompt;
    std::vector<std::uint16_t> noise_bf16;
    std::vector<float> vision_scale_ones;
    std::vector<float> encoder_scale_ones;
    std::vector<float> decoder_scale_ones;
    std::vector<std::vector<float>> vision_samples;
    std::vector<std::vector<float>> encoder_samples;
    std::vector<std::vector<float>> decoder_samples;
};

NativeRtxCalibrationSession::NativeRtxCalibrationSession(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

NativeRtxCalibrationSession::~NativeRtxCalibrationSession() = default;

std::unique_ptr<NativeRtxCalibrationSession>
NativeRtxCalibrationSession::create(
    const NativeCalibrationConfig& config,
    double percentile,
    modalities::Status* status) {
    if (!valid_native_calibration_config(config) ||
        !std::isfinite(percentile) || percentile < 0.0 ||
        percentile > 100.0) {
        if (status) *status = invalid("RTX calibration config is invalid");
        return nullptr;
    }
    std::unique_ptr<Impl> impl(
        new (std::nothrow) Impl(config, percentile));
    if (!impl) {
        if (status) *status = backend("RTX calibration allocation failed");
        return nullptr;
    }
    modalities::Status st = impl->initialize();
    if (!st.ok_status()) {
        if (status) *status = st;
        return nullptr;
    }
    std::unique_ptr<NativeRtxCalibrationSession> session(
        new (std::nothrow) NativeRtxCalibrationSession(std::move(impl)));
    if (!session) {
        if (status) {
            *status = backend("RTX calibration session allocation failed");
        }
        return nullptr;
    }
    if (status) *status = modalities::Status::ok();
    return session;
}

modalities::Status NativeRtxCalibrationSession::observe(
    const std::string& prompt,
    const float* state,
    std::uint64_t n_state,
    const std::vector<modalities::VisionFrame>& frames,
    const float* noise,
    std::uint64_t n_noise,
    std::uint64_t noise_seed) {
    return impl_ ? impl_->observe(
                       prompt, state, n_state, frames, noise, n_noise,
                       noise_seed)
                 : invalid("RTX calibration session is invalid");
}

modalities::Status NativeRtxCalibrationSession::finalize(
    const std::string& artifact_path) const {
    return impl_ ? impl_->finalize(artifact_path)
                 : invalid("RTX calibration session is invalid");
}

std::uint64_t NativeRtxCalibrationSession::sample_count() const {
    return impl_ ? impl_->vision_samples.size() : 0;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt
