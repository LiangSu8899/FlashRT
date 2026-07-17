#include "flashrt/cpp/models/pi05/plans/sm110/native_thor_calibration_session.h"

#include "flashrt/cpp/loader/safetensors.h"
#include "flashrt/cpp/loader/sha256.h"
#include "flashrt/cpp/models/pi05/model/io.h"
#include "flashrt/cpp/models/pi05/support/native_calibration.h"
#include "flashrt/cpp/models/pi05/plans/sm110/native_thor_fp8_forward.h"
#include "flashrt/cpp/models/pi05/plans/sm110/native_thor_style_precompute.h"
#include "flashrt/cpp/models/pi05/plans/sm110/native_thor_weight_materializer.h"
#include "flashrt/cpp/models/pi05/model/prompt_embed.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstring>
#include <future>
#include <new>

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

struct NativeThorCalibrationSession::Impl {
    explicit Impl(frt_ctx context, NativeThorCalibrationConfig value,
                  double requested_percentile)
        : config(std::move(value)),
          percentile(requested_percentile),
          ctx(context),
          weights(context),
          workspace(context),
          forward(&driver) {}

    ~Impl() {
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
        modalities::text_embedding_staging_destroy(&text_staging);
        modalities::vision_staging_destroy(&vision_staging);
        if (ctx) {
            frt_ctx_destroy(ctx);
            ctx = nullptr;
        }
    }

    modalities::Status initialize() {
        int device = 0;
        cudaDeviceProp properties{};
        cudaError_t rc = cudaGetDevice(&device);
        if (rc == cudaSuccess) {
            rc = cudaGetDeviceProperties(&properties, device);
        }
        if (rc != cudaSuccess) return backend(cudaGetErrorString(rc));
        if (properties.major != 11 || properties.minor != 0) {
            return modalities::Status::error(
                modalities::StatusCode::kUnsupported,
                "Pi0.5 FP8 calibration requires SM110");
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
        std::string hash_error;
        if (!loader::sha256_file(config.tokenizer_model_path,
                                 &tokenizer_sha256, &hash_error)) {
            return modalities::Status::error(
                modalities::StatusCode::kNotFound, hash_error);
        }
        loader::SafetensorsFile source;
        if (!source.open(weights_path)) {
            return modalities::Status::error(
                modalities::StatusCode::kNotFound, source.error());
        }
        NativeThorWeightMaterializer materializer(source, &weights);
        NativeThorMaterializationOptions options;
        options.num_steps = config.num_steps;
        options.include_embedding = true;
        modalities::Status st =
            materializer.materialize_all(options, &weight_scales);
        if (!st.ok_status()) return st;
        HashResult digest = weights_hash.get();
        if (!digest.ok) {
            return modalities::Status::error(
                modalities::StatusCode::kNotFound, digest.error);
        }
        weights_sha256 = std::move(digest.digest);

        NativeWorkspaceConfig workspace_config;
        workspace_config.num_views = config.num_views;
        workspace_config.max_prompt_tokens = config.max_prompt_tokens;
        workspace_config.chunk_size = config.chunk_size;
        workspace_config.num_steps = config.num_steps;
        workspace_config.vision_pool_factor = config.vision_pool_factor;
        workspace_config.flavor = NativeWorkspaceFlavor::kThorFp8;
        workspace_config.enable_calibration = true;
        st = workspace.allocate(workspace_config);
        if (!st.ok_status()) return st;
        st = workspace.expand_vision_position_embedding(weights);
        if (!st.ok_status()) return st;
        st = workspace.set_fixed_prompt_length(0);
        if (!st.ok_status()) return st;

        rc = cudaStreamCreate(&stream);
        if (rc != cudaSuccess) return backend(cudaGetErrorString(rc));
        NativeThorStylePrecomputer precomputer(&driver);
        st = precomputer.run(
            weights, &workspace, reinterpret_cast<std::uintptr_t>(stream));
        if (!st.ok_status()) return st;

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

        const NativeWorkspaceBuffer* image_buffer =
            workspace.find("observation_images_normalized");
        const NativeWorkspaceBuffer* noise_buffer =
            workspace.find("diffusion_noise");
        image_output = device_view(
            image_buffer, modalities::DType::kFloat16,
            modalities::Layout::kNHWC,
            {static_cast<std::uint64_t>(config.num_views), 224, 224, 3});
        action_output = device_view(
            noise_buffer, modalities::DType::kFloat16,
            modalities::Layout::kFlat,
            {static_cast<std::uint64_t>(config.chunk_size), 32});
        if (!image_output.data || !action_output.data) {
            return invalid("Thor calibration IO buffers are incomplete");
        }
        io.reset(new (std::nothrow) RuntimeIo(
            config.num_views, image_output, action_output, {}, {}, stream,
            config.chunk_size, 32, 32, modalities::DType::kFloat16,
            &vision_staging, nullptr, true));
        if (!io) return backend("Thor calibration IO allocation failed");

        const NativeDeviceWeight* embedding =
            weights.find("embedding_weight");
        const NativeWorkspaceBuffer* prompt =
            workspace.find("prompt_embedding");
        if (!embedding || !prompt ||
            embedding->dtype != NativeWeightDType::kFloat16 ||
            embedding->shape.size() != 2 || embedding->shape[1] != 2048) {
            return invalid("Thor calibration embedding buffers are invalid");
        }
        embedding_table.data = frt_buffer_dptr(embedding->buffer);
        embedding_table.bytes = frt_buffer_bytes(embedding->buffer);
        embedding_table.dtype = modalities::DType::kFloat16;
        embedding_table.place = modalities::MemoryPlace::kDevice;
        embedding_table.layout = modalities::Layout::kFlat;
        embedding_table.shape =
            {embedding->shape[0], embedding->shape[1]};
        prompt_output = device_view(
            prompt, modalities::DType::kFloat16,
            modalities::Layout::kFlat,
            {static_cast<std::uint64_t>(config.max_prompt_tokens), 2048});
        prompt_spec.vocab_size = embedding->shape[0];
        prompt_spec.hidden_dim = 2048;
        prompt_spec.max_tokens = config.max_prompt_tokens;
        prompt_spec.scale = std::sqrt(2048.0f);
        normalized_state.resize(config.state_dim);
        token_ids.reserve(static_cast<std::size_t>(config.max_prompt_tokens) + 1);
        const std::size_t max_prompt_bytes =
            static_cast<std::size_t>(config.max_prompt_tokens) * 8;
        formatted_prompt.reserve(
            max_prompt_bytes + static_cast<std::size_t>(config.state_dim) * 5 +
            32);
        noise_f16.resize(static_cast<std::size_t>(config.chunk_size) * 32);
        return modalities::Status::ok();
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
        std::uint64_t prompt_len = 0;
        st = embed_prompt(
            tokenizer, prompt_spec, prompt, normalized_state.data(),
            normalized_state.size(), embedding_table, prompt_output,
            &token_ids, &prompt_len, stream, &text_staging,
            &formatted_prompt);
        if (!st.ok_status()) return st;
        rc_check = cudaStreamSynchronize(stream);
        if (rc_check != cudaSuccess) return backend(cudaGetErrorString(rc_check));
        st = workspace.set_fixed_prompt_length(static_cast<int>(prompt_len));
        if (!st.ok_status()) return st;
        st = io->prepare_vision(frames);
        if (!st.ok_status()) return st;

        const NativeWorkspaceBuffer* encoder = workspace.find("encoder_x");
        const NativeWorkspaceBuffer* prompt_buffer =
            workspace.find("prompt_embedding");
        if (!encoder || !prompt_buffer) {
            return invalid("Thor calibration prompt window is missing");
        }
        const std::size_t prompt_offset =
            static_cast<std::size_t>(workspace.encoder_vision_sequence()) *
            2048 * sizeof(std::uint16_t);
        rc_check = cudaMemcpyAsync(
            static_cast<unsigned char*>(frt_buffer_dptr(encoder->buffer)) +
                prompt_offset,
            frt_buffer_dptr(prompt_buffer->buffer),
            frt_buffer_bytes(prompt_buffer->buffer), cudaMemcpyDeviceToDevice,
            stream);
        if (rc_check != cudaSuccess) return backend(cudaGetErrorString(rc_check));

        st = prepare_native_calibration_noise(
            noise, n_noise,
            noise_seed + static_cast<std::uint64_t>(encoder_samples.size()),
            static_cast<std::size_t>(config.chunk_size) * 32,
            modalities::DType::kFloat16, &noise_f16);
        if (!st.ok_status()) return st;
        rc_check = cudaMemcpyAsync(
            action_output.data, noise_f16.data(),
            noise_f16.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice,
            stream);
        if (rc_check != cudaSuccess) return backend(cudaGetErrorString(rc_check));

        const std::uintptr_t native_stream =
            reinterpret_cast<std::uintptr_t>(stream);
        st = forward.vision(weights, &workspace, weight_scales, native_stream);
        if (!st.ok_status()) return st;
        std::vector<float> encoder_scale;
        st = forward.calibrate_encoder(
            weights, &workspace, weight_scales, &encoder_scale, native_stream);
        if (!st.ok_status()) return st;
        std::vector<float> decoder_scale;
        st = forward.calibrate_decoder(
            weights, &workspace, &decoder_scale, native_stream);
        if (!st.ok_status()) return st;
        encoder_samples.push_back(std::move(encoder_scale));
        decoder_samples.push_back(std::move(decoder_scale));
        return modalities::Status::ok();
    }

    modalities::Status finalize(const std::string& artifact_path) const {
        if (encoder_samples.empty() ||
            encoder_samples.size() != decoder_samples.size()) {
            return invalid("Thor calibration has no complete samples");
        }
        NativeCalibrationArtifact artifact;
        artifact.hardware = hardware;
        artifact.weights_sha256 = weights_sha256;
        artifact.tokenizer_sha256 = tokenizer_sha256;
        artifact.num_views = config.num_views;
        artifact.max_prompt_tokens = config.max_prompt_tokens;
        artifact.state_dim = config.state_dim;
        artifact.chunk_size = config.chunk_size;
        artifact.num_steps = config.num_steps;
        artifact.vision_pool_factor = config.vision_pool_factor;
        artifact.sample_count = encoder_samples.size();
        artifact.percentile = percentile;
        modalities::Status st = reduce_native_calibration_samples(
            encoder_samples, percentile, &artifact.encoder_scales);
        if (!st.ok_status()) return st;
        st = reduce_native_calibration_samples(
            decoder_samples, percentile, &artifact.decoder_scales);
        if (!st.ok_status()) return st;
        return save_native_calibration_artifact(artifact_path, artifact);
    }

    NativeThorCalibrationConfig config;
    double percentile = 99.9;
    std::string hardware;
    std::string weights_sha256;
    std::string tokenizer_sha256;
    frt_ctx ctx = nullptr;
    NativeDeviceWeightStore weights;
    NativeWorkspace workspace;
    NativeThorKernelDriver driver;
    NativeThorFp8Forward forward;
    NativeThorWeightScales weight_scales;
    cudaStream_t stream = nullptr;
    cudaError_t rc_check = cudaSuccess;
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
    std::vector<std::uint16_t> noise_f16;
    std::vector<std::vector<float>> encoder_samples;
    std::vector<std::vector<float>> decoder_samples;
};

NativeThorCalibrationSession::NativeThorCalibrationSession(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

NativeThorCalibrationSession::~NativeThorCalibrationSession() = default;

std::unique_ptr<NativeThorCalibrationSession>
NativeThorCalibrationSession::create(
    const NativeThorCalibrationConfig& config,
    double percentile,
    modalities::Status* status) {
    if (!valid_native_calibration_config(config) ||
        config.vision_pool_factor != 1 || (config.max_prompt_tokens & 1) ||
        !std::isfinite(percentile) ||
        percentile < 0.0 || percentile > 100.0) {
        if (status) *status = invalid("Thor calibration config is invalid");
        return nullptr;
    }
    frt_ctx ctx = frt_ctx_create();
    if (!ctx) {
        if (status) *status = backend("Thor calibration context creation failed");
        return nullptr;
    }
    std::unique_ptr<Impl> impl(
        new (std::nothrow) Impl(ctx, config, percentile));
    if (!impl) {
        frt_ctx_destroy(ctx);
        if (status) *status = backend("Thor calibration allocation failed");
        return nullptr;
    }
    modalities::Status st = impl->initialize();
    if (!st.ok_status()) {
        if (status) *status = st;
        return nullptr;
    }
    std::unique_ptr<NativeThorCalibrationSession> session(
        new (std::nothrow) NativeThorCalibrationSession(std::move(impl)));
    if (!session) {
        if (status) *status = backend("Thor calibration session allocation failed");
        return nullptr;
    }
    if (status) *status = modalities::Status::ok();
    return session;
}

modalities::Status NativeThorCalibrationSession::observe(
    const std::string& prompt,
    const float* state,
    std::uint64_t n_state,
    const std::vector<modalities::VisionFrame>& frames,
    const float* noise,
    std::uint64_t n_noise,
    std::uint64_t noise_seed) {
    return impl_ ? impl_->observe(prompt, state, n_state, frames, noise,
                                 n_noise, noise_seed)
                 : invalid("Thor calibration session is invalid");
}

modalities::Status NativeThorCalibrationSession::finalize(
    const std::string& artifact_path) const {
    return impl_ ? impl_->finalize(artifact_path)
                 : invalid("Thor calibration session is invalid");
}

std::uint64_t NativeThorCalibrationSession::sample_count() const {
    return impl_ ? impl_->encoder_samples.size() : 0;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt
