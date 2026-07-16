#include "native_open_internal.h"

#if defined(FLASHRT_CPP_HAS_SENTENCEPIECE) && \
    (defined(FLASHRT_CPP_WITH_FA2) || defined(FLASHRT_CPP_WITH_THOR_FP8))

#include "config_map.h"
#include "flashrt/cpp/loader/sha256.h"
#include "flashrt/cpp/models/pi05/model_runtime.h"
#include "flashrt/cpp/models/pi05/native_graph_runtime.h"
#if defined(FLASHRT_CPP_WITH_FA2)
#include "flashrt/cpp/models/pi05/native_graph_owner.h"
#endif
#if defined(FLASHRT_CPP_WITH_THOR_FP8)
#include "flashrt/cpp/models/pi05/native_calibration.h"
#include "flashrt/cpp/models/pi05/native_thor_graph_owner.h"
#endif

#include <cuda_runtime_api.h>

#include <climits>
#include <cmath>
#include <future>
#include <memory>
#include <sstream>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

const NativeWorkspaceBuffer* workspace_buffer(
    const NativeGraphRuntime& owner,
    const char* name) {
    return owner.workspace().find(name);
}

void release_graph_owner(void* owner) {
    delete static_cast<NativeGraphRuntime*>(owner);
}

int update_prompt_length(void* owner, std::uint64_t prompt_len) {
    auto* graph = static_cast<NativeGraphRuntime*>(owner);
    if (!graph || prompt_len > static_cast<std::uint64_t>(INT_MAX)) return -1;
    return cface::status_code(
        graph->set_prompt_length(static_cast<int>(prompt_len)));
}

bool add_identity(frt_runtime_builder builder, const char* key,
                  const std::string& value) {
    return frt_runtime_builder_add_identity(builder, key, value.c_str()) == 0;
}

int unpublished_set_input(void*, uint32_t, const void*, uint64_t, int) {
    return -3;
}
int unpublished_get_output(void*, uint32_t, void*, uint64_t, uint64_t*, int) {
    return -3;
}

frt_model_runtime_verbs unpublished_verbs() {
    frt_model_runtime_verbs verbs{};
    verbs.struct_size = sizeof(verbs);
    verbs.set_input = unpublished_set_input;
    verbs.get_output = unpublished_get_output;
    return verbs;
}

int fail_builder(frt_runtime_builder builder, std::string* error,
                 const char* message) {
    frt_model_runtime_verbs discard_verbs = unpublished_verbs();
    frt_model_runtime_v1* discarded = frt_runtime_builder_finish_model(
        builder, &discard_verbs, nullptr, nullptr, nullptr, nullptr);
    if (discarded) discarded->release(discarded->owner);
    if (error) *error = message;
    return -6;
}

}  // namespace

int build_native_model_runtime(const NativeOpenConfig& config,
                               frt_model_runtime_v1** out,
                               std::string* error) {
    if (!out) return -1;
    *out = nullptr;
    int device = 0;
    cudaDeviceProp properties{};
    cudaError_t cuda_rc = cudaGetDevice(&device);
    if (cuda_rc == cudaSuccess) {
        cuda_rc = cudaGetDeviceProperties(&properties, device);
    }
    if (cuda_rc != cudaSuccess) {
        if (error) *error = cudaGetErrorString(cuda_rc);
        return -6;
    }
    const std::string hardware_id =
        "sm" + std::to_string(properties.major * 10 + properties.minor);
    enum class Precision { kBf16, kFp8E4M3Fn };
    Precision precision;
    if (config.precision == "auto") {
        if (properties.major == 12 && properties.minor == 0) {
            precision = Precision::kBf16;
        } else if (properties.major == 11 && properties.minor == 0) {
            precision = Precision::kFp8E4M3Fn;
        } else {
            if (error) {
                *error = "Pi0.5 native_v2 has no backend for " + hardware_id;
            }
            return -3;
        }
    } else if (config.precision == "bf16") {
        precision = Precision::kBf16;
    } else if (config.precision == "fp8_e4m3fn") {
        precision = Precision::kFp8E4M3Fn;
    } else {
        if (error) *error = "Pi0.5 native precision is invalid";
        return -1;
    }
    if (precision == Precision::kBf16 &&
        (properties.major != 12 || properties.minor != 0)) {
        if (error) *error = "Pi0.5 native BF16 requires SM120";
        return -3;
    }
    if (precision == Precision::kFp8E4M3Fn &&
        (properties.major != 11 || properties.minor != 0)) {
        if (error) *error = "Pi0.5 native FP8 requires SM110";
        return -3;
    }
#if !defined(FLASHRT_CPP_WITH_FA2)
    if (precision == Precision::kBf16) {
        if (error) *error = "Pi0.5 native BF16 backend is not built";
        return -3;
    }
#endif
#if !defined(FLASHRT_CPP_WITH_THOR_FP8)
    if (precision == Precision::kFp8E4M3Fn) {
        if (error) *error = "Pi0.5 native Thor FP8 backend is not built";
        return -3;
    }
#endif

    struct HashResult {
        bool ok = false;
        std::string digest;
        std::string error;
    };
    const std::string weights_path =
        config.checkpoint_path + "/model.safetensors";
    std::future<HashResult> weights_hash = std::async(
        std::launch::async, [weights_path] {
            HashResult result;
            result.ok = loader::sha256_file(
                weights_path, &result.digest, &result.error);
            return result;
        });
    std::string tokenizer_sha256;
    std::string hash_error;
    if (!loader::sha256_file(config.tokenizer_model_path, &tokenizer_sha256,
                             &hash_error)) {
        if (error) *error = hash_error;
        return -2;
    }

    NativeCalibrationArtifact calibration;
    std::string calibration_sha256;
    if (precision == Precision::kFp8E4M3Fn) {
#if defined(FLASHRT_CPP_WITH_THOR_FP8)
        if (config.calibration_path.empty()) {
            if (error) *error = "Pi0.5 native FP8 requires calibration_path";
            return -1;
        }
        modalities::Status calibration_status =
            load_native_calibration_artifact(config.calibration_path,
                                             &calibration);
        if (!calibration_status.ok_status()) {
            if (error) *error = calibration_status.message;
            return cface::status_code(calibration_status);
        }
        if (calibration.hardware != hardware_id ||
            calibration.tokenizer_sha256 != tokenizer_sha256 ||
            calibration.num_views != config.num_views ||
            calibration.max_prompt_tokens != config.max_prompt_tokens ||
            calibration.state_dim != config.state_dim ||
            calibration.chunk_size != config.chunk ||
            calibration.num_steps != config.num_steps ||
            calibration.vision_pool_factor != config.vision_pool_factor) {
            if (error) *error = "Pi0.5 calibration identity does not match config";
            return -2;
        }
        if (!loader::sha256_file(config.calibration_path,
                                 &calibration_sha256, &hash_error)) {
            if (error) *error = hash_error;
            return -2;
        }
#endif
    }

    NativeGraphConfig graph_config;
    graph_config.num_views = config.num_views;
    graph_config.max_prompt_tokens = config.max_prompt_tokens;
    graph_config.chunk_size = config.chunk;
    graph_config.num_steps = config.num_steps;
    graph_config.vision_pool_factor = config.vision_pool_factor;
    modalities::Status st;
    std::unique_ptr<NativeGraphRuntime> graph;
    if (precision == Precision::kBf16) {
#if defined(FLASHRT_CPP_WITH_FA2)
        graph = NativeGraphOwner::create(
            config.checkpoint_path, graph_config, &st);
#endif
    } else {
#if defined(FLASHRT_CPP_WITH_THOR_FP8)
        graph = NativeThorGraphOwner::create(
            config.checkpoint_path, graph_config, calibration, &st);
#endif
    }
    if (!graph) {
        if (error) *error = st.message;
        return cface::status_code(st);
    }
    HashResult weights_sha256 = weights_hash.get();
    if (!weights_sha256.ok) {
        if (error) *error = weights_sha256.error;
        return -2;
    }
    if (precision == Precision::kFp8E4M3Fn &&
        calibration.weights_sha256 != weights_sha256.digest) {
        if (error) *error = "Pi0.5 calibration checkpoint digest mismatch";
        return -2;
    }

    const NativeWorkspaceBuffer* images =
        workspace_buffer(*graph, "observation_images_normalized");
    const NativeWorkspaceBuffer* noise =
        workspace_buffer(*graph, "diffusion_noise");
    const NativeWorkspaceBuffer* encoder =
        workspace_buffer(*graph, "encoder_x");
    const NativeWorkspaceBuffer* previous =
        workspace_buffer(*graph, "rtc_prev_action_chunk");
    const NativeWorkspaceBuffer* prefix_weights =
        workspace_buffer(*graph, "rtc_prefix_weights");
    const NativeWorkspaceBuffer* guidance =
        workspace_buffer(*graph, "rtc_guidance_weight");
    const NativeWorkspaceBuffer* prompt =
        workspace_buffer(*graph, "prompt_embedding");
    const NativeDeviceWeight* embedding = graph->weights().find(
        "embedding_weight");
    if (!images || !noise || !encoder || !previous || !prefix_weights ||
        !guidance || !prompt || !embedding ||
        embedding->dtype != (precision == Precision::kBf16
                                 ? NativeWeightDType::kBf16
                                 : NativeWeightDType::kFloat16) ||
        embedding->shape.size() != 2 || embedding->shape[1] != 2048) {
        if (error) *error = "native graph export buffers are incomplete";
        return -6;
    }

    frt_runtime_builder builder = frt_runtime_builder_create(graph->context());
    if (!builder) {
        if (error) *error = "native runtime builder creation failed";
        return -6;
    }
    const frt_shape_key keys[] = {0};
    bool ok =
        frt_runtime_builder_add_stream(
            builder, "main", graph->stream_id(), 0,
            graph->native_stream()) == 0 &&
        frt_runtime_builder_add_graph(
            builder, "infer", graph->infer_graph(), 0, keys, 1,
            graph->stream_id()) == 0 &&
        frt_runtime_builder_add_buffer(
            builder, "observation_images_normalized", images->buffer,
            frt_buffer_bytes(images->buffer), FRT_RT_ROLE_INPUT) == 0 &&
        frt_runtime_builder_add_buffer(
            builder, "diffusion_noise", noise->buffer,
            frt_buffer_bytes(noise->buffer),
            FRT_RT_ROLE_INPUT | FRT_RT_ROLE_OUTPUT) == 0 &&
        frt_runtime_builder_add_buffer(
            builder, "encoder_x", encoder->buffer,
            frt_buffer_bytes(encoder->buffer),
            FRT_RT_ROLE_INPUT | FRT_RT_ROLE_STATE) == 0 &&
        frt_runtime_builder_add_buffer(
            builder, "rtc_prev_action_chunk", previous->buffer,
            frt_buffer_bytes(previous->buffer), FRT_RT_ROLE_INPUT) == 0 &&
        frt_runtime_builder_add_buffer(
            builder, "rtc_prefix_weights", prefix_weights->buffer,
            frt_buffer_bytes(prefix_weights->buffer), FRT_RT_ROLE_INPUT) == 0 &&
        frt_runtime_builder_add_buffer(
            builder, "rtc_guidance_weight", guidance->buffer,
            frt_buffer_bytes(guidance->buffer), FRT_RT_ROLE_INPUT) == 0 &&
        frt_runtime_builder_add_buffer(
            builder, "prompt_embedding", prompt->buffer,
            frt_buffer_bytes(prompt->buffer),
            FRT_RT_ROLE_INPUT | FRT_RT_ROLE_STATE) == 0;
    if (!ok) return fail_builder(builder, error, "native descriptor build failed");

    ok = frt_runtime_builder_add_region(
             builder, "rollout_boundary", noise->buffer, 0,
             frt_buffer_bytes(noise->buffer),
             FRT_RT_REGION_SNAPSHOT | FRT_RT_REGION_RESTORE) == 0;
    if (!ok) return fail_builder(builder, error, "native region build failed");

    const bool thor_fp8 = precision == Precision::kFp8E4M3Fn;
    const std::string precision_id = thor_fp8 ? "fp8_e4m3fn" : "bf16";
    const std::string pipeline_id = thor_fp8 ? "NativeThorFp8" : "NativeBf16";
    const std::string tensor_dtype = thor_fp8 ? "float16" : "bf16";
    ok = add_identity(builder, "model", "pi05") &&
         add_identity(builder, "producer", "native") &&
         add_identity(builder, "pipeline", pipeline_id) &&
         add_identity(builder, "hardware", hardware_id) &&
         add_identity(builder, "precision", precision_id) &&
         add_identity(builder, "tensor_dtype", tensor_dtype) &&
         add_identity(builder, "weights_sha256", weights_sha256.digest) &&
         add_identity(builder, "tokenizer_sha256", tokenizer_sha256) &&
         add_identity(builder, "io", "native_v2") &&
         add_identity(builder, "state_prompt_mode", "fixed") &&
         add_identity(builder, "num_views", std::to_string(config.num_views)) &&
         add_identity(builder, "max_prompt_len",
                      std::to_string(config.max_prompt_tokens)) &&
         add_identity(builder, "state_dim", std::to_string(config.state_dim)) &&
         add_identity(builder, "chunk_size", std::to_string(config.chunk)) &&
         add_identity(builder, "num_steps", std::to_string(config.num_steps)) &&
         add_identity(builder, "vision_pool_factor",
                      std::to_string(config.vision_pool_factor)) &&
         add_identity(builder, "model_action_dim", "32") &&
         add_identity(builder, "robot_action_dim",
                      std::to_string(config.action_q01.size()));
    if (ok && thor_fp8) {
        ok = add_identity(builder, "calibration_sha256", calibration_sha256);
    }
    if (!ok) return fail_builder(builder, error, "native identity build failed");

    std::ostringstream manifest;
    manifest << "{\"model\":\"pi05\",\"producer\":\"native\","
             << "\"hardware\":\"" << hardware_id
             << "\",\"precision\":\"" << precision_id
             << "\",\"io\":\"native_v2\","
             << "\"stage_plan\":{\"name\":\"full\","
             << "\"stages\":[{\"name\":\"infer\","
             << "\"graph\":\"infer\",\"after\":[]}]}}";
    if (frt_runtime_builder_set_manifest(builder, manifest.str().c_str()) != 0) {
        return fail_builder(builder, error, "native manifest build failed");
    }

    const int64_t prompt_shape[] = {-1};
    const int64_t state_shape[] = {config.state_dim};
    const int64_t image_shape[] = {config.num_views, 224, 224, 3};
    const int64_t raw_action_shape[] = {config.chunk, 32};
    const int64_t action_shape[] = {
        config.chunk, static_cast<int64_t>(config.action_q01.size())};
    const std::uint64_t action_bytes =
        static_cast<std::uint64_t>(config.chunk) *
        config.action_q01.size() * sizeof(float);
    const uint32_t io_dtype =
        thor_fp8 ? FRT_RT_DTYPE_F16 : FRT_RT_DTYPE_BF16;
    ok = frt_runtime_builder_add_port(
             builder, "prompt", FRT_RT_MOD_TEXT, FRT_RT_DTYPE_U8,
             FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_IN, FRT_RT_PORT_STAGED, 1,
             prompt_shape, 1, 0, nullptr, 0, 0) == 0 &&
         frt_runtime_builder_add_port(
             builder, "state", FRT_RT_MOD_STATE, FRT_RT_DTYPE_F32,
             FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_IN, FRT_RT_PORT_STAGED, 1,
             state_shape, 1, 0, nullptr, 0, 0) == 0 &&
         frt_runtime_builder_add_port(
             builder, "images", FRT_RT_MOD_IMAGE, io_dtype,
             FRT_RT_LAYOUT_NHWC, FRT_RT_PORT_IN, FRT_RT_PORT_STAGED, 1,
             image_shape, 4, 30, images->buffer, 0,
             frt_buffer_bytes(images->buffer)) == 0 &&
         frt_runtime_builder_add_port(
             builder, "noise", FRT_RT_MOD_TENSOR, io_dtype,
             FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_IN, FRT_RT_PORT_SWAP, 0,
             raw_action_shape, 2, 0, noise->buffer, 0,
             frt_buffer_bytes(noise->buffer)) == 0 &&
         frt_runtime_builder_add_port(
             builder, "actions", FRT_RT_MOD_ACTION, FRT_RT_DTYPE_F32,
             FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_OUT, FRT_RT_PORT_STAGED, 0,
             action_shape, 2, 0, nullptr, 0, action_bytes) == 0 &&
         frt_runtime_builder_add_port(
             builder, "actions_raw", FRT_RT_MOD_TENSOR, io_dtype,
             FRT_RT_LAYOUT_FLAT, FRT_RT_PORT_OUT, FRT_RT_PORT_SWAP, 0,
             raw_action_shape, 2, 0, noise->buffer, 0,
             frt_buffer_bytes(noise->buffer)) == 0 &&
         frt_runtime_builder_add_stage(builder, 0, nullptr, 0) == 0;
    if (!ok) return fail_builder(builder, error, "native port/stage build failed");

    NativeGraphRuntime* raw_graph = graph.release();
    /* This base is retained only by the verb override below and is never
     * returned to a consumer. The published object always has real verbs. */
    frt_model_runtime_verbs base_verbs = unpublished_verbs();
    frt_model_runtime_v1* base = frt_runtime_builder_finish_model(
        builder, &base_verbs, nullptr, raw_graph, nullptr,
        release_graph_owner);
    if (!base) {
        delete raw_graph;
        if (error) *error = "native integrated runtime finish failed";
        return -6;
    }

    std::vector<float> action_mean(config.action_q01.size());
    std::vector<float> action_stddev(config.action_q01.size());
    for (std::size_t i = 0; i < action_mean.size(); ++i) {
        action_stddev[i] =
            (config.action_q99[i] - config.action_q01[i] + 1e-6f) * 0.5f;
        action_mean[i] = config.action_q01[i] + action_stddev[i];
    }
    frt_pi05_runtime_config runtime_config{};
    runtime_config.struct_size = sizeof(runtime_config);
    runtime_config.num_views = config.num_views;
    runtime_config.chunk = config.chunk;
    runtime_config.model_action_dim = 32;
    runtime_config.robot_action_dim = static_cast<int>(action_mean.size());
    runtime_config.action_mean = action_mean.data();
    runtime_config.n_action_mean = action_mean.size();
    runtime_config.action_stddev = action_stddev.data();
    runtime_config.n_action_stddev = action_stddev.size();
    runtime_config.graph_name = "infer";
    runtime_config.image_buffer_name = "observation_images_normalized";
    runtime_config.action_buffer_name = "diffusion_noise";
    const int runtime_dtype = thor_fp8 ? FRT_PI05_DTYPE_FLOAT16
                                      : FRT_PI05_DTYPE_BFLOAT16;
    runtime_config.image_dtype = runtime_dtype;
    runtime_config.action_dtype = runtime_dtype;
    runtime_config.max_frame_width = config.max_frame_width;
    runtime_config.max_frame_height = config.max_frame_height;
    runtime_config.prompt_tokenizer_model_path =
        config.tokenizer_model_path.c_str();
    runtime_config.prompt_embedding_table_data =
        frt_buffer_dptr(embedding->buffer);
    runtime_config.prompt_embedding_table_bytes =
        frt_buffer_bytes(embedding->buffer);
    runtime_config.prompt_embedding_table_dtype = runtime_dtype;
    runtime_config.prompt_embedding_vocab_size = embedding->shape[0];
    runtime_config.prompt_embedding_hidden_dim = 2048;
    runtime_config.prompt_embedding_data = frt_buffer_dptr(prompt->buffer);
    runtime_config.prompt_embedding_bytes = frt_buffer_bytes(prompt->buffer);
    runtime_config.prompt_embedding_dtype = runtime_dtype;
    runtime_config.max_prompt_tokens = config.max_prompt_tokens;
    runtime_config.prompt_embedding_scale = std::sqrt(2048.0f);
    runtime_config.state_q01 = config.state_q01.data();
    runtime_config.n_state_q01 = config.state_q01.size();
    runtime_config.state_q99 = config.state_q99.data();
    runtime_config.n_state_q99 = config.state_q99.size();
    runtime_config.prompt_length_update = update_prompt_length;
    runtime_config.prompt_length_update_user = raw_graph;
    runtime_config.prompt_embedding_on_device = 1;

    frt_model_runtime_v1* model = nullptr;
    const int rc = frt_pi05_model_runtime_create_over(
        base, &runtime_config, &model);
    base->release(base->owner);
    if (rc != 0 || !model) {
        if (error) *error = "native Pi0.5 verb overlay failed";
        return rc != 0 ? rc : -6;
    }
    *out = model;
    if (error) error->clear();
    return 0;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#else

namespace flashrt {
namespace models {
namespace pi05 {

int build_native_model_runtime(const NativeOpenConfig&,
                               frt_model_runtime_v1** out,
                               std::string* error) {
    if (out) *out = nullptr;
    if (error) {
        *error = "native graph backend and SentencePiece are unavailable";
    }
    return -3;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif
