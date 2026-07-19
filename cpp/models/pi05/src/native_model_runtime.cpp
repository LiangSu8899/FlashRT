#include "native_open_internal.h"

#if defined(FLASHRT_CPP_HAS_SENTENCEPIECE) && \
    (defined(FLASHRT_CPP_WITH_PI05_SM120_TARGET) || defined(FLASHRT_CPP_WITH_PI05_SM110_TARGET))

#include "config_map.h"
#include "flashrt/cpp/loader/sha256.h"
#include "flashrt/cpp/models/pi05/model_runtime.h"
#include "flashrt/cpp/models/pi05/support/native_calibration.h"
#include "flashrt/cpp/models/pi05/model/native_session.h"
#include "flashrt/cpp/models/pi05/model/dims.h"
#include "flashrt/cpp/models/pi05/model/execution_plan.h"
#if defined(FLASHRT_CPP_WITH_PI05_SM120_TARGET)
#include "flashrt/cpp/models/pi05/targets/sm120/target.h"
#endif
#if defined(FLASHRT_CPP_WITH_PI05_SM110_TARGET)
#include "flashrt/cpp/models/pi05/targets/sm110/target.h"
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

void release_backend_session(void* owner) {
    delete static_cast<Pi05NativeSession*>(owner);
}

int update_prompt_length(void* owner, std::uint64_t prompt_len) {
    auto* session = static_cast<Pi05NativeSession*>(owner);
    if (!session || prompt_len > static_cast<std::uint64_t>(INT_MAX)) {
        return -1;
    }
    return cface::status_code(
        session->set_prompt_length(static_cast<int>(prompt_len)));
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
            precision = config.calibration_path.empty()
                            ? Precision::kBf16
                            : Precision::kFp8E4M3Fn;
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
        !((properties.major == 11 || properties.major == 12) &&
          properties.minor == 0)) {
        if (error) *error = "Pi0.5 native FP8 requires SM110 or SM120";
        return -3;
    }
#if !defined(FLASHRT_CPP_WITH_PI05_SM120_TARGET)
    if (precision == Precision::kBf16) {
        if (error) *error = "Pi0.5 native BF16 backend is not built";
        return -3;
    }
#endif
#if !defined(FLASHRT_CPP_WITH_PI05_SM110_TARGET)
    if (precision == Precision::kFp8E4M3Fn && properties.major == 11) {
        if (error) *error = "Pi0.5 native Thor FP8 backend is not built";
        return -3;
    }
#endif
#if !defined(FLASHRT_CPP_WITH_PI05_SM120_TARGET)
    if (precision == Precision::kFp8E4M3Fn && properties.major == 12) {
        if (error) *error = "Pi0.5 native RTX FP8 backend is not built";
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
            result.ok = loader::sha256_file_cached(
                weights_path, &result.digest, nullptr, &result.error);
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
            calibration.activation_dtype !=
                (properties.major == 11 ? "float16" : "bfloat16") ||
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
    }

    Pi05ShapeConfig shape_config;
    shape_config.num_views = config.num_views;
    shape_config.max_prompt_tokens = config.max_prompt_tokens;
    shape_config.chunk = config.chunk;
    shape_config.num_steps = config.num_steps;
    shape_config.vision_pool_factor = config.vision_pool_factor;
    shape_config.state_dim = config.state_dim;
    shape_config.robot_action_dim =
        static_cast<std::int64_t>(config.action_q01.size());
    modalities::Status st;
    Pi05ResolvedShape shape;
    st = resolve_pi05_shape(shape_config, &shape);
    if (!st.ok_status()) {
        if (error) *error = st.message;
        return cface::status_code(st);
    }

    const bool thor_fp8 = precision == Precision::kFp8E4M3Fn &&
                          properties.major == 11;
    const bool rtx_fp8 = precision == Precision::kFp8E4M3Fn &&
                         properties.major == 12;
    frt_ctx context = frt_ctx_create();
    if (!context) {
        if (error) *error = "native execution context creation failed";
        return -6;
    }
    std::unique_ptr<Pi05TargetBundle> target;
    if (precision == Precision::kBf16 || rtx_fp8) {
#if defined(FLASHRT_CPP_WITH_PI05_SM120_TARGET)
        targets::sm120::Sm120TargetConfig target_config;
        target_config.checkpoint_path = config.checkpoint_path;
        if (rtx_fp8) {
            target_config.execution_mode =
                targets::sm120::Sm120ExecutionMode::kStaticFp8E4M3;
            target_config.calibration = calibration;
        }
        target = targets::sm120::Sm120TargetBundle::create(
            context, shape, std::move(target_config), &st);
#endif
    } else if (thor_fp8) {
#if defined(FLASHRT_CPP_WITH_PI05_SM110_TARGET)
        targets::sm110::Sm110TargetConfig target_config;
        target_config.checkpoint_path = config.checkpoint_path;
        target_config.calibration = calibration;
        target = targets::sm110::Sm110TargetBundle::create(
            context, shape, std::move(target_config), &st);
#endif
    }
    if (!target) {
        frt_ctx_destroy(context);
        if (error) *error = st.message;
        return cface::status_code(st);
    }
    std::unique_ptr<Pi05NativeSession> session = Pi05NativeSession::create(
        context, shape, std::move(target), &st);
    if (!session) {
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

    const Pi05ResolvedResources& resources = session->resources();
    const Pi05ResolvedBuffer* images = &resources.buffers.images;
    const Pi05ResolvedBuffer* noise = &resources.buffers.noise;
    const Pi05ResolvedBuffer* encoder = &resources.buffers.encoder_state;
    const Pi05ResolvedBuffer* previous = &resources.buffers.previous_actions;
    const Pi05ResolvedBuffer* prefix_weights =
        &resources.buffers.prefix_weights;
    const Pi05ResolvedBuffer* guidance = &resources.buffers.guidance_weight;
    const Pi05ResolvedBuffer* prompt = &resources.buffers.prompt_embedding;
    const Pi05ResolvedWeight* embedding = &resources.weights.embedding_table;
    const Pi05ExecutionPlanDescriptor* execution_plan =
        pi05_execution_plan(config.stage_plan.c_str());
    if (!execution_plan || !execution_plan->stage_count) {
        if (error) *error = "Pi0.5 execution plan is invalid";
        return -1;
    }

    frt_runtime_builder builder =
        frt_runtime_builder_create(session->context());
    if (!builder) {
        if (error) *error = "native runtime builder creation failed";
        return -6;
    }
    const frt_shape_key keys[] = {0};
    bool ok = frt_runtime_builder_add_stream(
                  builder, "main", session->stream_id(), 0,
                  session->native_stream()) == 0;
    std::size_t graph_count = 0;
    const Pi05GraphDescriptor* graph_catalog =
        pi05_graph_catalog(&graph_count);
    for (std::size_t i = 0; ok && i < graph_count; ++i) {
        const Pi05GraphDescriptor& spec = graph_catalog[i];
        ok = static_cast<std::size_t>(spec.id) == i &&
             frt_runtime_builder_add_graph(
                 builder, spec.name, session->graph(spec.id), 0, keys, 1,
                 session->stream_id()) == 0;
    }
    ok = ok && frt_runtime_builder_add_buffer(
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

    const bool fp8 = precision == Precision::kFp8E4M3Fn;
    const std::string precision_id = fp8 ? "fp8_e4m3fn" : "bf16";
    const std::string pipeline_id = thor_fp8
                                        ? "NativeThorFp8"
                                        : rtx_fp8 ? "NativeRtxFp8"
                                                  : "NativeBf16";
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
         add_identity(builder, "stage_plan", execution_plan->name) &&
         add_identity(builder, "state_prompt_mode", "fixed") &&
         add_identity(builder, "num_views", std::to_string(config.num_views)) &&
         add_identity(builder, "max_prompt_len",
                      std::to_string(config.max_prompt_tokens)) &&
         add_identity(builder, "state_dim", std::to_string(config.state_dim)) &&
         add_identity(builder, "chunk_size", std::to_string(config.chunk)) &&
         add_identity(builder, "num_steps", std::to_string(config.num_steps)) &&
         add_identity(builder, "vision_pool_factor",
                      std::to_string(config.vision_pool_factor)) &&
         add_identity(builder, "model_action_dim",
                      std::to_string(kPi05ModelDims.action_width)) &&
         add_identity(builder, "robot_action_dim",
                      std::to_string(config.action_q01.size()));
    if (ok && fp8) {
        ok = add_identity(builder, "calibration_sha256", calibration_sha256);
    }
    if (!ok) return fail_builder(builder, error, "native identity build failed");

    std::ostringstream manifest;
    manifest << "{\"model\":\"pi05\",\"producer\":\"native\","
             << "\"hardware\":\"" << hardware_id
             << "\",\"precision\":\"" << precision_id
             << "\",\"io\":\"native_v2\",\"graphs\":[";
    for (std::size_t i = 0; i < graph_count; ++i) {
        if (i) manifest << ',';
        manifest << '\"' << graph_catalog[i].name << '\"';
    }
    manifest << "],\"stage_plan\":{\"name\":\""
             << execution_plan->name << "\",\"stages\":[";
    for (std::size_t i = 0; i < execution_plan->stage_count; ++i) {
        const Pi05StageDescriptor& stage = execution_plan->stages[i];
        const Pi05GraphDescriptor* graph = pi05_graph_descriptor(stage.graph);
        if (!graph) {
            return fail_builder(builder, error,
                                "native execution plan build failed");
        }
        if (i) manifest << ',';
        manifest << "{\"name\":\"" << stage.name
                 << "\",\"graph\":\"" << graph->name
                 << "\",\"after\":[";
        for (std::size_t d = 0; d < stage.after_count; ++d) {
            const std::uint32_t dependency = stage.after[d];
            if (dependency >= i) {
                return fail_builder(builder, error,
                                    "native execution plan build failed");
            }
            if (d) manifest << ',';
            manifest << '\"' << execution_plan->stages[dependency].name
                     << '\"';
        }
        manifest << "]}";
    }
    manifest << "]}}";
    if (frt_runtime_builder_set_manifest(builder, manifest.str().c_str()) != 0) {
        return fail_builder(builder, error, "native manifest build failed");
    }

    const int64_t prompt_shape[] = {-1};
    const int64_t state_shape[] = {config.state_dim};
    const int64_t image_shape[] = {
        config.num_views, kPi05ModelDims.image_height,
        kPi05ModelDims.image_width, kPi05ModelDims.image_channels};
    const int64_t raw_action_shape[] = {
        config.chunk, kPi05ModelDims.action_width};
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
             frt_buffer_bytes(noise->buffer)) == 0;
    for (std::size_t i = 0; ok && i < execution_plan->stage_count; ++i) {
        const Pi05StageDescriptor& stage = execution_plan->stages[i];
        ok = frt_runtime_builder_add_stage(
                 builder, static_cast<std::uint32_t>(stage.graph),
                 stage.after, stage.after_count) == 0;
    }
    if (!ok) return fail_builder(builder, error, "native port/stage build failed");

    Pi05NativeSession* raw_session = session.release();
    /* This base is retained only by the verb override below and is never
     * returned to a consumer. The published object always has real verbs. */
    frt_model_runtime_verbs base_verbs = unpublished_verbs();
    frt_model_runtime_v1* base = frt_runtime_builder_finish_model(
        builder, &base_verbs, nullptr, raw_session, nullptr,
        release_backend_session);
    if (!base) {
        delete raw_session;
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
    runtime_config.model_action_dim = kPi05ModelDims.action_width;
    runtime_config.robot_action_dim = static_cast<int>(action_mean.size());
    runtime_config.action_mean = action_mean.data();
    runtime_config.n_action_mean = action_mean.size();
    runtime_config.action_stddev = action_stddev.data();
    runtime_config.n_action_stddev = action_stddev.size();
    const Pi05GraphDescriptor* first_graph =
        pi05_graph_descriptor(execution_plan->stages[0].graph);
    runtime_config.graph_name = first_graph ? first_graph->name : nullptr;
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
        embedding->device_data;
    runtime_config.prompt_embedding_table_bytes = embedding->bytes;
    runtime_config.prompt_embedding_table_dtype = runtime_dtype;
    runtime_config.prompt_embedding_vocab_size = embedding->shape.dims[0];
    runtime_config.prompt_embedding_hidden_dim =
        kPi05ModelDims.encoder_width;
    runtime_config.prompt_embedding_data = frt_buffer_dptr(prompt->buffer);
    runtime_config.prompt_embedding_bytes = frt_buffer_bytes(prompt->buffer);
    runtime_config.prompt_embedding_dtype = runtime_dtype;
    runtime_config.max_prompt_tokens = config.max_prompt_tokens;
    runtime_config.prompt_embedding_scale =
        std::sqrt(static_cast<float>(kPi05ModelDims.encoder_width));
    runtime_config.state_q01 = config.state_q01.data();
    runtime_config.n_state_q01 = config.state_q01.size();
    runtime_config.state_q99 = config.state_q99.data();
    runtime_config.n_state_q99 = config.state_q99.size();
    runtime_config.prompt_length_update = update_prompt_length;
    runtime_config.prompt_length_update_user = raw_session;
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
