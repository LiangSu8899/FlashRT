#include "flashrt/cpp/models/pi05/native_bf16_forward.h"
#include "flashrt/cpp/models/pi05/native_calibration.h"
#include "flashrt/cpp/models/pi05/native_rtx_weight_packer.h"
#include "flashrt/cpp/models/pi05/native_weight_materializer.h"

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct CaptureArgs {
    const flashrt::models::pi05::NativeBf16Forward* forward = nullptr;
    const flashrt::models::pi05::NativeDeviceWeightStore* weights = nullptr;
    flashrt::models::pi05::NativeWorkspace* workspace = nullptr;
    flashrt::models::pi05::NativeRtxAttentionWorkspace* attention = nullptr;
    const flashrt::models::pi05::NativeRtxAttentionDriver* attention_driver =
        nullptr;
    bool recorded = false;
    std::string error;
};

void record_vision(void* user, void* stream) {
    auto* args = static_cast<CaptureArgs*>(user);
    const flashrt::modalities::Status st = args->forward->vision(
        *args->weights, args->workspace, args->attention,
        args->attention_driver, reinterpret_cast<std::uintptr_t>(stream));
    args->recorded = st.ok_status();
    args->error = st.message;
}

bool write_buffer(std::ofstream* file, const void* device,
                  std::size_t elements) {
    std::vector<std::uint16_t> host(elements);
    if (cudaMemcpy(host.data(), device, elements * sizeof(std::uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    file->write(reinterpret_cast<const char*>(host.data()),
                static_cast<std::streamsize>(host.size() *
                                             sizeof(std::uint16_t)));
    return file->good();
}

bool download_buffer(
    const void* device,
    std::size_t elements,
    std::vector<std::uint16_t>* host) {
    if (!device || !host) return false;
    host->resize(elements);
    return cudaMemcpy(
               host->data(), device, elements * sizeof(std::uint16_t),
               cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool upload_scales(
    flashrt::models::pi05::NativeWorkspace* workspace,
    const std::vector<float>& scales) {
    const auto* output = workspace->find("rtx_fp8_vision_scales");
    return output && output->shape ==
                         std::vector<std::uint64_t>({scales.size()}) &&
           cudaMemcpy(frt_buffer_dptr(output->buffer), scales.data(),
                      scales.size() * sizeof(float),
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

bool read_images(
    const char* path,
    std::vector<std::uint16_t>* images) {
    if (!path || !images) return false;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file ||
        file.tellg() != static_cast<std::streamoff>(
                            images->size() * sizeof(std::uint16_t))) {
        return false;
    }
    file.seekg(0);
    file.read(
        reinterpret_cast<char*>(images->data()),
        static_cast<std::streamsize>(
            images->size() * sizeof(std::uint16_t)));
    return file.good();
}

flashrt::modalities::Status write_vision_layer_diagnostics(
    const char* path,
    const flashrt::models::pi05::NativeKernelDriver& driver,
    const flashrt::models::pi05::NativeBf16Forward& forward,
    const flashrt::models::pi05::NativeDeviceWeightStore& weights,
    flashrt::models::pi05::NativeWorkspace* workspace,
    flashrt::models::pi05::NativeRtxAttentionWorkspace* attention,
    const flashrt::models::pi05::NativeRtxAttentionDriver& attention_driver,
    cudaStream_t stream) {
    using flashrt::models::pi05::NativeWorkspaceBuffer;
    const NativeWorkspaceBuffer* images =
        workspace->find("observation_images_normalized");
    const NativeWorkspaceBuffer* patches =
        workspace->find("vision_patches");
    const NativeWorkspaceBuffer* position =
        workspace->find("vision_pos_embed_expanded");
    const NativeWorkspaceBuffer* x = workspace->find("vision_x");
    const NativeWorkspaceBuffer* x_norm =
        workspace->find("vision_x_norm");
    const auto* patch_weight = weights.find("vision_patch_embedding_w");
    const auto* patch_bias = weights.find("vision_patch_embedding_b");
    const auto* norm_weight = weights.find("vision_pre_attn_norm_w_0");
    const auto* norm_bias = weights.find("vision_pre_attn_norm_b_0");
    if (!images || !patches || !position || !x || !x_norm ||
        !patch_weight || !patch_bias || !norm_weight || !norm_bias) {
        return flashrt::modalities::Status::error(
            flashrt::modalities::StatusCode::kInvalidArgument,
            "native vision diagnostic buffers are incomplete");
    }
    const auto ptr = [](const auto* value) {
        return frt_buffer_dptr(value->buffer);
    };
    const int sequence = workspace->vision_sequence();
    const std::uintptr_t native_stream =
        reinterpret_cast<std::uintptr_t>(stream);
    flashrt::modalities::Status st = driver.patch_im2col_16bit(
        ptr(images), ptr(patches), workspace->num_views(), native_stream);
    if (st.ok_status()) {
        st = driver.bf16_nn(
            ptr(patches), ptr(patch_weight), ptr(x), sequence, 1152, 588,
            native_stream);
    }
    if (st.ok_status()) {
        st = driver.bias_residual_bf16(
            ptr(x), ptr(position), ptr(patch_bias), sequence, 1152,
            native_stream);
    }
    if (st.ok_status()) {
        st = driver.layer_norm_bf16(
            ptr(x), ptr(norm_weight), ptr(norm_bias), ptr(x_norm), sequence,
            1152, 1.0e-5f, native_stream);
    }
    if (!st.ok_status()) return st;

    int diagnostic_layer = 0;
    if (const char* value =
            std::getenv("FLASHRT_VISION_DIAGNOSTIC_LAYER")) {
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (!end || *end != '\0' || parsed < 0 || parsed >= 27) {
            return flashrt::modalities::Status::error(
                flashrt::modalities::StatusCode::kInvalidArgument,
                "native vision diagnostic layer is invalid");
        }
        diagnostic_layer = static_cast<int>(parsed);
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    const std::size_t elements =
        static_cast<std::size_t>(sequence) * 1152;
    if (!output || cudaStreamSynchronize(stream) != cudaSuccess ||
        !write_buffer(&output, ptr(x), elements)) {
        return flashrt::modalities::Status::error(
            flashrt::modalities::StatusCode::kBackend,
            "native vision diagnostic output failed");
    }
    std::vector<std::uint16_t> layer_qkv;
    std::vector<std::uint16_t> layer_attention;
    std::vector<std::uint16_t> layer_hidden;
    for (int layer = 0; layer < 27; ++layer) {
        st = forward.vision_layer(
            layer, weights, workspace, attention, &attention_driver,
            native_stream);
        if (!st.ok_status()) return st;
        if (cudaStreamSynchronize(stream) != cudaSuccess ||
            !write_buffer(&output, ptr(x), elements)) {
            return flashrt::modalities::Status::error(
                flashrt::modalities::StatusCode::kBackend,
                "native vision layer diagnostic output failed");
        }
        if (layer == diagnostic_layer) {
            const NativeWorkspaceBuffer* qkv =
                workspace->find("vision_QKV");
            const NativeWorkspaceBuffer* hidden =
                workspace->find("vision_hidden");
            const auto* attention_output =
                attention->find("attn_vis_O");
            if (!qkv || !hidden || !attention_output ||
                !download_buffer(
                    ptr(qkv), static_cast<std::size_t>(sequence) * 3456,
                    &layer_qkv) ||
                !download_buffer(
                    frt_buffer_dptr(attention_output->buffer),
                    static_cast<std::size_t>(sequence) * 1152,
                    &layer_attention) ||
                !download_buffer(
                    ptr(hidden), static_cast<std::size_t>(sequence) * 4304,
                    &layer_hidden)) {
                return flashrt::modalities::Status::error(
                    flashrt::modalities::StatusCode::kBackend,
                    "native vision sublayer diagnostics failed");
            }
        }
    }
    for (const auto* values : {
             &layer_qkv,
             &layer_attention,
             &layer_hidden}) {
        output.write(
            reinterpret_cast<const char*>(values->data()),
            static_cast<std::streamsize>(
                values->size() * sizeof(std::uint16_t)));
        if (!output) {
            return flashrt::modalities::Status::error(
                flashrt::modalities::StatusCode::kBackend,
                "native vision diagnostic write failed");
        }
    }
    return flashrt::modalities::Status::ok();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 5) {
        std::cerr << "usage: pi05_native_vision_probe CHECKPOINT OUTPUT "
                     "[CALIBRATION INPUT_BF16]\n";
        return 2;
    }
    using namespace flashrt::models::pi05;
    const bool fp8 = argc == 5;
    NativeCalibrationArtifact calibration;
    if (fp8) {
        const flashrt::modalities::Status calibration_status =
            load_native_calibration_artifact(argv[3], &calibration);
        if (!calibration_status.ok_status() ||
            calibration.activation_dtype != "bfloat16" ||
            calibration.hardware != "sm120") {
            std::cerr << (calibration_status.ok_status()
                              ? "SM120 FP8 calibration is required"
                              : calibration_status.message)
                      << '\n';
            return 2;
        }
    }
    flashrt::loader::SafetensorsFile source;
    if (!source.open(std::string(argv[1]) + "/model.safetensors")) {
        std::cerr << source.error() << '\n';
        return 2;
    }
    frt_ctx ctx = frt_ctx_create();
    if (!ctx) return 1;
    NativeDeviceWeightStore weights(ctx);
    NativeWeightMaterializer materializer(source, &weights);
    flashrt::modalities::Status st = materializer.materialize_vision_globals();
    for (int layer = 0; layer < 27 && st.ok_status(); ++layer) {
        st = materializer.materialize_vision_layer(layer);
    }
    NativeKernelDriver driver;
    if (st.ok_status()) st = driver.status();
    if (fp8 && st.ok_status()) {
        NativeRtxWeightPacker packer(&weights, &driver);
        for (int layer = 0; layer < 27 && st.ok_status(); ++layer) {
            for (const char* stem : {
                     "vision_attn_qkv_w_", "vision_attn_o_w_",
                     "vision_ffn_up_w_", "vision_ffn_down_w_"}) {
                st = packer.pack_weight(
                    std::string(stem) + std::to_string(layer));
                if (!st.ok_status()) break;
            }
        }
        if (st.ok_status()) {
            st = packer.pack_weight(
                "encoder_multi_modal_projector_w", "vision_projector_w");
        }
    }
    NativeWorkspace workspace(ctx);
    NativeRtxAttentionWorkspace attention(ctx);
    NativeWorkspaceConfig workspace_config;
    if (fp8) {
        workspace_config.num_views = calibration.num_views;
        workspace_config.max_prompt_tokens = calibration.max_prompt_tokens;
        workspace_config.chunk_size = calibration.chunk_size;
        workspace_config.num_steps = calibration.num_steps;
        workspace_config.vision_pool_factor =
            calibration.vision_pool_factor;
        workspace_config.flavor = NativeWorkspaceFlavor::kRtxFp8;
    }
    NativeRtxAttentionConfig attention_config;
    if (st.ok_status()) st = workspace.allocate(workspace_config);
    if (st.ok_status()) {
        st = workspace.expand_vision_position_embedding(weights);
    }
    attention_config.num_views = workspace.num_views();
    attention_config.encoder_sequence = workspace.encoder_sequence();
    attention_config.encoder_vision_sequence =
        workspace.encoder_vision_sequence();
    attention_config.chunk_size = workspace.chunk_size();
    if (st.ok_status()) st = attention.allocate(attention_config);
    if (st.ok_status() && fp8 &&
        !upload_scales(&workspace, calibration.vision_scales)) {
        st = flashrt::modalities::Status::error(
            flashrt::modalities::StatusCode::kBackend,
            "native vision scale upload failed");
    }
    if (!st.ok_status()) {
        std::cerr << st.message << '\n';
        frt_ctx_destroy(ctx);
        return 1;
    }
    const auto* images = workspace.find("observation_images_normalized");
    const auto* vision_x = workspace.find("vision_x");
    const auto* encoder_x = workspace.find("encoder_x");
    std::vector<std::uint16_t> host_images(
        static_cast<std::size_t>(workspace.num_views()) * 224 * 224 * 3);
    if (fp8) {
        if (!read_images(argv[4], &host_images)) {
            std::cerr << "native vision input is invalid\n";
            frt_ctx_destroy(ctx);
            return 1;
        }
    } else {
        for (std::size_t i = 0; i < host_images.size(); ++i) {
            const float value =
                static_cast<float>(static_cast<int>(i % 257) - 128) /
                128.0f;
            host_images[i] = flashrt::modalities::float_to_bfloat16(value);
        }
    }
    if (!images || !vision_x || !encoder_x ||
        cudaMemcpy(frt_buffer_dptr(images->buffer), host_images.data(),
                   host_images.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        frt_ctx_destroy(ctx);
        return 1;
    }
    NativeRtxAttentionDriver attention_driver(&attention);
    NativeRtxLinear linear(
        &driver, fp8 ? NativeRtxLinearMode::kFp8Static
                     : NativeRtxLinearMode::kBf16);
    NativeBf16Forward forward(&driver, &linear);
    const int vision_sequence = workspace.vision_sequence();
    if (fp8 && std::getenv("FLASHRT_VISION_LAYER_DIAGNOSTICS")) {
        cudaStream_t diagnostic_stream = nullptr;
        if (cudaStreamCreate(&diagnostic_stream) != cudaSuccess) {
            frt_ctx_destroy(ctx);
            return 1;
        }
        st = write_vision_layer_diagnostics(
            argv[2], driver, forward, weights, &workspace, &attention,
            attention_driver, diagnostic_stream);
        cudaStreamDestroy(diagnostic_stream);
        frt_ctx_destroy(ctx);
        if (!st.ok_status()) {
            std::cerr << st.message << '\n';
            return 1;
        }
        std::cout << "PASS native vision layer diagnostics\n";
        return 0;
    }
    frt_graph graph = frt_graph_create(
        ctx, "native_vision", vision_sequence);
    cudaStream_t stream = nullptr;
    if (!graph || cudaStreamCreate(&stream) != cudaSuccess ||
        frt_graph_bind(graph, "images", images->buffer) != FRT_OK ||
        frt_graph_bind(graph, "vision_x", vision_x->buffer) != FRT_OK ||
        frt_graph_bind(graph, "encoder_x", encoder_x->buffer) != FRT_OK) {
        frt_ctx_destroy(ctx);
        return 1;
    }
    CaptureArgs capture{&forward, &weights, &workspace, &attention,
                        &attention_driver, false, {}};
    const int capture_rc = frt_graph_capture(
        graph, vision_sequence, record_vision, &capture);
    if (capture_rc != FRT_OK || !capture.recorded) {
        std::cerr << "vision capture failed: rc=" << capture_rc
                  << " status=" << capture.error << '\n';
        frt_graph_destroy(graph);
        cudaStreamDestroy(stream);
        frt_ctx_destroy(ctx);
        return 1;
    }
    const int stream_id = frt_ctx_wrap_stream(ctx, stream);
    for (int i = 0; i < 100; ++i) {
        if (cudaMemcpyAsync(frt_buffer_dptr(images->buffer), host_images.data(),
                            host_images.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess ||
            frt_graph_replay(graph, vision_sequence, stream_id) != FRT_OK) {
            frt_graph_destroy(graph);
            cudaStreamDestroy(stream);
            frt_ctx_destroy(ctx);
            return 1;
        }
    }
    if (frt_graph_variant_count(graph) != 1 ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        frt_graph_destroy(graph);
        cudaStreamDestroy(stream);
        frt_ctx_destroy(ctx);
        return 1;
    }
    std::ofstream file(argv[2], std::ios::binary | std::ios::trunc);
    const bool ok = file &&
        write_buffer(
            &file, frt_buffer_dptr(vision_x->buffer),
            static_cast<std::size_t>(vision_sequence) * 1152) &&
        write_buffer(
            &file, frt_buffer_dptr(encoder_x->buffer),
            static_cast<std::size_t>(workspace.encoder_vision_sequence()) *
                2048);
    frt_graph_destroy(graph);
    cudaStreamDestroy(stream);
    frt_ctx_destroy(ctx);
    if (!ok) return 1;
    std::cout << "PASS native vision 27 layers\n";
    return 0;
}
