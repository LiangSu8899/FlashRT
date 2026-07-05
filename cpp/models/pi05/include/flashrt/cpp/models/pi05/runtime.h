#ifndef FLASHRT_CPP_MODELS_PI05_RUNTIME_H
#define FLASHRT_CPP_MODELS_PI05_RUNTIME_H

#include "flashrt/cpp/families/vla/runtime.h"
#include "flashrt/cpp/models/pi05/io.h"

#include <string>

namespace flashrt {
namespace models {
namespace pi05 {

using ReplayFn = int (*)(frt_graph graph, frt_shape_key key, int stream_id,
                         void* user);

struct RuntimeConfig {
    int num_views = 3;
    int chunk = kDefaultChunk;
    int model_action_dim = kModelActionDim;
    int robot_action_dim = kLiberoActionDim;

    modalities::DType image_dtype = modalities::DType::kBFloat16;
    modalities::DType action_dtype = modalities::DType::kBFloat16;

    std::vector<float> action_mean;
    std::vector<float> action_stddev;

    std::string graph_name = "infer";
    std::string image_buffer_name = "observation_images_normalized";
    std::string action_buffer_name = "diffusion_noise";

    /* Persistent vision-staging capacity (see c_api.h): allocated once at
     * construction so prepare_vision never allocates on the hot path. */
    int max_frame_width = 1280;
    int max_frame_height = 720;

    /* Optional host/device overrides. If left null, Runtime derives tensor
     * views from the export's named buffers. The current CPU reference
     * preprocess/postprocess requires host tensors; GPU processors will use
     * the same RuntimeIo contract with device buffers. */
    modalities::TensorView image_input_override;
    modalities::TensorView action_output_override;

    ReplayFn replay_fn = nullptr;
    void* replay_user = nullptr;
};

class Runtime final : public families::vla::Runtime {
public:
    Runtime(const frt_runtime_export_v1* exp, RuntimeConfig config);
    ~Runtime() override;

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    bool ok() const { return status_.ok_status(); }
    const modalities::Status& status() const { return status_; }

    const frt_runtime_export_v1* export_runtime() const override {
        return exp_;
    }
    const families::vla::Manifest& manifest() const override {
        return manifest_;
    }

    int set_prompt(const char* text) override;
    modalities::Status prepare_vision(
        const std::vector<modalities::VisionFrame>& frames) override;
    int replay_tick() override;
    modalities::Status read_actions(std::vector<float>* robot_actions) override;

private:
    void retain_export();
    void release_export();
    modalities::Status bind();

    static int default_replay(frt_graph graph, frt_shape_key key,
                              int stream_id, void* user);

    const frt_runtime_export_v1* exp_ = nullptr;
    RuntimeConfig config_;
    families::vla::Manifest manifest_;
    modalities::Status status_;
    modalities::VisionStaging staging_;
    RuntimeIo io_;
    frt_graph graph_ = nullptr;
    frt_shape_key graph_key_ = 0;
    int stream_id_ = -1;
};

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_RUNTIME_H
