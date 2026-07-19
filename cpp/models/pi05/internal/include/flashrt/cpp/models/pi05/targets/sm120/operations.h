#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_OPERATIONS_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_OPERATIONS_H

#include "flashrt/cpp/models/pi05/support/native_resource_resolver.h"
#include "flashrt/cpp/models/pi05/targets/sm120/attention_driver.h"
#include "flashrt/cpp/models/pi05/targets/sm120/bf16_linear.h"
#include "flashrt/cpp/models/pi05/targets/sm120/bf16_scratch.h"
#include "flashrt/cpp/models/pi05/targets/sm120/fp8_linear.h"

#include <memory>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {

class Sm120Operations final {
public:
    static modalities::Status autotune_fp8(
        const Pi05ResolvedShape& shape,
        Pi05ResolvedResources* resources,
        const Sm120Bf16ScratchBacking& scratch,
        Sm120Fp8Linear* linear);

    static std::unique_ptr<Sm120Operations> create(
        const Pi05ResolvedShape& shape,
        const Pi05ResolvedResources& resources,
        const Pi05NativeSupportBuffers& support,
        const Sm120Bf16ScratchBacking& scratch,
        const Sm120AttentionBacking& attention,
        const Sm120AttentionDriver& attention_driver,
        const Sm120Bf16Linear& bf16_linear,
        Sm120Fp8Linear* fp8_linear,
        modalities::Status* status);

    modalities::Status compose_prompt(Pi05Stream stream) const;
    modalities::Status encoder_attention(int layer, Pi05Stream stream) const;
    modalities::Status encoder_mlp(int layer, Pi05Stream stream) const;
    modalities::Status encoder_cache_finalize(int layer,
                                               Pi05Stream stream) const;
    modalities::Status diffusion_input_project(int step,
                                                Pi05Stream stream) const;
    modalities::Status decoder_attention(int layer,
                                         int step,
                                         Pi05Stream stream) const;
    modalities::Status decoder_mlp(int layer,
                                   int step,
                                   Pi05Stream stream) const;
    modalities::Status action_project(int step, Pi05Stream stream) const;
    modalities::Status diffusion_update(int step, Pi05Stream stream) const;

private:
    Sm120Operations(const Pi05ResolvedShape& shape,
                    const Pi05ResolvedResources& resources,
                    const Pi05NativeSupportBuffers& support,
                    const Sm120Bf16ScratchBacking& scratch,
                    const Sm120AttentionBacking& attention,
                    const Sm120AttentionDriver& attention_driver,
                    const Sm120Bf16Linear& bf16_linear,
                    Sm120Fp8Linear* fp8_linear)
        : shape_(shape),
          resources_(resources),
          support_(support),
          scratch_(scratch),
          attention_(attention),
          attention_driver_(attention_driver),
          bf16_linear_(bf16_linear),
          fp8_linear_(fp8_linear) {}

    modalities::Status linear(
        const Pi05ResolvedWeight& weight,
        Pi05LinearWeightKey key,
        int step,
        const void* input,
        void* output,
        int rows,
        int input_width,
        int output_width,
        Pi05Stream stream,
        bool prequantized = false) const;
    float* scale(Pi05LinearWeightKey key, int step) const;
    bool static_fp8() const {
        return fp8_linear_ && !fp8_linear_->observing();
    }
    bool observed_fp8() const {
        return fp8_linear_ && fp8_linear_->observing();
    }

    const Pi05ResolvedShape& shape_;
    const Pi05ResolvedResources& resources_;
    const Pi05NativeSupportBuffers& support_;
    const Sm120Bf16ScratchBacking& scratch_;
    const Sm120AttentionBacking& attention_;
    const Sm120AttentionDriver& attention_driver_;
    const Sm120Bf16Linear& bf16_linear_;
    Sm120Fp8Linear* fp8_linear_ = nullptr;
};

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_OPERATIONS_H
