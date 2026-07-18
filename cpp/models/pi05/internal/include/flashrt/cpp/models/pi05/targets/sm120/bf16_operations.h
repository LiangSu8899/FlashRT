#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_BF16_OPERATIONS_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_BF16_OPERATIONS_H

#include "flashrt/cpp/models/pi05/support/native_resource_resolver.h"
#include "flashrt/cpp/models/pi05/targets/sm120/attention_driver.h"
#include "flashrt/cpp/models/pi05/targets/sm120/bf16_linear.h"
#include "flashrt/cpp/models/pi05/targets/sm120/bf16_scratch.h"

#include <memory>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {

class Sm120Bf16Operations final {
public:
    static std::unique_ptr<Sm120Bf16Operations> create(
        const Pi05ResolvedShape& shape,
        const Pi05ResolvedResources& resources,
        const Pi05NativeSupportBuffers& support,
        const Sm120Bf16ScratchBacking& scratch,
        const Sm120AttentionBacking& attention,
        const Sm120AttentionDriver& attention_driver,
        const Sm120Bf16Linear& linear,
        modalities::Status* status);

    modalities::Status compose_prompt(Pi05Stream stream) const;
    modalities::Status vision_embed(Pi05Stream stream) const;
    modalities::Status vision_attention(int layer, Pi05Stream stream) const;
    modalities::Status vision_mlp(int layer, Pi05Stream stream) const;
    modalities::Status vision_project(Pi05Stream stream) const;
    modalities::Status encoder_attention(int layer, Pi05Stream stream) const;
    modalities::Status encoder_mlp(int layer, Pi05Stream stream) const;
    modalities::Status encoder_cache_finalize(int layer,
                                               Pi05Stream stream) const;

private:
    Sm120Bf16Operations(const Pi05ResolvedShape& shape,
                        const Pi05ResolvedResources& resources,
                        const Pi05NativeSupportBuffers& support,
                        const Sm120Bf16ScratchBacking& scratch,
                        const Sm120AttentionBacking& attention,
                        const Sm120AttentionDriver& attention_driver,
                        const Sm120Bf16Linear& linear)
        : shape_(shape),
          resources_(resources),
          support_(support),
          scratch_(scratch),
          attention_(attention),
          attention_driver_(attention_driver),
          linear_(linear) {}

    const Pi05ResolvedShape& shape_;
    const Pi05ResolvedResources& resources_;
    const Pi05NativeSupportBuffers& support_;
    const Sm120Bf16ScratchBacking& scratch_;
    const Sm120AttentionBacking& attention_;
    const Sm120AttentionDriver& attention_driver_;
    const Sm120Bf16Linear& linear_;
};

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_BF16_OPERATIONS_H
