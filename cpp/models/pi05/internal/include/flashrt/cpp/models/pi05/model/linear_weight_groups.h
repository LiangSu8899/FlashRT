#ifndef FLASHRT_CPP_MODELS_PI05_MODEL_LINEAR_WEIGHT_GROUPS_H
#define FLASHRT_CPP_MODELS_PI05_MODEL_LINEAR_WEIGHT_GROUPS_H

#include "flashrt/cpp/models/pi05/model/resolved_resources.h"

#include <cstddef>
#include <cstdint>

namespace flashrt {
namespace models {
namespace pi05 {

inline constexpr std::size_t kPi05LinearScalesPerLayer = 4;

enum class Pi05LinearDomain : std::uint8_t {
    kVision = 0,
    kEncoder,
    kDecoder,
};

enum class Pi05LinearRole : std::uint8_t {
    kAttentionQkv = 0,
    kAttentionOutput,
    kMlpUp,
    kMlpGateUpGroup,
    kMlpDown,
    kProjector,
};

struct Pi05LinearWeightKey final {
    Pi05LinearDomain domain = Pi05LinearDomain::kVision;
    Pi05LinearRole role = Pi05LinearRole::kAttentionQkv;
    int layer = -1;
};

struct Pi05LinearWeightGroup final {
    Pi05LinearWeightKey key;
    Pi05ResolvedWeight* first = nullptr;
    // Gate/up groups expose both logical inputs; the target chooses whether
    // their physical layout remains separate or populates fused.
    Pi05ResolvedWeight* second = nullptr;
    Pi05ResolvedWeight* fused = nullptr;
};

struct Pi05LinearScaleLayout final {
    std::size_t vision = 0;
    std::size_t encoder = 0;
    std::size_t decoder = 0;

    std::size_t total() const { return vision + encoder + decoder; }
};

struct Pi05LinearActivationSite final {
    Pi05LinearDomain domain = Pi05LinearDomain::kVision;
    std::size_t index = 0;
};

class Pi05LinearWeightGroupSink {
public:
    virtual ~Pi05LinearWeightGroupSink() = default;
    virtual modalities::Status record(
        const Pi05LinearWeightGroup& group) = 0;
};

modalities::Status visit_pi05_linear_weight_groups(
    Pi05ResolvedWeights* weights,
    Pi05LinearWeightGroupSink* sink);

modalities::Status resolve_pi05_linear_scale_layout(
    const Pi05ResolvedShape& shape,
    Pi05LinearScaleLayout* out);

modalities::Status resolve_pi05_linear_activation_site(
    const Pi05LinearWeightKey& key,
    int step,
    const Pi05ResolvedShape& shape,
    Pi05LinearActivationSite* out);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_MODEL_LINEAR_WEIGHT_GROUPS_H
