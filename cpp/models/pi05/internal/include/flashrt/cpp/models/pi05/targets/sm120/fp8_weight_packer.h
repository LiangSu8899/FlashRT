#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_FP8_WEIGHT_PACKER_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_FP8_WEIGHT_PACKER_H

#include "flashrt/cpp/models/pi05/model/linear_weight_groups.h"
#include "flashrt/exec.h"

#include <cstddef>
#include <vector>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm120 {

struct Sm120Fp8PackedWeight final {
    Pi05LinearWeightKey key;
    frt_buffer values = nullptr;
    frt_buffer scale = nullptr;
    Pi05ResolvedWeight view;
};

class Sm120Fp8WeightPacker final : public Pi05LinearWeightGroupSink {
public:
    explicit Sm120Fp8WeightPacker(frt_ctx context,
                                  Pi05Stream stream = 0);

    modalities::Status record(
        const Pi05LinearWeightGroup& group) override;
    modalities::Status finish();

    std::size_t size() const { return packed_.size(); }
    const Sm120Fp8PackedWeight* packed_weight(std::size_t index) const;
    std::size_t merge_scratch_bytes() const { return merge_scratch_bytes_; }

private:
    modalities::Status pack_single(
        const Pi05LinearWeightGroup& group);
    modalities::Status pack_pair(
        const Pi05LinearWeightGroup& group);
    modalities::Status pack_view(
        const Pi05LinearWeightKey& key,
        const Pi05ResolvedWeight& source,
        Pi05ResolvedWeight* destination);
    modalities::Status ensure_merge_scratch(std::size_t bytes);
    modalities::Status fail(modalities::Status status);

    frt_ctx context_ = nullptr;
    Pi05Stream stream_ = 0;
    frt_buffer merge_scratch_ = nullptr;
    std::size_t merge_scratch_bytes_ = 0;
    std::size_t scratch_generation_ = 0;
    std::vector<Sm120Fp8PackedWeight> packed_;
    bool finished_ = false;
    bool failed_ = false;
};

}  // namespace sm120
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_TARGETS_SM120_FP8_WEIGHT_PACKER_H
