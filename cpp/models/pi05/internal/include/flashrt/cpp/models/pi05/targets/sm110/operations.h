#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM110_OPERATIONS_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM110_OPERATIONS_H

#include "flashrt/cpp/models/pi05/model/resolved_resources.h"
#include "flashrt/cpp/models/pi05/support/native_resource_resolver.h"

#include <memory>

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm110 {

class Sm110Fp8WeightPacker;
class Sm110OperationDriver;
class Sm110PhysicalResources;

class Sm110Operations final {
public:
    static std::unique_ptr<Sm110Operations> create(
        const Pi05ResolvedShape& shape,
        const Pi05ResolvedResources& resources,
        const Pi05NativeSupportBuffers& support,
        const Sm110PhysicalResources& physical,
        const Sm110Fp8WeightPacker& packer,
        const Sm110OperationDriver& driver,
        modalities::Status* status);

    modalities::Status time_mlp(int step, Pi05Stream stream) const;
    modalities::Status attention_style(int layer,
                                       int step,
                                       Pi05Stream stream) const;
    modalities::Status mlp_style(int layer,
                                 int step,
                                 Pi05Stream stream) const;
    modalities::Status final_style(int step, Pi05Stream stream) const;

private:
    Sm110Operations(const Pi05ResolvedShape& shape,
                    const Pi05ResolvedResources& resources,
                    const Pi05NativeSupportBuffers& support,
                    const Sm110PhysicalResources& physical,
                    const Sm110Fp8WeightPacker& packer,
                    const Sm110OperationDriver& driver)
        : shape_(shape),
          resources_(resources),
          support_(support),
          physical_(physical),
          packer_(packer),
          driver_(driver) {}

    modalities::Status style(const Pi05ResolvedWeight& weight,
                             const Pi05ResolvedWeight& bias,
                             const Pi05ResolvedBuffer& destination,
                             int layer,
                             int step,
                             Pi05Stream stream) const;

    const Pi05ResolvedShape& shape_;
    const Pi05ResolvedResources& resources_;
    const Pi05NativeSupportBuffers& support_;
    const Sm110PhysicalResources& physical_;
    const Sm110Fp8WeightPacker& packer_;
    const Sm110OperationDriver& driver_;
};

}  // namespace sm110
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_TARGETS_SM110_OPERATIONS_H
