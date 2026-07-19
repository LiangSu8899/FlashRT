#ifndef FLASHRT_CPP_MODELS_PI05_TARGETS_SM110_FRONTEND_BINDINGS_H
#define FLASHRT_CPP_MODELS_PI05_TARGETS_SM110_FRONTEND_BINDINGS_H

#include "flashrt/cpp/models/pi05/model/frontend_ops.h"
#include "flashrt/cpp/models/pi05/support/native_calibration.h"

namespace flashrt {
namespace models {
namespace pi05 {
namespace targets {
namespace sm110 {

class Sm110Fp8WeightPacker;
class Sm110OperationDriver;
class Sm110PhysicalResources;

struct Sm110FrontendBindings final {
    const Pi05ResolvedShape* shape = nullptr;
    const NativeCalibrationArtifact* calibration = nullptr;
    const Sm110PhysicalResources* physical = nullptr;
    const Sm110Fp8WeightPacker* weights = nullptr;
    Sm110OperationDriver* driver = nullptr;
};

modalities::Status initialize_sm110_frontend_ops(
    Sm110FrontendBindings* bindings,
    Pi05FrontendOps* out);

}  // namespace sm110
}  // namespace targets
}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_TARGETS_SM110_FRONTEND_BINDINGS_H
