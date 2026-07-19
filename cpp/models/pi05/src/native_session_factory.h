#ifndef FLASHRT_CPP_MODELS_PI05_NATIVE_SESSION_FACTORY_H
#define FLASHRT_CPP_MODELS_PI05_NATIVE_SESSION_FACTORY_H

#include "flashrt/cpp/models/pi05/model/native_session.h"
#include "flashrt/cpp/models/pi05/support/native_calibration.h"

#include <memory>
#include <optional>
#include <string>

namespace flashrt {
namespace models {
namespace pi05 {

enum class NativeSessionPrecision {
    kBf16 = 0,
    kFp8E4M3Fn,
};

struct NativeDeviceProfile final {
    int major = 0;
    int minor = 0;
    std::string hardware;
    modalities::DType activation_dtype = modalities::DType::kUInt8;
};

modalities::Status query_native_device(NativeDeviceProfile* out);

std::unique_ptr<Pi05NativeSession> create_native_session(
    const std::string& checkpoint_path,
    const Pi05ResolvedShape& shape,
    NativeSessionPrecision precision,
    std::optional<NativeCalibrationArtifact> calibration,
    Pi05SessionMode mode,
    NativeDeviceProfile* device,
    modalities::Status* status);

}  // namespace pi05
}  // namespace models
}  // namespace flashrt

#endif  // FLASHRT_CPP_MODELS_PI05_NATIVE_SESSION_FACTORY_H
