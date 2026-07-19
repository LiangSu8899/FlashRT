#include "native_session_factory.h"

#if defined(FLASHRT_CPP_WITH_PI05_SM120_TARGET)
#include "flashrt/cpp/models/pi05/targets/sm120/target.h"
#endif
#if defined(FLASHRT_CPP_WITH_PI05_SM110_TARGET)
#include "flashrt/cpp/models/pi05/targets/sm110/target.h"
#endif

#if defined(FLASHRT_CPP_WITH_PI05_SM120_TARGET) || \
    defined(FLASHRT_CPP_WITH_PI05_SM110_TARGET)
#include <cuda_runtime_api.h>
#endif

#include <utility>

namespace flashrt {
namespace models {
namespace pi05 {
namespace {

modalities::Status invalid(const char* message) {
    return modalities::Status::error(modalities::StatusCode::kInvalidArgument,
                                     message);
}

modalities::Status unsupported(const std::string& message) {
    return modalities::Status::error(modalities::StatusCode::kUnsupported,
                                     message);
}

void set_status(modalities::Status* destination,
                modalities::Status status) {
    if (destination) *destination = std::move(status);
}

}  // namespace

modalities::Status query_native_device(NativeDeviceProfile* out) {
    if (!out) return invalid("native device profile output is null");
#if defined(FLASHRT_CPP_WITH_PI05_SM120_TARGET) || \
    defined(FLASHRT_CPP_WITH_PI05_SM110_TARGET)
    int device = 0;
    cudaDeviceProp properties{};
    cudaError_t result = cudaGetDevice(&device);
    if (result == cudaSuccess) {
        result = cudaGetDeviceProperties(&properties, device);
    }
    if (result != cudaSuccess) {
        return modalities::Status::error(modalities::StatusCode::kBackend,
                                         cudaGetErrorString(result));
    }
    NativeDeviceProfile resolved;
    resolved.major = properties.major;
    resolved.minor = properties.minor;
    resolved.hardware =
        "sm" + std::to_string(properties.major * 10 + properties.minor);
    resolved.activation_dtype = properties.major == 11
                                    ? modalities::DType::kFloat16
                                    : modalities::DType::kBFloat16;
    *out = std::move(resolved);
    return modalities::Status::ok();
#else
    return unsupported("native PI0.5 target support is not built");
#endif
}

std::unique_ptr<Pi05NativeSession> create_native_session(
    const std::string& checkpoint_path,
    const Pi05ResolvedShape& shape,
    NativeSessionPrecision precision,
    std::optional<NativeCalibrationArtifact> calibration,
    Pi05SessionMode mode,
    NativeDeviceProfile* device,
    modalities::Status* status) {
    if (checkpoint_path.empty() ||
        (precision == NativeSessionPrecision::kFp8E4M3Fn &&
         ((mode == Pi05SessionMode::kCaptured) != calibration.has_value()))) {
        set_status(status,
                   invalid("native session calibration mode is invalid"));
        return nullptr;
    }
    NativeDeviceProfile resolved_device;
    modalities::Status result = query_native_device(&resolved_device);
    if (!result.ok_status()) {
        set_status(status, std::move(result));
        return nullptr;
    }
    if (precision == NativeSessionPrecision::kBf16) {
        if (mode != Pi05SessionMode::kCaptured || calibration ||
            resolved_device.major != 12 || resolved_device.minor != 0) {
            set_status(status, unsupported("native PI0.5 BF16 requires SM120"));
            return nullptr;
        }
    } else if (!((resolved_device.major == 11 ||
                  resolved_device.major == 12) &&
                 resolved_device.minor == 0)) {
        set_status(status,
                   unsupported("native PI0.5 FP8 requires SM110 or SM120"));
        return nullptr;
    }

    frt_ctx context = frt_ctx_create();
    if (!context) {
        set_status(status, modalities::Status::error(
                               modalities::StatusCode::kBackend,
                               "native execution context creation failed"));
        return nullptr;
    }
    std::unique_ptr<Pi05TargetBundle> target;
    if (resolved_device.major == 12) {
#if defined(FLASHRT_CPP_WITH_PI05_SM120_TARGET)
        targets::sm120::Sm120TargetConfig target_config;
        target_config.checkpoint_path = checkpoint_path;
        if (precision == NativeSessionPrecision::kFp8E4M3Fn) {
            target_config.execution_mode = calibration
                ? targets::sm120::Sm120ExecutionMode::kStaticFp8E4M3
                : targets::sm120::Sm120ExecutionMode::kObservedFp8E4M3;
            target_config.calibration = std::move(calibration);
        }
        target = targets::sm120::Sm120TargetBundle::create(
            context, shape, std::move(target_config), &result);
#else
        result = unsupported("native PI0.5 SM120 target is not built");
#endif
    } else if (resolved_device.major == 11) {
#if defined(FLASHRT_CPP_WITH_PI05_SM110_TARGET)
        targets::sm110::Sm110TargetConfig target_config;
        target_config.checkpoint_path = checkpoint_path;
        target_config.calibration = std::move(calibration);
        target = targets::sm110::Sm110TargetBundle::create(
            context, shape, std::move(target_config), &result);
#else
        result = unsupported("native PI0.5 SM110 target is not built");
#endif
    }
    if (!target) {
        frt_ctx_destroy(context);
        set_status(status, std::move(result));
        return nullptr;
    }
    std::unique_ptr<Pi05NativeSession> session = Pi05NativeSession::create(
        context, shape, std::move(target), &result, mode);
    if (!session) {
        set_status(status, std::move(result));
        return nullptr;
    }
    if (device) *device = std::move(resolved_device);
    set_status(status, modalities::Status::ok());
    return session;
}

}  // namespace pi05
}  // namespace models
}  // namespace flashrt
