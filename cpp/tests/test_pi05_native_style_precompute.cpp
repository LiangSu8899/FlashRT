#include "flashrt/cpp/models/pi05/native_style_precompute.h"

#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdio>

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        cudaGetLastError();
        std::printf("SKIP - no CUDA device\n");
        return 0;
    }
    frt_ctx ctx = frt_ctx_create();
    assert(ctx);
    flashrt::models::pi05::NativeWorkspace workspace(ctx);
    flashrt::models::pi05::NativeWorkspaceConfig config;
    assert(workspace.allocate(config).ok_status());
    flashrt::models::pi05::NativeDeviceWeightStore weights(ctx);
    flashrt::models::pi05::NativeKernelDriver driver;
    flashrt::models::pi05::NativeStylePrecomputer precomputer(&driver);
    assert(!precomputer.run(weights, &workspace, 0).ok_status());
    frt_ctx_destroy(ctx);
    std::printf("PASS - Pi0.5 native style precompute validation\n");
    return 0;
}
