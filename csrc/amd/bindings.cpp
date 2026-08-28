// ================================================================
// FlashRT AMD — pybind11 bindings (module: flash_rt_amd_kernels)
//
// Same ABI contract as csrc/bindings.cpp: every entry takes
// uintptr_t device pointers + a uintptr_t stream, never tensors.
// Entries keep the CUDA-side names and signatures so the pipeline
// layer is portable text across platforms.
//
// This is the AMD-only module. The CUDA module (flash_rt_kernels)
// is untouched; the two never build together.
// ================================================================

#include <pybind11/pybind11.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <cstdint>
#include <string>

namespace py = pybind11;

// ── Pointer helpers (mirror csrc/bindings.cpp) ──
static void* to_ptr(uintptr_t addr) { return reinterpret_cast<void*>(addr); }
template<typename T> static T* typed_ptr(uintptr_t addr) { return reinterpret_cast<T*>(addr); }
static hipStream_t to_stream(uintptr_t s) { return reinterpret_cast<hipStream_t>(s); }

// ── Kernel declarations (defined in kernels/*.hip) ──
void rms_norm(const __hip_bfloat16* x, const __hip_bfloat16* weight,
              __hip_bfloat16* out, int seq_len, int dim, float eps,
              hipStream_t stream);
void rms_norm_fp16(const __half* x, const __half* weight,
                   __half* out, int seq_len, int dim, float eps,
                   hipStream_t stream);
void rms_norm_inplace(const __hip_bfloat16* weight,
                      __hip_bfloat16* x, int seq_len, int dim, float eps,
                      hipStream_t stream);

PYBIND11_MODULE(flash_rt_amd_kernels, m) {
    m.doc() = "FlashRT AMD (ROCm/HIP) kernels — raw-pointer ABI";

    m.def("build_info", []() {
        py::dict info;
        info["platform"] = "hip";
#ifdef FLASHRT_AMD_GPU_ARCH
        info["gpu_arch"] = FLASHRT_AMD_GPU_ARCH;
#endif
        int rt_version = 0;
        (void)hipRuntimeGetVersion(&rt_version);
        info["hip_runtime_version"] = rt_version;
        return info;
    });

    m.def("device_arch", []() {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) return std::string("none");
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return std::string("unknown");
        return std::string(prop.gcnArchName);
    });

    // ── Norm ──
    m.def("rms_norm", [](uintptr_t x, uintptr_t weight, uintptr_t out,
                         int seq_len, int dim, float eps, uintptr_t stream) {
        rms_norm(typed_ptr<__hip_bfloat16>(x), typed_ptr<__hip_bfloat16>(weight),
                 typed_ptr<__hip_bfloat16>(out), seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("rms_norm_fp16", [](uintptr_t x, uintptr_t weight, uintptr_t out,
                              int seq_len, int dim, float eps, uintptr_t stream) {
        rms_norm_fp16(typed_ptr<__half>(x), typed_ptr<__half>(weight),
                      typed_ptr<__half>(out), seq_len, dim, eps, to_stream(stream));
    }, py::arg("x"), py::arg("weight"), py::arg("out"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    m.def("rms_norm_inplace", [](uintptr_t weight, uintptr_t x,
                                 int seq_len, int dim, float eps, uintptr_t stream) {
        rms_norm_inplace(typed_ptr<__hip_bfloat16>(weight), typed_ptr<__hip_bfloat16>(x),
                         seq_len, dim, eps, to_stream(stream));
    }, py::arg("weight"), py::arg("x"),
       py::arg("seq_len"), py::arg("dim"), py::arg("eps") = 1e-6f, py::arg("stream") = 0);

    (void)to_ptr;  // silence unused warning until pointer-only entries land
}
