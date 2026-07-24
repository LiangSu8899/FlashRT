#include "flashrt/providers/llama_cpp/c_api.h"
#include "flashrt/providers/llama_cpp/jetson_pi_engine.h"
#include "checkpoint_identity.h"

#if defined(_WIN32)
#define FRT_PROVIDER_EXPORT __declspec(dllexport)
#else
#define FRT_PROVIDER_EXPORT __attribute__((visibility("default")))
#endif

extern "C" FRT_PROVIDER_EXPORT int frt_model_runtime_open_v1(
        const char* config_json, frt_model_runtime_v1** out) {
    if (!out) return -1;
    *out = nullptr;
    if (!config_json) return -1;

    const frt_llama_cpp_engine_factory_v1* factory =
        frt_llama_cpp_default_engine_factory();
    if (!factory) return -3;

    int rc = frt_llama_cpp_pi0_runtime_open_with_engine_factory(
        config_json, factory, out);
    if (rc == 0) return 0;
    rc = frt_llama_cpp_llm_runtime_open_with_engine_factory(
        config_json, factory, out);
    if (rc == 0) return 0;
    rc = frt_llama_cpp_mllm_runtime_open_with_engine_factory(
        config_json, factory, out);
    if (rc == 0) return 0;

    flashrt::providers::llama_cpp::set_runtime_open_error(
        "configuration does not describe a supported provider model family");
    return rc;
}

#undef FRT_PROVIDER_EXPORT
