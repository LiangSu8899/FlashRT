#ifndef FLASHRT_PROVIDERS_LLAMA_CPP_JETSON_PI_ENGINE_H
#define FLASHRT_PROVIDERS_LLAMA_CPP_JETSON_PI_ENGINE_H

//
// Default Jetson-PI Pi0 engine factory.
//
// Returns a borrowed pointer to a process-global factory vtable backed by the
// Jetson-PI jetson_pi_pi0 policy library. Pass it to
// frt_llama_cpp_pi0_runtime_open_with_engine_factory to open a Pi0 runtime
// that drives a real Jetson-PI whole-graph infer.
//
// Returns NULL when FlashRT was built without FLASHRT_CPP_WITH_JETSON_PI.
// The pointer is valid for the process lifetime; do not release it.
//
// factory->create_pi0 is thread-safe (errors are reported through a
// thread-local sink queried via factory->last_error). Each successful
// create_pi0 yields one owned engine reference; the runtime-open wrapper
// transfers it to the v1 runtime, which releases it on drop.
//
// Engine error codes (returned by set_input/run_infer/get_output, surfaced
// through v1 verbs): 0 ok; -1 null self / precondition; -2 invalid config or
// model; -5 action output buffer too small; -7 actions not ready (run_infer
// did not complete since the last set_input); -8 generic infer failure.
// last_error returns a non-null string describing the most recent failure.
//

#include "flashrt/providers/llama_cpp/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

const frt_llama_cpp_engine_factory_v1*
frt_llama_cpp_default_engine_factory(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* FLASHRT_PROVIDERS_LLAMA_CPP_JETSON_PI_ENGINE_H */
