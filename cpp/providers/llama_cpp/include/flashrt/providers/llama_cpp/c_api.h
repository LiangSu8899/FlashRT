#ifndef FLASHRT_PROVIDERS_LLAMA_CPP_C_API_H
#define FLASHRT_PROVIDERS_LLAMA_CPP_C_API_H

#include "flashrt/model_runtime.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum frt_llama_cpp_model_family {
    FRT_LLAMA_CPP_MODEL_PI0  = 1,
    FRT_LLAMA_CPP_MODEL_LLM  = 2,
    FRT_LLAMA_CPP_MODEL_MLLM = 3,
};

enum frt_llama_cpp_pi0_port {
    FRT_LLAMA_CPP_PI0_PORT_IMAGES  = 0,
    FRT_LLAMA_CPP_PI0_PORT_PROMPT  = 1,
    FRT_LLAMA_CPP_PI0_PORT_STATE   = 2,
    FRT_LLAMA_CPP_PI0_PORT_ACTIONS = 3,
};

enum frt_llama_cpp_pi0_stage_index {
    FRT_LLAMA_CPP_PI0_STAGE_INDEX_INFER   = 0,
    FRT_LLAMA_CPP_PI0_STAGE_INDEX_CONTEXT = 1,
    FRT_LLAMA_CPP_PI0_STAGE_INDEX_ACTION  = 2,
};

enum frt_llama_cpp_llm_port {
    FRT_LLAMA_CPP_LLM_PORT_PROMPT     = 0,
    FRT_LLAMA_CPP_LLM_PORT_TEXT       = 1,
    FRT_LLAMA_CPP_LLM_PORT_NEXT_TOKEN = 2,
    FRT_LLAMA_CPP_LLM_PORT_LOGITS     = 3,
    FRT_LLAMA_CPP_LLM_PORT_IS_EOG     = 4,
    FRT_LLAMA_CPP_LLM_PORT_TOKENS     = 5,
};

enum frt_llama_cpp_llm_stage_index {
    FRT_LLAMA_CPP_LLM_STAGE_INDEX_INFER   = 0,
    FRT_LLAMA_CPP_LLM_STAGE_INDEX_RESET   = 1,
    FRT_LLAMA_CPP_LLM_STAGE_INDEX_PREFILL = 2,
    FRT_LLAMA_CPP_LLM_STAGE_INDEX_DECODE  = 3,
};

enum frt_llama_cpp_mllm_port {
    FRT_LLAMA_CPP_MLLM_PORT_IMAGES     = 0,
    FRT_LLAMA_CPP_MLLM_PORT_PROMPT     = 1,
    FRT_LLAMA_CPP_MLLM_PORT_TEXT       = 2,
    FRT_LLAMA_CPP_MLLM_PORT_NEXT_TOKEN = 3,
    FRT_LLAMA_CPP_MLLM_PORT_LOGITS     = 4,
    FRT_LLAMA_CPP_MLLM_PORT_IS_EOG     = 5,
};

enum frt_llama_cpp_mllm_stage_index {
    FRT_LLAMA_CPP_MLLM_STAGE_INDEX_INFER   = 0,
    FRT_LLAMA_CPP_MLLM_STAGE_INDEX_RESET   = 1,
    FRT_LLAMA_CPP_MLLM_STAGE_INDEX_PREFILL = 2,
    FRT_LLAMA_CPP_MLLM_STAGE_INDEX_DECODE  = 3,
};

typedef struct frt_llama_cpp_pi0_config {
    uint32_t struct_size;

    const char* model_path;
    const char* mmproj_path;
    const char* backend;

    uint32_t n_views;
    uint32_t image_height;
    uint32_t image_width;
    uint32_t image_channels;
    uint32_t action_steps;
    uint32_t action_dim;

    const char* model_identity;   /* append-only checkpoint content identity */
    const char* mmproj_identity;  /* append-only checkpoint content identity */
} frt_llama_cpp_pi0_config;

#define FRT_LLAMA_CPP_PI0_CONFIG_BASE_SIZE \
    (offsetof(frt_llama_cpp_pi0_config, model_identity))

typedef struct frt_llama_cpp_llm_config {
    uint32_t struct_size;

    const char* model_path;
    const char* backend;

    uint32_t n_ctx;            /* KV context size; 0 = from model             */
    int32_t  n_threads;        /* CPU threads; 0 = hardware_concurrency       */

    float    temp;             /* sampler temperature (<=0 = greedy)          */
    int32_t  top_k;            /* 0 = disabled                                */
    float    top_p;            /* 0 = disabled                                */
    uint32_t seed;             /* RNG seed                                    */
    uint32_t max_tokens;       /* cap on generated tokens per infer           */

    const char* model_identity; /* append-only checkpoint content identity     */
} frt_llama_cpp_llm_config;

#define FRT_LLAMA_CPP_LLM_CONFIG_BASE_SIZE \
    (offsetof(frt_llama_cpp_llm_config, model_identity))

typedef struct frt_llama_cpp_mllm_config {
    uint32_t struct_size;

    const char* model_path;
    const char* mmproj_path;
    const char* backend;

    uint32_t n_ctx;            /* KV context size; 0 = 4096 fallback          */
    int32_t  n_threads;        /* CPU threads; 0 = hardware_concurrency       */

    float    temp;             /* sampler temperature (<=0 = greedy)          */
    int32_t  top_k;            /* 0 = disabled                                */
    float    top_p;            /* 0 = disabled                                */
    uint32_t seed;             /* RNG seed                                    */
    uint32_t max_tokens;       /* cap on generated tokens per infer           */

    const char* model_identity;  /* append-only checkpoint content identity   */
    const char* mmproj_identity; /* append-only checkpoint content identity   */
} frt_llama_cpp_mllm_config;

#define FRT_LLAMA_CPP_MLLM_CONFIG_BASE_SIZE \
    (offsetof(frt_llama_cpp_mllm_config, model_identity))

typedef struct frt_llama_cpp_engine_v1 {
    uint32_t struct_size;
    /* Capability bits. Older engines initialize this reserved word to zero. */
    uint32_t reserved;

    void* self;
    /* retain/release must be both null for a borrowed engine, or both set for
     * a reference-counted engine. Asymmetric ownership hooks are rejected. */
    void (*retain)(void* self);
    void (*release)(void* self);

    int (*set_input)(void* self, uint32_t port,
                     const void* data, uint64_t bytes, int stream);
    int (*run_infer)(void* self);
    int (*get_output)(void* self, uint32_t port,
                      void* out, uint64_t capacity, uint64_t* written,
                      int stream);
    /* Should return a non-null borrowed string. If it violates that contract,
     * the wrapper reports a stable boundary error string. */
    const char* (*last_error)(void* self);

    /* Optional append-only tail. Engines exposing this callback support the
     * provider's model-family-specific stage indices (e.g. LLM reset/prefill/
     * decode). Older engines end at last_error and remain valid. */
    int (*run_stage)(void* self, uint32_t stage);
} frt_llama_cpp_engine_v1;

#define FRT_LLAMA_CPP_ENGINE_V1_BASE_SIZE \
    (offsetof(frt_llama_cpp_engine_v1, run_stage))
#define FRT_LLAMA_CPP_ENGINE_V1_RUN_STAGE_SIZE \
    (offsetof(frt_llama_cpp_engine_v1, run_stage) + \
     sizeof(((frt_llama_cpp_engine_v1*)0)->run_stage))

/* Stored in frt_llama_cpp_engine_v1::reserved. A callback alone is
 * insufficient to distinguish a real boundary from cached-result shims. */
#define FRT_LLAMA_CPP_ENGINE_CAP_PI0_REAL_CONTEXT_ACTION UINT32_C(1)

typedef struct frt_llama_cpp_engine_factory_v1 {
    uint32_t struct_size;
    uint32_t reserved;

    void* self;
    /* Returns one owned engine reference in out_engine. The config pointer
     * and its string fields are borrowed and valid only for the duration of
     * the callback; factories must copy anything they keep. The runtime-open
     * wrapper consumes the returned reference after retaining the model
     * runtime's own engine reference. Factory-created engines must provide
     * symmetric retain/release hooks; borrowed engines are only accepted by
     * the lower create_with_engine entry point. On nonzero return, no engine
     * ownership is transferred and out_engine is ignored. */
    int (*create_pi0)(void* self, const frt_llama_cpp_pi0_config* config,
                      frt_llama_cpp_engine_v1* out_engine);
    /* Same contract as create_pi0 but for a generic GGUF LLM (text in ->
     * text out). May be NULL if the factory only serves Pi0. */
    int (*create_llm)(void* self, const frt_llama_cpp_llm_config* config,
                      frt_llama_cpp_engine_v1* out_engine);
    /* Same contract as create_pi0 but for a multimodal LLM (images + text in
     * -> text out). May be NULL if the factory does not serve VLMs. */
    int (*create_mllm)(void* self, const frt_llama_cpp_mllm_config* config,
                       frt_llama_cpp_engine_v1* out_engine);
    const char* (*last_error)(void* self);
} frt_llama_cpp_engine_factory_v1;

/* Description of the most recent runtime-open failure on the calling thread.
 * Always returns a non-null borrowed string, valid until the next
 * frt_llama_cpp_*_runtime_open_with_engine_factory call on that thread. */
const char* frt_llama_cpp_runtime_open_error(void);

int frt_llama_cpp_pi0_runtime_create_with_engine(
    const frt_llama_cpp_pi0_config* config,
    const frt_llama_cpp_engine_v1* engine,
    frt_model_runtime_v1** out);

/* Provider-specific JSON open path for the current dependency-injection
 * boundary. Required JSON fields:
 *   model_family="pi0", model_path, mmproj_path, backend,
 *   n_views, image_height, image_width, image_channels,
 *   action_steps, action_dim.
 * No field has a default; missing or mismatched fields fail hard. */
int frt_llama_cpp_pi0_runtime_open_with_engine_factory(
    const char* config_json,
    const frt_llama_cpp_engine_factory_v1* factory,
    frt_model_runtime_v1** out);

int frt_llama_cpp_llm_runtime_create_with_engine(
    const frt_llama_cpp_llm_config* config,
    const frt_llama_cpp_engine_v1* engine,
    frt_model_runtime_v1** out);

/* Provider-specific JSON open path for generic GGUF LLM. Required JSON
 * fields:
 *   model_family="llm", model_path, backend,
 *   n_ctx, n_threads, temp, top_k, top_p, seed, max_tokens.
 * No field has a default; missing or mismatched fields fail hard. The factory
 * must provide create_llm. */
int frt_llama_cpp_llm_runtime_open_with_engine_factory(
    const char* config_json,
    const frt_llama_cpp_engine_factory_v1* factory,
    frt_model_runtime_v1** out);

int frt_llama_cpp_mllm_runtime_create_with_engine(
    const frt_llama_cpp_mllm_config* config,
    const frt_llama_cpp_engine_v1* engine,
    frt_model_runtime_v1** out);

/* Provider-specific JSON open path for multimodal LLM. Required JSON fields:
 *   model_family="mllm", model_path, mmproj_path, backend,
 *   n_ctx, n_threads, temp, top_k, top_p, seed, max_tokens.
 * No field has a default; missing or mismatched fields fail hard. The factory
 * must provide create_mllm. */
int frt_llama_cpp_mllm_runtime_open_with_engine_factory(
    const char* config_json,
    const frt_llama_cpp_engine_factory_v1* factory,
    frt_model_runtime_v1** out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* FLASHRT_PROVIDERS_LLAMA_CPP_C_API_H */
