#ifndef FLASHRT_CPP_MODELS_PI05_C_API_H
#define FLASHRT_CPP_MODELS_PI05_C_API_H

#include "flashrt/cpp/models/pi05/export.h"
#include "flashrt/runtime.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct frt_pi05_runtime_s frt_pi05_runtime;
typedef struct frt_pi05_calibration_session_s
    frt_pi05_calibration_session;

enum frt_pi05_pixel_format {
    FRT_PI05_PIXEL_RGB8  = 0,
    FRT_PI05_PIXEL_BGR8  = 1,
    FRT_PI05_PIXEL_RGBA8 = 2,
    FRT_PI05_PIXEL_BGRA8 = 3,
    FRT_PI05_PIXEL_GRAY8 = 4,
};

enum frt_pi05_dtype {
    FRT_PI05_DTYPE_DEFAULT  = 0,
    FRT_PI05_DTYPE_BFLOAT16 = 1,
    FRT_PI05_DTYPE_FLOAT16  = 2,
    FRT_PI05_DTYPE_FLOAT32  = 3,
};

typedef struct frt_pi05_runtime_config {
    uint32_t struct_size;

    int num_views;
    int chunk;
    int model_action_dim;
    int robot_action_dim;

    const float* action_mean;
    uint64_t n_action_mean;
    const float* action_stddev;
    uint64_t n_action_stddev;

    const char* graph_name;
    const char* image_buffer_name;
    const char* action_buffer_name;

    /* Optional ABI extension. Zero keeps the legacy v1 BF16 default. Complete
     * producers set these from their declared tensor windows: BF16 on SM120,
     * F16 on the SM110 FP8 producer. */
    int image_dtype;
    int action_dtype;

    /* Optional ABI extension: capacity of the persistent vision staging pool
     * (allocated once at create; the per-frame hot path never allocates).
     * Zero keeps the defaults (1280x720). A camera frame larger than the
     * capacity is a per-call error, never a fallback allocation. */
    int max_frame_width;
    int max_frame_height;

    /* Optional native prompt/state staging. All fields are ABI-tail
     * extensions; older adopted-export callers may stop before this block. */
    const char* prompt_tokenizer_model_path;
    const void* prompt_embedding_table_data;
    uint64_t prompt_embedding_table_bytes;
    int prompt_embedding_table_dtype;
    uint64_t prompt_embedding_vocab_size;
    uint64_t prompt_embedding_hidden_dim;
    void* prompt_embedding_data;
    uint64_t prompt_embedding_bytes;
    int prompt_embedding_dtype;
    uint64_t max_prompt_tokens;
    float prompt_embedding_scale;

    const float* state_q01;
    uint64_t n_state_q01;
    const float* state_q99;
    uint64_t n_state_q99;

    int (*prompt_length_update)(void* user, uint64_t prompt_len);
    void* prompt_length_update_user;
    int prompt_embedding_on_device;
} frt_pi05_runtime_config;

typedef struct frt_pi05_vision_frame {
    uint32_t struct_size;
    const char* name;
    const void* data;
    uint64_t bytes;
    int width;
    int height;
    int stride_bytes;
    int pixel_format;
    uint64_t timestamp_ns;
} frt_pi05_vision_frame;

/* One observation. Dataset calibration calls observe repeatedly; dataset
 * iteration and decoding remain host policy. Noise is optional f32
 * [chunk, 32]; omitted noise is generated deterministically from noise_seed
 * and the committed sample count. */
typedef struct frt_pi05_calibration_sample_v1 {
    uint32_t struct_size;
    const char* prompt;
    const float* state;
    uint64_t n_state;
    const frt_pi05_vision_frame* frames;
    uint64_t n_frames;
    const float* noise;
    uint64_t n_noise;
    uint64_t noise_seed;
} frt_pi05_calibration_sample_v1;

FLASHRT_PI05_C_API int frt_pi05_runtime_create(
    const frt_runtime_export_v1* exp,
    const frt_pi05_runtime_config* config,
    frt_pi05_runtime** out);
FLASHRT_PI05_C_API void frt_pi05_runtime_destroy(frt_pi05_runtime*);

FLASHRT_PI05_C_API int frt_pi05_runtime_set_prompt(
    frt_pi05_runtime*, const char* text);
FLASHRT_PI05_C_API int frt_pi05_runtime_set_prompt_state(
    frt_pi05_runtime*, const char* text,
    const float* state, uint64_t n_state);
FLASHRT_PI05_C_API int frt_pi05_runtime_prepare_vision(
    frt_pi05_runtime*, const frt_pi05_vision_frame* frames,
    uint64_t n_frames);
FLASHRT_PI05_C_API int frt_pi05_runtime_replay_tick(frt_pi05_runtime*);
FLASHRT_PI05_C_API int frt_pi05_runtime_read_actions(
    frt_pi05_runtime*, float* out_actions,
    uint64_t out_capacity, uint64_t* n_written);

FLASHRT_PI05_C_API const frt_runtime_export_v1* frt_pi05_runtime_export(
    frt_pi05_runtime*);
FLASHRT_PI05_C_API const char* frt_pi05_runtime_last_error(
    frt_pi05_runtime*);

/* Native FP8 calibration uses the same checkpoint, tokenizer, shape, and
 * state-normalization fields as frt_model_runtime_open_v1. The model forward
 * is the regular uncaptured semantic pipeline with target observers enabled. */
FLASHRT_PI05_C_API int frt_pi05_calibration_create_v1(
    const char* config_json,
    double percentile,
    frt_pi05_calibration_session** out);
FLASHRT_PI05_C_API int frt_pi05_calibration_observe_v1(
    frt_pi05_calibration_session*,
    const frt_pi05_calibration_sample_v1* sample);
FLASHRT_PI05_C_API int frt_pi05_calibration_finalize_v1(
    frt_pi05_calibration_session*,
    const char* artifact_path);
FLASHRT_PI05_C_API uint64_t frt_pi05_calibration_sample_count_v1(
    const frt_pi05_calibration_session*);
FLASHRT_PI05_C_API const char* frt_pi05_calibration_last_error_v1(
    const frt_pi05_calibration_session*);
FLASHRT_PI05_C_API const char* frt_pi05_calibration_create_last_error_v1(void);
FLASHRT_PI05_C_API void frt_pi05_calibration_destroy_v1(
    frt_pi05_calibration_session*);

#ifdef __cplusplus
}
#endif

#endif  // FLASHRT_CPP_MODELS_PI05_C_API_H
