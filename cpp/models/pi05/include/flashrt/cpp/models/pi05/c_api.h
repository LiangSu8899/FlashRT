#ifndef FLASHRT_CPP_MODELS_PI05_C_API_H
#define FLASHRT_CPP_MODELS_PI05_C_API_H

#include "flashrt/runtime.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct frt_pi05_runtime_s frt_pi05_runtime;

enum frt_pi05_pixel_format {
    FRT_PI05_PIXEL_RGB8  = 0,
    FRT_PI05_PIXEL_BGR8  = 1,
    FRT_PI05_PIXEL_RGBA8 = 2,
    FRT_PI05_PIXEL_BGRA8 = 3,
    FRT_PI05_PIXEL_GRAY8 = 4,
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

int frt_pi05_runtime_create(const frt_runtime_export_v1* exp,
                            const frt_pi05_runtime_config* config,
                            frt_pi05_runtime** out);
void frt_pi05_runtime_destroy(frt_pi05_runtime*);

int frt_pi05_runtime_set_prompt(frt_pi05_runtime*, const char* text);
int frt_pi05_runtime_prepare_vision(frt_pi05_runtime*,
                                    const frt_pi05_vision_frame* frames,
                                    uint64_t n_frames);
int frt_pi05_runtime_replay_tick(frt_pi05_runtime*);
int frt_pi05_runtime_read_actions(frt_pi05_runtime*,
                                  float* out_actions,
                                  uint64_t out_capacity,
                                  uint64_t* n_written);

const frt_runtime_export_v1* frt_pi05_runtime_export(frt_pi05_runtime*);
const char* frt_pi05_runtime_last_error(frt_pi05_runtime*);

#ifdef __cplusplus
}
#endif

#endif  // FLASHRT_CPP_MODELS_PI05_C_API_H
