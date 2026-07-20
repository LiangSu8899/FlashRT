/* Model-runtime v1 ABI baseline.
 *
 * This is a self-contained snapshot of the data ABI published before any
 * additive frt_model_runtime_v1 tail. Keep the original ABI type and enum
 * names: the baseline producer is compiled in an isolated translation unit,
 * while layout checks include this file inside a C++ namespace.
 *
 * Do not replace these declarations with includes from model_runtime.h.
 */
#ifndef FLASHRT_MODEL_RUNTIME_V1_ABI_BASELINE_H
#define FLASHRT_MODEL_RUNTIME_V1_ABI_BASELINE_H

#include <stddef.h>
#include <stdint.h>

#include "flashrt/runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FRT_MODEL_RUNTIME_ABI_VERSION 1u

enum frt_rt_modality {
    FRT_RT_MOD_TENSOR = 0,
    FRT_RT_MOD_IMAGE  = 1,
    FRT_RT_MOD_TEXT   = 2,
    FRT_RT_MOD_STATE  = 3,
    FRT_RT_MOD_ACTION = 4,
    FRT_RT_MOD_AUDIO  = 5,
    FRT_RT_MOD_DEPTH  = 6,
    FRT_RT_MOD_FORCE  = 7
};

enum frt_rt_dtype {
    FRT_RT_DTYPE_U8   = 0,
    FRT_RT_DTYPE_F32  = 1,
    FRT_RT_DTYPE_F16  = 2,
    FRT_RT_DTYPE_BF16 = 3,
    FRT_RT_DTYPE_I32  = 4,
    FRT_RT_DTYPE_I64  = 5
};

enum frt_rt_layout {
    FRT_RT_LAYOUT_FLAT = 0,
    FRT_RT_LAYOUT_HWC  = 1,
    FRT_RT_LAYOUT_NHWC = 2,
    FRT_RT_LAYOUT_CHW  = 3,
    FRT_RT_LAYOUT_NCHW = 4
};

enum frt_rt_pixel_format {
    FRT_RT_PIXEL_RGB8  = 0,
    FRT_RT_PIXEL_BGR8  = 1,
    FRT_RT_PIXEL_RGBA8 = 2,
    FRT_RT_PIXEL_BGRA8 = 3,
    FRT_RT_PIXEL_GRAY8 = 4
};

enum frt_rt_port_direction { FRT_RT_PORT_IN = 0, FRT_RT_PORT_OUT = 1 };

enum frt_rt_port_update {
    FRT_RT_PORT_SWAP   = 0,
    FRT_RT_PORT_STAGED = 1,
    FRT_RT_PORT_SETUP  = 2
};

typedef struct frt_image_view {
    uint32_t struct_size;
    uint32_t pixel_format;
    const void* data;
    uint64_t bytes;
    int32_t width, height, stride_bytes;
    uint32_t reserved;
    uint64_t timestamp_ns;
} frt_image_view;

typedef struct frt_runtime_port_desc {
    const char* name;
    uint32_t modality;
    uint32_t dtype;
    uint32_t layout;
    uint32_t direction;
    uint32_t update;
    uint32_t required;
    const int64_t* shape;
    uint32_t rank;
    uint32_t cadence_hint_hz;
    frt_buffer buffer;
    uint64_t offset;
    uint64_t bytes;
} frt_runtime_port_desc;

typedef struct frt_runtime_stage_desc {
    uint32_t graph;
    uint32_t n_after;
    const uint32_t* after;
} frt_runtime_stage_desc;

typedef struct frt_model_runtime_verbs {
    uint32_t struct_size;
    uint32_t reserved;
    int (*set_input)(void*, uint32_t, const void*, uint64_t, int);
    int (*get_output)(void*, uint32_t, void*, uint64_t, uint64_t*, int);
    int (*prepare)(void*, uint32_t, frt_shape_key);
    int (*step)(void*);
    const char* (*last_error)(void*);
} frt_model_runtime_verbs;

typedef struct frt_model_runtime_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const frt_runtime_export_v1* exp;
    const frt_runtime_port_desc* ports;
    uint64_t n_ports;
    const frt_runtime_stage_desc* stages;
    uint64_t n_stages;
    void* self;
    frt_model_runtime_verbs verbs;
    void* owner;
    void (*retain)(void* owner);
    void (*release)(void* owner);
} frt_model_runtime_v1;

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* FLASHRT_MODEL_RUNTIME_V1_ABI_BASELINE_H */
