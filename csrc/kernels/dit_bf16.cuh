#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

void add_bias_bf16(__nv_bfloat16* values,
                   const __nv_bfloat16* bias,
                   int rows,
                   int columns,
                   cudaStream_t stream = nullptr);
