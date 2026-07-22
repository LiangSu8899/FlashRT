// Cosmos3-Reasoner M=1 W4A16 GEMV (plain layout, e2m1 codes + per-16 bf16 scale).
#pragma once

#ifndef FLASHRT_HAVE_COSMOS3_REASONER
#error "cosmos3_reasoner_gemv.cuh requires FLASHRT_HAVE_COSMOS3_REASONER"
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::kernels {

void cosmos3_reasoner_gemv_w4a16_bf16(
    const uint8_t* w_packed,      // [n_rows, k/2]
    const __nv_bfloat16* w_scales,  // [n_rows, k/16]
    const __nv_bfloat16* a,       // [k]
    __nv_bfloat16* out,           // [n_rows]
    int n_rows,
    int k,
    cudaStream_t stream);

}  // namespace flash_rt::kernels
