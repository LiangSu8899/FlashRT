// SPDX-License-Identifier: Apache-2.0
//
// MMA rewrite of the WY K*K^T chunk kernel (see gated_delta_wy_kkt_mma.cu).
// Same argument surface and A layout as qwen36_gdn_wy_kkt_b64_bf16, so a
// consumer switches by entry name alone. Additive.
#pragma once
#include <cuda_runtime.h>

void qwen36_gdn_wy_kkt_b64_mma_bf16(
    const void* k16_l2,
    const void* beta,
    const void* g_cumsum,
    void*       A,
    int S,
    cudaStream_t stream);
