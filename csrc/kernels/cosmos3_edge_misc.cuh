// Cosmos3-Edge hot-path helpers.

#pragma once

#ifndef FLASHRT_HAVE_COSMOS3_EDGE
#error "cosmos3_edge_misc.cuh requires FLASHRT_HAVE_COSMOS3_EDGE"
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::kernels {

void cosmos3_edge_qk_norm_rope_bf16(
    const __nv_bfloat16* q_in,
    const __nv_bfloat16* k_in,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    int rows,
    int q_heads,
    int k_heads,
    int head_dim,
    int rope_dim,
    float eps,
    cudaStream_t stream);

void cosmos3_edge_fill_flat_velocity_bf16(
    const __nv_bfloat16* action,
    __nv_bfloat16* velocity,
    int flat_dim,
    int action_numel,
    cudaStream_t stream);

void cosmos3_edge_add_bias_zero_action_tail_bf16(
    __nv_bfloat16* action,
    const __nv_bfloat16* bias,
    int rows,
    int cols,
    int valid_cols,
    cudaStream_t stream);

void cosmos3_edge_scatter_rows_bf16(
    const __nv_bfloat16* src,
    __nv_bfloat16* dst,
    const int64_t* row_indices,
    int rows,
    int hidden,
    cudaStream_t stream);

void cosmos3_edge_gather_rows_bf16(
    const __nv_bfloat16* src,
    __nv_bfloat16* dst,
    const int64_t* row_indices,
    int rows,
    int hidden,
    cudaStream_t stream);

void cosmos3_edge_copy_action_tail_f32_to_bf16(
    const float* flat_noise,
    __nv_bfloat16* action,
    int flat_dim,
    int action_numel,
    cudaStream_t stream);

void cosmos3_edge_add_action_bias_timestep_bf16(
    __nv_bfloat16* x,
    const __nv_bfloat16* static_bias,
    const __nv_bfloat16* timestep,
    int rows,
    int hidden,
    cudaStream_t stream);

void cosmos3_edge_add_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    int numel,
    cudaStream_t stream);

void cosmos3_edge_avgdown3d_bf16(
    const __nv_bfloat16* x,
    __nv_bfloat16* out,
    int b,
    int c,
    int t,
    int h,
    int w,
    int out_c,
    int factor_t,
    int factor_s,
    int group_size,
    cudaStream_t stream);

void cosmos3_edge_unipc_step_f32_bf16(
    const float* sample,
    const __nv_bfloat16* velocity,
    const float* prev_m1,
    const float* prev_m2,
    const float* prev_last_sample,
    float* next_sample,
    float* current_m,
    float* current_last_sample,
    int numel,
    float sigma,
    int corrector_order,
    int predictor_order,
    float c_sample,
    float c_last,
    float c_prev_m1,
    float c_prev_m2,
    float c_curr_m,
    float p_sample,
    float p_curr_m,
    float p_prev_m1,
    cudaStream_t stream);

void cosmos3_edge_qk_norm_rope_strided_bf16(
    const __nv_bfloat16* q_in,
    const __nv_bfloat16* k_in,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_out,
    __nv_bfloat16* k_out,
    int rows,
    int q_heads,
    int k_heads,
    int q_in_row_stride,
    int k_in_row_stride,
    float eps,
    cudaStream_t stream);

void cosmos3_edge_relu2_to_fp8_static_bf16(
    const __nv_bfloat16* x,
    __nv_fp8_e4m3* out,
    const float* d_scale,
    int numel,
    cudaStream_t stream);

}  // namespace flash_rt::kernels
