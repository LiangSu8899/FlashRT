#include "cosmos3_edge_misc.cuh"

#ifndef FLASHRT_HAVE_COSMOS3_EDGE
#error "cosmos3_edge_misc.cu requires FLASHRT_HAVE_COSMOS3_EDGE"
#endif

#include "common.cuh"

#include <climits>

namespace flash_rt::kernels {
namespace {

__device__ __forceinline__ __nv_bfloat16 norm_value_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    int base,
    int col,
    float rms)
{
  return __float2bfloat16(static_cast<float>(x[base + col]) * rms * static_cast<float>(weight[col]));
}

__device__ __forceinline__ __nv_bfloat16 norm_rope_value_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    int row,
    int base,
    int col,
    int rope_dim,
    float rms)
{
  const __nv_bfloat16 xv_bf = norm_value_bf16(x, weight, base, col, rms);
  if (col >= rope_dim) {
    return xv_bf;
  }

  const int half = rope_dim >> 1;
  const int rot_col = (col < half) ? (col + half) : (col - half);
  float rot = static_cast<float>(norm_value_bf16(x, weight, base, rot_col, rms));
  if (col < half) {
    rot = -rot;
  }
  const float xv = static_cast<float>(xv_bf);
  const float cv = static_cast<float>(cos[row * rope_dim + col]);
  const float sv = static_cast<float>(sin[row * rope_dim + col]);
  const float rot_sin_bf = static_cast<float>(__float2bfloat16(rot * sv));
  return __float2bfloat16(rot_sin_bf + xv * cv);
}

__global__ void qk_norm_rope_bf16_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int rows,
    int q_heads,
    int k_heads,
    int head_dim,
    int rope_dim,
    float eps)
{
  const int q_vecs = rows * q_heads;
  const int vec = blockIdx.x;
  const bool is_q = vec < q_vecs;
  const int heads = is_q ? q_heads : k_heads;
  const int local_vec = is_q ? vec : vec - q_vecs;
  const int row = local_vec / heads;
  const int head = local_vec - row * heads;
  const __nv_bfloat16* x = is_q ? q_in : k_in;
  const __nv_bfloat16* weight = is_q ? q_weight : k_weight;
  __nv_bfloat16* out = is_q ? q_out : k_out;
  const int base = (row * heads + head) * head_dim;

  extern __shared__ float shared[];
  const int dim2 = head_dim >> 1;
  float local_sum = 0.0f;
  if (threadIdx.x < dim2) {
    const int c0 = threadIdx.x << 1;
    const int c1 = c0 + 1;
    const float v0 = static_cast<float>(x[base + c0]);
    const float v1 = static_cast<float>(x[base + c1]);
    local_sum = v0 * v0 + v1 * v1;
  }
  const float rms = rsqrtf(block_reduce_sum(local_sum, shared) / head_dim + eps);

  if (threadIdx.x < dim2) {
    const int c0 = threadIdx.x << 1;
    const int c1 = c0 + 1;
    out[base + c0] = norm_rope_value_bf16(x, weight, cos, sin, row, base, c0, rope_dim, rms);
    out[base + c1] = norm_rope_value_bf16(x, weight, cos, sin, row, base, c1, rope_dim, rms);
  }
}

// Fast path for head_dim == rope_dim == 128: one warp per (row, head) vector.
// Per-head RMSNorm via register warp-reduce (no shared memory, no barriers);
// the rotate-half partner values live 16 lanes away, exchanged with
// __shfl_xor_sync instead of re-reading global memory. All loads/stores are
// __nv_bfloat162-vectorized.
__global__ void qk_norm_rope_bf16_warp128_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int rows,
    int q_heads,
    int k_heads,
    float eps,
    int total_vecs,
    int q_in_row_stride,
    int k_in_row_stride)
{
  const int warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (warp >= total_vecs) {
    return;
  }
  const int lane = threadIdx.x & 31;
  const int q_vecs = rows * q_heads;
  const bool is_q = warp < q_vecs;
  const int heads = is_q ? q_heads : k_heads;
  const int local = is_q ? warp : warp - q_vecs;
  const int row = local / heads;
  const int head = local - row * heads;
  const __nv_bfloat16* x = is_q ? q_in : k_in;
  const __nv_bfloat16* w = is_q ? q_weight : k_weight;
  __nv_bfloat16* out = is_q ? q_out : k_out;
  const int in_stride = is_q ? q_in_row_stride : k_in_row_stride;
  const int in_base = row * in_stride + (head << 7);
  const int base = (row * heads + head) << 7;

  const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x + in_base);
  const int e0 = lane << 1;
  const __nv_bfloat162 a = x2[e0];
  const __nv_bfloat162 b = x2[e0 + 1];
  const float v0 = __bfloat162float(a.x), v1 = __bfloat162float(a.y);
  const float v2 = __bfloat162float(b.x), v3 = __bfloat162float(b.y);
  float ss = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, off);
  }
  const float rms = rsqrtf(ss * (1.0f / 128.0f) + eps);

  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(w);
  const __nv_bfloat162 wa = w2[e0], wb = w2[e0 + 1];
  const float n0 = v0 * rms * __bfloat162float(wa.x);
  const float n1 = v1 * rms * __bfloat162float(wa.y);
  const float n2 = v2 * rms * __bfloat162float(wb.x);
  const float n3 = v3 * rms * __bfloat162float(wb.y);

  const float p0 = __shfl_xor_sync(0xffffffffu, n0, 16);
  const float p1 = __shfl_xor_sync(0xffffffffu, n1, 16);
  const float p2 = __shfl_xor_sync(0xffffffffu, n2, 16);
  const float p3 = __shfl_xor_sync(0xffffffffu, n3, 16);
  const float sgn = (lane < 16) ? -1.0f : 1.0f;

  const __nv_bfloat162* c2 = reinterpret_cast<const __nv_bfloat162*>(cos + (row << 7));
  const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(sin + (row << 7));
  const __nv_bfloat162 ca = c2[e0], cb = c2[e0 + 1];
  const __nv_bfloat162 sa = s2[e0], sb = s2[e0 + 1];
  const float o0 = n0 * __bfloat162float(ca.x) + sgn * p0 * __bfloat162float(sa.x);
  const float o1 = n1 * __bfloat162float(ca.y) + sgn * p1 * __bfloat162float(sa.y);
  const float o2 = n2 * __bfloat162float(cb.x) + sgn * p2 * __bfloat162float(sb.x);
  const float o3 = n3 * __bfloat162float(cb.y) + sgn * p3 * __bfloat162float(sb.y);

  __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out + base);
  out2[e0] = __floats2bfloat162_rn(o0, o1);
  out2[e0 + 1] = __floats2bfloat162_rn(o2, o3);
}

__global__ void fill_flat_velocity_bf16_kernel(
    const __nv_bfloat16* __restrict__ action,
    __nv_bfloat16* __restrict__ velocity,
    int flat_dim,
    int action_numel)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= flat_dim) {
    return;
  }
  const int action_start = flat_dim - action_numel;
  velocity[idx] = (idx < action_start) ? __float2bfloat16(0.0f) : action[idx - action_start];
}

__global__ void add_bias_zero_action_tail_bf16_kernel(
    __nv_bfloat16* __restrict__ action,
    const __nv_bfloat16* __restrict__ bias,
    int total,
    int cols,
    int valid_cols)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int col = idx % cols;
  if (col >= valid_cols) {
    action[idx] = __float2bfloat16(0.0f);
    return;
  }
  action[idx] = __float2bfloat16(static_cast<float>(action[idx]) + static_cast<float>(bias[col]));
}

__global__ void scatter_rows_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    const int64_t* __restrict__ row_indices,
    int total,
    int hidden)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int row = idx / hidden;
  const int col = idx - row * hidden;
  const int64_t dst_row = row_indices[row];
  dst[dst_row * hidden + col] = src[idx];
}

__global__ void gather_rows_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    const int64_t* __restrict__ row_indices,
    int total,
    int hidden)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int row = idx / hidden;
  const int col = idx - row * hidden;
  const int64_t src_row = row_indices[row];
  dst[idx] = src[src_row * hidden + col];
}

__global__ void copy_action_tail_f32_to_bf16_kernel(
    const float* __restrict__ flat_noise,
    __nv_bfloat16* __restrict__ action,
    int flat_dim,
    int action_numel)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= action_numel) {
    return;
  }
  action[idx] = __float2bfloat16(flat_noise[flat_dim - action_numel + idx]);
}

__global__ void add_action_bias_timestep_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ static_bias,
    const __nv_bfloat16* __restrict__ timestep,
    int total,
    int hidden)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int col = idx % hidden;
  const __nv_bfloat16 with_static = __float2bfloat16(
      static_cast<float>(x[idx]) + static_cast<float>(static_bias[col]));
  x[idx] = __float2bfloat16(static_cast<float>(with_static) + static_cast<float>(timestep[col]));
}

__global__ void add_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int numel)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  out[idx] = __float2bfloat16(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
}

__global__ void avgdown3d_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int total,
    int c,
    int t,
    int h,
    int w,
    int out_c,
    int out_t,
    int out_h,
    int out_w,
    int factor_t,
    int factor_s,
    int group_size,
    int pad_t)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  int rem = idx;
  const int ow = rem % out_w;
  rem /= out_w;
  const int oh = rem % out_h;
  rem /= out_h;
  const int ot = rem % out_t;
  rem /= out_t;
  const int oc = rem % out_c;
  const int b = rem / out_c;

  const int factor = factor_t * factor_s * factor_s;
  float acc = 0.0f;
  #pragma unroll
  for (int g = 0; g < 8; ++g) {
    if (g >= group_size) {
      break;
    }
    int cf = oc * group_size + g;
    const int ws = cf % factor_s;
    cf /= factor_s;
    const int hs = cf % factor_s;
    cf /= factor_s;
    const int tt = cf % factor_t;
    const int ic = cf / factor_t;
    if (ic >= c || oc * group_size + g >= c * factor) {
      continue;
    }
    const int padded_t = ot * factor_t + tt;
    if (padded_t < pad_t) {
      continue;
    }
    const int it = padded_t - pad_t;
    if (it >= t) {
      continue;
    }
    const int ih = oh * factor_s + hs;
    const int iw = ow * factor_s + ws;
    const int64_t in_idx =
        (((static_cast<int64_t>(b) * c + ic) * t + it) * h + ih) * w + iw;
    acc += static_cast<float>(x[in_idx]);
  }
  out[idx] = __float2bfloat16(acc / static_cast<float>(group_size));
}

__global__ void unipc_step_f32_bf16_kernel(
    const float* __restrict__ sample,
    const __nv_bfloat16* __restrict__ velocity,
    const float* __restrict__ prev_m1,
    const float* __restrict__ prev_m2,
    const float* __restrict__ prev_last_sample,
    float* __restrict__ next_sample,
    float* __restrict__ current_m,
    float* __restrict__ current_last_sample,
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
    float p_prev_m1)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }

  const float sample_v = sample[idx];
  const float sigma_velocity = static_cast<float>(__float2bfloat16(sigma * static_cast<float>(velocity[idx])));
  const float curr_m = sample_v - sigma_velocity;

  float corrected = c_sample * sample_v + c_curr_m * curr_m;
  if (corrector_order >= 1) {
    corrected += c_last * prev_last_sample[idx] + c_prev_m1 * prev_m1[idx];
  }
  if (corrector_order >= 2) {
    corrected += c_prev_m2 * prev_m2[idx];
  }

  float next = p_sample * corrected + p_curr_m * curr_m;
  if (predictor_order >= 2) {
    next += p_prev_m1 * prev_m1[idx];
  }

  current_m[idx] = curr_m;
  current_last_sample[idx] = corrected;
  next_sample[idx] = next;
}

}  // namespace

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
    cudaStream_t stream)
{
  if (rows <= 0 || q_heads <= 0 || k_heads <= 0 || head_dim <= 0 || rope_dim <= 0) {
    return;
  }
  const int vecs = rows * (q_heads + k_heads);
  if (head_dim == 128 && rope_dim == 128) {
    constexpr int kWarpsPerBlock = 8;
    const int blocks = (vecs + kWarpsPerBlock - 1) / kWarpsPerBlock;
    qk_norm_rope_bf16_warp128_kernel<<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
        q_in, k_in, q_weight, k_weight, cos, sin, q_out, k_out,
        rows, q_heads, k_heads, eps, vecs,
        q_heads * head_dim, k_heads * head_dim);
    return;
  }
  qk_norm_rope_bf16_kernel<<<vecs, 256, 256 * sizeof(float), stream>>>(
      q_in, k_in, q_weight, k_weight, cos, sin, q_out, k_out,
      rows, q_heads, k_heads, head_dim, rope_dim, eps);
}

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
    cudaStream_t stream)
{
  if (rows <= 0 || q_heads <= 0 || k_heads <= 0) {
    return;
  }
  const int vecs = rows * (q_heads + k_heads);
  constexpr int kWarpsPerBlock = 8;
  const int blocks = (vecs + kWarpsPerBlock - 1) / kWarpsPerBlock;
  qk_norm_rope_bf16_warp128_kernel<<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
      q_in, k_in, q_weight, k_weight, cos, sin, q_out, k_out,
      rows, q_heads, k_heads, eps, vecs, q_in_row_stride, k_in_row_stride);
}

void cosmos3_edge_fill_flat_velocity_bf16(
    const __nv_bfloat16* action,
    __nv_bfloat16* velocity,
    int flat_dim,
    int action_numel,
    cudaStream_t stream)
{
  if (action == nullptr || velocity == nullptr || flat_dim <= 0 || action_numel <= 0 || action_numel > flat_dim) {
    return;
  }
  constexpr int kBlock = 256;
  fill_flat_velocity_bf16_kernel<<<(flat_dim + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      action, velocity, flat_dim, action_numel);
}

void cosmos3_edge_add_bias_zero_action_tail_bf16(
    __nv_bfloat16* action,
    const __nv_bfloat16* bias,
    int rows,
    int cols,
    int valid_cols,
    cudaStream_t stream)
{
  if (action == nullptr || bias == nullptr || rows <= 0 || cols <= 0 || valid_cols < 0 || valid_cols > cols) {
    return;
  }
  constexpr int kBlock = 256;
  const int total = rows * cols;
  add_bias_zero_action_tail_bf16_kernel<<<(total + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      action, bias, total, cols, valid_cols);
}

void cosmos3_edge_scatter_rows_bf16(
    const __nv_bfloat16* src,
    __nv_bfloat16* dst,
    const int64_t* row_indices,
    int rows,
    int hidden,
    cudaStream_t stream)
{
  if (src == nullptr || dst == nullptr || row_indices == nullptr || rows <= 0 || hidden <= 0) {
    return;
  }
  constexpr int kBlock = 256;
  const int total = rows * hidden;
  scatter_rows_bf16_kernel<<<(total + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      src, dst, row_indices, total, hidden);
}

void cosmos3_edge_gather_rows_bf16(
    const __nv_bfloat16* src,
    __nv_bfloat16* dst,
    const int64_t* row_indices,
    int rows,
    int hidden,
    cudaStream_t stream)
{
  if (src == nullptr || dst == nullptr || row_indices == nullptr || rows <= 0 || hidden <= 0) {
    return;
  }
  constexpr int kBlock = 256;
  const int total = rows * hidden;
  gather_rows_bf16_kernel<<<(total + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      src, dst, row_indices, total, hidden);
}

void cosmos3_edge_copy_action_tail_f32_to_bf16(
    const float* flat_noise,
    __nv_bfloat16* action,
    int flat_dim,
    int action_numel,
    cudaStream_t stream)
{
  if (flat_noise == nullptr || action == nullptr || flat_dim <= 0 || action_numel <= 0 || action_numel > flat_dim) {
    return;
  }
  constexpr int kBlock = 256;
  copy_action_tail_f32_to_bf16_kernel<<<(action_numel + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      flat_noise, action, flat_dim, action_numel);
}

void cosmos3_edge_add_action_bias_timestep_bf16(
    __nv_bfloat16* x,
    const __nv_bfloat16* static_bias,
    const __nv_bfloat16* timestep,
    int rows,
    int hidden,
    cudaStream_t stream)
{
  if (x == nullptr || static_bias == nullptr || timestep == nullptr || rows <= 0 || hidden <= 0) {
    return;
  }
  constexpr int kBlock = 256;
  const int total = rows * hidden;
  add_action_bias_timestep_bf16_kernel<<<(total + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      x, static_bias, timestep, total, hidden);
}

void cosmos3_edge_add_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* out,
    int numel,
    cudaStream_t stream)
{
  if (a == nullptr || b == nullptr || out == nullptr || numel <= 0) {
    return;
  }
  constexpr int kBlock = 256;
  add_bf16_kernel<<<(numel + kBlock - 1) / kBlock, kBlock, 0, stream>>>(a, b, out, numel);
}

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
    cudaStream_t stream)
{
  if (x == nullptr || out == nullptr || b <= 0 || c <= 0 || t <= 0 || h <= 0 || w <= 0 ||
      out_c <= 0 || factor_t <= 0 || factor_s <= 0 || group_size <= 0) {
    return;
  }
  const int factor = factor_t * factor_s * factor_s;
  if (c * factor != out_c * group_size || h % factor_s != 0 || w % factor_s != 0) {
    return;
  }
  const int pad_t = (factor_t - (t % factor_t)) % factor_t;
  const int out_t = (t + pad_t) / factor_t;
  const int out_h = h / factor_s;
  const int out_w = w / factor_s;
  const int64_t total64 = static_cast<int64_t>(b) * out_c * out_t * out_h * out_w;
  if (total64 <= 0 || total64 > static_cast<int64_t>(INT_MAX)) {
    return;
  }
  constexpr int kBlock = 256;
  const int total = static_cast<int>(total64);
  avgdown3d_bf16_kernel<<<(total + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      x, out, total, c, t, h, w, out_c, out_t, out_h, out_w,
      factor_t, factor_s, group_size, pad_t);
}

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
    cudaStream_t stream)
{
  if (sample == nullptr || velocity == nullptr || next_sample == nullptr ||
      current_m == nullptr || current_last_sample == nullptr || numel <= 0) {
    return;
  }
  if ((corrector_order >= 1 && (prev_m1 == nullptr || prev_last_sample == nullptr)) ||
      (corrector_order >= 2 && prev_m2 == nullptr) ||
      (predictor_order >= 2 && prev_m1 == nullptr)) {
    return;
  }
  constexpr int kBlock = 256;
  unipc_step_f32_bf16_kernel<<<(numel + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      sample, velocity, prev_m1, prev_m2, prev_last_sample, next_sample,
      current_m, current_last_sample, numel, sigma, corrector_order,
      predictor_order, c_sample, c_last, c_prev_m1, c_prev_m2, c_curr_m,
      p_sample, p_curr_m, p_prev_m1);
}

namespace {

__global__ void relu2_to_fp8_static_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ d_scale,
    int n2)
{
  const float inv_scale = 1.0f / (*d_scale);
  const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n2; i += gridDim.x * blockDim.x) {
    const __nv_bfloat162 v = x2[i];
    float a = fmaxf(__bfloat162float(v.x), 0.0f);
    float b = fmaxf(__bfloat162float(v.y), 0.0f);
    a = fminf(a * a * inv_scale, 448.0f);
    b = fminf(b * b * inv_scale, 448.0f);
    out[2 * i] = __nv_fp8_e4m3(a);
    out[2 * i + 1] = __nv_fp8_e4m3(b);
  }
}

}  // namespace

void cosmos3_edge_relu2_to_fp8_static_bf16(
    const __nv_bfloat16* x,
    __nv_fp8_e4m3* out,
    const float* d_scale,
    int numel,
    cudaStream_t stream)
{
  if (x == nullptr || out == nullptr || d_scale == nullptr || numel <= 0 || (numel & 1) != 0) {
    return;
  }
  constexpr int kBlock = 256;
  const int n2 = numel >> 1;
  const int blocks = min((n2 + kBlock - 1) / kBlock, 65535);
  relu2_to_fp8_static_kernel<<<blocks, kBlock, 0, stream>>>(x, out, d_scale, n2);
}

}  // namespace flash_rt::kernels
