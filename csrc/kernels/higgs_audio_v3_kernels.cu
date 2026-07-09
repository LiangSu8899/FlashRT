#include "higgs_audio_v3_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::kernels {
namespace {

__global__ void higgs_argmax_delay_kernel(
    const __nv_bfloat16* logits,
    int64_t* codes,
    int num_codebooks,
    int codebook_vocab,
    int delay,
    int boc) {
  const int cb = blockIdx.x;
  if (cb >= num_codebooks) return;
  if (delay < num_codebooks && cb > delay) {
    if (threadIdx.x == 0) codes[cb] = static_cast<int64_t>(boc);
    return;
  }

  extern __shared__ unsigned char smem[];
  float* vals = reinterpret_cast<float*>(smem);
  int* idxs = reinterpret_cast<int*>(vals + blockDim.x);

  const int tid = threadIdx.x;
  float best = -3.402823466e38f;
  int best_i = 0;
  const __nv_bfloat16* row = logits + cb * codebook_vocab;
  for (int i = tid; i < codebook_vocab; i += blockDim.x) {
    const float v = __bfloat162float(row[i]);
    if (v > best || (v == best && i < best_i)) {
      best = v;
      best_i = i;
    }
  }
  vals[tid] = best;
  idxs[tid] = best_i;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float ov = vals[tid + stride];
      const int oi = idxs[tid + stride];
      if (ov > vals[tid] || (ov == vals[tid] && oi < idxs[tid])) {
        vals[tid] = ov;
        idxs[tid] = oi;
      }
    }
    __syncthreads();
  }
  if (tid == 0) codes[cb] = static_cast<int64_t>(idxs[0]);
}

__global__ void higgs_embed_sum_kernel(
    const int64_t* codes,
    const __nv_bfloat16* codebook,
    __nv_bfloat16* embed,
    int num_codebooks,
    int codebook_vocab,
    int hidden) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= hidden) return;
  float acc = 0.0f;
  for (int cb = 0; cb < num_codebooks; ++cb) {
    const int code = static_cast<int>(codes[cb]);
    const int row = cb * codebook_vocab + code;
    acc += __bfloat162float(codebook[row * hidden + h]);
  }
  embed[h] = __float2bfloat16(acc);
}

}  // namespace

void higgs_audio_v3_argmax_delay_embed_bf16(
    const __nv_bfloat16* logits,
    const __nv_bfloat16* codebook,
    int64_t* codes_out,
    __nv_bfloat16* embed_out,
    int num_codebooks,
    int codebook_vocab,
    int hidden,
    int delay,
    int boc,
    cudaStream_t stream) {
  if (num_codebooks <= 0 || codebook_vocab <= 0 || hidden <= 0) return;
  const int arg_threads = 1024;
  const size_t smem = arg_threads * (sizeof(float) + sizeof(int));
  higgs_argmax_delay_kernel<<<num_codebooks, arg_threads, smem, stream>>>(
      logits, codes_out, num_codebooks, codebook_vocab, delay, boc);
  const int emb_threads = 256;
  const int emb_blocks = (hidden + emb_threads - 1) / emb_threads;
  higgs_embed_sum_kernel<<<emb_blocks, emb_threads, 0, stream>>>(
      codes_out, codebook, embed_out, num_codebooks, codebook_vocab, hidden);
}

}  // namespace flash_rt::kernels
