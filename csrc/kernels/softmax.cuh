// ================================================================
// FlashRT — Softmax kernel (FP16)
// Port of pi05 softmax_bf16_kernel. In-place row-wise softmax.
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

void softmax_fp16(__half* data, int rows, int cols, cudaStream_t stream = 0);

// Causal softmax: per-head row-wise softmax that masks upper-triangular
// positions. Logits buffer is laid out as (NH * S_q, S_kv_pad) — for
// global row index ``r``, the per-head Q index is ``q = r % S_q`` and we
// mask cols ``j > q`` (strict upper-triangular). Also masks pad columns
// at ``j >= pad_start`` (= S_kv) regardless of row.
// Used by the causal LLM attention path in the GROOT N1.7 pipeline.
void softmax_causal_fp16(__half* data, int rows, int cols,
                          int S_q, int pad_start,
                          cudaStream_t stream = 0);

// Softmax with state token masking: first `mask_rows` rows have cols [mask_start, cols) set to -inf.
// Eliminates separate mask kernel launch. Used by Pi0 state-masked attention.
// mask_rows: first N rows get mask_start applied
// mask_start: state token's key limit (enc_seq+1)
// pad_start: actual S_kv before padding (for pad column masking on all rows)
void softmax_state_masked_fp16(__half* data, int rows, int cols,
                                int mask_rows, int mask_start, int pad_start,
                                cudaStream_t stream = 0);
