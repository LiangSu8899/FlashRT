# Usage

## Building a host against the adapter

The reference host is the Jetson-PI-Edge llama.cpp tree, which carries the
integration side (CMake wiring, fuse-hook call sites, pi0 graph changes)
on its FlashRT branch and consumes this repository as a submodule at
`ggml/src/ggml-cuda/flashrt/flashrt-public`:

```bash
git clone --recursive -b feat/flashrt-thor-kernels <host repo>
cd Jetson-PI-Edge
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FLASHRT=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

Options:

- `GGML_CUDA_FLASHRT` (OFF by default) — enables the layer. Without it the
  build is stock llama.cpp; every integration point is compiled out.
- `GGML_CUDA_FLASHRT_PUBLIC_DIR` — path to a FlashRT checkout, overriding
  the submodule location.
- `GGML_CUDA_FLASHRT_CUTLASS_DIR` — CUTLASS override; defaults to
  `third_party/cutlass` inside this repository.

The adapter is built as a separate CMake OBJECT library with
`-arch=sm_110a` and CUTLASS headers; the rest of ggml-cuda compiles
unchanged. The AOT FlashAttention-4 windows enable automatically when the
`fa4_aot/*.o` artifacts are present (they are checked in; see
`fa4_aot/README.md` to regenerate).

## Model preparation

- LLM weights: quantize with the host's `llama-quantize` to the `NVFP4`
  target (exposed by the FlashRT branch). Setting `GGML_NVFP4_MSE=1`
  during quantization selects per-block scales by reconstruction-MSE
  search instead of plain absmax (slower to quantize, more accurate).
- The mmproj (vision tower) is quantized to NVFP4 the same way.

## Running

```bash
PI_MODEL=pi05 ./build/bin/llama-server -m <llm.gguf> --mmproj <mmproj.gguf> \
    -ngl 99 --flash-attn on --port <port>
```

The server exposes the host's action-chunk HTTP protocol (reset → images →
state → infer). Warm-up matters on Thor: latency reaches its steady state
after roughly 15 inferences.

## Runtime switches

All switches are environment variables; unset means enabled/default.

| variable | effect |
|---|---|
| `GGML_CUDA_FLASHRT_DISABLE=1` | disable the whole layer at runtime (stock kernels) |
| `GGML_FLASHRT_NO_RMS_GEMMA=1` | disable the Gemma norm-chain window |
| `GGML_FLASHRT_NO_QKV_PREFILL=1` | disable the fused prefill QKV window |
| `GGML_FLASHRT_NO_DEC_ATTN=1` | disable the decomposed decode attention |
| `GGML_FLASHRT_NO_VIT_FA4=1` | disable the AOT FA4 vision attention |
| `GGML_FLASHRT_NO_PREFILL_FA4=1` | disable the AOT FA4 prefill attention |
| `GGML_FLASHRT_NO_KV_TAIL=1` | disable the batched persistent-KV tail copies |
| `GGML_FLASHRT_NO_VIS_F16=1` | disable the vision QKV window's direct f16 K/V outputs |
| `GGML_CUDA_FLASHRT_NO_CACHE=1` | disable the pointer-keyed weight repack cache (required for `test-backend-ops`, see TESTING.md) |
| `GGML_FLASHRT_DEBUG=1` | print window-match failure diagnostics |
| `GGML_FLASHRT_DUMP=<file>` / `GGML_FLASHRT_DUMP_MAX=<n>` | dump the first n evaluated graphs' node sequences |

Host-side switches on the FlashRT branch (outside this repository):
`GGML_PI05_MOD_PRECOMP=0` disables the denoise-schedule modulation
precompute, `LLAMA_GRAPH_REUSE_DISABLE=1` disables graph reuse,
`GGML_CUDA_DISABLE_GRAPHS=1` disables CUDA graphs (useful for profiling:
kernels are invisible to nsys while CUDA graphs replay).

Every window degrades gracefully: when its predicate does not match (or
its switch is set) the nodes run on stock ggml kernels, so the switches
bisect regressions window by window.
