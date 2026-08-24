# ggml host adapter (native)

Native C++/CUDA host adapter that maps FlashRT structures onto ggml's CUDA
backend (llama.cpp family), targeting Jetson AGX Thor (SM110). Unlike the
Python runtime adapters (`vllm_engine.py`, `sglang_engine.py`), this adapter
is consumed at build time: the host's CMake compiles these translation units
inside its own build tree.

Layout:

- `fr_kernels.h` — pure C entry points (no ggml, no CUTLASS in the header).
- `fr_gemm_f32out.cu`, `fr_ada.cu`, `fr_qkv_post.cu`, `fr_quant_act.cu`,
  `fr_repack.cu` — framework-free CUDA kernels (NVFP4 wire format, repack,
  fused norm/modulation/rope, activation quantize).
- `fr_dispatch.cu`, `fr_ggml.cuh` — the ggml-facing half: subgraph window
  matchers over `ggml_tensor` chains, weight/activation caches, and kernel
  dispatch. Requires ggml-cuda's internal headers on the include path.
- GEMMs with fused epilogues are consumed from `csrc/gemm/fp4/` in this
  repository (GeGLU interleaved, SigLIP FFN f32-boundary pair); nothing is
  vendored.

Host-side integration (fuse-hook call sites, graph construction changes,
build wiring) lives in the host tree and points its build at this directory.
