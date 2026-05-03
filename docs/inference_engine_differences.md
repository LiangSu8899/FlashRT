# Differences from existing inference engines

> Pick by **workload**, not by tok/s. Each engine below is the right answer for some scenario; FlashRT is the right answer for small-batch realtime fixed-architecture workloads. Small batches (CFG / multi-policy / multi-frame) are supported; what's not the design point is high-batch throughput.

---

## 1. Workload coverage

|  | Small-batch realtime (latency tail) | Large-batch throughput (tokens/sec total) |
|---|---|---|
| **LLM-shaped** (chat, completion, agent) | — | vLLM, SGLang, TRT-LLM |
| **Fixed-arch multimodal** (VLA, world model, audio gen, realtime DiT) | **FlashRT** | TensorRT (after build) |

---

## 2. User workflow — checkpoint to running production

| Stage | TensorRT | vLLM / SGLang | FlashRT |
|---|---|---|---|
| Convert format | PyTorch → ONNX (debug ops, shapes, dynamic axes) | none — load HF directly | none — load safetensors / Orbax directly |
| Build / compile | `trtexec` 30 min – several hours per shape | none | none |
| Calibrate | baked into `.engine` at build time | n/a (LLM-side, online) | first call, ~3 s, JSON cached to `~/.flash_rt/calibration/` |
| Steady-state call | C++ engine runtime / Python TRT bindings | `LLM().generate()` or HTTP | `model.predict(obs, prompt)` |
| Reproduce after model change | rebuild engine | redeploy | restart process (~5 s) |
| Reproduce after driver bump | rebuild engine | redeploy | restart process |
| Reproduce after GPU swap | rebuild engine per arch | redeploy | restart process |
| Deployment artifact | `.engine` file (per shape × per GPU × per driver) | wheel + checkpoint | `.so` + checkpoint |

---

## 3. Developer workflow — adding a new model

| Step | TensorRT | vLLM / SGLang | FlashRT |
|---|---|---|---|
| Write model code | (none — comes from ONNX) | `model_executor/models/<m>.py`, 200–800 LOC, must satisfy `AttentionMetadata` | `frontends/<fw>/<m>_<arch>.py` frontend, 800–1500 LOC, direct kernel calls |
| Weight loading | implicit in ONNX | `load_weights()` 100–300 LOC, HF style | `WEIGHT_SPEC` 100–300 LOC declarative |
| Custom op | TRT plugin (C++/CUDA, separate `.so`, scoped outside Myelin's auto-fusion pass) | Triton kernel (DSL) or `csrc/` CUDA + rebuild wheel | `.cu` + pybind, rebuild kernel `.so` (~2 min) |
| Custom attention | (none — write a plugin) | new attention backend, 500–2000 LOC + scheduler integration | implement `AttentionBackend` protocol, 50–200 LOC |
| Calibration | rebuild engine | n/a | 4 lines around shared cache framework |
| Iterate (change forward) | rebuild engine (~30 min) | rebuild wheel if `.cu`, else restart | restart Python (~10 s) |
| Multi-tenant serving for free | no (single engine) | yes (continuous batching) | no — one process per model |
| Test conformance | per-engine | scheduler integration tests | `tests/test_*_attn_backend.py` + precision suite |

Pain points each stack accepts:

| Stack | What it accepts | What it gets in return |
|---|---|---|
| **TensorRT** | Long build per shape; per-driver / per-arch rebuild required; plugins compose with Myelin's auto-fusion in limited ways | Auto fusion + tactic search per shape |
| **vLLM / SGLang** | Pipeline must satisfy scheduler / KV-block abstractions | Continuous batching + paged KV + OpenAI HTTP |
| **FlashRT** | Hand-written forward per `(model, framework, hardware)` | No compile, no export, direct kernel control, fast dev loop |

---

## 4. Routing table — which stack for which workload

| Workload | Stack |
|---|---|
| LLM serving, large batches, OpenAI HTTP | vLLM / SGLang |
| LLM serving with structured generation (CFG, JSON, agents) | SGLang |
| Frozen model, frozen GPU, amortize build cost over millions of inferences | TensorRT |
| LLM on consumer / Mac / CPU | llama.cpp |
| LLM on Jetson via the official NVIDIA stack | TRT-LLM / MLC-LLM |
| **Small-batch realtime VLA / robotics / on-device DiT or audio gen** | **FlashRT** |

Multiple stacks coexisting in one project is normal, not a contradiction. A robot running Pi0.5 control on FlashRT at 23 Hz alongside an LLM agent on vLLM solves two different problems.

---

## 5. Today's scope

FlashRT today focuses on the small-batch realtime / fixed-architecture workload above. A few related capabilities are deliberately handled by other layers, with explicit room for future extension:

- **Graph compilation / auto-fusion** — the runtime executes the kernel sequence the frontend wrote. Compiler-driven fusion is a possible direction; today's shape works because the target shape space is small enough for hand-tuned composition to win.
- **Multi-tenant serving** — one process serves one captured graph. A fan-out serving layer on top of FlashRT workers is an explicit extension path for B≥1 realtime services; cross-request continuous batching is what vLLM / SGLang are shaped for.
- **Generic HuggingFace LLM loader** — current LLM-shaped support comes from manual integrations (Pi0-FAST's Gemma 2B AR decode, Qwen3 / DiT inside GROOT). Generic auto-loading is an extension path, not a current feature.
- **Kernel authoring DSL** — kernels are hand-written CUDA today; integrations with Triton / TileLang authoring are possible future surfaces, but not a current goal.

---

## 6. Read more

- [`architecture.md`](architecture.md) — FlashRT's eight infra components
- [`adding_new_model.md`](adding_new_model.md) — full integration walkthrough
- [`extension/weight_spec.md`](extension/weight_spec.md), [`extension/attention_backend.md`](extension/attention_backend.md), [`extension/calibration.md`](extension/calibration.md) — first-class API contracts

External: [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/) · [vLLM](https://docs.vllm.ai/) · [SGLang](https://github.com/sgl-project/sglang) · [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
