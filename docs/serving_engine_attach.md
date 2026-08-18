# Attaching structures to a serving engine (vLLM / SGLang)

The engine adapters seat this repo's kernels inside a running serving
engine — no fork, no model conversion, one hook installed before the
engine loads weights. Everything below is a measured configuration:
each command is the exact form its receipts were produced with.

## Requirements

These determine whether the seats bind at all — check them before
anything else.

**Hardware**
- An SM120 GPU (RTX 5090 class). The W4A4 GEMV, small-M Marlin, and
  paged FP8-KV attention tiers are `sm120a` builds; on other
  architectures those seats refuse cleanly and the host runs its own
  kernels (the adapter never blocks engine startup).
- The receipts below are from a single 32 GB card, `max_num_seqs=1`
  (single-stream serving). Batch>1 is untested on this line.

**Software**
- vLLM **0.27.x** (the adapter patches `GPUModelRunner.load_model` and
  reads the FlashInfer backend's metadata layout of that series).
- SGLang **0.5.x** with the FlashInfer attention backend, run from its
  official container image.
- torch **2.13 + cu130**: the published kernel build variant consumed
  here is `torch213-cxx11-cu130-x86_64-linux`. A different torch/CUDA
  pair needs the matching build variant on the hub (or mounted via
  `FRT_KERNEL_DIR_*`, see the SGLang section).
- `kernels` (huggingface) library for hub resolution on the vLLM host.
  Air-gapped SGLang containers skip it entirely through
  `FRT_KERNEL_DIR_*`.

**Model / checkpoint**
- Measured host: **Qwen3.8-27B** (Qwen3.5 backbone: 48 gated-delta +
  16 full-attention layers), served from a **ModelOpt mixed-precision
  checkpoint** — NVFP4-packed projections (uint8 nibble pairs + FP8
  block-16 scales + a per-tensor global scale), FP8 rows on the
  gated-delta projections, a W4A16 head, and FP8 KV cache
  (`kv_cache_dtype fp8_e4m3` resolved from the checkpoint config).
- The `auto`/`mirror` tiers *adopt* these packs, so this checkpoint
  family is what they expect; a dense BF16 checkpoint works through
  the re-grid tiers (`nvfp4`, `w8`) instead.
- Speculative arms: vLLM MTP (`method qwen3_5_mtp`,
  `num_speculative_tokens=6` — measured optimum for this model; K≥7
  loses throughput at every context) / SGLang DSpark with its
  published draft model.

Real code/text prompts throughout — repeated/synthetic prompts
inflate speculative acceptance and void any spec-arm number.

## vLLM

### Attach

```python
from flash_rt.structures.adapters import vllm_engine

vllm_engine.install_load_hook(
    seats=vllm_engine.DENSE_SEAT_SUFFIXES,  # projection positions
    precision="auto",     # adopt the host's packs; carry FP8 rows to W4
    consume=True,         # release replaced host weights (KV pool grows)
    head=True,            # LM head relay (see spec-decode note below)
    fused_mlp=True)

from vllm import LLM
llm = LLM(model=..., ...)   # boot normally; seats install during load
```

The hook patches the model runner's `load_model` and attaches between
weight load and the engine's first trace — the only window where a
compiled vLLM host accepts a module swap.

### Precision tiers

| `precision=` | behavior |
|---|---|
| `"auto"` | adopt NVFP4 packs zero-copy; re-grid the checkpoint's FP8 rows to W4 (a precision change, reported as such) |
| `"mirror"` | faithful per-position: NVFP4 adopted, FP8 stays FP8 |
| `"nvfp4"` | re-grid everything to W4 from dense rows |
| `"w8"` | per-channel weight-only INT8 from dense rows |

### Environment knobs

| env | default | meaning |
|---|---|---|
| `FRT_HEAD_MARLIN` | `1` (auto-`0` under spec decode) | relay the checkpoint's W4A16 head into the small-M Marlin layout |
| `FRT_ATTN_XQA` | `0` | route spec-verify attention through the paged BF16-Q/FP8-KV kernel |
| `FRT_MOE_BAND_T` | `16` | rows above this go to the host expert path |

### Speculative decode: the head relay steps aside

vLLM's MTP draft shares the target's `lm_head`, and acceptance is an
agreement test between draft and target. Relaying only the target's
head to a different (even more accurate) numeric path breaks that
agreement: judged paired over ten prompts, acceptance length landed
lower on all ten. With a `speculative_config` present the relay
therefore defaults off; set `FRT_HEAD_MARLIN=1` explicitly to force
it. Without spec decode it stays on — pure step-rate, no acceptance
to protect.

### Long-context serving (the released memory becomes KV)

With `consume=True` the replaced host weights are released and the KV
pool grows by that amount. Measured on the 27B host: stock vLLM tops
out near a 102K context on the 32 GB card; the attached engine boots
the model's native 262144 and generates at 200K.

Two settings matter at the top of that range:

```python
LLM(...,
    max_model_len=262144,
    kv_cache_memory_bytes=10_400_000_000,  # explicit KV budget
    max_num_batched_tokens=4096)           # halves prefill activation peak
```

Sizing KV implicitly through `gpu_memory_utilization` leaves too
little headroom for long-prefill activation transients (measured OOM
at 200K with utilization-based sizing); an explicit KV budget with
1.5 GB+ left free is the configuration that survives.

### Judging protocol (what the receipts require)

- Speculative arms drift to different greedy continuations per arm, and
  acceptance length rides content. Cross-arm AL comparisons are only
  valid **paired by prompt** (same context length list, per-point
  pairing); single-prompt A/B buries systematic effects in content
  variance.
- At long contexts (32K+), decode and AL swing across boots; judge on
  medians of three or more boots, and prefer the step-rate column
  (decode ÷ AL) where continuations differ.
- Long-context prompts must be real text end to end. Tiled/repeated
  prompts inflate draft acceptance far above honest values. Measure
  decode as a difference of two long generations
  (`(n₂-n₁)/(t₂-t₁)`, `ignore_eos=True`) so prefill jitter and early
  stops cannot pollute the number.

## SGLang

The scheduler spawns worker processes, so the hook travels as a
`sitecustomize.py` on `PYTHONPATH` and activates through env vars.
Seats are the linear-attention projections (the fused-MLP and LM-head
surfaces differ from vLLM's and are not seated). Inside an air-gapped
container the kernels load from mounted snapshot directories through
`FRT_KERNEL_DIR_<REPO_SLUG>` — no hub access needed at serve time.

```bash
C=/root/.cache/huggingface/hub
docker run -d --name sgl-attach --gpus all --shm-size 32g --ipc=host \
  --network host \
  -v <models-dir>:/models \
  -v <this-repo>:/frt:ro \
  -v <hook-dir>:/frt-hook:ro \
  -v <hf-cache>:/root/.cache/huggingface \
  -e PYTHONPATH=/frt-hook \
  -e FRT_SGLANG_ATTACH=1 \
  -e FRT_SGLANG_STRUCTURES_PATH=/frt \
  -e FRT_SGLANG_PRECISION=auto \
  -e FRT_SGLANG_RELEASE=1 \
  -e "FRT_SGLANG_SEATS=linear_attn.out_proj,linear_attn.in_proj_qkvz" \
  -e FRT_KERNEL_DIR_FLASHRT_FP4_GEMM=$C/kernels--flashrt--fp4-gemm/snapshots/<rev>/build/<variant> \
  -e FRT_KERNEL_DIR_FLASHRT_FP4_FUSED_OPS=$C/kernels--flashrt--fp4-fused-ops/snapshots/<rev>/build/<variant> \
  -e FRT_KERNEL_DIR_FLASHRT_FLASHRT_FP8_FFN=$C/kernels--flashrt--flashrt-fp8-ffn/snapshots/<rev>/build/<variant> \
  -e FRT_KERNEL_DIR_FLASHRT_FLASHRT_GEMM_EPILOGUES=$C/kernels--flashrt--flashrt-gemm-epilogues/snapshots/<rev>/build/<variant> \
  <sglang-image> \
  sglang serve --model-path /models/<model> --trust-remote-code \
  --mem-fraction-static 0.85 --attention-backend flashinfer \
  --chunked-prefill-size 2048 --disable-radix-cache \
  --max-running-requests 1 --host 0.0.0.0 --port 30000
```

`<hook-dir>` contains the bridge's `sitecustomize.py` (see
`flash_rt/structures/adapters/sglang_engine.py:install`, which writes
it). `<variant>` is the build directory for the serving container's
torch/CUDA pair, e.g. `torch213-cxx11-cu130-x86_64-linux`.

**Speculative serving (DSpark)**: add

```
--speculative-algorithm DSPARK \
--speculative-draft-model-path /models/<draft-model>
```

and raise `--mem-fraction-static` to **0.90–0.93** — at 0.85 the
draft weights leave no room for the KV pool and the server refuses to
boot (measured; 0.92 is the configuration the receipts used).

## Measured receipts (fresh paired baselines, single 5090)

Decode tok/s, real code/text prompts, greedy; spec = MTP K=6 (vLLM) /
DSpark (SGLang). All baselines re-measured in the same window as the
attached arms.

| | vLLM 2K | vLLM 32K | SGLang 2K | SGLang 32K |
|---|---|---|---|---|
| attach vs base (no spec) | +17–27% | +13–17% | +11% | +9–10% |
| attach vs base (spec) | +2–6% (paired) | code +52%, text −5–15% | +9–14% | code −2%, text +29% |
| TTFT | −11–21% | −11% | −14% | −8% |
| max context | — | 102K → 262144 native | — | — |
| 200K decode | — | 178 (code) / 175 (text) tok/s | — | — |

Spec-arm decode columns ride acceptance-length content variance (see
judging protocol); the step-rate column is uniformly +9–13% for the
attached arms.
