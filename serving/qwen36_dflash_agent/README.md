# serving/qwen36_dflash_agent

OpenAI-compatible serving host for Qwen3.6-27B NVFP4 with **DFlash
block-diffusion speculative decoding**
(see [`docs/qwen36_dflash.md`](../../docs/qwen36_dflash.md)).

This directory is the policy layer above the FlashRT execution
contract: it owns request shaping and telemetry only, adds no session
or KV verbs to `exec/`, and keeps the frontend API untouched.

## Scope

| | this host | [`serving/qwen36_agent`](../qwen36_agent) |
|---|---|---|
| decode path | DFlash drafter (K=15 block) | MTP chain (K<=6) |
| session state | stateless — full prefill per request | exact-prefix reuse, capsules |
| tool calling / SSE streaming | no | yes |
| concurrency | batch 1, serialized | batch 1, scheduled sessions |

Use this host for single-stream, short-context request/response
workloads (robot planners, structured-output services) where the
DFlash path measures fastest; use `qwen36_agent` for long-running
agent sessions.

## Quickstart

**Prerequisites**: FlashRT built for your GPU (`GPU_ARCH=110` on
Jetson AGX Thor), the Qwen3.6-27B NVFP4 checkpoint, the paired FP8
MTP checkpoint (frontend construction requires it), and the DFlash
drafter checkpoint:

```bash
hf download z-lab/Qwen3.6-27B-DFlash --local-dir /models/Qwen3.6-27B-DFlash
pip install fastapi uvicorn
```

**1. Start the server**

```bash
export FLASHRT_QWEN36_MTP_CKPT_DIR=/models/Qwen3.6-27B-FP8
export FLASHRT_QWEN36_DFLASH_CKPT_DIR=/models/Qwen3.6-27B-DFlash
export FLASHRT_QWEN36_LONG_KV_CACHE=fp8

python -m serving.qwen36_dflash_agent.server \
  --checkpoint /models/Qwen3.6-27B-NVFP4 \
  --max-seq 32768 --K 15 \
  --host 127.0.0.1 --port 8000
```

The frontend arch is auto-detected (SM110 -> Thor, otherwise RTX);
override with `--arch thor|rtx`.

**2. Check it is up**

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok","arch":"thor","path":"dflash","pertoken_window":true,...}
```

**3. Chat completion**

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "model": "qwen3.6-27b-dflash",
    "messages": [{"role": "user", "content":
      "Output a JSON action list to pick up the red cube and place it on the tray."}],
    "max_tokens": 256
  }'
```

The response carries a `flashrt` telemetry block with the speculation
cycle count, realized accept length, and end-to-end latency.

## Limits (v1)

- Greedy decode only; sampling parameters are accepted and ignored.
- `stream` is not supported; responses return complete.
- The DFlash loop generates the full `max_tokens` budget and the
  response is truncated at the first end token — budget generously
  but not extravagantly.
- Qwen thinking mode is off by default; pass `"enable_thinking": true`
  to opt in.

## Tuning

DFlash env knobs (`FLASHRT_QWEN36_DFLASH_PERTOKEN`, `..._WINDOW`,
`..._WINDOW_SEED`) are documented in
[`docs/qwen36_dflash.md`](../../docs/qwen36_dflash.md) together with
measured Thor performance.
