# Demo: one GPU, native-length context

Two terminals, the same `vllm serve` command. One refuses the model's
native 262144 context on a 32 GB card; the other — three environment
variables later — boots it and answers questions about a 200K-token
codebase at full speed.

Requirements are the engine-attach guide's
(`docs/serving_engine_attach.md`): an SM120 GPU, vLLM 0.27.x, and the
`RadixArk/Qwen3.8-27B-NVFP4` checkpoint.

## Scene 1 — the ceiling

```bash
MODEL=<ckpt> ./serve_stock.sh
```

The stock server refuses at boot: the KV cache the native context
needs does not fit next to the weights. The error message is the
demo's opening shot — the engine itself saying 262144 is out of
reach, and estimating ~102K as the best it could do.

## Scene 2 — three env vars

```bash
MODEL=<ckpt> FRT_REPO=<flashrt-checkout> ./serve_attached.sh
```

Same command underneath; the hook attaches during load, the replaced
weights are released, and the freed memory becomes the KV pool the
native context needs. The boot log shows the seats installing and the
server comes up at `max_model_len 262144`.

## Scene 3 — ask the codebase

```bash
python ask.py --corpus <some-repo> --ctx 200000 \
  --question "Summarize the architecture of this codebase."
```

The client streams the answer with a live decode meter. Measured on
one RTX 5090: 200K-token prompts answer at ~170-180 tok/s decode
after a ~68 s prefill; at 2K the same server does ~210 tok/s on code
continuations with TTFT around 120 ms.

The same attach also serves shorter contexts faster than the stock
boot (see the guide's tables) — the demo's closing line: nothing was
converted, nothing forked; detach and the host is untouched.
