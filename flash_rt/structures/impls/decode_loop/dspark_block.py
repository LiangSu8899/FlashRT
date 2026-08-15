"""Block-draft speculative decoding for the whole-step loop (DFlash/DSpark).

The draft is a small full-attention transformer that infills a block of
masked positions in one forward, conditioned on auxiliary features tapped
from a handful of the target's layers. One round costs one draft forward
plus one multi-token verify — against the autoregressive MTP draft chain
this removes the per-token draft steps that dominate that family's round.

Serving geometry (learned the hard way): the draft emits one logit row
per noise slot *including the seed slot* — the seed's own row, fed the
real token embedding, is the strongest draft of the block, and dropping
it (as the checkpoint's reference implementation does) halves the
acceptance length. gamma drafts come from gamma noise slots
``[seed, mask x (gamma-1)]``, the Markov bigram head is applied
*sequentially* (each slot's bias conditioned on the previously sampled
token, anchored at the seed), and the verify window is gamma+1 tokens.

The round is fully captured, following the MTP member's graph families:

- the draft forward is one graph — static context K/V buffers take the
  block's own keys by ``index_copy_`` at the block positions (the block
  region ``[start, start+gamma)`` always sits right past the appended
  context, so one masked SDPA over the static window covers context and
  block siblings alike), and the serial Markov head runs unrolled on
  device, so proposing costs zero host syncs;
- the gamma+1 verify is one graph with the gated-delta state snapshot
  captured at its head (the MTP draft graph's ``snap_states`` idiom);
- re-advance is gated-delta-only: the verify pass already produced
  everything else a rejected round needs — the bonus token is the
  verify logit at the cut (its prefix rows saw exactly the accepted
  tokens), and the KV rows for the accepted region were committed by
  the verify itself (same tokens, same positions, FP8 pages included).
  The only state a rejection actually loses is the gated-delta
  recurrence past the cut, so each rejected-prefix length gets a graph
  that restores the snapshot and re-drives just the gated-delta
  sublayers over their stashed verify inputs — forty-eight state
  layers instead of the whole model. A fully-accepted round skips
  rollback outright: the verify pass committed exactly the accepted
  stream;
- the arbiter costs the round's single host sync, same as the MTP loop.

Aux features are the *inputs* of the configured target layers (the
serving-side capture point for this host family), collected by forward
pre-hooks. Hooks only execute during eager passes and graph capture —
the tensors they see during the verify capture live in the graph's
private pool, which every replay rewrites in place, so the captured
references stay valid round after round and the accepted rows' features
read straight out of them.

Weights load from the draft checkpoint's single safetensors file; the
draft shares the target's embedding and lm_head (the draft was trained
against the target's own head). The confidence head ships in the
checkpoint but is not consumed here — the block length stays fixed.
"""

from __future__ import annotations

import json
import os
import pathlib

import torch
import torch.nn.functional as F

__all__ = ["DSparkBlockDraft", "DSparkRunner"]


@torch.library.custom_op("flashrt_native::dspark_draft_attn",
                         mutates_args=())
def _draft_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                seqused: torch.Tensor, scale: float) -> torch.Tensor:
    """Non-causal fa2 over the draft's static context window.

    ``seqused`` is the device-side valid-K length (context + block):
    the kernel reads it at run time, so the captured launch replays
    with whatever the round wrote there — no mask, no materialised
    attention math. q [S, Hq, D]; k/v [max, Hkv, D]."""
    from flash_rt import flash_rt_fa2 as fa2

    S, Hq, D = q.shape
    o = torch.empty_like(q)
    lse = torch.empty(1, Hq, S, device=q.device, dtype=torch.float32)
    # strides are (batch, token, head) with a unit dim stride — the
    # production call sites' convention for the vendored fa2 surface
    mk, hk = k.shape[0], k.shape[1]
    fa2.fwd_bf16_seqused(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
        lse.data_ptr(), seqused.data_ptr(), 1, S, mk, Hq, hk, D,
        (S * Hq * D, Hq * D, D),
        (mk * hk * D, hk * D, D),
        (mk * hk * D, hk * D, D),
        (S * Hq * D, Hq * D, D), scale, 0,
        torch.cuda.current_stream().cuda_stream)
    return o


@_draft_attn.register_fake
def _(q, k, v, seqused, scale):
    return torch.empty_like(q)


def _rms(x, w, eps=1e-6):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * w.float()).to(x.dtype)


class DSparkBlockDraft:
    """The draft network, functional form over checkpoint tensors.

    ``propose_`` is capture-ready: every input rides a persistent device
    buffer (seed token, block start), every write lands in a persistent
    buffer (the draft tokens), and no host value is consulted.
    """

    def __init__(self, draft_dir, embed, lm_head, max_len, device="cuda"):
        from safetensors.torch import load_file

        d = pathlib.Path(str(draft_dir))
        cfg = json.loads((d / "config.json").read_text())
        dc = cfg.get("dflash_config") or {}
        self.taps = list(dc["target_layer_ids"])
        self.mask_id = int(dc["mask_token_id"])
        #: gamma = draft tokens per round = noise slots (seed included)
        self.gamma = int(cfg["block_size"])
        self.n_layers = int(cfg["num_hidden_layers"])
        self.n_q = int(cfg["num_attention_heads"])
        self.n_kv = int(cfg["num_key_value_heads"])
        self.hd = int(cfg["head_dim"])
        self._embed = embed
        self._head = lm_head
        self._dev = device
        self._max = int(max_len)

        t = {k: v.to(device, torch.bfloat16)
             for k, v in load_file(str(d / "model.safetensors")).items()}
        self._t = t
        # draft projection precision: the verify pass anchors the
        # output stream regardless, so acceptance length alone judges
        # the draft's precision. Measured on the real-source coding
        # stream: W4A4 costs it (AL 2.93 -> 2.79, net negative
        # despite the halved weight stream) - the bf16 default
        # stands, the arms stay for other streams. FP8 (W8A8,
        # per-tensor scales) is the middle arm.
        mode = os.environ.get("FRT_DSPARK_DRAFT_QUANT", "bf16")
        self._proj = {}
        self._proj8 = {}
        names = [f"layers.{li}.self_attn.{nm}"
                 for li in range(self.n_layers)
                 for nm in ("q_proj", "k_proj", "v_proj", "o_proj")]
        names += [f"layers.{li}.mlp.{nm}"
                  for li in range(self.n_layers)
                  for nm in ("gate_proj", "up_proj", "down_proj")]
        if mode == "fp8":
            for nm in names:
                key = nm + ".weight"
                w = t[key].float()
                sw = (w.abs().amax() / 448.0).clamp_min(1e-12)
                self._proj8[nm] = (
                    (w / sw).to(torch.float8_e4m3fn).contiguous(),
                    sw.to(torch.float32))
                del t[key]
            torch.cuda.empty_cache()
        elif mode == "fp4":
            from ..linear_proj import nvfp4_dynamic
            for nm in names:
                key = nm + ".weight"
                seam, _rel = nvfp4_dynamic.bind_proj_seam(
                    {"w": t[key]})
                self._proj[nm] = seam
                del t[key]
            torch.cuda.empty_cache()

        # the checkpoint's own rope parameters through the host library's
        # rope init - yarn attention scaling included in cos/sin
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3RotaryEmbedding)
        rp = dict(cfg["rope_parameters"])
        qc = Qwen3Config(
            hidden_size=cfg["hidden_size"], head_dim=self.hd,
            num_attention_heads=self.n_q,
            num_key_value_heads=self.n_kv,
            max_position_embeddings=cfg["max_position_embeddings"],
            rope_scaling=rp, rope_theta=rp.get("rope_theta", 1e7))
        rot = Qwen3RotaryEmbedding(qc).to(device)
        pos = torch.arange(self._max, device=device)[None]
        cos, sin = rot(torch.empty(1, 1, device=device), pos)
        self._cos = cos[0].to(torch.bfloat16)     # [max_len, hd]
        self._sin = sin[0].to(torch.bfloat16)

        # per-layer context K/V (roped keys). Appended rows carry the
        # fc-projected target features; the block region past ``_len``
        # holds the propose pass's own keys until the next append
        # overwrites it — every stale row dies before it is read.
        self._ck = [torch.zeros(self._max, self.n_kv, self.hd,
                                device=device, dtype=torch.bfloat16)
                    for _ in range(self.n_layers)]
        self._cv = [torch.zeros(self._max, self.n_kv, self.hd,
                                device=device, dtype=torch.bfloat16)
                    for _ in range(self.n_layers)]
        self._len = 0

        g = self.gamma
        self._ids = torch.full((g,), self.mask_id, dtype=torch.long,
                               device=device)
        self._start = torch.zeros(1, dtype=torch.long, device=device)
        self._drafts = torch.zeros(g, dtype=torch.long, device=device)
        self._arg = torch.arange(g, device=device)
        self._armax = torch.arange(self._max, device=device)
        self._seqk = torch.zeros(1, dtype=torch.int32, device=device)

    def reset(self):
        self._len = 0

    def _mm(self, x, name):
        """Projection router: quantized arm when bound, bf16 otherwise."""
        seam = self._proj.get(name)
        if seam is not None:
            return seam(x)
        w8 = self._proj8.get(name)
        if w8 is not None:
            wq, sw = w8
            sx = (x.float().abs().amax() / 448.0).clamp_min(1e-12)
            xq = (x.float() / sx).to(torch.float8_e4m3fn)
            return torch._scaled_mm(
                xq, wq.t(), scale_a=sx, scale_b=sw,
                out_dtype=torch.bfloat16)
        return x @ self._t[name + ".weight"].T

    def _rope(self, x, pos):
        # x [S, H, hd]; standard interleaved-half rotation
        cos = self._cos[pos].unsqueeze(1)
        sin = self._sin[pos].unsqueeze(1)
        h = self.hd // 2
        x1, x2 = x[..., :h], x[..., h:]
        rot = torch.cat((-x2, x1), dim=-1)
        return x * cos + rot * sin

    @torch.no_grad()
    def append_ctx(self, feats, pos):
        """feats [T, taps*hidden] raw tap concat; pos [T] absolute."""
        t = self._t
        tgt = _rms(feats @ t["fc.weight"].T, t["hidden_norm.weight"])
        n = feats.shape[0]
        for li in range(self.n_layers):
            p = f"layers.{li}.self_attn."
            k = self._mm(tgt, p + "k_proj").view(n, self.n_kv, self.hd)
            k = self._rope(_rms(k, t[p + "k_norm.weight"]), pos)
            v = self._mm(tgt, p + "v_proj").view(n, self.n_kv, self.hd)
            self._ck[li].index_copy_(0, pos, k)
            self._cv[li].index_copy_(0, pos, v)
        self._len += n

    @torch.no_grad()
    def propose_(self):
        """One block forward, buffers in, buffers out (capture-ready).

        Reads ``_ids`` (slot 0 = seed) and ``_start``; writes the gamma
        draft tokens into ``_drafts``. The block's keys land in the
        context buffers at the block positions, so a single masked SDPA
        over the static window serves the dual-source attention: every
        noise slot sees all context and every noise sibling.
        """
        t = self._t
        g = self.gamma
        pos = self._start + self._arg
        h = self._embed(self._ids.unsqueeze(0))[0]      # [g, hidden]
        # device-side valid-K length: context + this block
        self._seqk.copy_((self._start + g).to(torch.int32))
        scale = 1.0 / (self.hd ** 0.5)
        for li in range(self.n_layers):
            p = f"layers.{li}."
            a = f"{p}self_attn."
            x = _rms(h, t[p + "input_layernorm.weight"])
            q = self._mm(x, a + "q_proj").view(g, self.n_q, self.hd)
            q = self._rope(_rms(q, t[a + "q_norm.weight"]), pos)
            kn = self._mm(x, a + "k_proj").view(g, self.n_kv, self.hd)
            kn = self._rope(_rms(kn, t[a + "k_norm.weight"]), pos)
            vn = self._mm(x, a + "v_proj").view(g, self.n_kv, self.hd)
            self._ck[li].index_copy_(0, pos, kn)
            self._cv[li].index_copy_(0, pos, vn)
            o = _draft_attn(q.contiguous(), self._ck[li], self._cv[li],
                            self._seqk, scale).reshape(g, -1)
            h = h + self._mm(o, a + "o_proj")
            x = _rms(h, t[p + "post_attention_layernorm.weight"])
            m = f"{p}mlp."
            h = h + self._mm(
                F.silu(self._mm(x, m + "gate_proj"))
                * self._mm(x, m + "up_proj"), m + "down_proj")
        logits = self._head(_rms(h, t["norm.weight"]).unsqueeze(0))[0]
        # serial markov, unrolled on device: each slot biased by the
        # previously sampled token, anchored at the seed — no host
        # round-trips inside the chain. The bias GEMV reads the [vocab,
        # r] factor once per slot; it stays bf16 (fp32 accumulate) so
        # the chain reads half the bytes
        w1 = t["markov_head.markov_w1.weight"]
        w2 = t["markov_head.markov_w2.weight"]
        prev = self._ids[0:1]
        for i in range(g):
            bias = w1.index_select(0, prev) @ w2.T
            tok = (logits[i].float() + bias[0].float()).argmax().view(1)
            self._drafts[i:i + 1].copy_(tok)
            prev = tok


class DSparkRunner:
    """Captured-round driver over a built whole-step loop."""

    def __init__(self, loop, draft_dir, model=None):
        self._loop = loop
        del model  # reserved: host handle for cross-vehicle diagnostics
        self._draft = DSparkBlockDraft(
            draft_dir, loop._embed, loop._head, loop._max,
            device=loop._embed.weight.device)
        self._tap_in = {}
        lm_layers = loop._layers
        self._hooks = []
        for slot, li in enumerate(self._draft.taps):
            self._hooks.append(lm_layers[li].register_forward_pre_hook(
                self._make_hook(slot)))
        # the gated-delta sublayers and their inputs: the re-advance
        # graphs re-drive exactly these modules over the rows the
        # verify pass showed them (input = the decoder layer's normed
        # hidden, which is what the sublayer consumes again)
        self._gdn_mods = {}
        self._gdn_in = {}
        for i, lyr in enumerate(lm_layers):
            if i in loop._full_set:
                continue
            for _name, mod in lyr.named_children():
                if hasattr(mod, "conv1d") and hasattr(mod, "A_log"):
                    self._gdn_mods[i] = mod
                    self._hooks.append(mod.register_forward_pre_hook(
                        self._make_gdn_hook(i), with_kwargs=True))
                    break
        g = self._draft.gamma
        dev = self._draft._dev
        self._vblk = torch.zeros(1, g + 1, dtype=torch.long, device=dev)
        self._vpos = torch.zeros(g + 1, dtype=torch.long, device=dev)
        self._ag1 = torch.arange(g + 1, device=dev)
        self._outbuf = torch.zeros(loop._max, dtype=torch.long,
                                   device=dev)
        self._gp = None
        self.last_acceptance = 0.0

    def _make_hook(self, slot):
        def hook(module, args):
            self._tap_in[slot] = args[0]
        return hook

    def _make_gdn_hook(self, idx):
        def hook(module, args, kwargs):
            self._gdn_in[idx] = args[0] if args else \
                kwargs["hidden_states"]
        return hook

    def _taps_rows(self, taps, rows):
        return torch.cat([taps[s][0, :rows] for s in
                          range(len(self._draft.taps))], dim=-1)

    def _capture(self):
        """Warm every round shape, then capture the graph families.

        Runs once, right after the first prompt pass, when every buffer
        holds live values. The warmups execute on a side stream (the
        capture-stream discipline the loop's own capture established);
        the gated-delta states they advance are rolled back before
        capture so the first replay starts from the real prompt state.
        """
        loop, draft = self._loop, self._draft
        g = draft.gamma
        if loop._kv_band is not None:
            # every round shape's fp8 mask exists before capture — a
            # lazy fill between warmup and capture flips state mid-graph
            loop._kv_band.prewarm(range(1, g + 2))
        # compile-then-capture, the MTP verify recipe: inductor's
        # elementwise fusion buys the multi-row passes the same third
        # it buys the plain step. Each round shape specialises once
        # during warmup; the loop's init already raised the recompile
        # budget for per-layer specialisation.
        if loop._use_compile:
            if getattr(loop, "_fwd_full_c", None) is None:
                loop._fwd_full_c = torch.compile(loop._fwd_full,
                                                 dynamic=False)
            fwd = loop._fwd_full_c
            if getattr(self, "_propose_c", None) is None:
                self._propose_c = torch.compile(draft.propose_,
                                                dynamic=False)
            prop = self._propose_c
        else:
            fwd = loop._fwd_full
            prop = draft.propose_
        self._snap = {
            i: (torch.empty_like(loop.cache.conv_states[i]),
                torch.empty_like(loop.cache.recurrent_states[i]))
            for i in loop._gdn_slots()}
        for i, (cb, rb) in self._snap.items():
            cb.copy_(loop.cache.conv_states[i])
            rb.copy_(loop.cache.recurrent_states[i])

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            prop()
            self._vblk[0, 1:].copy_(draft._drafts)
            fwd(self._vblk, self._vpos)
            for m in range(1, g + 1):
                fwd(self._vblk[:, :m], self._vpos[:m])
        torch.cuda.current_stream().wait_stream(side)
        for i, (cb, rb) in self._snap.items():
            loop.cache.conv_states[i].copy_(cb)
            loop.cache.recurrent_states[i].copy_(rb)

        self._gp = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._gp):
            # the verify inputs assemble in-graph: seed and start
            # already ride draft buffers, so the whole block-and-
            # positions setup costs zero host launches per round
            prop()
            self._vblk[0, 0].copy_(draft._ids[0])
            self._vblk[0, 1:].copy_(draft._drafts)
            self._vpos.copy_(self._ag1)
            self._vpos.add_(draft._start)
        # the pre-round state snapshot rides at the verify graph's head:
        # ninety-six host-launched little copies a round otherwise sit
        # squarely inside the per-round sync window
        self._gv = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._gv):
            for i, (cb, rb) in self._snap.items():
                cb.copy_(loop.cache.conv_states[i])
                rb.copy_(loop.cache.recurrent_states[i])
            self._vlg, self._vhn = fwd(self._vblk, self._vpos)
        # the verify capture just ran the hooks: these references live
        # in the verify graph's pool and every replay rewrites them in
        # place. Saved now — later captures would overwrite the hook
        # dicts with their own pools.
        self._vtaps = [self._tap_in[s]
                       for s in range(len(draft.taps))]
        gdn_v = dict(self._gdn_in)
        # gated-delta-only re-advance: restore the snapshot, then
        # re-drive just the state sublayers over the rows the verify
        # showed them. Everything else a rejected round needs already
        # exists — bonus logits sit in the verify output at the cut,
        # and the accepted region's KV rows were the verify's own
        # writes (same tokens, same positions).
        self._ra = {}
        for m in range(1, g + 1):
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for i, (cb, rb) in self._snap.items():
                    loop.cache.conv_states[i].copy_(cb)
                    loop.cache.recurrent_states[i].copy_(rb)
                for i, mod in self._gdn_mods.items():
                    mod(gdn_v[i][:, :m], loop.cache, None)
            self._ra[m] = gr

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens):
        loop, draft = self._loop, self._draft
        dev = input_ids.device
        L = int(input_ids.shape[1])
        g = draft.gamma
        if L + max_new_tokens + g + 2 > loop._max:
            raise ValueError("refused: window exceeds the static max")
        draft.reset()
        self._round_log = []
        loop.cache.frt_continue = False
        loop._rope_delta.zero_()
        if loop._kv_band is not None:
            loop._kv_band.reset()
        # canonical window, same as the loop's own generate: rows past
        # the prompt zero before any produced token
        for i in loop._full:
            loop.cache.key_cache[i][:, :, L:].zero_()
            loop.cache.value_cache[i][:, :, L:].zero_()
        logits, _ = loop._fwd_full(input_ids,
                                   torch.arange(L, device=dev))
        draft.append_ctx(
            self._taps_rows(self._tap_in, L),
            torch.arange(L, device=dev))
        loop.cache.frt_continue = True
        seed = logits[0, -1].float().argmax().view(1)
        self._outbuf[0].copy_(seed[0])
        produced = 1
        start = L
        rounds = 0
        accepted_total = 0
        if self._gp is None:
            self._vpos.copy_(self._ag1 + start)
            draft._ids[0].copy_(seed[0])
            draft._start.fill_(start)
            self._vblk[0, 0].copy_(seed[0])
            self._capture()
        while produced < max_new_tokens:
            draft._ids[0].copy_(seed[0])
            draft._start.fill_(start)
            self._gp.replay()
            self._gv.replay()
            post = self._vlg[0].float().argmax(-1)          # [g+1]
            match = (self._vblk[0, 1:] == post[:-1]).long()
            # device-side prefix match: the round's single host sync
            a = int(match.cumprod(0).sum().item())
            # the bonus is the verify logit at the cut either way: row
            # a's prefix saw exactly the accepted tokens
            bonus = post[a:a + 1]
            if a < g:
                # rejection loses only the gated-delta recurrence past
                # the cut: restore and re-drive the state sublayers
                self._ra[a + 1].replay()
            rows = a + 1
            draft.append_ctx(
                self._taps_rows(self._vtaps, rows),
                torch.arange(start, start + rows, device=dev))
            if a > 0:
                self._outbuf[produced:produced + a].copy_(
                    self._vblk[0, 1:a + 1])
            self._outbuf[produced + a].copy_(bonus[0])
            seed = bonus
            produced += rows
            start += rows
            accepted_total += rows
            rounds += 1
            if rounds <= 40:
                self._round_log.append(rows)
        self.last_acceptance = accepted_total / max(rounds, 1)
        self.last_rounds = self._round_log
        loop.cache.frt_continue = False
        return torch.cat(
            [input_ids, self._outbuf[:max_new_tokens].view(1, -1)], dim=1)

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
