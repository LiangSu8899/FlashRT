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

Aux features are the *inputs* of the configured target layers (the
serving-side capture point for this host family), collected by forward
pre-hooks that see every loop forward: the prompt pass and each
accepted-prefix re-advance feed the draft's context cache exactly once
per position.

The target-side round rides the loop's own machinery: ``_fwd_full`` for
the verify and the re-advance (offset row mask, in-place cache slots,
gated-delta continuation via the cache's ``frt_continue`` contract), and
the GDN state snapshot/restore idiom the MTP member established. Static
KV slots beyond the accepted position need no cropping — the row mask
never attends past the current row.

Weights load from the draft checkpoint's single safetensors file; the
draft shares the target's embedding and lm_head (the draft was trained
against the target's own head). The confidence head ships in the
checkpoint but is not consumed here — the block length stays fixed.
"""

from __future__ import annotations

import json
import pathlib

import torch
import torch.nn.functional as F

__all__ = ["DSparkBlockDraft", "DSparkRunner"]


def _rms(x, w, eps=1e-6):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * w.float()).to(x.dtype)


class DSparkBlockDraft:
    """The draft network, functional form over checkpoint tensors."""

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

        t = {k: v.to(device, torch.bfloat16)
             for k, v in load_file(str(d / "model.safetensors")).items()}
        self._t = t

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
        pos = torch.arange(max_len, device=device)[None]
        cos, sin = rot(torch.empty(1, 1, device=device), pos)
        self._cos = cos[0].to(torch.bfloat16)     # [max_len, hd]
        self._sin = sin[0].to(torch.bfloat16)

        # per-layer context K/V caches (roped keys), grown append-only
        self._ck = [torch.empty(max_len, self.n_kv, self.hd,
                                device=device, dtype=torch.bfloat16)
                    for _ in range(self.n_layers)]
        self._cv = [torch.empty(max_len, self.n_kv, self.hd,
                                device=device, dtype=torch.bfloat16)
                    for _ in range(self.n_layers)]
        self._len = 0

    def reset(self):
        self._len = 0

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
            k = (tgt @ t[p + "k_proj.weight"].T).view(
                n, self.n_kv, self.hd)
            k = self._rope(_rms(k, t[p + "k_norm.weight"]), pos)
            v = (tgt @ t[p + "v_proj.weight"].T).view(
                n, self.n_kv, self.hd)
            self._ck[li][self._len:self._len + n] = k
            self._cv[li][self._len:self._len + n] = v
        self._len += n

    @torch.no_grad()
    def propose(self, seed_id, start):
        """One block forward -> gamma draft tokens (serial markov)."""
        t = self._t
        g = self.gamma
        ids = torch.full((g,), self.mask_id, dtype=torch.long,
                         device=self._dev)
        ids[0] = seed_id
        pos = torch.arange(start, start + g, device=self._dev)
        h = self._embed(ids.unsqueeze(0))[0]          # [g, hidden]
        L = self._len
        rep = self.n_q // self.n_kv
        for li in range(self.n_layers):
            p = f"layers.{li}."
            a = f"{p}self_attn."
            x = _rms(h, t[p + "input_layernorm.weight"])
            q = (x @ t[a + "q_proj.weight"].T).view(g, self.n_q, self.hd)
            q = self._rope(_rms(q, t[a + "q_norm.weight"]), pos)
            kn = (x @ t[a + "k_proj.weight"].T).view(g, self.n_kv, self.hd)
            kn = self._rope(_rms(kn, t[a + "k_norm.weight"]), pos)
            vn = (x @ t[a + "v_proj.weight"].T).view(g, self.n_kv, self.hd)
            K = torch.cat([self._ck[li][:L], kn], dim=0)
            V = torch.cat([self._cv[li][:L], vn], dim=0)
            K = K.repeat_interleave(rep, dim=1)
            V = V.repeat_interleave(rep, dim=1)
            # dual-source full attention: every noise slot sees all
            # context and every noise sibling (bidirectional block)
            att = torch.einsum("qhd,khd->hqk", q.float(), K.float())
            att = torch.softmax(att / (self.hd ** 0.5), dim=-1)
            o = torch.einsum("hqk,khd->qhd", att,
                             V.float()).to(h.dtype).reshape(g, -1)
            h = h + o @ t[a + "o_proj.weight"].T
            x = _rms(h, t[p + "post_attention_layernorm.weight"])
            m = f"{p}mlp."
            h = h + (F.silu(x @ t[m + "gate_proj.weight"].T)
                     * (x @ t[m + "up_proj.weight"].T)) \
                @ t[m + "down_proj.weight"].T
        logits = self._head(_rms(h, t["norm.weight"]).unsqueeze(0))[0]
        # serial markov: each slot biased by the previously sampled token
        w1, w2 = t["markov_head.markov_w1.weight"], \
            t["markov_head.markov_w2.weight"]
        drafts = torch.empty(g, dtype=torch.long, device=self._dev)
        prev = int(seed_id)
        for i in range(g):
            bias = (w1[prev].float() @ w2.float().T)
            tok = int((logits[i].float() + bias).argmax())
            drafts[i] = tok
            prev = tok
        return drafts


class DSparkRunner:
    """Round driver over a built whole-step loop."""

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
        self.last_acceptance = 0.0

    def _make_hook(self, slot):
        def hook(module, args):
            self._tap_in[slot] = args[0]
        return hook

    def _taps_cat(self):
        d = self._draft
        return torch.cat([self._tap_in[s][0] for s in
                          range(len(d.taps))], dim=-1)

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
        logits, _ = loop._fwd_full(input_ids,
                                   torch.arange(L, device=dev))
        draft.append_ctx(self._taps_cat(),
                         torch.arange(L, device=dev))
        loop.cache.frt_continue = True
        seed = int(logits[0, -1].float().argmax())
        seq = input_ids[0].tolist() + [seed]
        start = L
        rounds = 0
        accepted_total = 0
        while len(seq) - L < max_new_tokens:
            drafts = draft.propose(seq[start], start)
            blk = torch.tensor([[seq[start]] + drafts.tolist()],
                               device=dev)
            pos = torch.arange(start, start + g + 1, device=dev)
            conv = {i: loop.cache.conv_states[i].clone()
                    for i in loop._gdn_slots()}
            rec = {i: loop.cache.recurrent_states[i].clone()
                   for i in loop._gdn_slots()}
            vlog, _ = loop._fwd_full(blk, pos)
            # 轮特征取自 verify 前向 (快照态下的正确前缀隐层, 与
            # serving 的 aux 捕获点同语义); re-advance 只养状态
            vfeats = self._taps_cat().clone()
            post = vlog[0].float().argmax(-1)
            match = (blk[0, 1:] == post[:-1]).long()
            a = int(match.cumprod(0).sum())
            bonus = int(post[a])
            for i, t in conv.items():
                loop.cache.conv_states[i].copy_(t)
            for i, t in rec.items():
                loop.cache.recurrent_states[i].copy_(t)
            rpos = torch.arange(start, start + a + 1, device=dev)
            loop._fwd_full(blk[:, :a + 1], rpos)
            draft.append_ctx(vfeats[:a + 1], rpos)
            seq.extend(blk[0, 1:a + 1].tolist() + [bonus])
            start += a + 1
            accepted_total += a + 1
            rounds += 1
            if not hasattr(self, "_round_log"):
                self._round_log = []
            if rounds <= 40:
                self._round_log.append(a + 1)
        self.last_acceptance = accepted_total / max(rounds, 1)
        self.last_rounds = getattr(self, "_round_log", None)
        return torch.tensor([seq[:L + max_new_tokens]], device=dev)

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
