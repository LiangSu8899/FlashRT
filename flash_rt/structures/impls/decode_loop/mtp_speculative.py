"""MTP speculative decode — the decode_loop family's second member.

Checkpoints in this family ship a one-layer DeepSeek-style draft head
(``mtp.safetensors``) that transformers hosts never use. This member
loads it, assembles the draft from the host's own module classes, and
runs the draft/verify loop around the whole-step form. Greedy spec
decode is exact by construction: the verify pass recomputes every
draft token with the main model, so the accepted stream is identical
to plain greedy decode — the gate checks token identity, not a band.

The draft head is carried in BF16 (its FP8 blocks are dequantised at
load); its attention layer gets one extra slot in the static cache.
The gated-delta states cannot roll back through a rejected suffix, so
the loop snapshots them before each verify and re-advances the
accepted prefix from the snapshot when a draft is cut short.
"""

from __future__ import annotations

import torch


def _dequant_block_fp8(w, scale_inv):
    n, k = w.shape
    bn, bk = n // scale_inv.shape[0], k // scale_inv.shape[1]
    wf = w.float().view(scale_inv.shape[0], bn, scale_inv.shape[1], bk)
    wf = wf * scale_inv.float().view(scale_inv.shape[0], 1,
                                     scale_inv.shape[1], 1)
    return wf.view(n, k).to(torch.bfloat16)


class MtpDraftHead(torch.nn.Module):
    """fc + one host-class decoder layer + norms; embed/head shared."""

    def __init__(self, model, lm, ckpt_dir, layer_slot: int):
        super().__init__()
        from safetensors import safe_open

        cfg = getattr(model.config, "text_config", model.config)
        full_idx = next(i for i, t in enumerate(cfg.layer_types)
                        if t == "full_attention")
        layer_cls = type(lm.layers[full_idx])
        norm_cls = type(lm.norm)
        hidden = int(cfg.hidden_size)
        dev = lm.norm.weight.device

        f = safe_open(str(ckpt_dir) + "/mtp.safetensors", "pt")
        t = {k[len("mtp."):]: f.get_tensor(k) for k in f.keys()}

        # assemble on CPU, move once — the draft loads while the host
        # still has headroom, and never doubles on the device
        self.layer = layer_cls(cfg, full_idx).to(torch.bfloat16)
        pre = "layers.0."
        with torch.no_grad():
            for name, mod in self.layer.named_modules():
                w = t.get(pre + name + ".weight")
                if w is None:
                    continue
                s = t.get(pre + name + ".weight_scale_inv")
                w = (_dequant_block_fp8(w, s) if s is not None
                     else w.to(torch.bfloat16))
                mod.weight.copy_(w)
            self.layer = self.layer.to(dev)
            self.fc = torch.nn.Linear(2 * hidden, hidden, bias=False,
                                      device=dev, dtype=torch.bfloat16)
            self.fc.weight.copy_(t["fc.weight"].to(torch.bfloat16))
            self.norm_h = norm_cls(hidden).to(dev, torch.bfloat16)
            self.norm_h.weight.copy_(
                t["pre_fc_norm_hidden.weight"].to(torch.bfloat16))
            self.norm_e = norm_cls(hidden).to(dev, torch.bfloat16)
            self.norm_e.weight.copy_(
                t["pre_fc_norm_embedding.weight"].to(torch.bfloat16))
            self.norm_out = norm_cls(hidden).to(dev, torch.bfloat16)
            self.norm_out.weight.copy_(t["norm.weight"].to(torch.bfloat16))
        self.slot = int(layer_slot)
        self._embed = lm.embed_tokens
        self._rotary = lm.rotary_emb
        self._head = model.lm_head
        self.eval()

    @torch.no_grad()
    def forward(self, prev_h, tok_ids, pos_t, cache, mask_row):
        """(logits, h_out) at ``pos_t``; writes the draft's KV slot."""
        e = self._embed(tok_ids)
        h = self.fc(torch.cat([self.norm_e(e), self.norm_h(prev_h)],
                              dim=-1))
        cache._cp = pos_t
        h = self.layer(h, position_embeddings=self._rotary(
                           h, pos_t.view(1, -1)),
                       attention_mask=mask_row,
                       past_key_values=_SlotView(cache, self.slot),
                       use_cache=True, cache_position=pos_t)
        h = self.norm_out(h)
        return self._head(h), h


class _SlotView:
    """Route the draft layer's cache traffic to its private slot."""

    def __init__(self, cache, slot):
        self._c = cache
        self._s = slot

    def update(self, k, v, layer_idx, cache_kwargs=None):
        return self._c.update(k, v, self._s, cache_kwargs)

    def __getattr__(self, name):
        return getattr(self._c, name)
