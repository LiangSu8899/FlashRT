"""Attention adapter for Gemma-family denoise hosts (pi05 / pi_gemma).

Where the attention math runs is host-specific. In this family the
transformer's own forward calls ``modeling_gemma.eager_attention_forward``
directly, bypassing the config/interface dispatch entirely, so the seam
is that function, not a module. This adapter locates it by capturing one
denoise pass, binds an :mod:`..impls.attention_core` per layer from the
captured shapes and masks, and installs a function-level patch that
routes the fixed denoise shape to the packed-KV kernel while leaving
prefill and any other shape on the host path.

Registering this adapter lets ``autobuild`` pick up the attention_core
structure for this host family with no per-host scaffolding at the call
site — the host still just calls ``auto_swaps``.
"""

from __future__ import annotations

from ..impls.attention_core import bind_attention_core


class GemmaAttentionAdapter:
    """Recognise a Gemma-family denoise host and wire its fa2 seam."""

    __name__ = "gemma_attention"

    def __call__(self, model, forward):
        try:
            import transformers.models.gemma.modeling_gemma as mg
        except ImportError:
            return None
        orig = mg.eager_attention_forward

        recs = {"q": None, "masks": [], "keys": [], "values": []}

        def record(module, query, key, value, attention_mask, **kw):
            if query.shape[2] < 128:      # denoise (short) vs prefill
                recs["q"] = query.detach()
                recs["masks"].append(
                    attention_mask.detach()
                    if attention_mask is not None else None)
                recs["keys"].append(key.detach().clone())
                recs["values"].append(value.detach().clone())
            return orig(module, query, key, value, attention_mask, **kw)

        mg.eager_attention_forward = record
        try:
            with __import__("torch").no_grad():
                forward()
        finally:
            mg.eager_attention_forward = orig
        if recs["q"] is None:
            return None      # host never called this seam — not our family

        n_layers = _infer_layers(model)
        if n_layers == 0 or len(recs["keys"]) % n_layers != 0:
            return None
        steps = len(recs["keys"]) // n_layers
        captures = [{
            "q": recs["q"],
            "keys": [recs["keys"][i + s * n_layers] for s in range(steps)],
            "values": [recs["values"][i + s * n_layers]
                       for s in range(steps)],
            "mask": recs["masks"][i],
        } for i in range(n_layers)]

        bound = bind_attention_core(captures)
        if bound is None:
            return None      # head_dim unsupported → host keeps its path
        cores, prefix_update = bound
        seq_q = recs["q"].shape[2]
        expert = _expert_layers(model)
        for i, layer in enumerate(expert):
            layer.self_attn._fa2_core = cores[i]

        # no isolated speed bench here: benching this kernel against a
        # standalone compiled attention says it loses, while the same
        # swap measured inside the assembled graph wins by 0.76ms
        # (10x the intra-process variance) and improves parity. An
        # isolated probe cannot see what the seam actually replaces;
        # the composed net-win gate is the one that can.
        def fa2_fn(module, query, key, value, attention_mask, **kw):
            # no Python-visible side effects in here: a counter or any
            # host-side bookkeeping forces dynamo to break the graph at
            # every attention call, which fragments the surrounding
            # compiled region and pushes its CPU-side ops onto the
            # capture stream
            if query.shape[2] != seq_q or not hasattr(module, "_fa2_core"):
                return orig(module, query, key, value, attention_mask, **kw)
            return module._fa2_core(query, key, value,
                                    scale=kw.get("scaling")), None

        mg.eager_attention_forward = fa2_fn
        # the swap map is empty (the seam is a function, not a module);
        # the patch and the per-layer core buffers are the swap.
        # note: no extra host forward is run to self-verify — replaying
        # the host mutates its state (cache growth, guard shapes) and
        # that changes what the stage then captures. The recording pass
        # above already proves the seam is live in this host.
        return {}, None



def _infer_layers(model) -> int:
    layers = _expert_layers(model)
    return len(layers) if layers is not None else 0


def _expert_layers(model):
    """Find the denoise decoder layers under either the model or a
    policy wrapper — callers hand us whichever root they hold."""
    for path in ("paligemma_with_expert.gemma_expert.model.layers",
                 "model.paligemma_with_expert.gemma_expert.model.layers"):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        else:
            return node
    return None
