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

        fired = {"n": 0}

        def fa2_fn(module, query, key, value, attention_mask, **kw):
            if query.shape[2] != seq_q or not hasattr(module, "_fa2_core"):
                return orig(module, query, key, value, attention_mask, **kw)
            fired["n"] += 1
            return module._fa2_core(query, key, value,
                                    scale=kw.get("scaling")), None

        mg.eager_attention_forward = fa2_fn
        # self-verify the patch actually takes over the seam on the same
        # forward that will be captured; if it never fires (the host's
        # captured entry uses a different attention shape than the one
        # we calibrated on) leave the host untouched rather than install
        # a patch that only adds a Python check and blocks fusion
        with __import__("torch").no_grad():
            forward()
        if fired["n"] == 0:
            mg.eager_attention_forward = orig
            for layer in expert:
                if hasattr(layer.self_attn, "_fa2_core"):
                    del layer.self_attn._fa2_core
            return None
        # the swap map is empty (the seam is a function, not a module);
        # the patch and the per-layer core buffers are the swap
        return {}, None


def _infer_layers(model) -> int:
    layers = _expert_layers(model)
    return len(layers) if layers is not None else 0


def _expert_layers(model):
    try:
        return model.paligemma_with_expert.gemma_expert.model.layers
    except AttributeError:
        return None
