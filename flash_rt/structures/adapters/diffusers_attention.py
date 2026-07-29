"""Attention adapter for Diffusers ``AttnProcessor2_0`` hosts."""

from __future__ import annotations

import torch

from ..impls.attention_core import bind_dense_attention


def _qkv(attn, hidden_states, encoder_hidden_states, attention_mask, temb):
    """Reproduce the projection half of Diffusers ``AttnProcessor2_0``."""
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)
    if hidden_states.ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(
            batch_size, channel, height * width).transpose(1, 2)
    batch_size, sequence_length, _ = (
        hidden_states.shape
        if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size)
        attention_mask = attention_mask.view(
            batch_size, attn.heads, -1, attention_mask.shape[-1])
    if attn.group_norm is not None:
        hidden_states = attn.group_norm(
            hidden_states.transpose(1, 2)).transpose(1, 2)
    query = attn.to_q(hidden_states)
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(
            encoder_hidden_states)
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)
    head_dim = key.shape[-1] // attn.heads
    query = query.view(
        batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(
        batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(
        batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)
    return query, key, value, attention_mask


class _Recorder:
    def __init__(self, original, rows):
        self.original = original
        self.rows = rows

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None,
        attention_mask=None, temb=None, *args, **kwargs,
    ):
        query, key, value, mask = _qkv(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        self.rows.append({
            "q": query.detach(),
            "key": key.detach(),
            "value": value.detach(),
            "mask": mask.detach() if mask is not None else None,
        })
        return self.original(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb,
            *args, **kwargs)


class _FlashRTAttnProcessor2_0:
    """Diffusers processor with only the SDPA body replaced by FA2."""

    def __init__(self, core):
        self.core = core

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None,
        attention_mask=None, temb=None, *args, **kwargs,
    ):
        del args, kwargs
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
        query, key, value, mask = _qkv(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        projection_dtype = query.dtype
        guard = getattr(self.core, "_frt_guard", None)
        accepted_dtypes = tuple(getattr(guard, "dtypes", ()) or ())
        if accepted_dtypes and projection_dtype not in accepted_dtypes:
            # A module swap upstream may change the projection output dtype
            # after this routed seam was observed. FA2 does not consume fp32;
            # adapt at the backend boundary and restore the host dtype before
            # the output projection.
            target_dtype = accepted_dtypes[0]
            query = query.to(target_dtype)
            key = key.to(target_dtype)
            value = value.to(target_dtype)
        hidden_states = self.core(query, key, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            query.shape[0], -1, attn.heads * query.shape[-1])
        hidden_states = hidden_states.to(projection_dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class DiffusersAttentionAdapter:
    """Route called Diffusers SDPA processors through stateless Hub FA2."""

    __name__ = "diffusers_attention"

    def __call__(self, model, forward, *, prefix_cadence: bool = False):
        del prefix_cadence
        sites = []
        for path, module in model.named_modules():
            processor = getattr(module, "processor", None)
            if type(processor).__name__ == "AttnProcessor2_0":
                sites.append((path, module, processor))
        if not sites:
            return None

        captures = [[] for _ in sites]
        for (_, module, original), rows in zip(sites, captures):
            module.processor = _Recorder(original, rows)
        try:
            with torch.no_grad():
                forward()
        finally:
            for _, module, original in sites:
                module.processor = original

        routes = []
        observed = {}
        for (path, module, original), rows in zip(sites, captures):
            if not rows:
                continue
            core = bind_dense_attention(rows)
            if core is None:
                continue
            routed = _FlashRTAttnProcessor2_0(core)
            routes.append((module, original, routed))
            observed[f"{path}.processor::fa2_core"] = core
        if not routes:
            return None

        def enable() -> None:
            for module, _, routed in routes:
                module.processor = routed

        def disable() -> None:
            for module, original, _ in routes:
                module.processor = original

        def revert() -> None:
            disable()

        enable()
        return {}, None, {
            "revert": [revert],
            "observed": observed,
            "toggle": (enable, disable),
        }
