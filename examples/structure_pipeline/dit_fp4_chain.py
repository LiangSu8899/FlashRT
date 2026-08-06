"""The DiT chain, transcribed from the native Thor pipeline.

The native FlashRT frontend (#163) runs every DiT GEMM as a
block-scaled NVFP4 GEMM with the elementwise neighbours fused into the
GEMM epilogues: the AdaLN and the pre-FFN norm emit FP4 directly, the
attention output projection and the FFN down-projection carry the
residual add in their epilogue, and the FFN up-projection emits
bias+GELU straight back to FP4 for the down GEMM. Eight kernels per
layer, no elementwise traffic between them — and the per-layer AdaLN
modulators come from one batched projection per step, not from a
per-layer linear.

This file is that chain as an explicit-tier specialist, transcribed at
the level the native code owns it: the DiT model's ``forward``. The
per-block chain alone measured slower than the seat assembly it
replaced (39.9 vs 35.8 ms on Thor) because each of the 128
block-step calls re-ran the tiny modulator linear and split the fused
QKV through three copies; the native form computes all 32 modulators
in one GEMM per step and reads Q/K/V as strided views.

Cross-attention K/V go through the block's own ``to_k``/``to_v``
calls — when the cadence family has seated its static banks there,
those calls return the per-frame buffers for free, and when it has
not, they compute; the chain does not care who holds the seat.

Two-phase on purpose: ``prepare`` quantizes weights from the pristine
host (before any seat holds copies), ``apply`` swaps the DiT forward
after the assembly is attached (so the cadence banks are already in
place). ``apply`` returns an undo callable restoring the host forward
bit-exact.
"""

from __future__ import annotations

from typing import Callable

import torch

from flash_rt.structures.impls import KernelUnavailable, hub_kernel


def _dit(model):
    dit = model.action_head.model
    blocks = list(dit.transformer_blocks)
    head = blocks[0]
    return dit, blocks, head.num_attention_heads, head.attention_head_dim


@torch.no_grad()
def prepare_dit_fp4_chain(model) -> dict | None:
    """Quantize every DiT GEMM weight to NVFP4 from the pristine host.

    Returns the per-block weight table (plus the stacked modulator
    projection), or a note dict when the hub packages are absent — the
    caller keeps the generic seats in that case.
    """
    try:
        kg = hub_kernel("flashrt/fp4-gemm", ">=1")
        kq = hub_kernel("flashrt/adaptive-layernorm-producers", ">=1")
    except KernelUnavailable as missing:
        return {"unavailable": str(missing)}
    for symbol in ("nvfp4_gemm_bias_bf16", "nvfp4_gemm_bias_residual_bf16",
                   "nvfp4_gemm_bias_gelu_nvfp4", "quantize_fp4_sfa_bf16"):
        if not hasattr(kg, symbol):
            return {"unavailable": f"fp4-gemm lacks {symbol}"}
    for symbol in ("ada_layer_norm_quant_nvfp4_swizzled_bf16",
                   "layer_norm_no_affine_quant_nvfp4_swizzled_bf16"):
        if not hasattr(kq, symbol):
            return {"unavailable": f"adaln producers lack {symbol}"}

    dit, blocks, nh, hd = _dit(model)

    def quant(w: torch.Tensor):
        packed, sfb = kg.quantize_fp4_sfa_bf16(
            w.detach().to("cuda", torch.bfloat16).contiguous(), is_sfb=True)
        return packed, sfb

    table = []
    for block in blocks:
        attn = block.attn1
        is_self = attn.to_k.in_features == attn.to_q.in_features
        entry = {"is_self": is_self}
        if is_self:
            w = torch.cat([attn.to_q.weight, attn.to_k.weight,
                           attn.to_v.weight], dim=0)
            entry["qkv"] = quant(w)
            entry["qkv_b"] = torch.cat(
                [attn.to_q.bias, attn.to_k.bias, attn.to_v.bias]
            ).detach().to("cuda", torch.bfloat16).contiguous()
        else:
            entry["q"] = quant(attn.to_q.weight)
            entry["q_b"] = attn.to_q.bias.detach().to(
                "cuda", torch.bfloat16).contiguous()
        entry["o"] = quant(attn.to_out[0].weight)
        entry["o_b"] = attn.to_out[0].bias.detach().to(
            "cuda", torch.bfloat16).contiguous()
        entry["up"] = quant(block.ff.net[0].proj.weight)
        entry["up_b"] = block.ff.net[0].proj.bias.detach().to(
            "cuda", torch.bfloat16).contiguous()
        entry["down"] = quant(block.ff.net[2].weight)
        entry["down_b"] = block.ff.net[2].bias.detach().to(
            "cuda", torch.bfloat16).contiguous()
        table.append(entry)

    # one modulator projection for all layers: silu(temb) @ [every
    # norm1 linear stacked] gives the 32 (scale, shift) rows in one GEMM
    ada_w = torch.cat([b.norm1.linear.weight for b in blocks], dim=0)
    ada_b = torch.cat([b.norm1.linear.bias for b in blocks], dim=0)
    torch.cuda.synchronize()
    return {"kg": kg, "kq": kq, "table": table, "nh": nh, "hd": hd,
            "ada_w": ada_w.detach().contiguous(),
            "ada_b": ada_b.detach().contiguous()}


@torch.no_grad()
def _step_tables(model, chain, probe):
    """Pre-compute the per-step modulators, native style.

    The AdaLN modulators and the output-projection styles depend only
    on the step's timestep — a fixed schedule — yet computing them
    in-graph reads the stacked modulator weights (~300 MB) once per
    step: the profile names that one skinny GEMM as the whole
    regression. So they are computed here once per timestep value seen
    in one probe forward, and the graph resolves the step by comparing
    the timestep against the table keys — an ``index_select``, exactly
    the step-table form the adaln_producer family already ships.
    """
    dit, blocks, _, _ = _dit(model)
    n_layers = len(blocks)
    dim = blocks[0].dim
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def note(_module, args, output):
        t = args[0].reshape(-1)[:1].float()
        if not any(torch.allclose(t, prev) for prev, _ in seen):
            seen.append((t.detach().clone(), output.detach().clone()))

    hook = dit.timestep_encoder.register_forward_hook(note)
    try:
        probe()
    finally:
        hook.remove()
    if not seen:
        raise RuntimeError("dit chain: probe saw no timestep encodings")

    silu = torch.nn.functional.silu
    keys, mods, tails = [], [], []
    for t, temb in seen:
        keys.append(t)
        mods.append(torch.nn.functional.linear(
            silu(temb), chain["ada_w"], chain["ada_b"])
            .view(n_layers, 2, dim).to(torch.bfloat16))
        shift, scale = dit.proj_out_1(silu(temb)).chunk(2, dim=1)
        tails.append(torch.stack(
            [shift.reshape(dim), scale.reshape(dim)]).to(torch.bfloat16))
    return (torch.cat(keys).contiguous(),
            torch.stack(mods).contiguous(),
            torch.stack(tails).contiguous())


def apply_dit_fp4_chain(model, chain: dict,
                        probe: Callable[[], object]) -> Callable[[], None]:
    """Swap the DiT model forward for the native kernel chain.

    ``probe`` runs one full host forward so the step tables can be
    captured. Returns an undo callable restoring the host forward.
    """
    kg, kq = chain["kg"], chain["kq"]
    nh, hd = chain["nh"], chain["hd"]
    table = chain["table"]
    dit, blocks, _, _ = _dit(model)
    n_layers = len(blocks)
    dim = blocks[0].dim
    t_keys, mods_table, tails_table = _step_tables(model, chain, probe)

    ada_fp4 = kq.ada_layer_norm_quant_nvfp4_swizzled_bf16
    ln_fp4 = kq.layer_norm_no_affine_quant_nvfp4_swizzled_bf16
    gemm_bias = kg.nvfp4_gemm_bias_bf16
    gemm_bias_res = kg.nvfp4_gemm_bias_residual_bf16
    gemm_gelu_fp4 = kg.nvfp4_gemm_bias_gelu_nvfp4
    quant_act = kg.quantize_fp4_sfa_bf16
    sdpa = torch.nn.functional.scaled_dot_product_attention
    kv_calls = [(b.attn1.to_k, b.attn1.to_v) for b in blocks]

    def layer(li, h, sa, scale, shift, enc, mask):
        entry = table[li]
        xp, xs = ada_fp4(h, scale, shift)
        if entry["is_self"]:
            qkv = gemm_bias(xp, entry["qkv"][0], xs, entry["qkv"][1],
                            entry["qkv_b"])
            packs = qkv.view(sa, 3, nh, hd).permute(1, 2, 0, 3)
            o = sdpa(packs[0].unsqueeze(0), packs[1].unsqueeze(0),
                     packs[2].unsqueeze(0))
        else:
            q = gemm_bias(xp, entry["q"][0], xs, entry["q"][1],
                          entry["q_b"])
            to_k, to_v = kv_calls[li]
            kb = to_k(enc)
            vb = to_v(enc)
            skv = kb.shape[-2]
            o = sdpa(q.view(1, sa, nh, hd).transpose(1, 2),
                     kb.view(1, skv, nh, hd).transpose(1, 2),
                     vb.view(1, skv, nh, hd).transpose(1, 2),
                     attn_mask=mask)
        o = o.transpose(1, 2).reshape(sa, dim).contiguous()
        op, osf = quant_act(o)
        h = gemm_bias_res(op, entry["o"][0], osf, entry["o"][1],
                          entry["o_b"], h)
        np_, ns = ln_fp4(h)
        hp, hs = gemm_gelu_fp4(np_, entry["up"][0], ns, entry["up"][1],
                               entry["up_b"])
        return gemm_bias_res(hp, entry["down"][0], hs, entry["down"][1],
                             entry["down_b"], h)

    every_n = getattr(dit, "attend_text_every_n_blocks", 2)

    def dit_forward(hidden_states, encoder_hidden_states, timestep=None,
                    encoder_attention_mask=None,
                    return_all_hidden_states: bool = False,
                    image_mask=None, backbone_attention_mask=None):
        t = timestep.reshape(-1)[:1].float()
        idx = (t_keys - t).abs().argmin().reshape(1)
        mods = mods_table.index_select(0, idx)[0]
        tail = tails_table.index_select(0, idx)[0]
        img = image_mask & backbone_attention_mask
        text = (~image_mask) & backbone_attention_mask
        img = img.reshape(1, 1, 1, -1)
        text = text.reshape(1, 1, 1, -1)

        b, sa, d = hidden_states.shape
        h = hidden_states.reshape(sa, d).to(torch.bfloat16).contiguous()
        all_h = [hidden_states] if return_all_hidden_states else None
        for li in range(n_layers):
            if li % 2 == 1:
                h = layer(li, h, sa, mods[li, 0], mods[li, 1], None, None)
            else:
                mask = text if li % (2 * every_n) == 0 else img
                h = layer(li, h, sa, mods[li, 0], mods[li, 1],
                          encoder_hidden_states, mask)
            if all_h is not None:
                all_h.append(h.reshape(b, sa, d))

        h = h.reshape(b, sa, d).type_as(hidden_states)
        h = (dit.norm_out(h) * (1 + tail[1].reshape(1, 1, d))
             + tail[0].reshape(1, 1, d))
        out = dit.proj_out_2(h)
        if return_all_hidden_states:
            return out, all_h
        return out

    saved = dit.forward
    dit.forward = dit_forward

    def undo():
        dit.forward = saved

    return undo
