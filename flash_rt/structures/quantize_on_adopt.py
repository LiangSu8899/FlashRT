"""Quantize-on-adopt: full-precision checkpoints too big for the card.

``adopt_prequantized`` serves checkpoints that arrive already packed in
someone else's layout. This module is the door for the opposite
situation: the checkpoint is full precision and *cannot fit the card at
all*, but nearly all of its weight mass sits in one structure family —
a sparse-MoE expert bank. Quantizing that family once, at load time,
into structure impls brings the whole model into card budget while the
attention, norms, and router stay in the host's own precision.

The second format serves the dense sibling of that situation: a
full-precision checkpoint whose weight mass sits in ordinary 2-D linear
projections (attention q/k/v/o and MLP gate/up/down). Each projection
packs to the Hub NVFP4 layout through the same seam binder the
pre-quantized door uses, so both doors produce the same executable
form and everything downstream of adoption is shared. Two families are
deliberately not adopted here, each because it carries its own door:
projections living inside a gated-delta state layer (recognised by the
``conv1d`` + ``A_log`` profile) belong to the ``gated_delta_core``
scheme, which packs them together with the fused layer and owns their
release semantics; and a vocabulary projection (a Linear whose shape
mirrors an embedding in the same tree) is logits-family precision — a
separate, explicit binder decision, never a bulk-adoption side effect.
Scope is the module tree you hand in: pass the language stack, not the
whole multimodal shell, when towers outside it should keep their own
precision.

The calling convention matches the sibling door: the model is expected
CPU-resident straight from its loader; each expert bank streams through
the GPU in slabs as it packs, so peak footprint is the dense checkpoint
plus one slab. Move the model to the device *after* adopting — by then
the dense banks are gone and the remainder fits.

Like adoption of a pre-quantized checkpoint, this is a load-time
transform, not an attachment: the dense expert weights are released as
each bank binds (holding them would defeat the footprint the door
exists for), so there is no ``detach()`` — undoing an adoption is
reloading the checkpoint. The pack-and-unpack relative L2 of every bank
is recorded in the returned report; the receipt claims a measured
conversion loss, not the absence of one.
"""

from __future__ import annotations

import torch

from .prequantized import AdoptionReport

__all__ = ["quantize_on_adopt"]

_FORMATS = ("moe_experts_nvfp4", "linear_proj_nvfp4")


def _is_moe_expert_bank(module: torch.nn.Module) -> bool:
    """An expert bank: stacked 3D projections plus the activation, the
    shape contract ``gate_up_proj [E, 2I, H]`` / ``down_proj [E, H, I]``."""
    gu = getattr(module, "gate_up_proj", None)
    dn = getattr(module, "down_proj", None)
    if not (torch.is_tensor(gu) and torch.is_tensor(dn)):
        return False
    if gu.dim() != 3 or dn.dim() != 3 or not hasattr(module, "act_fn"):
        return False
    return (gu.shape[0] == dn.shape[0]
            and gu.shape[2] == dn.shape[1]
            and gu.shape[1] == 2 * dn.shape[2])


def _is_gated_delta_layer(module: torch.nn.Module) -> bool:
    """The state-layer profile (``conv1d`` + ``A_log``): its projections
    are packed by the ``gated_delta_core`` scheme together with the
    fused layer, never adopted piecemeal here."""
    return hasattr(module, "conv1d") and hasattr(module, "A_log")


def _vocab_signatures(model: torch.nn.Module) -> set:
    """Shapes of every embedding in the tree: a Linear mirroring one is
    a vocabulary projection (logits family), refused by this door."""
    return {(m.num_embeddings, m.embedding_dim)
            for m in model.modules()
            if isinstance(m, torch.nn.Embedding)}


def _adopt_linear_projections(model: torch.nn.Module,
                              report, *, verbose: bool) -> None:
    from .impls.linear_proj import nvfp4_dynamic

    vocab_sigs = _vocab_signatures(model)
    for name, module in list(model.named_modules()):
        if _is_gated_delta_layer(module):
            continue
        for child_name, child in list(module.named_children()):
            if not isinstance(child, torch.nn.Linear):
                continue
            w = getattr(child, "weight", None)
            if (w is None or w.dim() != 2
                    or not w.is_floating_point()
                    or hasattr(child, "weight_packed")):
                continue
            path = f"{name}.{child_name}" if name else child_name
            if tuple(w.shape) in vocab_sigs:
                report.retained[path] = "vocabulary projection"
                continue
            bias = getattr(child, "bias", None)
            try:
                bound, rel = nvfp4_dynamic.bind_proj_seam(
                    {"w": w.detach(),
                     "b": None if bias is None else bias.detach()})
            except ValueError as exc:
                report.retained[path] = str(exc)
                if verbose:
                    print(f"[quantize_on_adopt] {path}: retained "
                          f"({str(exc)[:80]})", flush=True)
                continue
            # release the dense projection before moving on: the
            # streaming bind is only slab-peak if retired weights go
            child.weight = None
            setattr(module, child_name, bound)
            report.replaced.append(path)
            report.conversion_rel_l2[path] = rel
            if verbose:
                print(f"[quantize_on_adopt] {path}: relL2={rel:.4f}",
                      flush=True)
    if not report.replaced:
        raise ValueError(
            "no dense projections found: this tree carries nothing "
            "linear_proj_nvfp4 adopts (all out of profile, or already "
            "packed)")


@torch.no_grad()
def quantize_on_adopt(model: torch.nn.Module,
                      fmt: str = "moe_experts_nvfp4", *,
                      verbose: bool = False) -> AdoptionReport:
    """Quantize every discovered expert bank of ``model`` into a
    structure impl; returns the adoption report for the receipt."""
    if fmt not in _FORMATS:
        raise ValueError(
            f"unknown quantize-on-adopt format {fmt!r}; supported: "
            f"{', '.join(_FORMATS)}")

    report = AdoptionReport(fmt=fmt)
    if fmt == "linear_proj_nvfp4":
        _adopt_linear_projections(model, report, verbose=verbose)
        # sibling projections that read the same normed hidden share one
        # activation quantization (identity-keyed, bit-identical by
        # construction) — the adopted checkpoint's default execution form
        from .impls.linear_proj import nvfp4_dynamic as _nv
        n_shared = _nv.link_shared_producers(model)
        torch.cuda.empty_cache()
        if verbose:
            print(f"[quantize_on_adopt] {report.summary()}; "
                  f"{n_shared} shared-producer groups", flush=True)
        return report

    from .impls.moe_experts import nvfp4_dynamic

    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not _is_moe_expert_bank(child):
                continue
            path = f"{name}.{child_name}" if name else child_name
            bound, rels = nvfp4_dynamic.bind_experts_seam(
                {"gate_up_proj": child.gate_up_proj.detach(),
                 "down_proj": child.down_proj.detach()},
                child.act_fn)
            # release the dense bank before moving on: the streaming
            # bind is only slab-peak if the retired experts actually go
            child.gate_up_proj = None
            child.down_proj = None
            setattr(module, child_name, bound)
            report.replaced.append(path)
            for stack, rel in rels.items():
                report.conversion_rel_l2[f"{path}.{stack}"] = rel
            if verbose:
                print(f"[quantize_on_adopt] {path}: "
                      + ", ".join(f"{k} relL2={v:.4f}"
                                  for k, v in rels.items()),
                      flush=True)
    if not report.replaced:
        raise ValueError(
            "no expert banks found: this model does not carry the "
            f"stacked-3D MoE structure {fmt} adopts")
    torch.cuda.empty_cache()
    if verbose:
        print(f"[quantize_on_adopt] {report.summary()}", flush=True)
    return report
