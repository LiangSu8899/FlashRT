"""Structure discovery — find catalog structures inside a host model.

Walks the module tree and matches region-structure seams by shape, not
by model name: a gated gate/up/down MLP is a ``decoder_ffn`` seam, a
fc1/fc2 MLP with a sibling LayerNorm is a ``vision_ffn`` seam. The
result is the same information a hand-written binding file carries
(paths, dims, variant), derived from the model object itself; bindings
become generated receipts instead of required inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from torch import nn

_DECODER_PROJ = ("gate_proj", "up_proj", "down_proj")
_VISION_PROJ = (("fc1", "fc2"), ("linear_fc1", "linear_fc2"))
_NORM_ATTRS = ("post_attention_layernorm", "layer_norm2", "norm2")
_ATTN_PROJ = (("q_proj", "k_proj", "v_proj", "o_proj"),
              ("q_proj", "k_proj", "v_proj", "out_proj"))
_PROJ_WEIGHT_FLOOR = 262144  # candidacy filter only; impls add their own
                             # work-based qualification and gates decide


@dataclass
class Seam:
    """One replaceable site: a structure instance found in the host."""

    structure: str
    path: str                 # dotted path of the swappable module
    parent_path: str
    norm_attr: str | None
    dims: dict[str, int]
    variant: dict[str, str]
    fc_attrs: tuple[str, str] | None = None
    proj_attr: str | None = None      # linear_proj: attr name in parent
    family: str = ""
    layer_index: int = -1
    m_profile: list[int] = field(default_factory=list)


def _activation_of(module: nn.Module) -> str | None:
    fn = None
    for attr in ("act_fn", "activation_fn", "act", "activation"):
        fn = getattr(module, attr, None)
        if fn is not None:
            break
    if fn is None:
        return None
    label = " ".join(
        [getattr(fn, "__name__", ""), type(fn).__name__, repr(fn)]).lower()
    if "silu" in label or "swish" in label:
        return "silu"
    if "gelu" in label:
        return "gelu"
    return None


def _resolve(root: nn.Module, path: str) -> nn.Module:
    node = root
    for part in path.split("."):
        if part:
            node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node


def _family_key(path: str) -> tuple[str, int]:
    """Template the trailing layer index: a.layers.12.mlp -> a.layers.{i}.mlp."""
    matches = list(re.finditer(r"\.(\d+)\.", "." + path + "."))
    if not matches:
        return path, -1
    m = matches[-1]
    start, end = m.start(1) - 1, m.end(1) - 1  # offsets in original path
    return path[:start] + "{i}" + path[end:], int(m.group(1))


def _norm_attr_of(root: nn.Module, parent_path: str) -> str | None:
    try:
        parent = _resolve(root, parent_path)
    except (AttributeError, IndexError, KeyError):
        return None
    for attr in _NORM_ATTRS:
        if isinstance(getattr(parent, attr, None), nn.Module):
            return attr
    return None


def discover(
    model: nn.Module,
    structures: tuple[str, ...] = ("decoder_ffn", "vision_ffn"),
) -> list[Seam]:
    """Find every region-structure seam in ``model``."""
    seams: list[Seam] = []
    for path, module in model.named_modules():
        if not path:
            continue
        parent_path = path.rsplit(".", 1)[0] if "." in path else ""
        if "decoder_ffn" in structures and all(
            isinstance(getattr(module, a, None), nn.Linear)
            for a in _DECODER_PROJ
        ):
            gate = module.gate_proj
            act = _activation_of(module) or "silu"
            family, idx = _family_key(path)
            seams.append(Seam(
                structure="decoder_ffn", path=path, parent_path=parent_path,
                norm_attr=_norm_attr_of(model, parent_path),
                dims={"D": gate.in_features, "F": gate.out_features},
                variant={"activation": act, "norm_weight_mode": "direct"},
                family=family, layer_index=idx))
            continue
        if "linear_proj" in structures:
            for group in _ATTN_PROJ:
                projs = [getattr(module, a, None) for a in group]
                if not all(isinstance(p, nn.Linear) for p in projs):
                    continue
                for attr, proj in zip(group, projs):
                    if proj.weight.numel() < _PROJ_WEIGHT_FLOOR:
                        continue
                    family, idx = _family_key(path)
                    seams.append(Seam(
                        structure="linear_proj",
                        path=path + "." + attr, parent_path=path,
                        norm_attr=None, proj_attr=attr,
                        dims={"K": proj.in_features,
                              "N": proj.out_features},
                        variant={"bias": ("add" if proj.bias is not None
                                          else "none"),
                                 "epilogue": "none", "in_dtype": "bf16"},
                        family=family + "." + attr, layer_index=idx))
                break
        if "vision_ffn" in structures:
            for fc1_attr, fc2_attr in _VISION_PROJ:
                fc1 = getattr(module, fc1_attr, None)
                fc2 = getattr(module, fc2_attr, None)
                if not (isinstance(fc1, nn.Linear)
                        and isinstance(fc2, nn.Linear)):
                    continue
                if (fc1.out_features != fc2.in_features
                        or fc1.in_features != fc2.out_features
                        or fc1.bias is None or fc2.bias is None):
                    continue
                norm_attr = _norm_attr_of(model, parent_path)
                if norm_attr is None:
                    continue  # vision_ffn boundary includes a LayerNorm
                act = _activation_of(module) or "gelu"
                family, idx = _family_key(path)
                seams.append(Seam(
                    structure="vision_ffn", path=path,
                    parent_path=parent_path, norm_attr=norm_attr,
                    dims={"D": fc1.in_features, "F": fc1.out_features},
                    variant={"activation": act}, fc_attrs=(fc1_attr, fc2_attr),
                    family=family, layer_index=idx))
                break
    return seams


def group_families(seams: list[Seam]) -> dict[str, list[Seam]]:
    """Group seams into families (same template path), index-sorted."""
    families: dict[str, list[Seam]] = {}
    for seam in seams:
        families.setdefault(seam.family, []).append(seam)
    for members in families.values():
        members.sort(key=lambda s: s.layer_index)
    return families


def seam_weights(model: nn.Module, seam: Seam) -> dict[str, torch.Tensor]:
    """Extract the impl-facing weight dict for one seam."""
    module = _resolve(model, seam.path)
    norm = (_resolve(model, seam.parent_path + "." + seam.norm_attr)
            if seam.norm_attr else None)
    if seam.structure == "linear_proj":
        return {"w": module.weight.detach(),
                "b": (module.bias.detach()
                      if module.bias is not None else None)}
    if seam.structure == "decoder_ffn":
        w_norm = (norm.weight.detach() if norm is not None
                  and getattr(norm, "weight", None) is not None
                  else torch.ones(seam.dims["D"]))
        return {
            "w_norm": w_norm,
            "w_gate": module.gate_proj.weight.detach().t().contiguous(),
            "w_up": module.up_proj.weight.detach().t().contiguous(),
            "w_down": module.down_proj.weight.detach().t().contiguous(),
        }
    fc1_attr, fc2_attr = seam.fc_attrs
    fc1, fc2 = getattr(module, fc1_attr), getattr(module, fc2_attr)
    return {
        "w_norm": norm.weight.detach(),
        "b_norm": norm.bias.detach(),
        "w_fc1": fc1.weight.detach(),
        "b_fc1": fc1.bias.detach(),
        "w_fc2": fc2.weight.detach(),
        "b_fc2": fc2.bias.detach(),
    }
