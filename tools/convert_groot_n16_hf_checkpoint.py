#!/usr/bin/env python3
"""Offline converter: HF GR00T N1.6 checkpoint -> FlashRT weight layout.

Phase-1b of the N1.6 parity plan. Converts a fine-tuned (or base) HF GR00T
N1.6 checkpoint into the exact tensor layout the FlashRT N1.6 kernels
consume, emitting BOTH the converted ``.safetensors`` and a JSON manifest
recording, for every tensor: HF key, HF shape, transform, FlashRT key,
FlashRT shape. This makes every weight mapping explicit and auditable, so a
backbone/weight-layout mismatch can be localized to a specific rule instead
of guessed from the final action.

Layout rules (one explicit rule per weight family):

  A. SigLIP2  (``backbone.model.vision_model.vision_model.*``)
       - attention q/k/v/o and FFN fc1/fc2: HF ``[out,in]`` -> ``[in,out]``
         (FlashRT GEMMs take ``[in,out]``); QKV kept separate, order Q,K,V.
       - layernorm / position_embedding / patch_embedding bias: passthrough.
       - (the double ``vision_model`` prefix is part of the HF key and kept).

  B. Qwen3  (``backbone.model.language_model.model.layers.*``)
       - q ``[2048,2048]``, k ``[1024,2048]``, v ``[1024,2048]`` are fused as
         ``cat([q,k,v], dim=0).T.contiguous()`` -> ``[2048, 4096]``
         (Q first, then K, then V; NO interleaving).
       - FFN ``cat([gate_proj, up_proj], dim=0).T.contiguous()`` with order
         ``gate | up`` (must not be swapped); down_proj transposed.
       - layernorm / q_norm / k_norm: passthrough.

  C. DiT  (``action_head.model.transformer_blocks.{l}.*``)
       - even block = cross-attention, odd block = self-attention.
       - self-attn (odd): QKV fused ``cat([q,k,v],0).T`` -> ``[1536, 4608]``
         (K/V input dim 1536).
       - cross-attn (even): q transposed; k/v ``[1536,2048]`` transposed,
         kept separate (NOT fused).
       - FFN is GELU (not GEGLU): net.0.proj and net.2 transposed.
       - norm1.linear (AdaLN) transposed; norm1.norm / norm3 / norm_out have
         NO affine parameters (absent from the checkpoint); output
         conditioning chunk order is (shift, scale).
       - proj_out_1 / proj_out_2 / timestep_encoder linears transposed.

  D. Embodiment  (``action_head.{action_encoder,state_encoder,action_decoder}``)
       - CategorySpecificLinear ``W`` is already ``[num_categories,in,out]``;
         after selecting ``W[eid]`` it is ``[in,out]`` already -> NO extra
         transpose. Biases passthrough.

Usage:
    python tools/convert_groot_n16_hf_checkpoint.py \
        --src /mnt/lerobot_so101_sim_v1_gr00t_n1d6_sim_fruits_cubes_10w \
        --dst /mnt/.../n1d6_flashrt_layout \
        [--dtype fp16]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_src(src: Path) -> dict:
    sd = {}
    for f in sorted(src.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                sd[k] = sf.get_tensor(k)
    return sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dtype", default="keep", choices=["keep", "fp16", "bf16"])
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    sd = load_src(src)

    cast = {"keep": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    def maybe_cast(t: torch.Tensor) -> torch.Tensor:
        return t.to(cast) if (cast is not None and t.is_floating_point()) else t

    out: dict[str, torch.Tensor] = {}
    manifest: list[dict] = []

    def emit(hf_key: str, fr_key: str, tensor: torch.Tensor, transform: str) -> None:
        hf_shape = list(sd[hf_key].shape) if hf_key in sd else None
        tensor = maybe_cast(tensor)
        out[fr_key] = tensor
        manifest.append({
            "hf_key": hf_key, "hf_shape": hf_shape, "transform": transform,
            "fr_key": fr_key, "fr_shape": list(tensor.shape),
        })

    T = lambda t: t.T.contiguous()  # [out,in] -> [in,out]

    for k, v in sd.items():
        # ── A. SigLIP2 ──
        if k.startswith("backbone.model.vision_model.vision_model."):
            if any(k.endswith(s) for s in (".to_q.weight", ".to_k.weight",
                                           ".to_v.weight", ".to_out.0.weight",
                                           ".mlp.fc1.weight", ".mlp.fc2.weight")):
                emit(k, k, T(v), "transpose[out,in]->[in,out]")
            else:
                emit(k, k, v, "passthrough")

        # ── B. Qwen3 ──
        elif k.startswith("backbone.model.language_model.model.layers."):
            if k.endswith(".self_attn.q_proj.weight"):
                # fuse when the sibling k/v are present (handled at q key)
                pre = k[: -len(".self_attn.q_proj.weight")]
                q = sd[f"{pre}.self_attn.q_proj.weight"]
                kk = sd[f"{pre}.self_attn.k_proj.weight"]
                vv = sd[f"{pre}.self_attn.v_proj.weight"]
                fused = torch.cat([q, kk, vv], dim=0).T.contiguous()
                emit(k, f"{pre}.self_attn.qkv_fused", fused,
                     "cat([q,k,v],0).T -> [in,out], order Q,K,V")
            elif k.endswith((".self_attn.k_proj.weight",
                             ".self_attn.v_proj.weight")):
                continue  # already fused into qkv_fused
            elif k.endswith(".self_attn.q_proj.bias"):
                pre = k[: -len(".self_attn.q_proj.bias")]
                q = sd.get(f"{pre}.self_attn.q_proj.bias")
                if q is not None:
                    kk = sd[f"{pre}.self_attn.k_proj.bias"]; vv = sd[f"{pre}.self_attn.v_proj.bias"]
                    emit(k, f"{pre}.self_attn.qkv_bias", torch.cat([q, kk, vv], 0),
                         "cat([qb,kb,vb],0)")
                else:
                    emit(k, k, v, "passthrough")
            elif k.endswith((".self_attn.k_proj.bias", ".self_attn.v_proj.bias")):
                pre = k[: -len(".self_attn.k_proj.bias")]
                if f"{pre}.self_attn.q_proj.bias" in sd:
                    continue  # fused into qkv_bias
                emit(k, k, v, "passthrough")
            elif k.endswith(".mlp.gate_proj.weight"):
                pre = k[: -len(".mlp.gate_proj.weight")]
                g = sd[f"{pre}.mlp.gate_proj.weight"]
                u = sd[f"{pre}.mlp.up_proj.weight"]
                emit(k, f"{pre}.mlp.gate_up_fused",
                     torch.cat([g, u], dim=0).T.contiguous(),
                     "cat([gate,up],0).T -> [in,out], order gate|up")
            elif k.endswith(".mlp.up_proj.weight"):
                continue  # fused into gate_up_fused
            elif k.endswith(".mlp.down_proj.weight"):
                emit(k, k, T(v), "transpose[out,in]->[in,out]")
            else:
                emit(k, k, v, "passthrough")

        # ── C. DiT ──
        elif k.startswith("action_head.model.transformer_blocks."):
            parts = k.split(".")
            l = int(parts[3])
            is_self = (l % 2 == 1)
            if k.endswith(".attn1.to_q.weight"):
                pre = k[: -len(".attn1.to_q.weight")]
                q = sd[f"{pre}.attn1.to_q.weight"]
                if is_self:
                    kk = sd[f"{pre}.attn1.to_k.weight"]
                    vv = sd[f"{pre}.attn1.to_v.weight"]
                    emit(k, f"{pre}.attn1.qkv_fused",
                         torch.cat([q, kk, vv], dim=0).T.contiguous(),
                         "self-attn cat([q,k,v],0).T -> [in,out]")
                else:
                    emit(k, k, T(q), "cross-attn q transpose")
            elif k.endswith(".attn1.to_k.weight") or k.endswith(".attn1.to_v.weight"):
                if is_self:
                    continue  # fused
                emit(k, k, T(v), "cross-attn k/v transpose [1536,2048]->[2048,1536]")
            elif k.endswith((".attn1.to_q.bias",)):
                pre = k[: -len(".attn1.to_q.bias")]
                q = sd[f"{pre}.attn1.to_q.bias"]; kk = sd[f"{pre}.attn1.to_k.bias"]; vv = sd[f"{pre}.attn1.to_v.bias"]
                if is_self:
                    emit(k, f"{pre}.qkv_bias", torch.cat([q, kk, vv], 0), "cat([qb,kb,vb],0)")
                else:
                    emit(k, k, v, "passthrough")
            elif k.endswith((".attn1.to_k.bias", ".attn1.to_v.bias")):
                if is_self:
                    continue
                emit(k, k, v, "passthrough")
            elif k.endswith(".ff.net.0.proj.weight") or k.endswith(".ff.net.2.weight") \
                    or k.endswith(".norm1.linear.weight") or k.endswith(".attn1.to_out.0.weight"):
                emit(k, k, T(v), "transpose[out,in]->[in,out] (GELU FFN, not GEGLU)")
            else:
                emit(k, k, v, "passthrough")

        # ── DiT top-level linears ──
        elif k.startswith("action_head.model.") and k.endswith(".weight") \
                and any(s in k for s in ("proj_out_1", "proj_out_2", "timestep_embedder")):
            emit(k, k, T(v), "transpose[out,in]->[in,out]")

        # ── D. Embodiment CategorySpecificLinear (NO transpose) ──
        elif k.startswith("action_head.") and any(
                s in k for s in ("action_encoder", "state_encoder", "action_decoder")):
            emit(k, k, v, "passthrough (W[eid] already [in,out]; no transpose)")

        # ── everything else (embeddings, vlln, mlp1, norms, etc.) ──
        else:
            emit(k, k, v, "passthrough")

    save_file(out, str(dst / "model_flashrt.safetensors"))
    with open(dst / "flashrt_layout_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {dst/'model_flashrt.safetensors'} ({len(out)} tensors)")
    print(f"wrote {dst/'flashrt_layout_manifest.json'} ({len(manifest)} rules)")
    # summary of transforms
    from collections import Counter
    c = Counter(m["transform"] for m in manifest)
    for t, n in c.most_common():
        print(f"  {n:5d}  {t}")


if __name__ == "__main__":
    main()
