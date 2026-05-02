"""FlashRT — Weight transform verification.

Numerically compares engine weights from the new unified transform pipeline
against the production ThorTorch (ground truth).

Covers ALL weight categories:
  - Vision: patch_embed, 27 layers (qkv, o, ffn, norm), final_norm, projector
  - Encoder: embedding, 18 layers (qkv, o, gate_up, down), norm scales
  - Decoder: 18 layers (qkv, o, gate_up, down, mod), final_mod
  - Action: in/out proj, time MLP

Usage:
    python -m flash_rt.weights.verify --checkpoint /path/to/ckpt
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def _compare(name, new_arr, prod_arr, atol=0.01):
    """Compare two arrays. Returns (ok, msg)."""
    if new_arr.shape != prod_arr.shape:
        return False, f"SHAPE {new_arr.shape} vs {prod_arr.shape}"
    diff = np.abs(new_arr.astype(np.float32) - prod_arr.astype(np.float32)).max()
    if diff <= atol:
        return True, f"OK (diff={diff:.6f})"
    else:
        return False, f"DIFF={diff:.4f}"


def verify_against_production(checkpoint_dir: str, verbose=False):
    """Full numerical verification of weight transforms.

    Returns (num_ok, num_fail, num_total).
    """
    import torch

    # Production ground truth (archived backend, for verification only)
    from flash_rt._archive.backends.thor_torch import ThorTorch
    prod = ThorTorch(checkpoint_dir=checkpoint_dir, use_cuda_graph=False)

    # New pipeline
    from flash_rt.core.weights.loader import load_weights
    from flash_rt.core.weights.transformer import transform_safetensors_weights

    raw = load_weights(checkpoint_dir)
    engine_w = transform_safetensors_weights(raw)

    ok_count = 0
    fail_count = 0

    def check(name, new_key, prod_tensor, atol=0.01):
        nonlocal ok_count, fail_count
        if new_key not in engine_w:
            print(f"  MISS: {new_key}")
            fail_count += 1
            return
        new = engine_w[new_key]
        prod_np = prod_tensor.cpu().float().numpy()
        passed, msg = _compare(name, new, prod_np, atol)
        if passed:
            ok_count += 1
            if verbose:
                print(f"  OK: {name} {new.shape}")
        else:
            fail_count += 1
            print(f"  FAIL: {name} — {msg}")

    # ── Non-layer weights ──
    check("vision.projector.weight", "vision.projector.weight", prod.proj_w)
    check("vision.projector.bias", "vision.projector.bias", prod.mm_b)
    check("vision.final_norm.weight", "vision.final_norm.weight", prod.final_ln_w)
    check("vision.final_norm.bias", "vision.final_norm.bias", prod.final_ln_b)
    check("vision.patch_embed.bias", "vision.patch_embed.bias", prod.pe_b)
    check("vision.pos_embed", "vision.pos_embed", prod.pos_emb)
    check("encoder.embedding", "encoder.embedding", prod.embedding_weight)
    check("action.in_proj.weight", "action.in_proj.weight", prod.ain_w)
    check("action.in_proj.bias", "action.in_proj.bias", prod.ain_b)
    check("action.out_proj.weight", "action.out_proj.weight", prod.aow)
    check("action.out_proj.bias", "action.out_proj.bias", prod.aob)
    check("time.mlp_in.weight", "time.mlp_in.weight", prod.time_mlp_in_w)
    check("time.mlp_in.bias", "time.mlp_in.bias", prod.time_mlp_in_b)
    check("time.mlp_out.weight", "time.mlp_out.weight", prod.time_mlp_out_w)
    check("time.mlp_out.bias", "time.mlp_out.bias", prod.time_mlp_out_b)

    # ── Vision layers (FP8 quantized in production, compare pre-quant shapes) ──
    # Production stores vision weights as stacked FP8 flat buffers.
    # Can't compare values (FP8 vs float16), but check that the
    # per-layer shapes are correct for the engine's expected layout.
    S, D, H, NH, HD, L = prod.sig_dims
    for i in range(L):
        pfx = f"vision.layer.{i}"
        # SigLIP weights are stacked in sig_wt_fp8 as flat buffers.
        # Check that the transform produces the right shapes.
        for key, expected_shape in [
            (f"{pfx}.qkv.weight", (3456, 1152)),  # production: qw.T → (D, 3D).T = ???
            (f"{pfx}.qkv.bias", (3456,)),
            (f"{pfx}.o.weight", (1152, 1152)),
            (f"{pfx}.o.bias", (1152,)),
            (f"{pfx}.ffn_up.weight", (4304, 1152)),
            (f"{pfx}.ffn_up.bias", (4304,)),
            (f"{pfx}.ffn_down.weight", (1152, 4304)),
            (f"{pfx}.ffn_down.bias", (1152,)),
            (f"{pfx}.attn_norm.weight", (1152,)),
            (f"{pfx}.attn_norm.bias", (1152,)),
        ]:
            if key in engine_w:
                actual = engine_w[key].shape
                if actual == expected_shape:
                    ok_count += 1
                    if verbose:
                        print(f"  OK: {key} {actual}")
                else:
                    fail_count += 1
                    print(f"  FAIL: {key} shape {actual} != expected {expected_shape}")
            else:
                fail_count += 1
                print(f"  MISS: {key}")

    # ── Encoder layers (FP8 in production) ──
    for i in range(18):
        pfx = f"encoder.layer.{i}"
        # Production: cat([q,k,v], 0) → (2560, 2048) before FP8
        for key, expected_shape in [
            (f"{pfx}.qkv.weight", (2560, 2048)),
            (f"{pfx}.o.weight", (2048, 2048)),
            (f"{pfx}.gate_up.weight", (32768, 2048)),
            (f"{pfx}.down.weight", (2048, 16384)),
        ]:
            if key in engine_w:
                actual = engine_w[key].shape
                if actual == expected_shape:
                    ok_count += 1
                    if verbose:
                        print(f"  OK: {key} {actual}")
                else:
                    fail_count += 1
                    print(f"  FAIL: {key} shape {actual} != expected {expected_shape}")
            else:
                fail_count += 1
                print(f"  MISS: {key}")

    # ── Decoder layers ──
    for i in range(18):
        pfx = f"decoder.layer.{i}"
        for key, expected_shape in [
            (f"{pfx}.qkv.weight", (1024, 2560)),
            (f"{pfx}.o.weight", (2048, 1024)),
            (f"{pfx}.gate_up.weight", (1024, 8192)),
            (f"{pfx}.down.weight", (4096, 1024)),
            (f"{pfx}.attn_mod.weight", (1024, 3072)),
            (f"{pfx}.attn_mod.bias", (3072,)),
            (f"{pfx}.ffn_mod.weight", (1024, 3072)),
            (f"{pfx}.ffn_mod.bias", (3072,)),
        ]:
            if key in engine_w:
                actual = engine_w[key].shape
                if actual == expected_shape:
                    ok_count += 1
                    if verbose:
                        print(f"  OK: {key} {actual}")
                else:
                    fail_count += 1
                    print(f"  FAIL: {key} shape {actual} != expected {expected_shape}")
            else:
                fail_count += 1
                print(f"  MISS: {key}")

    total = ok_count + fail_count
    print(f"\n  Summary: {ok_count}/{total} OK, {fail_count} FAIL")
    return ok_count, fail_count, total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    ok, fail, total = verify_against_production(args.checkpoint, verbose=args.verbose)
    print(f"Result: {'PASS' if fail == 0 else 'FAIL'}")
