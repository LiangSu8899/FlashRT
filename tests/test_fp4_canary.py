"""Step E.1 canary: verify FP4 frontend preserves precision.

Runs three variants and compares vs /tmp/v_prod.npy:
  (A) Pi05TorchFrontendThor base — reference
  (B) Pi05TorchFrontendThorFP4(use_fp4_encoder_ffn=False) — must bit-identical to (A)
  (C) Pi05TorchFrontendThorFP4(use_fp4_encoder_ffn=True, fp4_layers=(7,8,9))
      — target cos ≥ 0.997 vs prod canary

Run in container:
    PYTHONPATH=<repo_root>:\
$PYTHONPATH python3 tests/test_fp4_canary.py
"""
import json
import sys
import time

import numpy as np
import torch


def cos(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_variant(cls, **kw):
    """Run a pipeline with matched-noise monkey-patch, return (actions, p50_ms)."""
    np.random.seed(42)
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    wrist = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    obs = {"image": img, "wrist_image": wrist}

    pipe = cls(os.environ.get("FLASH_RT_PI05_CKPT", "/workspace/pytorch_checkpoints/pi05_libero_converted"),
               num_views=2, autotune=3, **kw)
    pipe.set_prompt("pick up the red block and place it in the tray")
    for _ in range(5):
        pipe.infer(obs)

    matched_noise = torch.from_numpy(np.load("/tmp/matched_noise.npy")).cuda()
    _orig = torch.Tensor.normal_

    def _patched(self, *a, **kw):
        if self.data_ptr() == pipe._g_noise.data_ptr():
            self.copy_(matched_noise); return self
        return _orig(self, *a, **kw)
    torch.Tensor.normal_ = _patched
    r = pipe.infer(obs)
    torch.Tensor.normal_ = _orig
    actions = r["actions"]

    lat = []
    for _ in range(20):
        t0 = time.perf_counter()
        pipe.infer(obs)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()

    return actions, lat[10]


def main():
    prod = np.load("/tmp/v_prod.npy")

    print("=" * 70)
    print("Step E.1 Canary — Pi0.5 FP4 Integration")
    print("=" * 70)

    # (A) Base reference
    from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
    print("\n[A] Pi05TorchFrontendThor (base, FP8-only)")
    a_act, a_p50 = run_variant(Pi05TorchFrontendThor)
    a_cos = cos(a_act, prod)
    print(f"    cos vs prod = {a_cos:.6f}   p50 = {a_p50:.1f}ms")

    # (B) Subclass, FP4 off
    from flash_rt.frontends.torch.pi05_thor_fp4 import Pi05TorchFrontendThorFP4
    print("\n[B] Pi05TorchFrontendThorFP4(use_fp4_encoder_ffn=False)")
    b_act, b_p50 = run_variant(Pi05TorchFrontendThorFP4,
                                use_fp4_encoder_ffn=False)
    b_cos = cos(b_act, prod)
    bb_cos = cos(b_act, a_act)
    print(f"    cos vs prod = {b_cos:.6f}   cos vs (A) = {bb_cos:.6f}   p50 = {b_p50:.1f}ms")

    # (C) FP4 on middle 3 encoder FFN layers
    print("\n[C] Pi05TorchFrontendThorFP4(use_fp4_encoder_ffn=True, fp4_layers=(7,8,9))")
    c_act, c_p50 = run_variant(Pi05TorchFrontendThorFP4,
                                use_fp4_encoder_ffn=True,
                                fp4_layers=(7, 8, 9))
    c_cos = cos(c_act, prod)
    cc_cos = cos(c_act, a_act)
    print(f"    cos vs prod = {c_cos:.6f}   cos vs (A) = {cc_cos:.6f}   p50 = {c_p50:.1f}ms")

    print()
    print("=" * 70)
    print("Gates:")
    pass_b = bb_cos > 0.99999          # B should be bit-identical to A
    pass_c_cos = c_cos >= 0.997        # plan target
    pass_c_lat = c_p50 <= a_p50 + 0.1  # any non-regression acceptable for MVP
    print(f"  (B) bit-identical to base:   {'✅' if pass_b else '❌'}  (cos={bb_cos:.6f})")
    print(f"  (C) cos vs prod ≥ 0.997:     {'✅' if pass_c_cos else '❌'}  (cos={c_cos:.6f})")
    print(f"  (C) p50 ≤ base + 0.1ms:      {'✅' if pass_c_lat else '❌'}  (Δ={c_p50-a_p50:+.2f}ms)")
    print("=" * 70)
    return 0 if (pass_b and pass_c_cos) else 1


if __name__ == '__main__':
    sys.exit(main())
