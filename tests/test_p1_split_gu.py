"""P1 split-GU validation: cos vs FP8 prod + latency A/B.

Three configs:
  (A) Pi0.5 FP8 baseline               — reference
  (B) Pi0.5 FP4 L7-9 + AWQ             — current production
  (C) Pi0.5 FP4 L7-9 + P1 split-GU     — new path

Gates:
  cos vs prod (B) >= 0.99
  cos vs prod (C) >= 0.99
  P50(C) <= P50(B) + 0.5 (allow some headroom; any ≤ saving is a win)

Run in container (Thor cool, serial):
    PYTHONPATH=<repo_root> \
      python3 tests/test_p1_split_gu.py
"""
from __future__ import annotations
import sys, time
import numpy as np
import torch


def cos(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_variant(label, build_fn):
    np.random.seed(42)
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    wrist = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    obs = {"image": img, "wrist_image": wrist}

    print(f"\n[{label}] Building...")
    t0 = time.perf_counter()
    pipe = build_fn()
    print(f"  build = {(time.perf_counter()-t0):.1f}s")
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


def cool(secs=60):
    print(f"  cool {secs}s..."); time.sleep(secs)


def main():
    from flash_rt.frontends.torch.pi05_thor_fp4 import Pi05TorchFrontendThorFP4
    CKPT = os.environ.get("FLASH_RT_PI05_CKPT", "/workspace/pytorch_checkpoints/pi05_libero_converted")
    prod = np.load("/tmp/v_prod.npy")
    v_torch = np.load("/tmp/v_torch.npy")

    # ─ Apples-to-apples: L7-9 + AWQ for both. Only path differs.
    # (Note: 18-layer scope requires AWQ on the Down channel; P1 geglu_two
    #  doesn't yet implement Down-AWQ, so we limit scope to L7-9 where AWQ
    #  is not strictly required for cos.)
    import os
    LAYERS_STR = os.environ.get("P1_LAYERS", "7,8,9")
    LAYERS = tuple(int(x) for x in LAYERS_STR.split(","))

    # (B') no-P1 baseline = current AWQ path
    b_act, b_p50 = run_variant(
        f"B': FP4 layers={LAYERS} AWQ=on  (current AWQ path)",
        lambda: Pi05TorchFrontendThorFP4(
            CKPT, num_views=2, autotune=3,
            use_fp4_encoder_ffn=True, fp4_layers=LAYERS,
            use_awq=True, awq_alpha=0.5, use_p1_split_gu=False))
    b_cos_t = cos(b_act, v_torch); b_cos_p = cos(b_act, prod)
    print(f"[B'] cos vs torch={b_cos_t:.6f}  cos vs prod={b_cos_p:.6f}  P50={b_p50:.2f}ms")

    # ─ Multi-trial A/B for tactic-noise filtering ─
    # Pattern: B C B C B C with cool between each
    b_p50s, c_p50s = [b_p50], []
    b_cos_p_first = b_cos_p
    c_cos_p_last = None

    for trial in range(3):
        cool(60)
        c_act, c_p50 = run_variant(
            f"C trial {trial+1}: FP4 layers={LAYERS} AWQ=on + P1 split-GU",
            lambda: Pi05TorchFrontendThorFP4(
                CKPT, num_views=2, autotune=3,
                use_fp4_encoder_ffn=True, fp4_layers=LAYERS,
                use_awq=True, awq_alpha=0.5, use_p1_split_gu=True))
        c_cos_t = cos(c_act, v_torch); c_cos_p = cos(c_act, prod)
        print(f"[C{trial+1}] cos vs torch={c_cos_t:.6f}  cos vs prod={c_cos_p:.6f}  P50={c_p50:.2f}ms")
        c_p50s.append(c_p50)
        c_cos_p_last = c_cos_p

        if trial < 2:
            cool(60)
            _, b_p50 = run_variant(
                f"B' trial {trial+2}: FP4 layers={LAYERS} AWQ=on  (current path)",
                lambda: Pi05TorchFrontendThorFP4(
                    CKPT, num_views=2, autotune=3,
                    use_fp4_encoder_ffn=True, fp4_layers=LAYERS,
                    use_awq=True, awq_alpha=0.5, use_p1_split_gu=False))
            print(f"[B'{trial+2}] P50={b_p50:.2f}ms")
            b_p50s.append(b_p50)

    b_med = sorted(b_p50s)[len(b_p50s)//2]
    c_med = sorted(c_p50s)[len(c_p50s)//2]
    print()
    print("="*60)
    print(f"  B' P50 trials: {[f'{x:.2f}' for x in b_p50s]}  median={b_med:.2f}ms")
    print(f"  C  P50 trials: {[f'{x:.2f}' for x in c_p50s]}  median={c_med:.2f}ms")
    print(f"  Median Δ (C - B'): {c_med - b_med:+.2f}ms")
    g1 = c_cos_t >= 0.99
    g2 = c_cos_p_last >= 0.99
    g3 = c_med < b_med
    print(f"  P1 cos vs torch >= 0.99   : {'PASS' if g1 else 'FAIL'} ({c_cos_t:.6f})")
    print(f"  P1 cos vs prod  >= 0.99   : {'PASS' if g2 else 'FAIL'} ({c_cos_p_last:.6f})")
    print(f"  P1 P50 median saving > 0   : {'PASS' if g3 else 'FAIL'} ({b_med-c_med:+.2f}ms)")
    return 0 if (g1 and g2 and g3) else 1


if __name__ == "__main__":
    sys.exit(main())
