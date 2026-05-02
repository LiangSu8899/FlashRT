"""Normal A/B: autotune=3, warmup + 20 iter, 3 trials.

No hammering, production-settings."""
import gc, sys, time
import numpy as np
import torch


def cos(a, b):
    a = a.flatten().astype(np.float64); b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    np.random.seed(42)
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    wrist = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    obs = {"image": img, "wrist_image": wrist}
    CKPT = "<your_pi05_torch_ckpt>"

    from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
    from flash_rt.frontends.torch.pi05_thor_fp4 import Pi05TorchFrontendThorFP4

    for trial in range(3):
        print(f"\n── Trial {trial+1}/3 ──")
        pipe_A = Pi05TorchFrontendThor(CKPT, num_views=2, autotune=3)
        pipe_A.set_prompt("pick up the red block and place it in the tray")
        pipe_C = Pi05TorchFrontendThorFP4(CKPT, num_views=2, autotune=3,
                                           use_fp4_encoder_ffn=True, fp4_layers=(7,8,9))
        pipe_C.set_prompt("pick up the red block and place it in the tray")

        # Warmup
        for _ in range(5):
            pipe_A.infer(obs); pipe_C.infer(obs)
        torch.cuda.synchronize()

        # 20 iters interleaved
        at, ct = [], []
        for _ in range(20):
            t0 = time.perf_counter(); pipe_A.infer(obs); at.append((time.perf_counter()-t0)*1000)
            t0 = time.perf_counter(); pipe_C.infer(obs); ct.append((time.perf_counter()-t0)*1000)

        at = np.sort(at); ct = np.sort(ct)
        print(f"  base p50={at[10]:.2f}ms  p25={at[5]:.2f}  p75={at[15]:.2f}")
        print(f"  FP4  p50={ct[10]:.2f}ms  p25={ct[5]:.2f}  p75={ct[15]:.2f}")
        print(f"  Δ p50 = {ct[10]-at[10]:+.2f}ms")

        # precision check once
        if trial == 0:
            matched_noise = torch.from_numpy(np.load("/tmp/matched_noise.npy")).cuda()
            prod = np.load("/tmp/v_prod.npy")
            _orig = torch.Tensor.normal_
            def make_patch(p):
                def _p(self, *a, **kw):
                    if self.data_ptr() == p._g_noise.data_ptr():
                        self.copy_(matched_noise); return self
                    return _orig(self, *a, **kw)
                return _p
            torch.Tensor.normal_ = make_patch(pipe_A)
            a_out = pipe_A.infer(obs)["actions"]
            torch.Tensor.normal_ = _orig
            torch.Tensor.normal_ = make_patch(pipe_C)
            c_out = pipe_C.infer(obs)["actions"]
            torch.Tensor.normal_ = _orig
            print(f"  precision: base cos={cos(a_out, prod):.4f}  FP4 cos={cos(c_out, prod):.4f}")

        del pipe_A, pipe_C; gc.collect(); torch.cuda.empty_cache()

    return 0

if __name__ == '__main__':
    sys.exit(main())
