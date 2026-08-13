"""Pi0.5 explicit structure pipeline — one seat book, every device.

The explicit tier of the three-tier rule: maximum replacement, built
entirely from structure primitives (region chains, lowering pins, the
cross-region wire, capture) — never a hand-written forward. Device
difference enters exactly once, at the kernel layer: each region walks
its band ladder in native-correspondence order and binds the first
band whose package facts qualify on this box. The same file serves
RTX 5090 and Thor; what differs is which rung answers.

Environment (same names as the auto probes):
    PI05_LEROBOT_SRC  lerobot source tree (PR #3870 line)
    PI05_CKPT         pi05_libero_pytorch checkpoint dir
    PI05_TOKENIZER    local paligemma tokenizer dir
    PI05_OBS_BUNDLE   real-observation bundle .pt

Usage:
    python pi05.py            # build, capture, min-of-3, parity
"""

from __future__ import annotations

import os
import pathlib
import statistics
import sys

import torch

_LEROBOT = os.environ.get("PI05_LEROBOT_SRC")
if _LEROBOT:
    sys.path.insert(0, _LEROBOT)
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parents[2]))

def _need(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"set {name} (see module docstring)")
    return value


CKPT = _need("PI05_CKPT")
VIEWS = int(os.environ.get("PI05_VIEWS", "2"))
OBS3 = os.environ.get("PI05_OBS3_NPZ")  # npz: img_0/wrist_0/wrist_right_0
TOKENIZER = _need("PI05_TOKENIZER")
BUNDLE = _need("PI05_OBS_BUNDLE")

#: the explicit seat book: per region family, the band ladder in
#: native-correspondence order. Binding facts (package symbols, shape
#: probes, smoke floors) decide which rung answers on this device —
#: the table itself is device-free.
SEAT_BOOK = {
    "adarms_stack": ("fp4_full", "fp4", "fp8"),
    "prefill_tower": ("fp4_p1", "fp4_p1_mid", "fp4", "fp8"),
    "vision_tower": ("fp4", "fp8"),
}

REGION_IMPLS = {
    "adarms_stack": ("flash_rt.structures.impls.adarms_stack.fp8_chain",
                     "bind_adarms_fp8_chain"),
    "prefill_tower": ("flash_rt.structures.impls.prefill_tower.fp8_chain",
                      "bind_prefill_fp8_chain"),
    "vision_tower": ("flash_rt.structures.impls.vision_tower.fp8_chain",
                     "bind_vision_fp8_chain"),
}


# ---------------------------------------------------------------------
# host loading — the pairing shims are signature-probed and no-ops on
# trees where the names already agree (see records: §61)
# ---------------------------------------------------------------------

def _shim_kernels_layer_pairing():
    try:
        import kernels.layer as kl
    except Exception:  # noqa: BLE001
        return

    def patch(cls):
        if cls is None or getattr(cls, "_frt_rev_default", False):
            return
        orig = cls.__init__

        def compat(self, *args, _orig=orig, **kwargs):
            try:
                _orig(self, *args, **kwargs)
            except ValueError:
                if (kwargs.get("revision") is not None
                        or kwargs.get("version") is not None):
                    raise
                kwargs.pop("version", None)
                kwargs["revision"] = "main"
                _orig(self, *args, **kwargs)

        cls.__init__ = compat
        cls._frt_rev_default = True

    for n in dir(kl):
        if n.endswith("Repository"):
            patch(getattr(kl, n, None))
    try:
        import kernels.layer.func as kf
        for n in dir(kf):
            if n.endswith("Repository"):
                patch(getattr(kf, n, None))
    except Exception:  # noqa: BLE001
        pass


def _shim_host_mask_api():
    import inspect

    import lerobot.policies.pi_gemma as pg
    fn = getattr(pg, "create_causal_mask", None)
    if fn is None or getattr(fn, "_frt_mask_shim", False):
        return
    if "cache_position" in inspect.signature(fn).parameters:
        return

    def compat(*args, **kwargs):
        kwargs.pop("cache_position", None)
        return fn(*args, **kwargs)

    compat._frt_mask_shim = True
    pg.create_causal_mask = compat


def _shim_vision_state_keys(policy) -> int:
    msd = policy.state_dict()
    want = [k for k in msd if ".vision_tower." in k
            and ".vision_model." not in k]
    if not want:
        return 0
    import safetensors.torch as st
    raw = st.load_file(f"{CKPT}/model.safetensors")
    fixed = {}
    for k, v in raw.items():
        if ".vision_model." not in k:
            continue
        flat = k.replace(".vision_model.", ".")
        for mk in want:
            if mk.endswith(flat) or flat.endswith(mk):
                fixed[mk] = v.to(msd[mk].dtype)
                break
    if fixed:
        policy.load_state_dict(fixed, strict=False)
    return len(fixed)


def load_host():
    _shim_kernels_layer_pairing()
    import json

    import transformers.utils as tu
    if not hasattr(tu, "torch_compilable_check"):
        tu.torch_compilable_check = (
            lambda *a, **k: a[0] if a and callable(a[0])
            else (lambda fn: fn))

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi052.configuration_pi052 import PI052Config
    from lerobot.policies.pi052.modeling_pi052 import PI052Policy
    _shim_host_mask_api()

    raw = json.load(open(f"{CKPT}/config.json"))
    cfg = PI052Config(
        action_expert_variant=raw.get("action_expert_variant",
                                      "gemma_300m"),
        paligemma_variant=raw.get("paligemma_variant", "gemma_2b"),
        dtype=raw.get("dtype", "float32"))
    policy = PI052Policy.from_pretrained(CKPT, config=cfg)
    n = _shim_vision_state_keys(policy)
    if n:
        print(f"[pi05] vision tower keys re-targeted: {n}", flush=True)
    feats = {
        "observation.images.image": PolicyFeature(
            FeatureType.VISUAL, (3, 224, 224)),
        "observation.images.image2": PolicyFeature(
            FeatureType.VISUAL, (3, 224, 224)),
        "observation.state": PolicyFeature(FeatureType.STATE, (32,)),
    }
    if VIEWS >= 3:
        feats["observation.images.image3"] = PolicyFeature(
            FeatureType.VISUAL, (3, 224, 224))
    elif VIEWS == 1:
        del feats["observation.images.image2"]
    policy.config.input_features = feats
    policy.config.output_features = {
        "action": PolicyFeature(FeatureType.ACTION, (32,))}
    policy.config.compile_model = False
    policy.config.dtype = "bfloat16"
    policy.model.paligemma_with_expert.to_bfloat16_for_selected_params(
        "bfloat16")
    return policy.to("cuda").eval()


def build_inputs(policy):
    import json

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS)

    raw = json.load(open(f"{CKPT}/config.json"))
    cfg05 = PI05Config(
        action_expert_variant=raw.get("action_expert_variant",
                                      "gemma_300m"),
        paligemma_variant=raw.get("paligemma_variant", "gemma_2b"),
        dtype=raw.get("dtype", "float32"))
    cfg05.input_features = policy.config.input_features
    cfg05.output_features = policy.config.output_features
    pre, _ = make_pre_post_processors(
        cfg05, pretrained_path=CKPT,
        preprocessor_overrides={"tokenizer_processor": {
            "tokenizer_name": TOKENIZER}})
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    obs = {"observation.state": bundle["state"].float().unsqueeze(0),
           "task": bundle["prompt"]}
    if VIEWS >= 3 and OBS3:
        # the native three-view protocol: image / wrist / wrist_right,
        # a real observation set (never synthetic views)
        import numpy as np
        fx = np.load(OBS3)
        for slot, key in (("image", "img_0"), ("image2", "wrist_0"),
                          ("image3", "wrist_right_0")):
            im = torch.from_numpy(fx[key]).float() / 255.0
            obs[f"observation.images.{slot}"] = (
                im.permute(2, 0, 1).unsqueeze(0).contiguous())
    else:
        imgs = ((bundle["input_images"].float() + 1.0) / 2.0).clamp(0, 1)
        imgs = imgs.permute(0, 3, 1, 2).contiguous()
        obs["observation.images.image"] = imgs[0:1]
        if VIEWS != 1:
            obs["observation.images.image2"] = imgs[1:2]
    batch = pre(obs)
    batch = {k: (v.to("cuda") if torch.is_tensor(v) else v)
             for k, v in batch.items()}

    with torch.no_grad():
        images, img_masks = policy._preprocess_images(dict(batch))
    n_views = len(images)
    imgs_win = torch.empty((n_views,) + tuple(images[0].shape[1:]),
                           device="cuda", dtype=images[0].dtype)
    for v in range(n_views):
        imgs_win[v].copy_(images[v][0])
    views = [imgs_win[v:v + 1] for v in range(n_views)]
    img_masks = [m.clone() for m in img_masks]
    tokens = batch[OBS_LANGUAGE_TOKENS].clone().contiguous()
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK].clone().contiguous()
    gen = torch.Generator(device="cuda").manual_seed(7)
    noise = torch.randn(1, policy.config.chunk_size,
                        policy.config.max_action_dim, device="cuda",
                        dtype=torch.float32, generator=gen)
    return views, img_masks, tokens, masks, noise


# ---------------------------------------------------------------------
# explicit build: pins + region ladder + wire
# ---------------------------------------------------------------------

def build(policy, hot):
    """Bind the explicit seat book; returns (handles, notes, undo)."""
    import importlib

    from flash_rt.structures import autobuild
    from flash_rt.structures.impls.graph_lowering.pi052_denoise import (
        Pi05DenoiseGraphLoweringAdapter)

    model = policy.model
    notes: dict = {"regions_bound": [], "regions_refused": []}
    undo: list = []

    lowering = Pi05DenoiseGraphLoweringAdapter().lower(model, hot)
    if lowering is not None:
        undo.append(lowering.undo)
        notes["pins"] = list(lowering.pins)
        print(f"[pi05] pins: {lowering.pins}", flush=True)

    extras = {"observed": {}, "revert": undo, "seams": [],
              "toggles": []}
    for family, ladder in SEAT_BOOK.items():
        mod_name, fn_name = REGION_IMPLS[family]
        mod = importlib.import_module(mod_name)
        region = importlib.import_module(
            mod_name.rsplit(".", 1)[0] + ".region")
        roots = region.identify(model)
        if not roots:
            notes["regions_refused"].append((family, "no root"))
            continue
        bind = getattr(mod, fn_name)
        for root in roots:
            bound = None
            for band in ladder:
                try:
                    result = bind(model, root, hot, band=band)
                except Exception as exc:  # noqa: BLE001 — next rung
                    result = {"refused": f"{type(exc).__name__}: {exc}"}
                if result and not result.get("refused"):
                    bound = (band, result)
                    break
                notes["regions_refused"].append(
                    (f"{family}:{band}",
                     str((result or {}).get("refused"))[:160]))
                print(f"[pi05] {family}:{band} refused: "
                      f"{str((result or {}).get('refused'))[:200]}",
                      flush=True)
            if bound is None:
                print(f"[pi05] {family}@{root}: every rung refused",
                      flush=True)
                continue
            band, result = bound
            extras["observed"].update(result.get("observed", {}))
            undo.extend(result.get("revert", ()))
            notes["regions_bound"].append(
                {"family": family, "root": root, "band": band,
                 "smoke": result.get("smoke_cos")})
            print(f"[pi05] {family}@{root}: {band} bound "
                  f"(smoke {result.get('smoke_cos'):.5f})", flush=True)

    autobuild._wire_region_products(
        extras, notes, lambda m: print(f"[pi05] {m}", flush=True))

    # ---- residual seats: everything the bound regions left behind --
    # The seat families fill the rest of the model (embeds, glue,
    # whatever tower a refused ladder left seated). Regions are pinned
    # seated for this pass so the scan never double-binds a tower the
    # book already owns, and seams under a bound root are dropped.
    from flash_rt.structures import auto_swaps, swap
    for family in SEAT_BOOK:
        os.environ["FRT_REGION_" + family.upper()] = "seated"
    roots_bound = [r["root"] for r in notes["regions_bound"]]

    def run_once():
        with torch.inference_mode():
            hot()

    plan = auto_swaps(model, run_once)
    keep = {path: mod for path, mod in plan.swaps.items()
            if not any(path == rt or path.startswith(rt + ".")
                       for rt in roots_bound)}
    handle = swap.attach(model, keep, observe=plan.observed,
                         revert=plan.revert)
    undo.append(handle.detach_all
                if hasattr(handle, "detach_all") else (lambda: None))
    notes["seats"] = len(keep)
    notes["seats_dropped_under_regions"] = len(plan.swaps) - len(keep)
    print(f"[pi05] seats: {len(keep)} attached "
          f"({len(plan.swaps) - len(keep)} claimed by regions)",
          flush=True)
    return extras, notes, undo


def main():
    from flash_rt import structures

    torch.manual_seed(0)
    policy = load_host()
    views, img_masks, tokens, masks, noise = build_inputs(policy)
    model = policy.model
    seed_noise = noise.clone()

    def hot():
        return model.sample_actions(views, img_masks, tokens, masks,
                                    noise=noise.clone())

    with torch.inference_mode():
        reference = hot().detach().float().clone()

    extras, notes, undo = build(policy, lambda: hot())

    with torch.inference_mode():
        treated = hot().detach().float().clone()
    eager_cos = torch.nn.functional.cosine_similarity(
        treated.flatten(), reference.flatten(), dim=0)
    print(f"[pi05] eager parity: {float(eager_cos):.6f}", flush=True)

    # the production window form (the shape the auto probe measures):
    # the noise buffer is a true in-out window — consumed as the seed,
    # rewritten with the actions, so a replay is one graph launch
    torch._dynamo.reset()
    noise_win = torch.zeros_like(seed_noise)

    def hot_body():
        acts = model.sample_actions(views, img_masks, tokens, masks,
                                    noise=noise_win)
        noise_win.copy_(acts)
        return noise_win

    hot_cap = torch.compile(hot_body)
    times = []
    stage = None
    for _ in range(3):
        noise_win.copy_(seed_noise)
        stage = structures.capture(hot_cap, model=policy,
                                   windows={"noise": noise_win},
                                   reference=None, gate_cos=0,
                                   min_speedup=0)
        vals = []
        for _ in range(9):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(5):
                stage.replay()
            e.record()
            torch.cuda.synchronize()
            vals.append(s.elapsed_time(e) / 5)
        times.append(round(statistics.median(vals), 3))
    noise_win.copy_(seed_noise)
    stage.replay()
    torch.cuda.synchronize()
    cap_cos = torch.nn.functional.cosine_similarity(
        noise_win.detach().float().flatten(), reference.flatten(),
        dim=0)
    print(f"[pi05] captured: {min(times)} ms (min of {times})  "
          f"parity {float(cap_cos):.6f}", flush=True)
    print(f"[pi05] book: {[(r['family'], r['band'])
                           for r in notes['regions_bound']]}",
          flush=True)


if __name__ == "__main__":
    main()
