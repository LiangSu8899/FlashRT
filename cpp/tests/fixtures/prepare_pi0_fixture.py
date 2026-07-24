#!/usr/bin/env python3
"""Convert a LIBERO Pi0 .npz fixture into the PNG/bin/txt files the C++ engine
test consumes.

Output files (in --out-dir):
  image.png         primary camera, 224x224x3 uint8
  wrist_image.png   wrist camera, 224x224x3 uint8
  state.bin         action_dim float32 (real state zero-padded to action_dim)
  prompt.txt        task prompt, trailing newline trimmed

The .npz key names are probed defensively; pass --image-key / --wrist-key /
--state-key to override.
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--action-dim", type=int, default=32)
    ap.add_argument("--image-key", default=None)
    ap.add_argument("--wrist-key", default=None)
    ap.add_argument("--state-key", default=None)
    args = ap.parse_args()

    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow is required: pip install pillow")

    d = np.load(args.npz)
    keys = list(d.files)
    print(f"npz keys: {keys}")

    def pick(candidates, override):
        if override:
            if override not in keys:
                sys.exit(f"override key {override!r} not in npz; have {keys}")
            return d[override]
        for c in candidates:
            if c in keys:
                return d[c]
        sys.exit(f"none of {candidates} found in npz; have {keys}")

    img = pick(["image", "agentview_image", "observation.images.image"],
               args.image_key)
    wrist = pick(["wrist_image", "wristview_image",
                  "observation.images.wrist_image"], args.wrist_key)
    state = pick(["state", "robot_state", "observation.state", "qpos"],
                 args.state_key)

    def to_hwc_uint8(a):
        a = np.asarray(a)
        if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
            a = np.transpose(a, (1, 2, 0))  # CHW -> HWC
        a = a.astype(np.uint8, copy=False)
        if a.ndim == 2:  # grayscale -> RGB
            a = np.stack([a, a, a], axis=-1)
        if a.shape[-1] == 4:
            a = a[..., :3]
        return a

    img = to_hwc_uint8(img)
    wrist = to_hwc_uint8(wrist)
    print(f"image {img.shape} {img.dtype}; wrist {wrist.shape} {wrist.dtype}; "
          f"state {np.asarray(state).shape} {np.asarray(state).dtype}")

    os.makedirs(args.out_dir, exist_ok=True)
    Image.fromarray(img).save(os.path.join(args.out_dir, "image.png"))
    Image.fromarray(wrist).save(os.path.join(args.out_dir, "wrist_image.png"))

    state = np.asarray(state).astype(np.float32).reshape(-1)
    if state.size > args.action_dim:
        sys.exit(f"state has {state.size} values, more than action_dim="
                 f"{args.action_dim}")
    padded = np.zeros(args.action_dim, dtype=np.float32)
    padded[:state.size] = state
    padded.tofile(os.path.join(args.out_dir, "state.bin"))

    with open(args.prompt) as f:
        prompt = f.read().rstrip("\n")
    with open(os.path.join(args.out_dir, "prompt.txt"), "w") as f:
        f.write(prompt)

    print(f"wrote image.png, wrist_image.png, state.bin ({padded.size} f32), "
          f"prompt.txt to {args.out_dir}")


if __name__ == "__main__":
    main()
