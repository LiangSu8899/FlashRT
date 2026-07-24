"""End-to-end smoke test for the Jetson-PI Pi0 provider through the Python
``flash_rt.load_model(framework="jetson_pi", ...)`` entry.

Skips (returns early) when the weights / fixture env vars are unset, so CI
without weights still passes.

Env:
  FLASHRT_PI0_MODEL        path to Pi0 policy GGUF
  FLASHRT_PI0_MMPROJ       path to VIT mmproj GGUF
  FLASHRT_PI0_FIXTURE_DIR  dir with image.png, wrist_image.png, state.bin, prompt.txt;
                          state.bin contains policy proprioception values
                          (8 floats for PI0.5), not action_dim padding
  FLASHRT_PI0_LIB          (optional) path to libflashrt_cpp_llama_cpp_provider_c.so
  FLASHRT_PI0_ACTION_STEPS required model action horizon (pi0_base is 50).
  FLASHRT_PI0_ACTION_DIM   required model action width (pi0_base is 32).
  FLASHRT_PI0_BACKEND      required backend for the Jetson-PI engine, e.g.
                            "cpu" or "cuda". CUDA_VISIBLE_DEVICES selects the
                            physical card for a CUDA run.

Run from the repository root after installing the provider or adding its
install directory to the platform dynamic-loader search path.
"""

import os
import sys


def _skip(msg):
    print(f"SKIP - {msg}")
    return 0


def main():
    model_env = os.environ.get("FLASHRT_PI0_MODEL")
    mmproj_env = os.environ.get("FLASHRT_PI0_MMPROJ")
    fixture_env = os.environ.get("FLASHRT_PI0_FIXTURE_DIR")
    backend = os.environ.get("FLASHRT_PI0_BACKEND")
    action_steps_env = os.environ.get("FLASHRT_PI0_ACTION_STEPS")
    action_dim_env = os.environ.get("FLASHRT_PI0_ACTION_DIM")
    if not model_env or not os.path.exists(model_env):
        return _skip("FLASHRT_PI0_MODEL not set or missing")
    if not mmproj_env or not os.path.exists(mmproj_env):
        return _skip("FLASHRT_PI0_MMPROJ not set or missing")
    if not fixture_env or not os.path.isdir(fixture_env):
        return _skip("FLASHRT_PI0_FIXTURE_DIR not set or missing")
    if not backend:
        return _skip("FLASHRT_PI0_BACKEND not set")
    if not action_steps_env or not action_dim_env:
        return _skip("FLASHRT_PI0_ACTION_STEPS/ACTION_DIM not set")
    for name in ("image.png", "wrist_image.png", "state.bin", "prompt.txt"):
        if not os.path.exists(os.path.join(fixture_env, name)):
            return _skip(f"fixture {name} missing in {fixture_env}")

    import numpy as np
    from PIL import Image

    action_steps = int(action_steps_env)
    action_dim = int(action_dim_env)

    import flash_rt
    model = flash_rt.load_model(
        model_env,
        framework="jetson_pi",
        mmproj_path=mmproj_env,
        backend=backend,
        num_views=2,
        action_steps=action_steps,
        action_dim=action_dim,
        lib_path=os.environ.get("FLASHRT_PI0_LIB"))

    image = np.asarray(Image.open(os.path.join(fixture_env, "image.png")).convert("RGB"), dtype=np.uint8)
    wrist = np.asarray(Image.open(os.path.join(fixture_env, "wrist_image.png")).convert("RGB"), dtype=np.uint8)
    with open(os.path.join(fixture_env, "state.bin"), "rb") as f:
        state = np.frombuffer(f.read(), dtype=np.float32)
    if state.size == 0:
        print("FAIL: state.bin must contain at least one float")
        return 1
    with open(os.path.join(fixture_env, "prompt.txt")) as f:
        prompt = f.read().rstrip("\n")

    actions = model.predict([image, wrist], prompt=prompt, state=state)

    failed = 0
    def check(cond, msg):
        nonlocal failed
        if cond:
            print(f"ok  : {msg}")
        else:
            print(f"FAIL: {msg}")
            failed = 1

    check(actions.shape == (action_steps, action_dim),
          f"actions shape == ({action_steps},{action_dim}), got {actions.shape}")
    if not np.any(np.isnan(actions)) and not np.any(np.isinf(actions)):
        check(True, "actions contain no NaN/Inf")
    else:
        check(False, "actions contain NaN/Inf")
    check(bool(np.any(actions != 0)), "actions are not all zero")

    model._pipe.context({"images": [image, wrist], "state": state})
    split_actions = model._pipe.action()
    check(np.array_equal(split_actions, actions),
          "context/action stages are bit-identical to predict")

    del model

    print("\n== JETSON_PI PYTHON " + ("PASSED" if not failed else "FAILED") + " ==")
    return failed


if __name__ == "__main__":
    sys.exit(main())
