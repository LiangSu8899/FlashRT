#!/usr/bin/env python3
"""
FlashRT Thor — LIBERO benchmark.

Usage:
    # Quick test (3 tasks x 3 episodes):
    python examples/thor/eval_libero.py \
        --checkpoint /path/to/pi05_checkpoint \
        --task_suite libero_spatial --quick

    # JAX framework:
    python examples/thor/eval_libero.py \
        --checkpoint /path/to/orbax_checkpoint \
        --framework jax --task_suite libero_spatial --quick

    # Full evaluation (10 tasks x 50 episodes):
    python examples/thor/eval_libero.py \
        --checkpoint /path/to/pi05_checkpoint \
        --task_suite libero_spatial

    # LIBERO-10:
    python examples/thor/eval_libero.py \
        --checkpoint /path/to/pi05_checkpoint \
        --task_suite libero_10
"""

import argparse
import collections
import json
import logging
import os
import pathlib
import subprocess as _sp
import sys
import tempfile
import time
from datetime import datetime

import numpy as np

# MuJoCo rendering setup (must be before any GL imports)
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def _patch_egl_cleanup():
    """Patch EGL cleanup to prevent CUDA memory corruption on Jetson unified memory."""
    try:
        import robosuite.renderers.context.egl_context as _egl
        _egl.EGLGLContext.free = lambda self: None
        _egl.EGLGLContext.__del__ = lambda self: None
    except Exception:
        pass
    try:
        import robosuite.utils.binding_utils as _bu
        _bu.MjRenderContext.__del__ = lambda self: None
    except Exception:
        pass

_patch_egl_cleanup()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──
LIBERO_ENV_RESOLUTION = 256
DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS_DICT = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def resize_with_pad(img, target_h, target_w):
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    result = np.zeros((target_h, target_w, 3), dtype=img.dtype)
    result[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return result


def get_libero_env(task, resolution, seed):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task_bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def run_episode(env, model, task_description, max_steps, replan_steps=5, num_steps_wait=10, obs=None):
    """Run a single LIBERO episode."""
    action_plan = collections.deque()
    if obs is None:
        obs = env.reset()
    t = 0

    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, reward, done, info = env.step(DUMMY_ACTION)
            t += 1
            continue

        if len(action_plan) == 0:
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = resize_with_pad(img, 224, 224)
            wrist_img = resize_with_pad(wrist_img, 224, 224)

            actions = model.predict(
                images=[img, wrist_img],
                prompt=task_description,
            )
            action_chunk = actions[:replan_steps]
            action_plan.extend(action_chunk)

        action = action_plan.popleft()
        if hasattr(action, 'tolist'):
            action = action.tolist()
        obs, reward, done, info = env.step(action)
        if done:
            return True
        t += 1
    return False


def eval_single_task(args, task_id):
    """Evaluate a single task (called in subprocess)."""
    from libero.libero import benchmark
    import tqdm

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    max_steps = MAX_STEPS_DICT[args.task_suite]

    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    logger.info(f"Task {task_id}: {task_description}")

    # Load model via unified API
    import flash_rt
    model = flash_rt.load_model(
        checkpoint=args.checkpoint,
        framework=args.framework,
        num_views=2,
        autotune=args.autotune,
    )

    successes = 0
    latencies = []
    for trial in tqdm.tqdm(range(args.num_trials), desc=f"Task {task_id}"):
        env.reset()
        obs = env.set_init_state(initial_states[trial % len(initial_states)])

        t0 = time.perf_counter()
        success = run_episode(env, model, task_description, max_steps,
                              replan_steps=args.replan_steps, obs=obs)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        if success:
            successes += 1
        logger.info(f"  Trial {trial}: {'SUCCESS' if success else 'FAIL'} ({elapsed:.1f}s)")

    rate = successes / args.num_trials
    env.close()
    logger.info(f"Task {task_id}: {successes}/{args.num_trials} = {rate:.1%}")

    return {
        "task_id": task_id,
        "task_description": task_description,
        "successes": successes,
        "num_trials": args.num_trials,
        "success_rate": rate,
        "mean_episode_time": float(np.mean(latencies)),
    }


def main():
    parser = argparse.ArgumentParser(description="FlashRT LIBERO benchmark")
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint dir (safetensors for torch, Orbax for jax)")
    parser.add_argument("--task_suite", default="libero_spatial",
                        choices=list(MAX_STEPS_DICT.keys()))
    parser.add_argument("--framework", default="torch", choices=["torch", "jax"])
    parser.add_argument("--autotune", type=int, default=3,
                        help="CUDA Graph autotune trials (0=off, 3=default, 5=thorough)")
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true", help="Quick: 3 tasks x 3 trials")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        task_end = 3
        args.num_trials = 3
    else:
        from libero.libero import benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite]()
        task_end = task_suite.n_tasks

    if args.output is None:
        args.output = f"libero_{args.task_suite}_{args.framework}_results.json"

    print("=" * 60)
    print(f"FlashRT LIBERO Benchmark")
    print(f"  Suite:      {args.task_suite}")
    print(f"  Framework:  {args.framework}")
    print(f"  Tasks:      0..{task_end-1}")
    print(f"  Trials:     {args.num_trials}")
    print(f"  Autotune:   {args.autotune}")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 60)

    all_results = []
    total_s, total_e = 0, 0

    for tid in range(task_end):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            out_path = f.name

        env = os.environ.copy()
        env['_FLASHVLA_SUBTASK_OUTPUT'] = out_path
        if args.framework == "jax":
            env['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0'
            env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

        cmd = [sys.executable, __file__,
               '--checkpoint', args.checkpoint,
               '--framework', args.framework,
               '--autotune', str(args.autotune),
               '--task_suite', args.task_suite,
               '--num_trials', str(args.num_trials),
               '--replan_steps', str(args.replan_steps),
               '--seed', str(args.seed),
               '--_task_id', str(tid)]

        logger.info(f"Launching task {tid}...")
        ret = _sp.run(cmd, env=env, timeout=3600)
        if ret.returncode != 0:
            logger.error(f"Task {tid} failed (exit {ret.returncode})")
            continue

        try:
            with open(out_path) as f:
                result = json.load(f)
            all_results.append(result)
            total_s += result["successes"]
            total_e += result["num_trials"]
            logger.info(f"Task {tid}: {result['successes']}/{result['num_trials']} "
                        f"= {result['success_rate']:.1%} — {result['task_description']}")
        except Exception as e:
            logger.error(f"Task {tid}: failed to read results: {e}")
        finally:
            os.unlink(out_path)

    # Summary
    overall_rate = total_s / total_e if total_e > 0 else 0
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.task_suite} ({args.framework})")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  Task {r['task_id']:2d}: {r['successes']:2d}/{r['num_trials']:2d} "
              f"= {r['success_rate']:.1%}  {r['task_description']}")
    print(f"{'='*60}")
    print(f"  Overall: {total_s}/{total_e} = {overall_rate:.1%}")
    print(f"{'='*60}")

    output_data = {
        "task_suite": args.task_suite,
        "framework": args.framework,
        "checkpoint": args.checkpoint,
        "autotune": args.autotune,
        "overall_success_rate": overall_rate,
        "total_successes": total_s,
        "total_episodes": total_e,
        "tasks": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    if '--_task_id' in sys.argv:
        idx = sys.argv.index('--_task_id')
        task_id = int(sys.argv[idx + 1])
        sys.argv = sys.argv[:idx] + sys.argv[idx+2:]

        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--task_suite", default="libero_spatial")
        parser.add_argument("--framework", default="torch")
        parser.add_argument("--autotune", type=int, default=3)
        parser.add_argument("--num_trials", type=int, default=50)
        parser.add_argument("--replan_steps", type=int, default=5)
        parser.add_argument("--seed", type=int, default=7)
        args = parser.parse_args()

        result = eval_single_task(args, task_id)

        out_path = os.environ.get('_FLASHVLA_SUBTASK_OUTPUT')
        if out_path:
            with open(out_path, 'w') as f:
                json.dump(result, f)
    else:
        main()
