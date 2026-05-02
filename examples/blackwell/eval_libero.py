#!/usr/bin/env python3
"""
FlashRT Blackwell — LIBERO benchmark.

Uses Pi05FastInference backend (pybind11 + FlashAttention) on RTX 5090.

Usage:
    python examples/blackwell/eval_libero.py \
        --checkpoint /path/to/pi05_libero_pytorch \
        --task_suite libero_spatial

For the full LIBERO evaluation with env integration, see:
    test-5090/libero_eval.py
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def main():
    parser = argparse.ArgumentParser(description="FlashRT Blackwell LIBERO benchmark")
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--task_suite', default='libero_spatial',
                        choices=['libero_spatial', 'libero_object', 'libero_goal',
                                 'libero_10', 'libero_90'])
    parser.add_argument('--num_views', type=int, default=2)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print(f"FlashRT Blackwell — LIBERO {args.task_suite}")
    print("=" * 60)

    print(f"\nFor full LIBERO evaluation, use the production script:")
    print(f"  cd test-5090/")
    print(f"  python libero_eval.py \\")
    print(f"    --checkpoint {args.checkpoint} \\")
    print(f"    --task_suite {args.task_suite}")
    if args.quick:
        print(f"    --quick")
    print()

    print("Library usage example:")
    print("```python")
    print("from flash_rt.backends.x86_sm120 import Pi05FastInference")
    print()
    print("model = Pi05FastInference(")
    print("    checkpoint=checkpoint_dict,")
    print(f"    num_views={args.num_views},")
    print("    chunk_size=10,")
    print(")")
    print("model.record_infer_graph()")
    print()
    print("# Per-step inference in LIBERO loop:")
    print("actions = model.forward(")
    print("    observation_images_normalized=obs_images,")
    print("    diffusion_noise=noise,")
    print("    task_prompt='pick up the red block',")
    print("    state_tokens=state,")
    print(")")
    print("```")


if __name__ == '__main__':
    main()
