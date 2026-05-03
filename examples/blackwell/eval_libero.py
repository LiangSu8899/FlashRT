#!/usr/bin/env python3
"""
FlashRT Blackwell — LIBERO benchmark pointer.

The unified ``flash_rt.load_model(...)`` API is hardware-agnostic, so
the Thor LIBERO evaluation script at ``examples/thor/eval_libero.py``
runs unchanged on Blackwell (RTX 5090, SM120) once cmake auto-detects
the right gencode flag. This stub prints the canonical command + a
minimal library-usage snippet so users land on the right entry point.

Usage:
    python examples/blackwell/eval_libero.py \
        --checkpoint /path/to/pi05_libero_pytorch \
        --task_suite libero_spatial
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="FlashRT Blackwell LIBERO benchmark pointer")
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

    print("\nThe shared LIBERO eval script lives at examples/thor/eval_libero.py")
    print("and runs unchanged on RTX 5090 (Blackwell, SM120):")
    print()
    print(f"  python examples/thor/eval_libero.py \\")
    print(f"    --checkpoint {args.checkpoint} \\")
    print(f"    --task_suite {args.task_suite} \\")
    print(f"    --framework torch \\")
    print(f"    --num_views {args.num_views}", end="")
    if args.quick:
        print(" \\\n    --quick")
    else:
        print()
    print()

    print("Library-usage snippet (matches README quickstart):")
    print("```python")
    print("import flash_rt")
    print()
    print("model = flash_rt.load_model(")
    print(f"    checkpoint=\"{args.checkpoint}\",")
    print("    framework=\"torch\",     # or \"jax\"")
    print("    config=\"pi05\",")
    print(f"    num_views={args.num_views},")
    print(")")
    print()
    print("# First call: ~3 s (calibration + CUDA Graph capture).")
    print("# Subsequent calls: graph replay (~17 ms on RTX 5090, 2 views).")
    print("actions = model.predict(images=[base_img, wrist_img],")
    print("                        prompt=\"pick up the red block\")")
    print("```")


if __name__ == '__main__':
    main()
