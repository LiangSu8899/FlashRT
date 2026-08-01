#!/usr/bin/env python3
"""How many rows a warp should take, measured rather than assumed.

The choice trades three things against each other and they do not resolve the
same way on every part: more rows per warp means fewer redundant reads of the
activation, more registers, and fewer blocks. On a part whose limit is
bandwidth the register cost dominates and few rows win; on one whose limit is
arithmetic the redundancy dominates and more rows win. A value tuned on the
first part is the wrong value on the second, so this sweeps it in one run.

    python benchmarks/w4a16_packed_sweep.py
    python benchmarks/w4a16_packed_sweep.py --rows 1 2 4 --group 32

Each setting runs in its own process, because the kernel reads the choice from
the environment once.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# The projections a 4B decode step issues, and how many layers issue each.
SHAPES = [
    ("mlp gate+up", 18432, 2560, 32, True),
    ("mlp down", 2560, 9216, 32, False),
    ("gdn in_proj", 12288, 2560, 24, False),
    ("gdn out", 2560, 4096, 24, False),
    ("attn qkv", 10240, 2560, 8, False),
    ("attn o", 2560, 4096, 8, False),
]

CHILD = r'''
import json, os, sys, time
import torch
from flash_rt import flash_rt_kernels as fvk

shapes = json.loads(sys.argv[1])
group = int(sys.argv[2])
device = sys.argv[3]
stream = torch.cuda.current_stream(device).cuda_stream
out = {}
for name, n, k, _count, gated in shapes:
    packed = torch.randint(-(2 ** 31), 2 ** 31 - 1, (n, k // 8),
                           dtype=torch.int32, device=device)
    scale = (torch.rand(n, k // group, device=device) * 0.02).to(torch.bfloat16)
    x = torch.randn(1, k, dtype=torch.bfloat16, device=device)
    width = n // 2 if gated else n
    result = torch.empty(1, width, dtype=torch.bfloat16, device=device)
    call = (fvk.w4a16_packed_matvec_gated_bf16 if gated
            else fvk.w4a16_packed_matvec_bf16)

    def run():
        call(x.data_ptr(), packed.data_ptr(), scale.data_ptr(),
             result.data_ptr(), n, k, group, stream)

    for _ in range(30):
        run()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(200):
        run()
    torch.cuda.synchronize(device)
    seconds = (time.perf_counter() - started) / 200
    nbytes = n * k // 2 + n * (k // group) * 2
    out[name] = {"us": seconds * 1e6, "gb_s": nbytes / seconds / 1e9,
                 "bytes": nbytes}
    del packed, scale, x, result
print("RESULT " + json.dumps(out))
'''


def measure(rows: int | None, group: int, device: str) -> dict:
    environment = dict(os.environ)
    if rows is not None:
        environment["FLASHRT_W4A16_ROWS"] = str(rows)
    else:
        environment.pop("FLASHRT_W4A16_ROWS", None)
    finished = subprocess.run(
        [sys.executable, "-c", CHILD, json.dumps(SHAPES), str(group), device],
        capture_output=True, text=True, env=environment)
    for line in finished.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(finished.stderr.strip()[-2000:] or "no result")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--group", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    args = parser.parse_args()

    settings: list[tuple[str, int | None]] = [("default", None)]
    settings += [(f"rows={rows}", rows) for rows in args.rows]

    table: dict[str, dict] = {}
    for label, rows in settings:
        table[label] = measure(rows, args.group, args.device)
        print(f"measured {label}")

    names = [shape[0] for shape in SHAPES]
    print(f"\n{'setting':<10}" + "".join(f"{name:>14}" for name in names))
    for label in table:
        row = "".join(f"{table[label][name]['gb_s']:>13.0f} " for name in names)
        print(f"{label:<10}{row}")

    print("\nbest per shape:")
    for name in names:
        best = min(table, key=lambda label: table[label][name]["us"])
        rate = table[best][name]["gb_s"]
        against = table["default"][name]["gb_s"]
        print(f"  {name:<14}{best:<10}{rate:>7.0f} GB/s"
              f"   ({rate / against - 1:+.1%} against default)")

    # What the step would cost if every shape got its best setting.
    total = sum(min(table[label][name]["us"] for label in table)
                * next(s[3] for s in SHAPES if s[0] == name)
                for name in names)
    now = sum(table["default"][name]["us"]
              * next(s[3] for s in SHAPES if s[0] == name) for name in names)
    print(f"\nprojections per token: {now / 1e3:.2f} ms now, "
          f"{total / 1e3:.2f} ms if each shape took its best setting")
    print("A setting is per launch, so a shape can keep the one that suits it.")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(table, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
