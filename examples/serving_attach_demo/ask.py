"""Streaming demo client with a live tok/s meter.

Feeds the server a real corpus (a directory of source files) up to
--ctx tokens, asks one question about it, and streams the answer with
a running decode-rate readout — the visible half of the demo.

  python ask.py --corpus <dir> --ctx 200000 \
      --question "Summarize the architecture of this codebase."
"""

import argparse
import json
import pathlib
import time
import urllib.request


def build_prompt(corpus: str, ctx_chars: int) -> str:
    parts, total = [], 0
    for p in sorted(pathlib.Path(corpus).rglob("*")):
        if p.suffix not in (".py", ".md", ".h", ".cu", ".cpp", ".txt"):
            continue
        try:
            t = p.read_text()
        except Exception:  # noqa: BLE001
            continue
        if not t.strip():
            continue
        parts.append(f"\n===== {p.name} =====\n{t}")
        total += len(t)
        if total >= ctx_chars:
            break
    return "".join(parts)[:ctx_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ctx", type=int, default=200000,
                    help="approximate prompt size in tokens")
    ap.add_argument("--question",
                    default="Summarize the architecture of this "
                            "codebase in ten bullet points.")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    body = build_prompt(args.corpus, args.ctx * 4)   # ~4 chars/token
    prompt = (f"Here is a codebase:\n{body}\n\n{args.question}\n")
    payload = {"model": "demo", "prompt": prompt, "stream": True,
               "max_tokens": args.max_tokens, "temperature": 0}
    req = urllib.request.Request(
        args.url + "/v1/completions", json.dumps(payload).encode(),
        {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    n = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            txt = chunk["choices"][0].get("text", "")
            if not txt:
                continue
            now = time.perf_counter()
            if first is None:
                first = now
                print(f"\n[TTFT {first - t0:.1f}s]\n", flush=True)
            n += 1
            rate = n / (now - first) if now > first else 0.0
            print(txt, end="", flush=True)
            if n % 32 == 0:
                print(f"  \033[36m[{rate:.0f} tok/s]\033[0m",
                      end="", flush=True)
    if first is not None:
        rate = n / (time.perf_counter() - first)
        print(f"\n\n[done: {n} tokens, decode {rate:.1f} tok/s, "
              f"TTFT {first - t0:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
