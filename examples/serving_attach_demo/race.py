"""The context race: one growing conversation until the server's
ceiling — run once against each arm and cut the recordings side by
side (one 32 GB card cannot hold two copies of the weights at once).

Each turn appends the next slice of a real codebase and asks for the
running summary to be updated; with prefix caching on, every turn
prefills only its increment, so the pace stays conversational. The
stock arm dies at its ceiling with the server's own error — that row
prints as the crash line and the run stops; the attached arm walks on
to the model's native maximum.

  python race.py --arm stock  --corpus <repo>
  python race.py --arm attach --corpus <repo>
"""

import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request


def corpus_slices(corpus, chars_per_step):
    buf = []
    for p in sorted(pathlib.Path(corpus).rglob("*")):
        if p.suffix not in (".py", ".md", ".h", ".cu", ".cpp", ".txt"):
            continue
        try:
            t = p.read_text()
        except Exception:  # noqa: BLE001
            continue
        if t.strip():
            buf.append(f"\n===== {p.name} =====\n{t}")
    text = "".join(buf)
    for i in range(0, len(text), chars_per_step):
        yield text[i:i + chars_per_step]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("stock", "attach"),
                    required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--step-tokens", type=int, default=8000)
    ap.add_argument("--answer-tokens", type=int, default=64)
    ap.add_argument("--max-ctx", type=int, default=262144)
    args = ap.parse_args()

    slices = corpus_slices(args.corpus, args.step_tokens * 4)
    history = ""
    approx = 0
    print(f"{'turn':>4} {'~context':>9}  {'TTFT':>6}  {'rate':>9}  "
          f"status", flush=True)
    turn = 0
    while approx < args.max_ctx - args.step_tokens:
        try:
            history += next(slices)
        except StopIteration:
            break
        turn += 1
        approx += args.step_tokens
        prompt = (f"Codebase so far:\n{history}\n\nUpdate your "
                  f"one-paragraph running summary of everything "
                  f"above.\n")
        payload = {"model": "demo", "prompt": prompt,
                   "max_tokens": args.answer_tokens,
                   "min_tokens": max(16, args.answer_tokens // 2),
                   "temperature": 0}
        req = urllib.request.Request(
            args.url + "/v1/completions",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:120]
            print(f"{turn:>4} {approx:>9}  {'—':>6}  {'—':>9}  "
                  f"\033[31m💥 {e.code}: {msg}\033[0m", flush=True)
            print(f"\n[{args.arm}] ceiling at ~{approx} tokens.",
                  flush=True)
            return
        dt = time.perf_counter() - t0
        n = out["usage"]["completion_tokens"]
        rate = n / dt if dt > 0 else 0.0
        print(f"{turn:>4} {approx:>9}  {dt:>5.1f}s  "
              f"{rate:>6.1f}t/s  \033[32mok\033[0m", flush=True)
    print(f"\n[{args.arm}] reached ~{approx} tokens — native window "
          f"served end to end.", flush=True)


if __name__ == "__main__":
    main()
