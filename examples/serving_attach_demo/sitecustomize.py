"""Attach hook for a stock ``vllm serve`` process, env-gated.

Put this file's directory on PYTHONPATH and set FRT_VLLM_ATTACH=1;
the server's processes install the load hook on their own. Without
the gate the file is inert.
"""
import os

if os.environ.get("FRT_VLLM_ATTACH") == "1":
    import sys
    p = os.environ.get("FRT_VLLM_STRUCTURES_PATH")
    if p and p not in sys.path:
        sys.path.insert(0, p)
    try:
        from flash_rt.structures.adapters import vllm_engine
        vllm_engine.install_load_hook(
            verbose=True,
            seats=vllm_engine.DENSE_SEAT_SUFFIXES,
            precision=os.environ.get("FRT_VLLM_PRECISION", "auto"),
            consume=os.environ.get("FRT_VLLM_CONSUME", "1") == "1",
            seat_draft=False,
            head=os.environ.get("FRT_VLLM_HEAD", "1") == "1",
            fused_mlp=True)
        print("[flash_rt] vllm serve attach hook installed", flush=True)
    except Exception as e:  # noqa: BLE001 — the server must still boot
        print(f"[flash_rt] attach hook failed: {e!r}", flush=True)
