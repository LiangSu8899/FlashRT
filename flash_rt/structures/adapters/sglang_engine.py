"""SGLang engine family: explicit assembly across a process boundary.

SGLang shares vLLM's module lineage — projections return
``(out, bias)``, seams answer the same structural predicates — but its
scheduler is a **spawned subprocess**: a patch applied in the launcher
process never reaches the model. The carrier that does is
``sitecustomize``: :func:`install` writes a hook module into a
temporary directory, prepends it to ``PYTHONPATH``, and every
interpreter the engine spawns runs it at startup. The hook is gated on
an environment flag and is inert anywhere else.

Two further engine facts ride here:

- **Quantized checkpoints hold FP8 block weights.** A dense binder fed
  raw FP8 bytes produces garbage that no bind-time smoke can catch
  (finite, right shape, wrong scale). Dense seating therefore
  dequantizes through the module's own ``weight_scale_inv`` in row
  slabs before packing — cross-format regrids carry their scale
  semantics or they do not run.
- **The radix prefix cache breaks repeat-determinism on hybrid
  (linear-attention) models** — measured, host-side, seam exonerated.
  Serving such a model with seats attached should disable the radix
  cache until the host fixes the interplay.

Usage, before constructing the engine::

    from flash_rt.structures.adapters import sglang_engine
    sglang_engine.install()
    llm = sgl.Engine(model_path=...)
    llm.generate(...)

Scope: dense projection seams (the measured 2x-class win on this
engine). The fused-MoE and LM-head surfaces differ from vLLM's and are
refused until profiled, not approximated.
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import torch

_ATTACH_FLAG = "FRT_SGLANG_ATTACH"
_PATH_VAR = "FRT_SGLANG_STRUCTURES_PATH"
_SEATS_VAR = "FRT_SGLANG_SEATS"

#: dense projection seams by dataflow position; the qwen3_5 family rows
#: plus the engine-wide decoder conventions. Overridable per install.
DENSE_SEAT_SUFFIXES = (
    "linear_attn.out_proj", "linear_attn.in_proj_qkvz",
    "self_attn.qkv_proj", "self_attn.o_proj",
    "shared_expert.gate_up_proj", "shared_expert.down_proj",
    "mlp.gate_up_proj", "mlp.down_proj",
)


def _dense_weight(module):
    """The module's weight as a dense BF16 matrix.

    FP8 block-quantized modules carry ``weight_scale_inv``; the dequant
    runs in 4096-row slabs so the transient stays bounded on tight
    cards. A quantized weight without its scale is refused."""
    w = module.weight.data
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if w.dtype != fp8:
        return w
    scale = getattr(module, "weight_scale_inv",
                    getattr(module, "weight_scale", None))
    if scale is None:
        raise ValueError("refused: fp8 weight without a block scale")
    sd = scale.data.float()
    n, k = w.shape
    bn = -(-n // sd.shape[0])
    bk = -(-k // sd.shape[1])
    rows = sd.repeat_interleave(bn, 0)[:n]
    out = torch.empty(n, k, device=w.device, dtype=torch.bfloat16)
    for i in range(0, n, 4096):
        j = min(i + 4096, n)
        out[i:j] = (w[i:j].float()
                    * rows[i:j].repeat_interleave(bk, 1)[:, :k]
                    ).to(torch.bfloat16)
    return out


def attach_engine(model, *, seats=DENSE_SEAT_SUFFIXES, use_gemv=None,
                  release=False, verbose=True):
    """Seat an SGLang model's dense projections; returns the handle."""
    from .. import swap as _swap
    from ..impls.linear_proj import nvfp4_dynamic as _linear
    from .vllm_engine import _ProjSeat, _is_projection

    if use_gemv is None:
        use_gemv = torch.cuda.get_device_capability() >= (12, 0)
    if not use_gemv:
        orig_init = _linear.LinearProjNvfp4Dynamic.__init__

        def _init(self, *a, **kw):
            orig_init(self, *a, **kw)
            self._gemv = None
        _linear.LinearProjNvfp4Dynamic.__init__ = _init

    refused = []
    targets = [(n, m) for n, m in model.named_modules()
               if any(n.endswith(s) for s in seats) and _is_projection(m)]
    targets.sort(key=lambda t: t[1].weight.numel())
    model.eval()

    # On a tight card the relief must land while binding continues, not
    # after it: with release, seats attach in slabs of original bytes
    # and each slab's originals move to the weight store before the
    # next slab binds. Without release, one handle carries everything.
    GROUP = 512 << 20
    handles, swaps, group_bytes, seated = [], {}, 0, 0

    def flush():
        nonlocal swaps, group_bytes, seated
        if not swaps:
            return
        handle = _swap.attach(model, swaps)
        if release:
            handle.consume()
        handles.append(handle)
        seated += len(swaps)
        swaps, group_bytes = {}, 0

    for name, mod in targets:
        try:
            seam, _ = _linear.bind_proj_seam({"w": _dense_weight(mod)})
        except Exception as e:
            if not refused:
                print(f"[structures.sglang] first refusal {name}: "
                      f"{e!r}"[:180], flush=True)
            refused.append((name, repr(e)[:120]))
            continue
        swaps[name] = _ProjSeat(seam)
        group_bytes += mod.weight.numel() * mod.weight.element_size()
        if release and group_bytes >= GROUP:
            flush()
    flush()
    if verbose:
        print(f"[structures.sglang] {seated} seats "
              f"({len(handles)} handles), {len(refused)} refused",
              flush=True)
    for h in handles:
        h.notes = {"refused": refused}
    return handles if len(handles) != 1 else handles[0]


def _patch_runner():
    """Runs inside the spawned scheduler, via the sitecustomize hook."""
    import sglang.srt.model_executor.model_runner as mr

    orig = mr.ModelRunner.load_model

    def load_model(self, *a, **kw):
        orig(self, *a, **kw)
        seats = tuple(s for s in os.environ.get(_SEATS_VAR, "").split(",")
                      if s) or DENSE_SEAT_SUFFIXES
        try:
            attach_engine(self.model, seats=seats,
                          release=os.environ.get("FRT_SGLANG_RELEASE") == "1")
        except Exception as e:
            print(f"[structures.sglang] attach refused: {e!r}", flush=True)
    mr.ModelRunner.load_model = load_model


_HOOK = """\
import os
if os.environ.get({flag!r}) == "1":
    try:
        import sys
        sys.path.insert(0, os.environ[{path!r}])
        from flash_rt.structures.adapters import sglang_engine
        sglang_engine._patch_runner()
    except Exception as e:
        print(f"[structures.sglang] hook inert: {{e!r}}", flush=True)
"""


def install(*, seats=None, structures_path=None, release=False):
    """Arm the spawn hook; call before constructing the engine.

    Writes a ``sitecustomize`` into a temporary directory, prepends it
    to ``PYTHONPATH`` and flags the attach on — every interpreter the
    engine spawns picks it up; other processes see a dormant flag."""
    if structures_path is None:
        structures_path = str(
            pathlib.Path(__file__).resolve().parents[3])
    hook_dir = tempfile.mkdtemp(prefix="frt-sglang-hook-")
    hook = pathlib.Path(hook_dir) / "sitecustomize.py"
    hook.write_text(_HOOK.format(flag=_ATTACH_FLAG, path=_PATH_VAR))
    os.environ[_PATH_VAR] = structures_path
    os.environ[_ATTACH_FLAG] = "1"
    if seats:
        os.environ[_SEATS_VAR] = ",".join(seats)
    if release:
        os.environ["FRT_SGLANG_RELEASE"] = "1"
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (hook_dir + (":" + prev if prev else ""))
    # the launcher itself may import sitecustomize-late; patch it too so
    # single-process embeddings behave the same way
    if "sglang" in sys.modules:
        _patch_runner()
    return hook_dir
