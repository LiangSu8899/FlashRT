"""JAX-side WeightLoader concretes (stage 7.5).

Concrete ``WeightSource`` / ``WeightSink`` / ``Transform`` implementations
for the Orbax/engine_w weight-upload path used by Thor JAX frontends
(``Pi05JaxFrontendThor``, ``Pi0JaxFrontendThor``).

Design differences vs the torch adapter (stage 7.2):

  * Source is a plain ``dict[str, numpy.ndarray]`` (openpi ``engine_w``)
    rather than a safetensors handle. QKV / gate_up pairs are already
    pre-fused during the openpi export, so no ``FusedQKV`` composite is
    needed on this side — items just read the packed tensor directly.
  * Sinks build ``CudaBuffer`` objects (not torch tensors). The per-layer
    concatenation pattern (``_jax_to_cb``) is captured by
    ``CudaBufferFlat``, which accumulates per-layer JAX uint8 arrays and
    on ``finalize`` does ``jnp.concatenate(...).reshape(-1)`` →
    ``CudaBuffer.device_empty`` → ``copy_from_jax``.
  * Every buffer populated here is additionally recorded into
    ``target._cache_blobs`` under a caller-supplied name, so
    ``_save_to_cache`` on the frontend can dump them without a
    GPU→CPU round trip (XLA context makes download unsafe).
  * ``JaxQuant`` is the JAX equivalent of ``Quant`` — calls the
    per-tensor amax / clip / fp8 cast on the GPU via JAX, stashes the
    scale into ``ctx.scratch['_pending_scale']`` like the torch side.

The frontend is responsible for creating ``self._cache_blobs = {}``
before running the loader; sinks mutate it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from flash_rt.executors.weight_loader import LoaderContext, ModelWeightSpec


# ════════════════════════════════════════════════════════════════════
#  Source
# ════════════════════════════════════════════════════════════════════

class OrbaxDictSource:
    """WeightSource backed by the openpi ``engine_w`` dict.

    Values are numpy arrays (openpi's ``transform_jax_weights`` produces
    host-side numpy before uploads). No caching — each ``get`` returns
    the underlying array reference directly.
    """

    def __init__(self, engine_w: dict):
        self._w = engine_w

    def get(self, key: str):
        return self._w[key]

    def has(self, key: str) -> bool:
        return key in self._w


# ════════════════════════════════════════════════════════════════════
#  Transforms
# ════════════════════════════════════════════════════════════════════

class Transpose:
    """numpy/jax ``.T`` on a 2-D tensor."""

    def apply(self, x, ctx):
        return x.T


class Astype:
    def __init__(self, dtype):
        self.dtype = dtype

    def apply(self, x, ctx):
        return x.astype(self.dtype)


class Contiguous:
    """Force a contiguous copy (equivalent to torch's ``.contiguous()``)
    — used after ``.T`` to avoid downstream layout surprises."""

    def apply(self, x, ctx):
        return np.ascontiguousarray(x)


class JaxQuant:
    """FP8 E4M3 per-tensor quant on GPU via JAX. Stashes scale in ctx.

    Accepts a numpy array (openpi engine_w values). Returns a JAX uint8
    array (the fp8 bytes reinterpreted). Scale is stashed into
    ``ctx.scratch['_pending_scale']`` so the runner can route it into
    the spec's ``scale_into`` list.
    """

    def apply(self, x, ctx):
        import ml_dtypes
        import jax.numpy as jnp
        w_jax = jnp.array(x, dtype=jnp.float32)
        amax = float(jnp.abs(w_jax).max())
        scale = max(amax / 448.0, 1e-12)
        w_scaled = jnp.clip(w_jax / scale, -448.0, 448.0)
        fp8_jax = w_scaled.astype(ml_dtypes.float8_e4m3fn)
        ctx.scratch["_pending_scale"] = float(scale)
        return fp8_jax.view(jnp.uint8)


# ════════════════════════════════════════════════════════════════════
#  Sinks
# ════════════════════════════════════════════════════════════════════

def _record_cache(target, name: str, data: bytes) -> None:
    """Mutate ``target._cache_blobs`` — initialised lazily if missing."""
    if not hasattr(target, "_cache_blobs") or target._cache_blobs is None:
        target._cache_blobs = {}
    target._cache_blobs[name] = data


class NumpyAttr:
    """Assign a numpy array directly to ``target.<name>``.

    Used for weights that live on the CPU side (time_mlp, mod Dense,
    embedding, RoPE tables) — the frontend runs CPU compute on them in
    ``set_prompt`` before any GPU upload.
    """

    def __init__(self, name: str):
        self.name = name
        self._bound_target = None
        self._stored = False

    def _bind(self, target):
        self._bound_target = target

    def store(self, tensor, *, scale=None):
        if self._stored:
            raise RuntimeError(f"NumpyAttr sink '{self.name}' stored twice")
        setattr(self._bound_target, self.name, tensor)
        self._stored = True

    def finalize(self) -> None:
        return None


class NumpyList:
    """Append-per-iteration into ``target.<name>`` (a numpy array list)."""

    def __init__(self, name: str):
        self.name = name
        self._bound_target = None
        self._list: list | None = None

    def _bind(self, target):
        self._bound_target = target
        self._list = []
        setattr(target, self.name, self._list)

    def store(self, tensor, *, scale=None):
        assert self._list is not None
        self._list.append(tensor)

    def finalize(self) -> None:
        return None


class CudaBufferAttr:
    """Assign a single ``CudaBuffer`` (wrapped from numpy) to ``target.<name>``.

    Records the raw bytes into ``target._cache_blobs[cache or name]`` so
    the weight-cache path can avoid a GPU→CPU download later.
    """

    def __init__(self, name: str, *, cache: str | None = None):
        self.name = name
        self.cache = cache if cache is not None else name
        self._bound_target = None
        self._stored = False

    def _bind(self, target):
        self._bound_target = target

    def store(self, tensor, *, scale=None):
        if self._stored:
            raise RuntimeError(f"CudaBufferAttr sink '{self.name}' stored twice")
        from flash_rt.core.cuda_buffer import CudaBuffer
        arr = np.ascontiguousarray(tensor)
        buf = CudaBuffer.from_numpy(arr)
        setattr(self._bound_target, self.name, buf)
        _record_cache(self._bound_target, self.cache, arr.tobytes())
        self._stored = True

    def finalize(self) -> None:
        return None


class CudaBufferFlat:
    """Per-layer accumulator → concatenated ``CudaBuffer`` on finalize.

    Matches the ``_jax_to_cb(jax_list, cache_name)`` pattern in the
    legacy jax frontends: per-layer JAX (or numpy) arrays are collected
    during the LayerBlock loop; finalize concatenates them flat, copies
    to a single ``CudaBuffer`` device allocation, and records the raw
    bytes into ``target._cache_blobs[cache]``.
    """

    def __init__(self, name: str, *, cache: str | None = None):
        self.name = name
        self.cache = cache if cache is not None else name
        self._bound_target = None
        self._parts: list | None = None

    def _bind(self, target):
        self._bound_target = target
        self._parts = []

    def store(self, tensor, *, scale=None):
        assert self._parts is not None
        self._parts.append(tensor)

    def finalize(self) -> None:
        import jax
        import jax.numpy as jnp

        from flash_rt.core.cuda_buffer import CudaBuffer

        parts = self._parts
        # Normalise each part to a JAX array so ``jnp.concatenate`` works
        # regardless of whether the upstream transform returned numpy or jax.
        flat = jnp.concatenate([jnp.asarray(p).reshape(-1) for p in parts])
        jax.block_until_ready(flat)
        buf = CudaBuffer.device_empty(flat.size, np.dtype(flat.dtype))
        buf.copy_from_jax(flat)
        setattr(self._bound_target, self.name, buf)
        _record_cache(self._bound_target, self.cache, np.array(flat).tobytes())


# ════════════════════════════════════════════════════════════════════
#  Convenience factories
# ════════════════════════════════════════════════════════════════════

def quant_flat_item(*, name: str, key: str, sink_attr: str,
                    cache: str, scale_into: str):
    """Shorthand: ``engine_w[key] -> JaxQuant -> CudaBufferFlat(sink_attr)``.

    Covers the 95% case in the jax encoder/decoder blocks:
    ``q8, qs = qfp8(engine_w[f"{pfx}.qkv.weight"]); dec_qkv.append(q8)``.
    """
    from flash_rt.executors.weight_loader import Item
    return Item(
        name=name,
        key=key,
        transforms=[JaxQuant()],
        sink=CudaBufferFlat(sink_attr, cache=cache),
        scale_into=scale_into,
    )


__all__ = [
    "OrbaxDictSource",
    "Transpose",
    "Astype",
    "Contiguous",
    "JaxQuant",
    "NumpyAttr",
    "NumpyList",
    "CudaBufferAttr",
    "CudaBufferFlat",
    "quant_flat_item",
]
