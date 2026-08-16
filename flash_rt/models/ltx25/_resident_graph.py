"""Resident-transformer support: unlock CUDA-graph capture on a single GPU.

Upstream refuses ``CompilationConfig(capture=True)`` with the single-GPU
builder because every stage call rebuilds the transformer onto fresh GPU
storages and ``gpu_model`` disposes it afterwards -- captured graphs would
replay against freed weight pointers.

``ResidentSwapBuilder`` closes that gap:

    * the first ``build`` delegates to the inner builder, applies the FlashRT
      swap installers, neuters ``dispose`` on the built model, and caches it;
    * every later ``build`` returns the cached model -- weight storages never
      move, so captured graphs stay valid;
    * ``keeps_gpu_resident_weights`` reports True, which satisfies the
      upstream capture precondition.

Memory contract: the transformer (plus repacked FFN weights) stays resident
(~14GB for the 48-block model). The text encoder (~26GB bf16) is loaded and
freed inside one prompt encode, and the two cannot coexist on a single
consumer part. So residency is not permanent by construction: it is a lease
that ``release`` ends, and every path that needs the encoder again (a prompt
outside the embedding cache) ends it first and lets the next build take a
fresh one. That is why both classes below are written around release rather
than around caching alone -- a cache that has no way to step aside turns a
second prompt into an out-of-memory error.
"""

from __future__ import annotations

import gc
import logging

import torch

from flash_rt.models.ltx25._nvfp4_ffn_swap import SwapInstallingBuilder

logger = logging.getLogger(__name__)


_x0_dispose_patched = False


def _patch_x0_dispose() -> None:
    """Make X0Model.dispose a no-op when it wraps a resident velocity model.

    The Disposable mixin's dispose walks the wrapper's own named_parameters,
    so neutering dispose on the inner model alone does not protect its
    storages -- a fresh X0Model wraps it on every stage build.
    """
    global _x0_dispose_patched
    if _x0_dispose_patched:
        return
    from ltx_core.model.transformer.model import X0Model

    original = X0Model.dispose

    def dispose(self):
        inner = getattr(self, "velocity_model", None)
        if getattr(inner, "_flash_rt_resident", False):
            return
        original(self)

    X0Model.dispose = dispose
    _x0_dispose_patched = True


class ResidentSwapBuilder(SwapInstallingBuilder):
    """SwapInstallingBuilder that builds once and keeps the model resident.

    The stage's ``_prepared_builder`` derives fresh rewrapped builders from
    the original on every stage call, so the cache lives in a mutable holder
    shared by reference across every rewrap -- the second stage must see the
    model the first stage built, not build a sibling.
    """

    def __init__(self, inner, installers, _holder=None) -> None:
        super().__init__(inner, installers)
        self._holder = _holder if _holder is not None else {}

    @property
    def keeps_gpu_resident_weights(self) -> bool:
        return True

    def build(self, **kwargs):
        model = self._holder.get("model")
        if model is not None:
            return model
        model = super().build(**kwargs)
        # gpu_model() disposes the X0Model wrapper after every stage, and the
        # Disposable mixin walks named_parameters -- which reach through to
        # this model's storages. Mark the model resident and teach X0Model's
        # dispose to skip wrappers holding a resident velocity model.
        model._flash_rt_resident = True
        model.dispose = lambda: None
        _patch_x0_dispose()
        self._holder["model"] = model
        logger.info("[ltx25] transformer resident: %.1fGB allocated",
                    torch.cuda.memory_allocated() / 2 ** 30)
        return model

    def _rewrap(self, inner):
        return ResidentSwapBuilder(inner, self._installers, self._holder)

    @property
    def is_resident(self) -> bool:
        return self._holder.get("model") is not None

    def release(self) -> int:
        """End the residency lease. Idempotent; returns bytes freed.

        Undoes exactly what ``build`` established, in the reverse order:
        the resident mark first (so ``X0Model.dispose`` stops skipping this
        model), then the instance-level ``dispose`` override (so the class's
        own dispose runs), then the disposal itself. The captured graphs need
        no separate teardown: the runner is reachable only through the
        patched block loop on this model, so dropping the model drops the
        graphs and their pool with it.

        A later ``build`` sees an empty holder and builds a fresh resident
        model, which is what makes a second prompt possible at all.
        """
        model = self._holder.pop("model", None)
        if model is None:
            return 0
        before = torch.cuda.memory_allocated()
        model._flash_rt_resident = False
        model.__dict__.pop("dispose", None)
        dispose = getattr(model, "dispose", None)
        if callable(dispose):
            dispose()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        freed = before - torch.cuda.memory_allocated()
        logger.info("[ltx25] resident transformer released: %.1fGB freed",
                    freed / 2 ** 30)
        return freed


class CachingPromptEncoder:
    """Wraps the pipeline's PromptEncoder with an embedding cache.

    Repeat prompts must not re-run the ~26GB encoder while a transformer is
    resident, and a cache serves that. What a cache cannot serve is the miss:
    a prompt nobody encoded yet needs the encoder loaded, which needs the
    residency lease to end first. ``on_miss`` is that call, made before the
    inner encoder runs and only when the cache has nothing -- so a repeat
    prompt keeps the resident model and a new prompt pays a rebuild instead
    of running the host out of memory.
    """

    def __init__(self, inner, on_miss=None) -> None:
        self._inner = inner
        self._cache: dict[tuple, object] = {}
        self._on_miss = on_miss

    def __call__(self, prompts, **kwargs):
        key = (tuple(prompts), tuple(sorted(kwargs.items())))
        hit = self._cache.get(key)
        if hit is None:
            if self._on_miss is not None:
                self._on_miss()
            hit = self._inner(prompts, **kwargs)
            self._cache[key] = hit
        return hit

    def clear(self) -> None:
        """Drop cached embeddings. Idempotent."""
        self._cache.clear()

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_inner"), item)
