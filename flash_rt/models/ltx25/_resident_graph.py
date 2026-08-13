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
(~14GB for the 48-block model). The text encoder (~26GB bf16) cannot coexist
with it, so the frontend must encode the prompt before the first transformer
build and cache the embeddings for later calls (see
``CachingPromptEncoder``).
"""

from __future__ import annotations

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


class CachingPromptEncoder:
    """Wraps the pipeline's PromptEncoder with an embedding cache.

    With a resident transformer the ~26GB bf16 text encoder no longer fits
    alongside it, so repeat prompts must not re-run the encoder. The first
    encode of each prompt list runs before the transformer's first build
    (pipeline order already guarantees that on the first call); later calls
    with the same prompts hit the cache.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self._cache: dict[tuple, object] = {}

    def __call__(self, prompts, **kwargs):
        key = (tuple(prompts), tuple(sorted(kwargs.items())))
        hit = self._cache.get(key)
        if hit is None:
            hit = self._inner(prompts, **kwargs)
            self._cache[key] = hit
        return hit

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_inner"), item)
