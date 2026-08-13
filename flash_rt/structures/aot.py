"""Whole-graph ahead-of-time packaging for swapped modules.

Graph integrity is a first-class property of the structures runtime:
a swapped module's hot path carries no Python bookkeeping (guards
qualify at bind time and step aside under compilation), so the whole
forward exports as one graph with the Hub kernels riding along as
``torch.library`` ops. This module turns that property into an
artifact: ``aot_package`` exports the module and compiles it with
AOTInductor into a self-contained package on disk; ``aot_load`` brings
it back as a callable that replays the compiled graph with no dynamo
in the loop and no JIT cost at first call.

The scoring suite treats an AoT arm like any other treated form:
stepwise parity, repeat chains, detach, and dual-baseline timing —
the package is a faster body for the same declared plan, never a
change of plan.
"""

from __future__ import annotations

import pathlib

import torch

#: True while ``torch.export`` traces a module through this door.
#: Export functionalization supports in-place writes only into
#: *registered* buffers (it lifts them as graph outputs); structure
#: code holding pool-leased plain-attribute tensors checks this flag
#: and re-homes them as owned registered buffers before writing.
_EXPORTING = False


def is_exporting() -> bool:
    return _EXPORTING


def _own_leased_stashes(module: torch.nn.Module) -> int:
    """Re-home pool-leased stash tensors as registered buffers.

    The pack heads write later siblings into stash tensors with a
    plain ``copy_``. Export functionalization lifts an in-place write
    only when its target is a *registered buffer* (the mutation
    becomes a graph output); a pool-leased plain-attribute tensor is
    inexpressible and kills the export. Re-homing has to happen here,
    before tracing — inside a traced forward the module would be
    mutated with fake tensors. The owned copy detaches the tensor
    from the workspace pool for this module only; readers resolve the
    same attribute name and see the registered buffer.
    """
    owned = 0
    for m in module.modules():
        splits = getattr(m, "splits", None)
        if not isinstance(splits, (list, tuple)) or len(splits) < 2:
            continue
        for i in range(1, len(splits)):
            name = f"stash{i}"
            t = m.__dict__.get(name)
            if t is not None and torch.is_tensor(t):
                del m.__dict__[name]
                m.register_buffer(name, t.detach().clone(),
                                  persistent=False)
                owned += 1
    return owned

__all__ = ["aot_package", "aot_package_external", "aot_load", "AotModule"]


def aot_package(module: torch.nn.Module, args=(), kwargs=None,
                package_path="module_aot.pt2",
                external_weights: bool = False,
                inductor_configs=None) -> str:
    """Export ``module`` on example inputs and AOT-compile the graph.

    Returns the package path. Raises on graph breaks or export
    failure — a partial graph is a defect to fix at the seam, not a
    fallback to hide.

    ``external_weights=True`` keeps the constants out of the compiled
    binary: the package carries the graph alone and the caller supplies
    the weights at load time (:func:`aot_load` with ``weights=``). This
    is the form for a module that must keep serving from its live
    parameters — baking would put a second copy of every weight on the
    card — and doubles as the deployment shape where one graph binary
    serves many checkpoints.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "refused: AOT packaging compiles for the present GPU; "
            "no CUDA device is visible")
    kwargs = dict(kwargs or {})
    _own_leased_stashes(module)
    global _EXPORTING
    _EXPORTING = True
    try:
        with torch.no_grad():
            exported = torch.export.export(module, args=tuple(args),
                                           kwargs=kwargs)
    finally:
        _EXPORTING = False
    configs = dict(inductor_configs or {})
    if external_weights:
        configs["aot_inductor.package_constants_in_so"] = False
    out = torch._inductor.aoti_compile_and_package(
        exported, package_path=str(pathlib.Path(package_path)),
        inductor_configs=configs or None)
    return str(out)


def aot_package_external(module: torch.nn.Module, args=(), kwargs=None,
                         package_path="module_aot.pt2",
                         inductor_configs=None):
    """External-weights packaging: returns ``(path, weights)``.

    ``weights`` is the complete name→tensor map the package will ask
    for at load — parameters, buffers persistent or not, and the
    tensor constants export lifted (plain-attribute tensors an impl
    holds). Handing exactly this map to :func:`aot_load` makes the
    runtime borrow every one of them in place: no second copy of any
    weight, and buffer mutations land in the caller's tensors.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "refused: AOT packaging compiles for the present GPU; "
            "no CUDA device is visible")
    kwargs = dict(kwargs or {})
    _own_leased_stashes(module)
    global _EXPORTING
    _EXPORTING = True
    try:
        with torch.no_grad():
            exported = torch.export.export(module, args=tuple(args),
                                           kwargs=kwargs)
    finally:
        _EXPORTING = False
    weights = dict(module.named_parameters())
    weights.update(dict(module.named_buffers()))
    weights.update({k: v for k, v in (exported.constants or {}).items()
                    if torch.is_tensor(v)})
    configs = dict(inductor_configs or {})
    configs["aot_inductor.package_constants_in_so"] = False
    out = torch._inductor.aoti_compile_and_package(
        exported, package_path=str(pathlib.Path(package_path)),
        inductor_configs=configs)
    return str(out), weights


def aot_load(package_path: str, weights=None):
    """Load an AOT package back as a callable graph.

    ``weights`` is a name→tensor mapping (a module ``state_dict``) for
    packages built with ``external_weights=True``. The runtime borrows
    the tensors in place (``user_managed``): no copy is made, and a
    graph that mutates a buffer mutates the caller's tensor — which is
    the point, for state the rest of the pipeline keeps reading.
    Missing names fail loudly with the exact FQNs.
    """
    compiled = torch._inductor.aoti_load_package(str(package_path))
    if weights is not None:
        fqns = compiled.get_constant_fqns()
        missing = [f for f in fqns if f not in weights]
        if missing:
            raise ValueError(
                f"aot_load: {len(missing)} constant(s) absent from the "
                f"supplied weights, first: {missing[:5]}")
        compiled.load_constants(
            {f: weights[f] for f in fqns},
            check_full_update=True, user_managed=True)
    return compiled


class AotModule(torch.nn.Module):
    """Drop-in stand-in that replays the packaged graph.

    Attribute lookups fall through to the host module, so pipeline
    glue that introspects config/dtype keeps working; ``host`` gives
    the original back for detach.
    """

    def __init__(self, compiled, host: torch.nn.Module):
        super().__init__()
        object.__setattr__(self, "_compiled", compiled)
        object.__setattr__(self, "host", host)

    def forward(self, *args, **kwargs):
        return self._compiled(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "host"), name)
