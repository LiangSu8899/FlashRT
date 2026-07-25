"""Auto-assembly: discover seams, calibrate them in one pass, bind them.

This is the distribution layer. Given a host model and a way to run it,
it finds every structure seam (:mod:`.discover`), captures exactly the
calibration each one needs in a single forward pass, and binds each
through its library impl. The caller gets a ``path -> module`` swap map
and any outside-cadence update functions — the same thing the hand
recipes produced, derived from the model object rather than written by
hand. A host integrates by importing and calling; it writes no
per-seam scaffolding.

The calibration each structure needs, captured structure-aware:
  linear_proj / qkv_pack : the shared input the projection(s) see, and
                           its per-tensor amax (the static act scale)
  adaln_producer         : the (cond, style) pairs the conditioning
                           projection emits across the tick, for the
                           step table and its fingerprint locator
  decoder_ffn / vision_ffn : the normed input the MLP sees

Seam negotiation is resolved here: when an adaln_producer feeds a
sibling qkv_pack under the same parent, the producer emits fp8 and the
pack takes the shared act scale, skipping its own input quantization.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .discover import (Seam, _resolve, discover, group_families,
                       seam_weights)

_FP8 = torch.float8_e4m3fn
_FP8_CHAIN_MAX_ROWS = 256  # fp8 producer chain qualifies at denoise
                          # M (bandwidth-bound); large-M prefill skips

# Host-family attention adapters. Attention seams are not a static
# module pattern — where the attention math actually runs is
# host-specific (a function in one host, a processor in another), so
# auto-discovery of the attention_core structure is delegated to
# registered adapters. Each adapter, given the model and a way to run
# it, returns (swaps, update) or None (this host is not its family).
_ATTENTION_ADAPTERS: list = []


def register_attention_adapter(adapter) -> None:
    """Register a host-family attention adapter (callable)."""
    _ATTENTION_ADAPTERS.append(adapter)


@dataclass
class AutoPlan:
    """Discovered + calibrated swaps, ready to stage."""

    swaps: dict[str, torch.nn.Module] = field(default_factory=dict)
    updates: list[Callable[[], None]] = field(default_factory=list)
    seams: list[Seam] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


def _amax(tensors) -> float:
    return max(t.float().abs().max().item() for t in tensors)


def _rows(t: torch.Tensor) -> int:
    return int(t.reshape(-1, t.shape[-1]).shape[0])


def auto_swaps(
    model: torch.nn.Module,
    forward: Callable[[], Any],
    *,
    structures: tuple[str, ...] = ("decoder_ffn", "vision_ffn",
                                   "qkv_pack", "adaln_producer",
                                   "linear_proj", "norm_fused",
                                   "attention_core", "decoder_block"),
    negotiate_fp8: bool = True,
    frames: int = 1,
    verbose: bool = False,
) -> AutoPlan:
    """Discover, calibrate in one pass, and bind every applicable seam."""

    def say(msg: str) -> None:
        if verbose:
            print(f"[autobuild] {msg}", flush=True)

    seams = discover(model, structures)
    # a packed group owns its q/k/v; linear_proj keeps only what the
    # pack does not take (the output projection), so the two structures
    # compose instead of fighting over the same module
    packed = {s.path + "." + a for s in seams
              if s.structure == "qkv_pack" for a in (s.pack_attrs or ())}
    seams = [s for s in seams
             if not (s.structure == "linear_proj" and s.path in packed)]
    say(f"discovered {len(seams)} seam(s)")
    adapter_only = not seams and "attention_core" in structures
    if not seams and not adapter_only:
        return AutoPlan()

    # ---- one calibration pass, structure-aware capture ----
    caps: dict[str, dict[str, Any]] = {}
    hooks = []

    def cap_input(path):
        def hook(module, args, kwargs, out):
            x = args[0] if args else next(iter(kwargs.values()))
            caps[path].setdefault("x", []).append(x.detach())
            caps[path]["rows"] = _rows(x)
            return None
        return hook

    # observed call order across the whole calibration pass. Anything
    # that has to know which seam runs first (a stream-scoped buffer
    # needs a writer, and the writer has to be the one the host calls
    # first) reads it from here rather than assuming the module tree's
    # order matches the forward's.
    call_order = itertools.count()

    def cap_cond(path):
        def hook(module, args, out):
            cap = caps[path]
            if "order" not in cap:
                cap["order"] = next(call_order)
            cap.setdefault("pairs", []).append(
                (args[0].detach().clone(), out.detach().clone()))
            return None
        return hook

    def cap_shape(path):
        # a block seam needs no tensors of its own, only the host's
        # return convention (bare tensor or 1-tuple)
        def hook(module, args, kwargs, out):
            caps[path]["returns_tuple"] = isinstance(out, tuple)
            return None
        return hook

    for seam in seams:
        caps[seam.path] = {}
        target = _resolve(model, seam.path)
        if seam.structure == "decoder_block":
            hooks.append(target.register_forward_hook(
                cap_shape(seam.path), with_kwargs=True))
        elif seam.structure == "adaln_producer":
            # the norm's own input gives rows; the cond projection gives
            # the (cond, style) pairs the table is built from
            hooks.append(target.register_forward_hook(
                cap_input(seam.path), with_kwargs=True))
            hooks.append(getattr(target, seam.cond_attr)
                         .register_forward_hook(cap_cond(seam.path)))
        elif seam.structure == "qkv_pack":
            first = getattr(target, seam.pack_attrs[0])
            hooks.append(first.register_forward_hook(
                cap_input(seam.path), with_kwargs=True))
        else:
            hooks.append(target.register_forward_hook(
                cap_input(seam.path), with_kwargs=True))

    if hooks:
        with torch.no_grad():
            for _ in range(max(1, frames)):
                forward()
        for h in hooks:
            h.remove()
        say("calibration pass done")

    # ---- fp8 seam negotiation: the load-bearing structure combination.
    # A single kernel need not win alone (fp8 qkv at M=50 is marginal,
    # fa2 in a bf16 stack loses); the *chain* wins — an adaln producer
    # that emits fp8 lets the qkv pack skip its own input quantization
    # and hands a clean fp8 seam down to the attention core. Bind the
    # producer→pack pair together with one shared act scale wherever a
    # producer feeds a pack under the same parent layer. ----
    act_scales: dict[str, torch.Tensor] = {}
    negotiated: dict[str, dict[str, Seam]] = {}
    if negotiate_fp8:
        by_parent: dict[str, dict[str, Seam]] = {}
        for seam in seams:
            layer = _layer_of(seam.path)
            if seam.structure == "adaln_producer":
                # a layer has two producer→consumer seams: the norm
                # before attention feeds the projections, the norm after
                # it feeds the MLP. Both can hand fp8 downstream.
                slot = ("producer" if _feeds_attention(seam.path)
                        else "producer_ffn")
                by_parent.setdefault(layer, {})[slot] = seam
            elif seam.structure == "qkv_pack":
                by_parent.setdefault(layer, {})["pack"] = seam
            elif seam.structure == "decoder_ffn":
                by_parent.setdefault(layer, {})["ffn"] = seam
        # the chain wins at small M (denoise): fp8 is bandwidth-bound and
        # pays there, while a large-M prefill GEMM is compute-bound and
        # fp8 buys little — and an fp8 producer feeding a big compiled
        # prefill region is where the triton fp8 codegen chokes. Qualify
        # on the calibrated row count, not on host names.
        dev = next(model.parameters()).device
        blocks = {s.path for s in seams if s.structure == "decoder_block"}
        for lay, g in by_parent.items():
            # the attention pack is always negotiated. The FFN chain is
            # negotiated only where a decoder_block owns the boundary,
            # and the reason is the boundary rather than the kernel: at
            # the norm seam the fused producer costs a kernel
            # (gate_residual, +180 launches) plus its style
            # materialization (+180) to save the FFN's own input
            # quantize (-180) — measured net +0.17ms, so it is refused
            # there. Inside a block the same kernel *replaces* the
            # host's gated residual add instead of adding to it, which
            # is the whole point of owning the block.
            pairs = [("producer", "pack")]
            if lay in blocks:
                pairs.append(("producer_ffn", "ffn"))
            keep = {}
            for p_slot, c_slot in pairs:
                if p_slot not in g or c_slot not in g:
                    continue
                c_cap = caps.get(g[c_slot].path, {})
                rows = c_cap.get("rows", 1 << 30)
                if not c_cap.get("x") or rows > _FP8_CHAIN_MAX_ROWS:
                    continue
                # the consumer's input == the producer's output; its amax
                # is the one static scale both sides share
                keep[p_slot], keep[c_slot] = g[p_slot], g[c_slot]
                act_scales[f"{lay}|{c_slot}"] = torch.tensor(
                    [max(_amax(c_cap["x"]) / 448.0, 1e-8)], device=dev)
            if keep:
                negotiated[lay] = keep

    # ---- the negotiated chain binds as one unit ----
    # producer and consumer must agree on the seam dtype: a pack bound
    # for fp8 input whose producer failed to bind would be handed BF16,
    # and the host would silently grow a quantize fused into whatever
    # produced it. Bind the pair together, or leave both on BF16.
    plan = AutoPlan(seams=seams)
    handled: set[str] = set()
    for lay, g in negotiated.items():
        for p_slot, c_slot in (("producer", "pack"),
                               ("producer_ffn", "ffn")):
            if p_slot not in g or c_slot not in g:
                continue
            p_seam, c_seam = g[p_slot], g[c_slot]
            p_cap = caps.get(p_seam.path, {})
            c_cap = caps.get(c_seam.path, {})
            if not (p_cap.get("pairs") and c_cap.get("x")):
                continue
            try:
                pair = _bind_negotiated(
                    model, p_seam, c_seam, p_cap, c_cap,
                    act_scales[f"{lay}|{c_slot}"], plan)
            except (ValueError, RuntimeError) as refusal:
                plan.notes.setdefault("refused", []).append(
                    (f"{lay} [{c_slot} chain]", str(refusal)[:80]))
                continue
            plan.swaps.update(pair)
            handled.update({p_seam.path, c_seam.path})
    plan.notes["negotiated_layers"] = sorted(
        lay for lay, g in negotiated.items()
        if any(sm.path in handled for sm in g.values()))

    # ---- bind the remaining seams individually ----
    for name, members in group_families(seams).items():
        for seam in members:
            if seam.path in handled:
                continue
            cap = caps.get(seam.path, {})
            try:
                bound = _bind_auto(model, seam, cap, plan, act_scales,
                                   negotiate_fp8)
            except (ValueError, RuntimeError) as refusal:
                plan.notes.setdefault("refused", []).append(
                    (seam.path, str(refusal)[:80]))
                continue
            if bound is None:
                continue
            if isinstance(bound, dict):
                plan.swaps.update(bound)
            else:
                plan.swaps[seam.path] = bound
    # ---- attention_core: host-family adapters (fa2 seam) ----
    if "attention_core" in structures:
        from . import adapters as _adapters  # noqa: F401 (registers)
        for adapter in _ATTENTION_ADAPTERS:
            try:
                result = adapter(model, forward)
            except (ValueError, RuntimeError) as refusal:
                plan.notes.setdefault("refused", []).append(
                    ("attention_core", str(refusal)[:80]))
                continue
            if result is None:
                continue
            att_swaps, update = result
            plan.swaps.update(att_swaps)
            if update is not None:
                plan.updates.append(update)
            plan.notes["attention_adapter"] = type(adapter).__name__ \
                if hasattr(adapter, "__name__") else str(adapter)
            break

    # ---- one step-scoped style materialisation per conditioning stream
    # Every adaptive-norm producer on one stream resolves the same step,
    # so the whole stream's styles are fixed for the step's duration.
    # Materialising them once beats materialising them per call by the
    # launch count, which is what that work actually costs. Runs before
    # the block assembly: a block holds its producers directly and drops
    # them from the swap map, so afterwards they are no longer findable
    # here.
    _attach_brokers(caps, plan, say)

    # ---- decoder_block: compose the bound sublayers into one block ----
    # last, because it is assembled from what the region structures
    # produced. The swaps it absorbs are dropped from the plan: the
    # block holds those modules directly, and a swap that also targeted
    # the host child would leave two live copies of the same seam.
    for seam in (s for s in seams if s.structure == "decoder_block"):
        try:
            block = _bind_block(model, seam, caps.get(seam.path, {}), plan)
        except (ValueError, RuntimeError) as refusal:
            plan.notes.setdefault("refused", []).append(
                (seam.path + " [block]", str(refusal)[:80]))
            continue
        if block is None:
            continue
        for child in _BLOCK_OWNED:
            plan.swaps.pop(seam.path + "." + child, None)
        plan.swaps[seam.path] = block

    say(f"bound {len(plan.swaps)} seam(s), "
        f"{len(plan.notes.get('refused', []))} refused")
    return plan


def _attach_brokers(caps, plan, say) -> None:
    from .impls.adaln_producer import AdaLNProducer, bind_style_broker

    groups: dict[tuple, list] = {}
    for path, module in plan.swaps.items():
        if not isinstance(module, AdaLNProducer):
            continue
        cap = caps.get(path, {})
        order = cap.get("order")
        if order is None or not cap.get("pairs"):
            continue
        # one broker per (stream, style width, row count): producers
        # that differ in any of those cannot share a buffer
        key = (_stream_key(cap["pairs"]), int(module.styles.shape[-1]),
               int(module.resid.shape[0]))
        groups.setdefault(key, []).append((order, path, module))

    for key, members in groups.items():
        # the writer is the producer the host calls first, taken from the
        # observed order of the calibration pass — not from the module
        # tree's order, which need not match the forward's
        members.sort(key=lambda entry: entry[0])
        try:
            broker = bind_style_broker([m for _, _, m in members], key[2])
        except (ValueError, RuntimeError) as refusal:
            plan.notes.setdefault("refused", []).append(
                (f"style_broker[{key[1]}x{key[2]}]", str(refusal)[:80]))
            continue
        if broker is None:
            continue
        plan.notes.setdefault("brokers", []).append(
            {"slots": broker.slots, "rows": key[2], "width": key[1],
             "writer": members[0][1]})
        say(f"style broker: {broker.slots} producer(s) share one "
            f"step-scoped materialisation (writer {members[0][1]})")


_BLOCK_OWNED = ("input_layernorm", "post_attention_layernorm", "mlp")


def _cond_kw(host) -> str:
    """The keyword the host threads its conditioning through."""
    import inspect
    try:
        params = list(inspect.signature(host.forward).parameters)
    except (TypeError, ValueError):
        params = []
    for name in ("adarms_cond", "cond", "temb", "emb"):
        if name in params:
            return name
    return "adarms_cond"


def _bind_block(model, seam, cap, plan):
    """Assemble one decoder_block from its already-bound sublayers."""
    from .impls.decoder_block import bind_decoder_block

    prod_in = plan.swaps.get(seam.path + ".input_layernorm")
    prod_out = plan.swaps.get(seam.path + ".post_attention_layernorm")
    ffn = plan.swaps.get(seam.path + ".mlp")
    if prod_in is None or prod_out is None or ffn is None:
        # a sublayer that did not bind leaves the host block intact:
        # the block structure adds composition, it does not substitute
        # for the region seams it is made of
        return None
    host = _resolve(model, seam.path)
    # the attention sublayer is family-specific (where the attention runs
    # and which rotary form it uses), so it comes from the same adapters
    # that bound the attention core. None keeps the host's attention
    # module, which is the pre-block behaviour.
    attn = None
    for adapter in _ATTENTION_ADAPTERS:
        builder = getattr(adapter, "sublayer", None)
        if builder is None:
            continue
        attn = builder(host)
        if attn is not None:
            break
    if attn is not None:
        _alias_kv_region(plan, seam.path, attn)
    return bind_decoder_block(
        host, prod_in, prod_out, ffn, cond_kw=_cond_kw(host),
        returns_tuple=bool(cap.get("returns_tuple")), attn=attn)


def _alias_kv_region(plan, path: str, sublayer) -> None:
    """Let the packed projections write into the core's packed KV region.

    Both sides can express this (see ``beta.joins``); the qualification
    is that nothing transforms the tensor in between. Value goes straight
    from the projection to the kernel and qualifies. Key does not on this
    family: a rotary embedding runs after the projection, so aliasing it
    would leave untransformed keys in the packed region — writing the
    transformed ones back is the copy this was meant to remove. Hosts
    without a rotary step qualify for both; the attribute is general and
    the qualification is per join.
    """
    from .impls.qkv_pack import PackedLinear

    head = plan.swaps.get(path + ".self_attn.q_proj")
    core = getattr(sublayer, "core", None)
    if not isinstance(head, PackedLinear) or core is None:
        return
    if not hasattr(core, "alias_suffix"):
        return
    _, v_region = core.alias_suffix(key=False, value=True)
    if v_region is None:
        return
    try:
        head.alias_stash(2, v_region)          # sibling order q, k, v
    except (ValueError, RuntimeError) as refusal:
        core._alias_v = False
        plan.notes.setdefault("refused", []).append(
            (path + " [kv alias]", str(refusal)[:80]))
        return
    plan.notes.setdefault("aliased_kv", []).append(path)


def _bind_auto(model, seam, cap, plan, act_scales, negotiate_fp8):
    """Route one seam to its impl with the captured calibration."""
    from .handle import get

    if seam.structure in ("decoder_ffn", "vision_ffn"):
        if not cap.get("x"):
            return None
        h = get(seam.structure)
        return h.bind(_resolve(model, seam.path), cap["x"], gate_cos=0.0)

    if seam.structure == "norm_fused":
        from .impls.norm_fused import bind_norm_fused
        return bind_norm_fused(_resolve(model, seam.path),
                               calibration=cap.get("x"))

    if seam.structure == "linear_proj":
        if not cap.get("x"):
            return None
        from .impls.linear_proj import fp8_static as proj_impl
        return proj_impl.bind_proj_seam(
            seam_weights(model, seam), calibration=cap["x"],
            original=_resolve(model, seam.path))

    if seam.structure == "qkv_pack":
        from .impls.qkv_pack import bind_attn_block, bind_qkv_pack
        if not cap.get("x"):
            return None
        block = _resolve(model, seam.path)
        act_scale = torch.tensor(
            [max(_amax(cap["x"]) / 448.0, 1e-8)],
            device=getattr(block, seam.pack_attrs[0]).weight.device)
        if seam.variant.get("bind") == "module":
            # the whole block: packed projections *and* the attention
            # compute dtype (hosts that run SDPA in fp32 pay for it)
            return {seam.path: bind_attn_block(
                block, act_scale, rows=cap["rows"],
                sdpa_dtype=torch.bfloat16)}
        mods = [getattr(block, a) for a in seam.pack_attrs]
        parts = bind_qkv_pack(mods, act_scale, rows=cap["rows"],
                              in_dtype="bf16_fused_quant")
        return {seam.path + "." + a: m
                for a, m in zip(seam.pack_attrs, parts)}

    if seam.structure == "adaln_producer":
        from .impls.adaln_producer import (bind_adaln_producer,
                                           bind_style_table)
        if not cap.get("pairs"):
            return None
        norm = _resolve(model, seam.path)
        proj = getattr(norm, seam.cond_attr)
        key = _stream_key(cap["pairs"])
        loc = plan.notes.setdefault("_locators", {}).get(key)
        table = bind_style_table(proj, cap["pairs"], locator=loc)
        plan.notes["_locators"][key] = table.locator
        return {seam.path + "." + seam.cond_attr: table}

    return None


class _Eager(torch.nn.Module):
    """Wrap a module so its forward runs outside the compiled region.

    An fp8-emitting seam's arithmetic, if traced by inductor, gets fused
    into fp8 math (illegal on sm120 triton) — and the quantize even
    reaches back across the boundary, so inductor casts the host's own
    gated residual to fp8 to feed it. The hand recipes never hit this
    because the whole denoise block froze to eager. A swapped-in module
    does not inherit that freezing, so fp8 seams declare it. Overriding
    the instance ``forward`` is not enough (dynamo inlines the class
    forward); the disable must sit on a class method, which is what this
    wrapper provides. The kernels are opaque either way, so eager here
    is a graph break, not real work.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    @torch._dynamo.disable
    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)


def _eager(module):
    return _Eager(module)


def _bind_negotiated(model, p_seam, k_seam, p_cap, c_cap, scale, plan):
    """Bind an fp8 producer and the pack it feeds as one chain.

    This is the combination the structure layer exists for: neither half
    is worth much alone (a small-M fp8 projection barely beats BF16, a
    producer that only reshapes styles saves nothing), but together the
    producer's fused quantize removes the consumer's input quantization
    entirely and hands a clean fp8 seam downstream.
    """
    from .impls.adaln_producer import bind_adaln_producer
    from .impls.qkv_pack import bind_qkv_pack

    norm = _resolve(model, p_seam.path)
    consumer = _resolve(model, k_seam.path)
    key = _stream_key(p_cap["pairs"])
    loc = plan.notes.setdefault("_locators", {}).get(key)
    style_width = p_cap["pairs"][0][1].shape[-1]
    prod = bind_adaln_producer(
        norm, p_cap["pairs"], act_scale=scale,
        rows=p_cap.get("rows") or c_cap["rows"],
        dim=getattr(norm, "dim", style_width // 3),
        locator=loc, norm="rms")
    plan.notes["_locators"][key] = prod.locator

    swaps = {p_seam.path: prod}
    if k_seam.structure == "decoder_ffn":
        from .impls.decoder_ffn import fp8_static as ffn_impl
        # the calibration samples are the BF16 activations the host
        # produced; the fp8 entry needs them in the seam dtype, and the
        # scale is the one the producer upstream will quantize with
        w = seam_weights(model, k_seam)
        bound = ffn_impl.bind_mlp_seam(
            w, variant={**k_seam.variant, "in_dtype": "fp8_static"},
            calibration_normed=[t for t in c_cap["x"]],
            original=consumer)
        swaps[k_seam.path] = bound
        return swaps
    mods = [getattr(consumer, a) for a in k_seam.pack_attrs]
    parts = bind_qkv_pack(mods, scale, rows=c_cap["rows"],
                          in_dtype="fp8_static")
    swaps.update({k_seam.path + "." + a: m
                  for a, m in zip(k_seam.pack_attrs, parts)})
    return swaps


def _stream_key(pairs) -> str:
    """Identify the conditioning stream a producer was calibrated on.

    Locators were keyed by seam family, which gives every family its own
    lookup even when they all read the same conditioning — the two norms
    of one block among them. Keying by the observed conditioning instead
    shares one locator across the whole stream. It is safe by
    construction rather than by convention: the key is a digest of the
    conditioning rows themselves, so two seams share a locator only when
    they saw byte-identical inputs, and identical inputs resolve to
    identical indices whichever seam built the table.
    """
    import hashlib

    digest = hashlib.blake2b(digest_size=16)
    for cond, _ in pairs:
        c = cond.detach().reshape(-1, cond.shape[-1]).to(torch.float32)
        digest.update(c.cpu().numpy().tobytes())
    return digest.hexdigest()


def _layer_of(path: str) -> str:
    """The parent layer key: a.layers.7.self_attn -> a.layers.7."""
    import re
    m = re.search(r"(.*\.layers\.\d+)\.", path)
    return m.group(1) if m else path.rsplit(".", 1)[0]


def _feeds_attention(path: str) -> bool:
    """An adaln producer that feeds attention (input_layernorm) rather
    than the MLP (post_attention_layernorm)."""
    leaf = path.rsplit(".", 1)[-1]
    return "input" in leaf or leaf in ("norm1", "ln1")
