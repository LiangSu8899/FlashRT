"""adaln_producer — conditioning-driven norm, resolved per step.

Diffusion hosts modulate every layer with a projection of the current
timestep embedding. Two facts make this a structure rather than a plain
GEMM: the conditioning vector takes one of a small fixed set of values
over a tick (it is a function of the step), and the norm that consumes
it can fuse modulation and output quantization into one kernel.

This implementation splits those two concerns:

- :class:`StepLocator` resolves "which step is this" from the
  conditioning tensor using a few high-separation coordinates — a
  fingerprint — instead of a full-width matmul against every stored
  vector. It is pure tensor work (index_select, squared distance,
  argmax), so it traces under dynamo and captures into a graph without
  host-side state. Sibling producers fed by the same conditioning
  stream share one locator, letting the compiler fold the repeated
  lookups.
- :class:`AdaLNProducer` replaces the host's adaptive norm: it looks up
  the precomputed style row for the current step and runs the fused
  norm+modulate(+static FP8 quantize) kernel, emitting either BF16 or
  FP8 plus the host's gate. The FP8 form is the upstream half of a
  producer→consumer seam: the shared ``act_scale`` lets a packed
  projection skip its own input quantization.

Qualification: the conditioning must actually be step-quantized (more
distinct vectors than ``max_steps`` means it depends on more than the
step, and a table would alias different inputs onto one row), and the
chosen fingerprint coordinates must separate the stored vectors by a
real margin. Either failure raises ``ValueError`` — the host keeps its
own producer.
"""

from __future__ import annotations

import torch

from .. import hub_kernel


def _dedup(pairs, max_steps, rtol):
    conds, outs = [], []
    for cond, out in pairs:
        c = cond.detach().reshape(-1, cond.shape[-1])
        o = out.detach().reshape(-1, out.shape[-1])
        for row in range(c.shape[0]):
            cr = c[row]
            if any(torch.allclose(cr, seen, rtol=rtol,
                                  atol=1e-6 * cr.abs().max().item() + 1e-12)
                   for seen in conds):
                continue
            conds.append(cr.clone())
            outs.append(o[row].clone())
            if len(conds) > max_steps:
                raise ValueError(
                    f"adaln_producer: >{max_steps} distinct conditioning "
                    "vectors — not step-quantized, keeping the host path")
    if not conds:
        raise ValueError("adaln_producer: no calibration pairs")
    return torch.stack(conds), torch.stack(outs)


class StepLocator(torch.nn.Module):
    """Resolve the current step index from the conditioning tensor."""

    def __init__(self, conds: torch.Tensor, n_dims: int = 8,
                 rel_margin: float = 1e-3):
        super().__init__()
        c = conds.float()
        steps = c.shape[0]
        if steps == 1:
            dims = torch.zeros(1, dtype=torch.long, device=c.device)
        else:
            diffs = (c.unsqueeze(0) - c.unsqueeze(1)).abs()
            eye = torch.eye(steps, device=c.device, dtype=torch.bool)
            diffs = diffs.masked_fill(eye.unsqueeze(-1), float("inf"))
            minsep = diffs.amin(dim=(0, 1))
            k = min(n_dims, c.shape[1])
            dims = minsep.topk(k).indices.sort().values
            margin = minsep[dims].min().item()
            if margin < rel_margin * c.abs().mean().item():
                raise ValueError(
                    "adaln_producer: conditioning vectors are not "
                    "separable on any coordinate subset")
        self.register_buffer("fp_dims", dims)
        self.register_buffer("fp_conds", c.index_select(
            1, dims).contiguous())

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        flat = cond.reshape(-1, cond.shape[-1]).float()
        c = flat.index_select(1, self.fp_dims)
        scores = -((c.unsqueeze(1) - self.fp_conds) ** 2).sum(-1)
        return scores.argmax(-1)


class StyleTable(torch.nn.Module):
    """Replace only the conditioning projection with its step table.

    The narrower of the two bind forms: the host keeps its own norm
    (often already fused by the compiler) and only the per-step style
    projection is memoized. Prefer this wherever the norm itself is not
    being upgraded — measurement decides, and the fused form is worth
    its kernel only when it also serves a downstream seam.
    """

    def __init__(self, host_proj: torch.nn.Module, styles: torch.Tensor,
                 locator: StepLocator):
        super().__init__()
        self.host_linear = host_proj
        self.locator = locator
        self.register_buffer("table", styles.contiguous())

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        out = self.table.index_select(0, self.locator(cond))
        return out.reshape(*cond.shape[:-1], out.shape[-1])

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("host_linear"), name)


class AdaLNProducer(torch.nn.Module):
    """Adaptive norm replacement: step lookup + fused norm/quantize."""

    def __init__(self, host_norm: torch.nn.Module,
                 styles: torch.Tensor, locator: StepLocator,
                 act_scale: torch.Tensor | None, rows: int, dim: int,
                 norm: str = "rms"):
        super().__init__()
        self.host_norm = host_norm
        self.locator = locator
        self.norm = norm
        self.register_buffer("styles",
                             styles.to(torch.bfloat16).contiguous())
        self.out_fp8 = act_scale is not None
        dev = styles.device
        if norm == "layer":
            # LayerNorm hosts (DiT AdaLayerNorm): style is (scale,
            # shift), no gate, and the fused kernel takes the raw
            # scale — it applies the (1 + scale) itself.
            if not self.out_fp8:
                raise ValueError("adaln_producer: layer norm form "
                                 "currently requires fp8 output")
            kq = hub_kernel("flashrt/adaptive-layernorm-producers", ">=1")
            self._fn = kq.ada_layer_norm_quant_fp8_bf16
            self.register_buffer("act_scale", act_scale)
        else:
            ka = hub_kernel("flashrt/flashrt-adaptive-norms", ">=1")
            if self.out_fp8:
                self._fn = ka.gate_residual_ada_norm_fp8_static_bf16
                self.register_buffer("act_scale", act_scale)
            else:
                self._fn = ka.ada_rms_norm_style_bf16
        # residual=0 / gate=1 turn the gated-residual kernel into a
        # plain modulated norm; both are preallocated for graph replay.
        # The kernel writes the residual buffer in place, so the zero
        # has to be re-established on every call — a buffer that is
        # merely allocated zeroed drifts silently from the second tick
        # onward, and the drift compounds.
        self.register_buffer("w_ones", torch.ones(
            dim, device=dev, dtype=torch.bfloat16))
        self.register_buffer("resid", torch.zeros(
            rows, dim, device=dev, dtype=torch.bfloat16))
        self.register_buffer("gate_ones", torch.ones(
            rows, dim, device=dev, dtype=torch.bfloat16))

    # ---- block-facing entries -------------------------------------
    # A caller that owns the whole block (see ``impls.decoder_block``)
    # can do two things a norm-boundary caller cannot: resolve the step
    # once and share it across the producers on the same conditioning
    # stream, and hand this producer the residual that is still pending
    # from the previous sublayer. The kernel already computes
    # ``residual + x * gate`` before it norms — at the norm boundary
    # there is nothing to hand it, so the residual is zeroed and the
    # host pays a separate elementwise add. These entries expose the
    # wider contract without changing the standalone one below.

    @property
    def can_absorb(self) -> bool:
        """Whether this producer can fold a pending gated residual."""
        return self.out_fp8 and self.norm == "rms"

    def resolve(self, cond: torch.Tensor) -> torch.Tensor:
        """Step index for this conditioning — shareable across siblings."""
        return self.locator(cond)

    def _style2d(self, idx: torch.Tensor) -> torch.Tensor:
        style = self.styles.index_select(0, idx)
        return style.expand(self.resid.shape[0], -1).contiguous()

    def produce(self, x: torch.Tensor, idx: torch.Tensor):
        """Normed output plus the full-width gate, both 2D.

        The standalone ``forward`` slices the gate down to one row for
        the host's broadcast add; a block caller keeps the full rows so
        it can feed the gate straight back into :meth:`absorb`.
        """
        style2d = self._style2d(idx)
        x2d = x.reshape(-1, x.shape[-1])
        if self.out_fp8:
            self.resid.zero_()      # in-place residual: reset per call
            _, y, gate = self._fn(self.resid, x2d, self.gate_ones,
                                  self.w_ones, style2d, self.act_scale)
        else:
            y, gate = self._fn(x2d, self.w_ones, style2d)
        return y, gate

    def absorb(self, residual: torch.Tensor, x: torch.Tensor,
               gate: torch.Tensor, idx: torch.Tensor):
        """Fold a pending ``residual + x * gate`` into this norm.

        Returns the updated residual stream, the normed (fp8) output and
        this producer's own gate. The residual is copied into the
        kernel's in-place buffer rather than written through: the caller
        owns its tensor and a hidden mutation of it is exactly the kind
        of silent aliasing that only shows up as drift.
        """
        if not self.can_absorb:
            raise ValueError(
                "adaln_producer: absorb needs the rms form with fp8 "
                "output — the plain entry has no residual argument")
        style2d = self._style2d(idx)
        shape = self.resid.shape
        self.resid.copy_(residual.reshape(shape))
        return self._fn(self.resid, x.reshape(shape), gate,
                        self.w_ones, style2d, self.act_scale)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        idx = self.locator(cond)
        if self.norm == "layer":
            style = self.styles.index_select(0, idx)
            scale, shift = style[0].chunk(2, dim=-1)
            y = self._fn(
                x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
                .contiguous(),
                scale.contiguous(), shift.contiguous(),
                self.act_scale)
            return y.reshape(x.shape)
        y, gate = self.produce(x, idx)
        return (y.reshape(x.shape),
                gate[:1].reshape(1, 1, gate.shape[-1]).to(x.dtype))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("host_norm"), name)


def bind_step_locator(pairs, *, max_steps: int = 64,
                      dedup_rtol: float = 1e-5, n_dims: int = 8):
    """Build a locator and the step table from ``(cond, out)`` pairs."""
    conds, outs = _dedup(pairs, max_steps, dedup_rtol)
    return StepLocator(conds, n_dims=n_dims), outs


def bind_style_table(host_proj: torch.nn.Module, pairs, *,
                     locator: StepLocator | None = None,
                     max_steps: int = 64) -> StyleTable:
    """Bind the table-only form onto the conditioning projection."""
    built, styles = bind_step_locator(pairs, max_steps=max_steps)
    return StyleTable(host_proj, styles, locator or built)


def bind_adaln_producer(host_norm: torch.nn.Module, pairs, *,
                        act_scale: torch.Tensor | None = None,
                        rows: int, dim: int,
                        locator: StepLocator | None = None,
                        max_steps: int = 64, norm: str = "rms"):
    """Bind an adaptive-norm producer from real ``(cond, style)`` pairs.

    ``pairs`` come from hooking the host's own conditioning projection
    over at least one full tick, so the stored style rows are exactly
    what the host computed. Pass ``act_scale`` to emit FP8 for a
    downstream packed projection; pass ``locator`` to share the step
    lookup with sibling producers on the same conditioning stream.
    """
    built, styles = bind_step_locator(pairs, max_steps=max_steps)
    return AdaLNProducer(host_norm, styles, locator or built,
                         act_scale, rows, dim, norm=norm)
