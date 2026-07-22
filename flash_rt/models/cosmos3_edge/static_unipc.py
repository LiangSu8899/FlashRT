"""Static UniPC scheduler step for the Cosmos3-Edge 30-step AV path."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch

from flash_rt.models.cosmos3_edge.dump_replay import EDGE_NUM_TRAIN_TIMESTEPS, EDGE_SHIFT
from flash_rt.models.cosmos3_video.fm_solvers_unipc import FlowUniPCMultistepScheduler


@dataclass(frozen=True)
class EdgeUniPCCoefficients:
    sigma: float
    corrector_order: int
    predictor_order: int
    c_sample: float
    c_last: float
    c_prev_m1: float
    c_prev_m2: float
    c_curr_m: float
    p_sample: float
    p_curr_m: float
    p_prev_m1: float


def _lambda(sigma: torch.Tensor) -> torch.Tensor:
    if float(sigma.item()) == 0.0:
        return torch.tensor(float("inf"), dtype=torch.float32)
    alpha = torch.tensor(1.0, dtype=torch.float32) - sigma.to(dtype=torch.float32)
    return torch.log(alpha) - torch.log(sigma.to(dtype=torch.float32))


def _predictor_coefficients(sigmas: torch.Tensor, step: int, order: int) -> tuple[float, float, float]:
    sigma_t = sigmas[step + 1].to(dtype=torch.float32)
    sigma_s0 = sigmas[step].to(dtype=torch.float32)
    alpha_t = torch.tensor(1.0, dtype=torch.float32) - sigma_t
    h = _lambda(sigma_t) - _lambda(sigma_s0)
    hh = -h
    h_phi_1 = torch.expm1(hh)
    b_h = torch.expm1(hh)
    sample_coeff = float((sigma_t / sigma_s0).item())
    curr_m_coeff = float((-alpha_t * h_phi_1).item())
    prev_m1_coeff = 0.0

    if order == 2:
        rk = (_lambda(sigmas[step - 1]) - _lambda(sigma_s0)) / h
        coeff = -alpha_t * b_h * torch.tensor(0.5, dtype=torch.float32) / rk
        prev_m1_coeff += float(coeff.item())
        curr_m_coeff += float((-coeff).item())
    elif order != 1:
        raise ValueError(f"unsupported UniPC predictor order: {order}")

    return sample_coeff, curr_m_coeff, prev_m1_coeff


def _corrector_coefficients(sigmas: torch.Tensor, step: int, order: int) -> tuple[float, float, float, float]:
    sigma_t = sigmas[step].to(dtype=torch.float32)
    sigma_s0 = sigmas[step - 1].to(dtype=torch.float32)
    alpha_t = torch.tensor(1.0, dtype=torch.float32) - sigma_t
    h = _lambda(sigma_t) - _lambda(sigma_s0)
    hh = -h
    h_phi_1 = torch.expm1(hh)
    b_h = torch.expm1(hh)

    last_coeff = float((sigma_t / sigma_s0).item())
    prev_m1_coeff = float((-alpha_t * h_phi_1).item())
    prev_m2_coeff = 0.0
    curr_m_coeff = 0.0

    if order == 1:
        coeff = -alpha_t * b_h * torch.tensor(0.5, dtype=torch.float32)
        curr_m_coeff += float(coeff.item())
        prev_m1_coeff += float((-coeff).item())
    elif order == 2:
        rk = (_lambda(sigmas[step - 2]) - _lambda(sigma_s0)) / h
        rks = torch.tensor([rk.item(), 1.0], dtype=torch.float32)
        h_phi_k = h_phi_1 / hh - 1
        factorial_i = 1
        matrix_rows = []
        rhs = []
        for power in range(1, order + 1):
            matrix_rows.append(torch.pow(rks, power - 1))
            rhs.append(h_phi_k * factorial_i / b_h)
            factorial_i *= power + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        rhos = torch.linalg.solve(torch.stack(matrix_rows), torch.tensor(rhs, dtype=torch.float32))
        coeff = -alpha_t * b_h
        prev_m2_coeff += float((coeff * rhos[0] / rk).item())
        prev_m1_coeff += float((coeff * rhos[0] * (-1.0 / rk)).item())
        curr_m_coeff += float((coeff * rhos[1]).item())
        prev_m1_coeff += float((-coeff * rhos[1]).item())
    else:
        raise ValueError(f"unsupported UniPC corrector order: {order}")

    return last_coeff, prev_m1_coeff, prev_m2_coeff, curr_m_coeff


def _precompute_coefficients(num_steps: int, *, shift: float) -> tuple[torch.Tensor, tuple[EdgeUniPCCoefficients, ...]]:
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=EDGE_NUM_TRAIN_TIMESTEPS,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(num_steps, device="cpu", shift=shift)
    sigmas = scheduler.sigmas.to(dtype=torch.float32)
    coeffs: list[EdgeUniPCCoefficients] = []
    for step in range(num_steps):
        if step == 0:
            corrector_order = 0
            c_sample, c_last, c_prev_m1, c_prev_m2, c_curr_m = 1.0, 0.0, 0.0, 0.0, 0.0
        else:
            corrector_order = 1 if step == 1 else 2
            c_last, c_prev_m1, c_prev_m2, c_curr_m = _corrector_coefficients(
                sigmas,
                step,
                corrector_order,
            )
            c_sample = 0.0

        predictor_order = min(2, num_steps - step)
        predictor_order = min(predictor_order, step + 1)
        p_sample, p_curr_m, p_prev_m1 = _predictor_coefficients(sigmas, step, predictor_order)
        coeffs.append(
            EdgeUniPCCoefficients(
                sigma=float(sigmas[step].item()),
                corrector_order=corrector_order,
                predictor_order=predictor_order,
                c_sample=c_sample,
                c_last=c_last,
                c_prev_m1=c_prev_m1,
                c_prev_m2=c_prev_m2,
                c_curr_m=c_curr_m,
                p_sample=p_sample,
                p_curr_m=p_curr_m,
                p_prev_m1=p_prev_m1,
            )
        )
    return scheduler.timesteps, tuple(coeffs)


class EdgeStaticUniPCScheduler:
    """One-launch native UniPC update for fixed Cosmos3-Edge dump geometry."""

    def __init__(self, num_steps: int, *, device: torch.device, shift: float = EDGE_SHIFT):
        self.device = torch.device(device)
        self.timesteps, self.coefficients = _precompute_coefficients(num_steps, shift=shift)
        self.timesteps = self.timesteps.to(device=self.device, dtype=torch.int64)
        self.native_step = None
        if self.device.type == "cuda" and os.environ.get("FLASHRT_COSMOS3_EDGE_NATIVE_UNIPC", "1") != "0":
            try:
                import flash_rt.flash_rt_kernels as fvk

                self.native_step = getattr(fvk, "cosmos3_edge_unipc_step_f32_bf16", None)
            except Exception:
                self.native_step = None
        self.prev_m1: torch.Tensor | None = None
        self.prev_m2: torch.Tensor | None = None
        self.last_sample: torch.Tensor | None = None
        self.current_m: torch.Tensor | None = None
        self.current_last_sample: torch.Tensor | None = None

    @property
    def native_available(self) -> bool:
        return self.native_step is not None

    def reset(self, sample: torch.Tensor) -> None:
        self.prev_m1 = torch.empty_like(sample)
        self.prev_m2 = torch.empty_like(sample)
        self.last_sample = torch.empty_like(sample)
        self.current_m = torch.empty_like(sample)
        self.current_last_sample = torch.empty_like(sample)

    def step(self, sample: torch.Tensor, velocity: torch.Tensor, step_index: int) -> torch.Tensor:
        if self.native_step is None:
            raise RuntimeError("native UniPC step binding is not available")
        if sample.dtype != torch.float32 or velocity.dtype != torch.bfloat16:
            raise TypeError(f"expected sample=float32 and velocity=bf16, got {sample.dtype} and {velocity.dtype}")
        if sample.device.type != "cuda" or velocity.device.type != "cuda":
            raise TypeError("native UniPC step requires CUDA tensors")
        if sample.shape != velocity.shape:
            raise ValueError(
                f"native UniPC step requires matching shapes, got {sample.shape} and {velocity.shape}"
            )
        if not sample.is_contiguous() or not velocity.is_contiguous():
            raise ValueError("native UniPC step requires contiguous sample and velocity tensors")
        if step_index < 0 or step_index >= len(self.coefficients):
            raise IndexError(f"UniPC step_index {step_index} is outside the configured schedule")
        if self.prev_m1 is None or self.prev_m2 is None or self.last_sample is None:
            self.reset(sample)
        assert self.prev_m1 is not None
        assert self.prev_m2 is not None
        assert self.last_sample is not None
        assert self.current_m is not None
        assert self.current_last_sample is not None

        coeff = self.coefficients[step_index]
        prev_m1_ptr = self.prev_m1.data_ptr() if coeff.corrector_order >= 1 or coeff.predictor_order >= 2 else 0
        prev_m2_ptr = self.prev_m2.data_ptr() if coeff.corrector_order >= 2 else 0
        last_ptr = self.last_sample.data_ptr() if coeff.corrector_order >= 1 else 0
        self.native_step(
            sample.data_ptr(),
            velocity.data_ptr(),
            prev_m1_ptr,
            prev_m2_ptr,
            last_ptr,
            sample.data_ptr(),
            self.current_m.data_ptr(),
            self.current_last_sample.data_ptr(),
            sample.numel(),
            coeff.sigma,
            coeff.corrector_order,
            coeff.predictor_order,
            coeff.c_sample,
            coeff.c_last,
            coeff.c_prev_m1,
            coeff.c_prev_m2,
            coeff.c_curr_m,
            coeff.p_sample,
            coeff.p_curr_m,
            coeff.p_prev_m1,
            torch.cuda.current_stream().cuda_stream,
        )

        old_prev_m2 = self.prev_m2
        old_last = self.last_sample
        self.prev_m2 = self.prev_m1
        self.prev_m1 = self.current_m
        self.current_m = old_prev_m2
        self.last_sample = self.current_last_sample
        self.current_last_sample = old_last
        return sample
