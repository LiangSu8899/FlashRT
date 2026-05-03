"""FP32 PyTorch eager reference for Pi0.5 CFG inference (oracle R_fp32).

Implements the classifier-free guidance flow from the π*0.6 paper
(arXiv:2511.14759, Appendix E, Eq. 13) on top of the upstream openpi
``PI0Pytorch`` model. The reference runs the model TWICE per
denoising step — once with the conditioned prompt
(``"<task>\\nAdvantage: positive"``) and once with the unconditioned
prompt (``"<task>"``) — and combines the two velocity predictions
into a single guided velocity:

    v_guided = (1 - beta) * v_uncond + beta * v_cond            (Eq. 13)
    a^{k+1}  = a^k + dt * v_guided                              (Euler step)

with ``a^0 ~ N(0, I)`` and ``dt = -1 / num_steps``. Boundary
condition matches openpi's :meth:`PI0Pytorch.sample_actions`.

This module's outputs are the ground-truth FlashRT verifies its
optimised serial / batched FP8 pipelines against (correctness
contract C2/C3/C5 in ``docs/precision_spec.md``). The reference is
intentionally slow and dependency-heavy; it is **only** imported by
tests and offline fixture generation, never by production code.

Usage
-----

    from flash_rt.refs.pi05_cfg_reference import Pi05CFGReference

    ref = Pi05CFGReference(
        config_name="pi05_libero",
        checkpoint_dir="<ckpts>/pi05_libero_pytorch",
        device="cuda",
    )
    out = ref.infer(
        obs={"observation/image": ..., "observation/wrist_image": ...,
             "observation/state": ..., "prompt": "pick up the cup"},
        beta=1.5,
        noise=np.ndarray((10, 32), dtype=np.float32),  # optional
    )
    out["actions"]               # (action_horizon, action_dim) np.float32
    out["v_cond_per_step"]       # (num_steps, action_horizon, action_dim)
    out["v_uncond_per_step"]     # (num_steps, action_horizon, action_dim)
    out["noise_per_step"]        # (num_steps + 1, action_horizon, action_dim)
                                  # noise_per_step[0] = initial noise
                                  # noise_per_step[k+1] = noise after step k
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ACP prompt format — matches flash_rt.core.rl.acp_tags. Duplicated
# here so the reference has zero dependency on flash_rt internals.
_ACP_POSITIVE_TAG = "Advantage: positive"


def _build_cond_prompt(task: str | None) -> str:
    base = task or ""
    return f"{base}\n{_ACP_POSITIVE_TAG}" if base else _ACP_POSITIVE_TAG


def _build_uncond_prompt(task: str | None) -> str:
    return task or ""


class Pi05CFGReference:
    """Independent PyTorch eager CFG reference for Pi0.5 / RECAP.

    Wraps the upstream ``openpi`` ``PI0Pytorch`` model with a manual
    classifier-free-guidance denoising loop. Per-step velocity tensors
    are exposed for fine-grained correctness checks against FlashRT's
    serial and batched paths.
    """

    def __init__(
        self,
        *,
        config_name: str = "pi05_libero",
        checkpoint_dir: str,
        device: str = "cuda",
        dtype: str = "bf16",
    ) -> None:
        try:
            from openpi.policies import policy_config
            from openpi.training import config as _openpi_config
        except ImportError as e:
            raise RuntimeError(
                "Pi05CFGReference requires the upstream openpi package "
                "(installed at /workspace/openpi in the pi0-stablehlo "
                "container). Reference inference is offline-only — do "
                "not use this class from production code paths.") from e

        # Patch openpi's load_pytorch to be tolerant of safetensors
        # version drift (mirrors torch_sample/pi05_inference.py).
        self._patch_openpi_loader()

        self._cfg_config = _openpi_config.get_config(config_name)
        self._policy = policy_config.create_trained_policy(
            self._cfg_config, checkpoint_dir,
            pytorch_device=device, default_prompt=None)
        self._model = self._policy._model  # openpi PI0Pytorch instance
        self._device = device
        if dtype == "bf16":
            self._compute_dtype = torch.bfloat16
        elif dtype == "fp32":
            self._compute_dtype = torch.float32
        else:
            raise ValueError(f"unsupported dtype {dtype!r}")
        self._num_steps = int(getattr(self._model.config, "num_steps", 10))
        self._action_dim = int(self._model.config.action_dim)
        self._action_horizon = int(self._model.config.action_horizon)

    # ── one-time openpi loader patch (mirrors torch_sample) ──

    @staticmethod
    def _patch_openpi_loader() -> None:
        if getattr(Pi05CFGReference, "_patched_loader", False):
            return
        import safetensors.torch as _st
        from openpi.models_pytorch import pi0_pytorch as _pi0pt

        def _load_pytorch_patched(self, train_config, weight_path: str):
            model = _pi0pt.PI0Pytorch(config=train_config.model)
            state_dict = _st.load_file(weight_path)
            model.load_state_dict(state_dict, strict=False)
            return model

        from openpi.models import model as _model_mod
        for _cls in vars(_model_mod).values():
            if isinstance(_cls, type) and hasattr(_cls, "load_pytorch"):
                _cls.load_pytorch = _load_pytorch_patched
        Pi05CFGReference._patched_loader = True

    # ─────────────────────────────────────────────────────────────────
    # Internal: build a prefix kv-cache for one prompt
    # ─────────────────────────────────────────────────────────────────

    def _prepare_inputs(self, obs: dict, prompt_text: str) -> dict:
        """Replicate ``Policy.infer`` input transforms for one prompt.

        Returns a dict with model-ready tensors on the configured device.
        """
        import jax  # openpi's transforms use jax.tree.map

        inputs = jax.tree.map(lambda x: x, {**obs, "prompt": prompt_text})
        inputs = self._policy._input_transform(inputs)
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.asarray(x)).to(self._device)[None, ...],
            inputs)
        return inputs

    def _build_prefix_cache(self, inputs: dict):
        """Embed prefix (vision + lang) and return (kv_cache, pad_mask).

        Mirrors the kv-cache construction inside
        :meth:`PI0Pytorch.sample_actions` (lines 383-399) so each prompt
        gets its own cached keys/values that the per-step decoder can
        cross-attend to.
        """
        from openpi.models import model as _model_mod
        observation = _model_mod.Observation.from_dict(inputs)
        images, img_masks, lang_tokens, lang_masks, state = (
            self._model._preprocess_observation(observation, train=False))

        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self._model.embed_prefix(images, img_masks, lang_tokens, lang_masks))
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._model._prepare_attention_masks_4d(
            prefix_att_2d_masks)

        # Force eager attention so kv-cache can be reused for multi-step
        # cross-attention (matches openpi's sample_actions setup).
        self._model.paligemma_with_expert.paligemma.language_model.config\
            ._attn_implementation = "eager"
        _, past_key_values = self._model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return past_key_values, prefix_pad_masks, state

    # ─────────────────────────────────────────────────────────────────
    # Public: CFG inference
    # ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def infer(
        self,
        obs: dict,
        *,
        beta: float = 1.5,
        noise: Optional[np.ndarray] = None,
    ) -> dict:
        """Run CFG denoising with per-step trace.

        Args:
            obs: openpi-style observation dict (must contain
                ``observation/image``, ``observation/wrist_image``,
                ``observation/state``, and ``prompt``).
            beta: CFG guidance strength (>= 1.0). ``beta == 1.0``
                collapses to cond-only (formula identity:
                ``v_uncond + 1*(v_cond - v_uncond) = v_cond``).
            noise: optional fixed initial noise of shape
                ``(action_horizon, action_dim)`` or
                ``(1, action_horizon, action_dim)``. When ``None``
                a fresh ``N(0, I)`` sample is drawn on-device.

        Returns:
            dict with keys ``actions``, ``v_cond_per_step``,
            ``v_uncond_per_step``, ``noise_per_step``,
            all ``np.float32`` numpy arrays with the batch dim
            stripped.
        """
        if beta < 1.0:
            raise ValueError(f"beta must be >= 1.0; got {beta}")
        task = obs.get("prompt", "")
        cond_inputs = self._prepare_inputs(obs, _build_cond_prompt(task))
        uncond_inputs = self._prepare_inputs(obs, _build_uncond_prompt(task))

        cond_cache, cond_pad, state_cond = self._build_prefix_cache(cond_inputs)
        uncond_cache, uncond_pad, state_uncond = self._build_prefix_cache(
            uncond_inputs)
        # State is identical for both prompts (only ``prompt`` differs).
        # Use the cond branch's state tensor as canonical.
        state = state_cond
        bsize = state.shape[0]

        # Initial noise. To compare apples-to-apples with FlashRT, the
        # reference samples noise via the *same* RNG path that
        # ``Pi05TorchFrontendRtx`` uses internally — a CUDA BF16
        # ``.normal_()`` — and casts to FP32 only for the denoising
        # math. Under a shared ``torch.manual_seed`` this guarantees
        # both implementations enter step 0 with byte-equivalent
        # initial noise. An explicit ``noise=`` argument bypasses this
        # and lets callers inject any seed-source they want (e.g. a
        # numpy array regenerated from a fixture).
        actions_shape = (bsize, self._action_horizon, self._action_dim)
        if noise is not None:
            noise_arr = np.asarray(noise, dtype=np.float32)
            if noise_arr.ndim == 2:
                noise_arr = noise_arr[None, ...]
            x_t = torch.from_numpy(noise_arr).to(
                device=self._device, dtype=torch.float32)
        else:
            x_t_bf16 = torch.empty(
                actions_shape, dtype=torch.bfloat16, device=self._device)
            x_t_bf16.normal_()
            x_t = x_t_bf16.to(torch.float32)

        dt = torch.tensor(-1.0 / self._num_steps, dtype=torch.float32,
                          device=self._device)
        time = torch.tensor(1.0, dtype=torch.float32, device=self._device)

        v_cond_traj = []
        v_uncond_traj = []
        noise_traj = [x_t.detach().cpu().numpy().copy()]

        # Force eager attention on the action-expert side too so the
        # kv-cache reuse path matches sample_actions.
        self._model.paligemma_with_expert.gemma_expert.model.config\
            ._attn_implementation = "eager"

        step = 0
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_cond = self._model.denoise_step(
                state, cond_pad, cond_cache, x_t, expanded_time)
            v_uncond = self._model.denoise_step(
                state, uncond_pad, uncond_cache, x_t, expanded_time)
            # Float32 combine (paper math has no quantization here).
            v_guided = (1.0 - beta) * v_uncond + beta * v_cond
            x_t = x_t + dt * v_guided
            time = time + dt
            v_cond_traj.append(v_cond.detach().cpu().numpy().copy())
            v_uncond_traj.append(v_uncond.detach().cpu().numpy().copy())
            noise_traj.append(x_t.detach().cpu().numpy().copy())
            step += 1

        actions_raw = x_t[0].detach().cpu().numpy().astype(np.float32)

        # Apply the policy's output transforms (unnormalize + slice to
        # robot DOF) so the returned ``actions`` match what FlashRT's
        # ``infer()`` produces. Per-step velocities and noise are kept
        # in the model's native (normalized, full-action_dim) space —
        # that is where C2/C3 oracle comparisons happen.
        actions_post = self._policy._output_transform({
            "state": cond_inputs["state"][0].detach().cpu().numpy(),
            "actions": actions_raw,
        })["actions"]

        return {
            "actions": actions_post.astype(np.float32),
            "actions_raw": actions_raw,
            "v_cond_per_step": np.stack(
                [v[0] for v in v_cond_traj]).astype(np.float32),
            "v_uncond_per_step": np.stack(
                [v[0] for v in v_uncond_traj]).astype(np.float32),
            "noise_per_step": np.stack(
                [n[0] for n in noise_traj]).astype(np.float32),
        }
