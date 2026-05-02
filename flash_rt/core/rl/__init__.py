"""Algorithm primitives for VLA reinforcement learning.

Framework-agnostic: no torch / jax / hardware assumptions in the
prompt-builder side; the value-function and reward modules are
PyTorch-only because the VF model is a torch ``nn.Module``. Used by:

* the inference path (``flash_vla.frontends.*``) — ACP tag builder,
  CFG sampler;
* the training path (``training/rl/``) — value targets, dense
  rewards, soft-bin loss, N-step advantage, per-task threshold,
  distributional VF heads.

The split mirrors the π\\*0.6 paper sections:

* ``acp_tags`` — Section V-B prompt tag conventions
* ``cfg_sampler`` — Appendix E classifier-free guidance combine
* ``reward`` — Equation 5 + Section IV-A soft-bin VF loss
* ``advantage`` — Appendix F N-step advantage + threshold
* ``value_function`` — Section IV-A distributional VF heads
"""

from .acp_tags import (
    ACP_NEGATIVE_TAG,
    ACP_NEGATIVE_VALUE,
    ACP_POSITIVE_TAG,
    ACP_POSITIVE_VALUE,
    ACP_TAG_KEY,
    build_acp_tagged_task,
    build_unconditioned_task,
)
from .advantage import (
    binarize_advantages,
    compute_nstep_advantages,
    compute_per_task_thresholds,
)
from .cfg_sampler import CFGSampler
from .reward import (
    DEFAULT_BIN_MAX,
    DEFAULT_BIN_MIN,
    DEFAULT_NUM_BINS,
    build_bin_centers,
    compute_dense_rewards_from_targets,
    compute_episode_value_targets,
    compute_normalized_value_targets,
    compute_soft_value_loss,
    expected_value_from_logits,
    project_values_to_bins,
)
from .value_function import StandaloneValueFunction, ValueFunctionHead

__all__ = [
    "ACP_NEGATIVE_TAG",
    "ACP_NEGATIVE_VALUE",
    "ACP_POSITIVE_TAG",
    "ACP_POSITIVE_VALUE",
    "ACP_TAG_KEY",
    "CFGSampler",
    "DEFAULT_BIN_MAX",
    "DEFAULT_BIN_MIN",
    "DEFAULT_NUM_BINS",
    "StandaloneValueFunction",
    "ValueFunctionHead",
    "binarize_advantages",
    "build_acp_tagged_task",
    "build_bin_centers",
    "build_unconditioned_task",
    "compute_dense_rewards_from_targets",
    "compute_episode_value_targets",
    "compute_normalized_value_targets",
    "compute_nstep_advantages",
    "compute_per_task_thresholds",
    "compute_soft_value_loss",
    "expected_value_from_logits",
    "project_values_to_bins",
]
