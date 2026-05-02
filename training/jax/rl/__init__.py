"""JAX-side RL primitives mirroring ``training.rl`` for parity.

Most of the algorithm — ACP tags (`flash_rt.core.rl.acp_tags`),
N-step advantage, per-task threshold (`flash_rt.core.rl.advantage`),
and the numpy reward primitives (`compute_episode_value_targets`,
`compute_dense_rewards_from_targets` in `flash_rt.core.rl.reward`)
is **framework-agnostic** and imported directly from the shared
package; this module only re-implements the bits that touched
torch tensors (distributional value-function head + soft loss).
"""
