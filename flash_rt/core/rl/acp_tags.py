"""ACP (Advantage-Conditioned Policy) prompt tags.

Shared tag definitions used by both training and inference. The
advantage indicator is injected as text appended to the task prompt,
matching the formulation in the π*0.6 paper (arXiv:2511.14759, Section
V-B). At inference time, the conditioned and unconditioned prompts
provide the two distributions that classifier-free guidance interpolates
between.

Training prompts:
    "task description\\nAdvantage: positive"   when I_t = 1
    "task description\\nAdvantage: negative"   when I_t = 0
    "task description"                          when advantage dropout fires

Inference prompts (β > 1 CFG):
    Conditioned:    "task description\\nAdvantage: positive"
    Unconditioned:  "task description"
"""

ACP_TAG_KEY = "Advantage"
ACP_POSITIVE_VALUE = "positive"
ACP_NEGATIVE_VALUE = "negative"

ACP_POSITIVE_TAG = f"{ACP_TAG_KEY}: {ACP_POSITIVE_VALUE}"
ACP_NEGATIVE_TAG = f"{ACP_TAG_KEY}: {ACP_NEGATIVE_VALUE}"


def build_acp_tagged_task(task: str | None, is_positive: bool) -> str:
    """Build a task string with the ACP advantage tag appended.

    Args:
        task: Base task description (e.g. "pick up the red cup"). May be
            empty or ``None``; the returned string then consists of the
            advantage tag alone.
        is_positive: ``True`` to append the positive advantage tag,
            ``False`` for the negative tag.

    Returns:
        The original task with ``"\\nAdvantage: <positive|negative>"``
        appended, or just the tag when ``task`` is empty.
    """
    tag = ACP_POSITIVE_TAG if is_positive else ACP_NEGATIVE_TAG
    base_task = task or ""
    if not base_task:
        return tag
    return f"{base_task}\n{tag}"


def build_unconditioned_task(task: str | None) -> str:
    """Return the unconditioned prompt (no advantage tag) for CFG inference."""
    return task or ""
