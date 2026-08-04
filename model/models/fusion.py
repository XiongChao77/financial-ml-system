import torch


def probabilities_to_logits(probabilities: torch.Tensor) -> torch.Tensor:
    """Convert normalized probabilities to numerically safe log-probabilities."""
    return torch.log(probabilities.clamp_min(1e-8))


def fuse_trigger_direction_logits(
    trigger_logits: torch.Tensor,
    direction_logits: torch.Tensor,
) -> torch.Tensor:
    """Fuse trigger/direction logits into [Short, Neutral, Long] probabilities."""
    trigger_probs = torch.softmax(trigger_logits, dim=1)
    direction_probs = torch.softmax(direction_logits, dim=1)

    p_action = trigger_probs[:, 1]
    p_short = p_action * direction_probs[:, 0]
    p_neutral = trigger_probs[:, 0]
    p_long = p_action * direction_probs[:, 1]
    return torch.stack((p_short, p_neutral, p_long), dim=1)
