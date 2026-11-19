import torch
import torch.nn.functional as F


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    """Weighted sum of BCE and soft-Dice loss."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    p = torch.sigmoid(logits)
    num = 2 * (p * targets).sum(dim=(2, 3))
    den = p.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1e-7
    return bce_weight * bce + (1 - bce_weight) * (1 - num / den).mean()
