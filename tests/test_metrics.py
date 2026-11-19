import torch

from denorank.utils.losses import dice_bce_loss
from denorank.utils.metrics import dice_iou, hd95_batch


def test_dice_iou_perfect_and_disjoint():
    a = torch.zeros(1, 1, 8, 8); a[..., :4, :] = 1
    b = torch.zeros(1, 1, 8, 8); b[..., 4:, :] = 1
    d, i = dice_iou(a, a)
    assert d.item() > 0.999 and i.item() > 0.999
    d, _ = dice_iou(a, b)
    assert d.item() < 1e-3


def test_hd95_identical_is_zero():
    a = torch.zeros(1, 1, 16, 16); a[..., 4:12, 4:12] = 1
    assert hd95_batch(a, a) == 0.0


def test_loss_decreases_with_better_logits():
    t = torch.zeros(1, 1, 8, 8); t[..., :4, :] = 1
    good = (t * 2 - 1) * 5
    assert dice_bce_loss(good, t) < dice_bce_loss(-good, t)
