"""Ensemble-based evidence scores (baselines): prediction consistency and epistemic uncertainty."""
import cv2
import numpy as np
import torch


def _binary_entropy(p, eps: float = 1e-7):
    return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))


def _count_area_match(pred_np, gt_np):
    out = []
    for p, g in zip(pred_np, gt_np):
        n_p = cv2.connectedComponents(p[0].astype(np.uint8))[0] - 1
        n_g = cv2.connectedComponents(g[0].astype(np.uint8))[0] - 1
        count = 1.0 - min(1.0, abs(n_p - n_g) / max(n_g, 1))
        a_p, a_g = float(p.sum()), float(g.sum())
        out.append(0.5 * count + 0.5 * min(a_p, a_g) / max(a_p, a_g, 1.0))
    return np.array(out, dtype=np.float32)


@torch.no_grad()
def ensemble_scores(models, imgs_u8, masks_u8, device, bs: int = 4):
    """Returns (consistency, epistemic).

    consistency: higher = ensemble prediction agrees with the conditioning mask (better)
    epistemic:   mutual information between prediction and ensemble member (lower = better)
    """
    for m in models:
        m.eval()
    cons, epi = [], []
    for i in range(0, len(imgs_u8), bs):
        x = imgs_u8[i:i + bs].float().div(255).to(device)
        g = masks_u8[i:i + bs].float().to(device)
        P = torch.stack([torch.sigmoid(m(x)) for m in models])
        mean = P.mean(0)
        mi = (_binary_entropy(mean) - _binary_entropy(P).mean(0)).mean(dim=(1, 2, 3))
        pred = (mean > 0.5).float()
        inter = (pred * g).sum(dim=(1, 2, 3))
        dice = (2 * inter) / (pred.sum(dim=(1, 2, 3)) + g.sum(dim=(1, 2, 3)) + 1e-7)
        cm = _count_area_match(pred.cpu().numpy(), g.cpu().numpy())
        cons.append(0.7 * dice.cpu().numpy() + 0.3 * cm)
        epi.append(mi.cpu().numpy())
    return np.concatenate(cons), np.concatenate(epi)
