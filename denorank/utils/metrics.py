import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def dice_iou(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7):
    """Per-sample Dice and IoU for binary (B,1,H,W) tensors."""
    inter = (pred * gt).sum(dim=(1, 2, 3))
    ps, gs = pred.sum(dim=(1, 2, 3)), gt.sum(dim=(1, 2, 3))
    return (2 * inter) / (ps + gs + eps), (inter + eps) / (ps + gs - inter + eps)


def _hd95_single(pred_np: np.ndarray, gt_np: np.ndarray) -> float:
    pred, gt = pred_np.astype(bool), gt_np.astype(bool)
    diag = float(np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2))
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return diag
    dt_pred, dt_gt = distance_transform_edt(~pred), distance_transform_edt(~gt)
    k = np.ones((3, 3), np.uint8)
    s_pred = pred & ~cv2.erode(pred.astype(np.uint8), k).astype(bool)
    s_gt = gt & ~cv2.erode(gt.astype(np.uint8), k).astype(bool)
    if not s_pred.any() or not s_gt.any():
        return diag
    return float(np.percentile(np.concatenate([dt_gt[s_pred], dt_pred[s_gt]]), 95))


def hd95_batch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean 95th-percentile Hausdorff distance over a batch of (B,1,H,W) masks."""
    p, g = pred.cpu().numpy(), gt.cpu().numpy()
    return float(np.mean([_hd95_single(p[i, 0], g[i, 0]) for i in range(p.shape[0])]))
