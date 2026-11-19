import numpy as np
import torch
import torch.nn.functional as F


def pixel_copy_guard(syn_imgs, real_imgs, thresh: float = 0.10):
    """Flag synthetic images that are near-copies of a real labelled patch (32x32 L2).

    Returns (keep_mask, min_distance).
    """
    def ds(x):
        return F.interpolate(x.float() / 255, size=(32, 32), mode="bilinear").reshape(len(x), -1)

    D = torch.cdist(ds(syn_imgs), ds(real_imgs))
    min_d = D.min(dim=1).values.numpy()
    return min_d >= thresh, min_d
