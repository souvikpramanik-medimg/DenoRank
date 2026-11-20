"""Tile loading, patch extraction and the patch dataset."""
import math
import os
import random
from glob import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def resolve_paths(root: str) -> Dict[str, Optional[str]]:
    """Resolve the standard layout::

        root/
          train/{images,masks}
          validation/{images,masks}   (optional; "val" also accepted)
          test/{images,masks}
    """
    def _p(split, kind):
        return os.path.join(root, split, kind)

    val = "validation" if os.path.isdir(os.path.join(root, "validation")) else "val"
    paths = {
        "train_img": _p("train", "images"), "train_mask": _p("train", "masks"),
        "val_img": _p(val, "images"), "val_mask": _p(val, "masks"),
        "test_img": _p("test", "images"), "test_mask": _p("test", "masks"),
    }
    if not os.path.isdir(paths["val_img"]):
        paths["val_img"] = paths["val_mask"] = None   # a validation split is carved from train
    return paths


def load_tiles(img_dir: str, mask_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    ip = sorted(glob(os.path.join(img_dir, "*")))
    mp = sorted(glob(os.path.join(mask_dir, "*")))
    assert len(ip) == len(mp) > 0, f"image/mask count mismatch in {img_dir} / {mask_dir}"
    imgs, masks = [], []
    for a, b in zip(ip, mp):
        im, mk = cv2.imread(a), cv2.imread(b, cv2.IMREAD_GRAYSCALE)
        assert im is not None and mk is not None, f"could not read {a} / {b}"
        imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        masks.append(mk)
    return imgs, masks


def tile_to_patches(img: np.ndarray, mask: np.ndarray, tile: int, patch: int):
    """Resize to tile x tile, binarise the mask, cut non-overlapping patches."""
    img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (tile, tile), interpolation=cv2.INTER_NEAREST)
    thr = 127 if mask.max() > 1 else 0
    mask = (mask > thr).astype(np.uint8)
    ps, ms = [], []
    for y in range(0, tile - patch + 1, patch):
        for x in range(0, tile - patch + 1, patch):
            ps.append(img[y:y + patch, x:x + patch])
            ms.append(mask[y:y + patch, x:x + patch])
    return ps, ms


def build_patch_tensors(imgs, masks, tile: int, patch: int):
    """Returns uint8 tensors: images (N,3,P,P) and masks (N,1,P,P)."""
    P, M = [], []
    for im, mk in zip(imgs, masks):
        ps, ms = tile_to_patches(im, mk, tile, patch)
        P += ps; M += ms
    return (torch.from_numpy(np.stack(P)).permute(0, 3, 1, 2).contiguous(),
            torch.from_numpy(np.stack(M)).unsqueeze(1).contiguous())


def choose_labeled_tiles(n_tiles: int, frac: float, seed: int) -> List[int]:
    """Pick ceil(frac * n_tiles) labelled tiles, deterministically."""
    n = max(1, int(math.ceil(frac * n_tiles)))
    return sorted(np.random.RandomState(seed).permutation(n_tiles)[:n].tolist())


class PatchDataset(Dataset):
    """uint8 patch tensors -> float tensors with optional flip/rot/intensity augmentation."""

    def __init__(self, imgs_u8: torch.Tensor, masks_u8: torch.Tensor, augment: bool = False):
        self.imgs, self.masks, self.augment = imgs_u8, masks_u8, augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = self.imgs[i].float() / 255.0
        m = self.masks[i].float()
        if self.augment:
            k = random.randint(0, 3)
            x, m = torch.rot90(x, k, (1, 2)), torch.rot90(m, k, (1, 2))
            if random.random() < 0.5:
                x, m = x.flip(2), m.flip(2)
            if random.random() < 0.5:
                x, m = x.flip(1), m.flip(1)
            if random.random() < 0.7:
                x = x * (0.8 + 0.4 * random.random())
                mean = x.mean()
                x = ((x - mean) * (0.8 + 0.4 * random.random()) + mean).clamp(0, 1)
        return x, m
