import torch

from denorank.data.dataset import tile_to_patches
from denorank.models import build_segmenter, register_segmenter
import numpy as np


def test_unet_forward_shape():
    net = build_segmenter("unet", base=8)
    assert net(torch.randn(2, 3, 64, 64)).shape == (2, 1, 64, 64)


def test_register_custom_segmenter():
    register_segmenter("tiny", lambda base=8: torch.nn.Conv2d(3, 1, 1))
    assert build_segmenter("tiny")(torch.randn(1, 3, 16, 16)).shape == (1, 1, 16, 16)


def test_tile_to_patches_count():
    img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    mk = (np.random.rand(500, 500) > 0.5).astype(np.uint8) * 255
    ps, ms = tile_to_patches(img, mk, 512, 128)
    assert len(ps) == len(ms) == 16 and ps[0].shape == (128, 128, 3)
