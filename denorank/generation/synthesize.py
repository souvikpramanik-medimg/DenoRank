"""Mask-conditioned synthetic H&E generation."""
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def random_deform(mask_u8: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Random rot90 / flip / small affine warp of binary masks (B,1,H,W) -> uint8."""
    g = torch.Generator().manual_seed(seed)
    m = mask_u8.float()
    out = []
    for i in range(m.shape[0]):
        mi = m[i:i + 1]
        k = int(torch.randint(0, 4, (1,), generator=g))
        mi = torch.rot90(mi, k, (2, 3))
        if torch.rand(1, generator=g) < 0.5:
            mi = mi.flip(3)
        ang = (torch.rand(1, generator=g).item() * 2 - 1) * math.radians(15)
        sc = 0.9 + 0.2 * torch.rand(1, generator=g).item()
        tx, ty = [(torch.rand(1, generator=g).item() * 2 - 1) * 0.1 for _ in range(2)]
        theta = torch.tensor([[math.cos(ang) / sc, -math.sin(ang) / sc, tx],
                              [math.sin(ang) / sc, math.cos(ang) / sc, ty]],
                             dtype=torch.float32).unsqueeze(0)
        grid = F.affine_grid(theta, mi.shape, align_corners=False)
        mi = (F.grid_sample(mi, grid, mode="bilinear", padding_mode="reflection",
                            align_corners=False) > 0.5).float()
        out.append(mi)
    return torch.cat(out).to(torch.uint8)


@torch.no_grad()
def generate_from_masks(pipe, masks_u8: torch.Tensor, cfg) -> torch.Tensor:
    """Inpaint nuclei regions of a blank tissue-coloured canvas; returns uint8 (N,3,P,P)."""
    outs = []
    size = masks_u8.shape[-1]
    for i in range(len(masks_u8)):
        mask_pil = Image.fromarray(masks_u8[i, 0].numpy() * 255).resize((512, 512), Image.NEAREST)
        init = Image.new("RGB", (512, 512), (245, 235, 235))
        res = pipe(prompt=cfg.prompt, negative_prompt=cfg.neg_prompt, image=init,
                   mask_image=mask_pil, num_inference_steps=cfg.gen_steps,
                   guidance_scale=cfg.gen_guidance).images[0]
        arr = np.array(res.resize((size, size), Image.LANCZOS))
        outs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(outs)
