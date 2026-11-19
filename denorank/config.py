"""Central experiment configuration."""
from dataclasses import dataclass, field
from typing import List, Tuple

GEN_PROMPT = "H&E stained histopathology image of cell nuclei"
GEN_NEG_PROMPT = "blurry, cartoon, low quality, distorted"


@dataclass
class Config:
    # --- data ---
    tile: int = 512                 # tiles are resized to tile x tile
    patch: int = 128                # non-overlapping patches cut from each tile
    label_fractions: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.25])
    seeds: List[int] = field(default_factory=lambda: [42])

    # --- generator (Stable Diffusion + LoRA) ---
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    prompt: str = GEN_PROMPT
    neg_prompt: str = GEN_NEG_PROMPT
    gen_steps: int = 25
    gen_guidance: float = 7.5
    gen_mult: int = 4               # synthetic masks per real labelled patch
    min_mask_frac: float = 0.01     # drop deformed masks with < 1% foreground
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_lr: float = 1e-4
    lora_epochs: int = 20

    # --- denoising-residual score ---
    den_timesteps: Tuple[int, ...] = (100, 300, 500)   # 3 forward passes
    den_unet: str = "lora"          # "lora" (fine-tuned) or "base" (pretrained)

    # --- segmenter ---
    segmenter: str = "unet"
    seg_base: int = 32
    seg_bs: int = 4
    seg_lr: float = 1e-4
    ens_k: int = 2                  # reference ensemble size
    ens_epochs: int = 12
    final_epochs: int = 20

    # --- selection ---
    copy_l2_thresh: float = 0.10
    max_keep_frac: float = 0.70
    min_keep_frac: float = 0.20
    default_keep_frac: float = 0.50

    # --- misc ---
    out_dir: str = "runs"
    time_budget_hours: float = 0.0  # 0 = unlimited
    visualize: bool = True
