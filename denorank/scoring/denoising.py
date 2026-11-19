"""Gradient-free denoising-residual score.

For a synthetic image x with VAE latent z, and noise eps ~ N(0, I) at timestep t,

    r(x) = (1/|T|) * sum_{t in T}  || eps_theta(z_t, t, c) - eps ||^2 ,   T = {100, 300, 500}

Lower residual = the sample looks more like the real-data distribution the generator
was fine-tuned on. Cost: |T| = 3 UNet forward passes per image, and no gradients through
(or even access to) the segmenter.
"""
import torch
import torch.nn.functional as F


def _get_base_unet(model_id: str, device):
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16).to(device)
    unet.eval().requires_grad_(False)
    return unet


@torch.no_grad()
def denoise_residual_scores(pipe, imgs_u8, cfg, device, bs: int = 4, seed: int = 0):
    """Returns a (N,) numpy array of residuals (lower = better).

    ``cfg.den_unet`` selects the epsilon-predictor: ``"lora"`` uses the LoRA-fine-tuned UNet
    held by ``pipe``; ``"base"`` uses the pretrained UNet without adaptation.
    """
    from diffusers import DDPMScheduler

    unet = pipe.unet if cfg.den_unet == "lora" else _get_base_unet(cfg.sd_model_id, device)
    unet.eval()
    vae, tok, tex = pipe.vae, pipe.tokenizer, pipe.text_encoder
    sched = DDPMScheduler.from_config(pipe.scheduler.config)

    tokens = tok(cfg.prompt, padding="max_length", max_length=77,
                 truncation=True, return_tensors="pt").input_ids.to(device)
    text_emb = tex(tokens)[0]
    gen = torch.Generator(device=device).manual_seed(seed)

    res = []
    for i in range(0, len(imgs_u8), bs):
        x = imgs_u8[i:i + bs].float().div(255).mul(2).sub(1).to(device)
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False).half()
        B = x.shape[0]
        lat = vae.encode(x).latent_dist.sample(generator=gen) * 0.18215
        enc = text_emb.expand(B, -1, -1).contiguous()
        errs = []
        for t in cfg.den_timesteps:
            tt = torch.full((B,), int(t), device=device, dtype=torch.long)
            noise = torch.randn(lat.shape, generator=gen, device=device, dtype=lat.dtype)
            pred = unet(sched.add_noise(lat, noise, tt), tt, enc).sample
            errs.append(((pred.float() - noise.float()) ** 2).mean(dim=(1, 2, 3)))
        res.append(torch.stack(errs).mean(0).cpu())
    return torch.cat(res).numpy()
