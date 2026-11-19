"""LoRA fine-tuning of the Stable Diffusion UNet on real H&E patches."""
import os

import torch
import torch.nn.functional as F

_PIPE_CACHE = {}


def load_sd_lora(real_imgs_u8: torch.Tensor, ckpt_dir: str, cfg, device):
    """Load Stable Diffusion (inpainting pipeline) and fine-tune the UNet with LoRA.

    The adapter is cached in ``ckpt_dir``; later calls reuse it. The epsilon-prediction
    objective is the standard denoising loss  E || eps - eps_theta(x_t, t, c) ||^2.
    """
    if "pipe" in _PIPE_CACHE:
        return _PIPE_CACHE["pipe"]
    from diffusers import DDPMScheduler, StableDiffusionInpaintPipeline
    from peft import LoraConfig, get_peft_model

    os.makedirs(ckpt_dir, exist_ok=True)
    lora_path = os.path.join(ckpt_dir, "lora_unet")
    lora_done = os.path.join(ckpt_dir, "lora_done.flag")

    print(f"[sd] loading {cfg.sd_model_id} ...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        cfg.sd_model_id, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False).to(device)
    pipe.set_progress_bar_config(disable=True)

    if os.path.exists(lora_done):
        print("[sd] loading cached LoRA")
        pipe.unet.load_attn_procs(lora_path)
    else:
        print(f"[sd] LoRA training on {len(real_imgs_u8)} patches ...")
        unet = get_peft_model(pipe.unet, LoraConfig(
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"], lora_dropout=0.05))
        unet.train()
        vae, text_enc, tokenizer = pipe.vae, pipe.text_encoder, pipe.tokenizer
        vae.requires_grad_(False)
        text_enc.requires_grad_(False)
        sched = DDPMScheduler.from_pretrained(cfg.sd_model_id, subfolder="scheduler")
        opt = torch.optim.AdamW(unet.parameters(), lr=cfg.lora_lr)
        tokens = tokenizer(cfg.prompt, padding="max_length", max_length=77,
                           truncation=True, return_tensors="pt").input_ids.to(device)
        for ep in range(cfg.lora_epochs):
            total = 0.0
            for i in torch.randperm(len(real_imgs_u8)):
                img = real_imgs_u8[i:i + 1].float().div(255).mul(2).sub(1).to(device)
                img = F.interpolate(img, size=(512, 512), mode="bilinear", align_corners=False).half()
                with torch.no_grad():
                    lat = vae.encode(img).latent_dist.sample() * 0.18215
                    enc = text_enc(tokens)[0]
                noise = torch.randn_like(lat)
                t = torch.randint(0, sched.config.num_train_timesteps, (1,), device=device).long()
                noisy = sched.add_noise(lat, noise, t)
                loss = F.mse_loss(unet(noisy, t, enc).sample.float(), noise.float())
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            print(f"  [lora] epoch {ep + 1}/{cfg.lora_epochs} loss {total / len(real_imgs_u8):.4f}")
        unet.save_pretrained(lora_path)
        open(lora_done, "w").close()

    _PIPE_CACHE["pipe"] = pipe
    return pipe
