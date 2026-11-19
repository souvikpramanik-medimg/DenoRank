"""Segmenter training and evaluation."""
import torch
from torch.utils.data import DataLoader

from denorank.data.dataset import PatchDataset
from denorank.models.segmenter import build_segmenter
from denorank.utils.losses import dice_bce_loss
from denorank.utils.metrics import dice_iou, hd95_batch
from denorank.utils.seed import seed_everything


@torch.no_grad()
def predict_probs(model, imgs_u8, device, bs: int = 4):
    model.eval()
    out = []
    for i in range(0, len(imgs_u8), bs):
        x = imgs_u8[i:i + bs].float().div(255.0).to(device)
        out.append(torch.sigmoid(model(x)).cpu())
    return torch.cat(out)


@torch.no_grad()
def evaluate_model(model, imgs_u8, masks_u8, device, bs: int = 4):
    """Returns (dice, iou, hd95) averaged over patches."""
    probs = predict_probs(model, imgs_u8, device, bs)
    pred, gt = (probs > 0.5).float(), masks_u8.float()
    dice, iou = dice_iou(pred, gt)
    return dice.mean().item(), iou.mean().item(), hd95_batch(pred, gt)


def fit_segmenter(train_imgs, train_masks, val_imgs, val_masks, device, *,
                  name: str = "unet", base: int = 32, epochs: int = 20, bs: int = 4,
                  lr: float = 1e-4, seed: int = 0, verbose: bool = True, tag: str = ""):
    """Train a segmenter; the best-validation-Dice checkpoint is restored."""
    seed_everything(seed)
    model = build_segmenter(name, base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ds = PatchDataset(train_imgs, train_masks, augment=True)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(ds) > bs, num_workers=0)
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for x, m in loader:
            x, m = x.to(device), m.to(device)
            opt.zero_grad()
            loss = dice_bce_loss(model(x), m)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        if val_imgs is not None:
            vd, vi, vh = evaluate_model(model, val_imgs, val_masks, device, bs)
            if vd > best:
                best = vd
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
                print(f"    [{tag}] ep {ep}/{epochs} loss {tot / max(1, len(loader)):.4f} "
                      f"val dice {vd:.4f} iou {vi:.4f} hd95 {vh:.2f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best
