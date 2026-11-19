"""TracIn-style influence (reference signal used to validate the gradient-free score)."""
import numpy as np
import torch

from denorank.utils.losses import dice_bce_loss


def compute_tracin_influence(model, val_imgs, val_masks, syn_imgs, syn_masks, device, bs: int = 4):
    """Influence of each synthetic sample on the validation loss at a single checkpoint.

    influence(z) = <grad L(z), g_val> / ||g_val||, where g_val is the mean validation-batch gradient.
    Higher = a gradient step on z moves the model more along the validation-descent direction.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    ref, n_ref = None, 0
    for i in range(0, len(val_imgs), bs):
        x = val_imgs[i:i + bs].float().div(255.0).to(device)
        m = val_masks[i:i + bs].float().to(device)
        grads = torch.autograd.grad(dice_bce_loss(model(x), m), model.parameters())
        if ref is None:
            ref = [g.detach().clone() for g in grads]
        else:
            for rg, g in zip(ref, grads):
                rg += g.detach()
        n_ref += 1
    for rg in ref:
        rg /= max(1, n_ref)
    ref_flat = torch.cat([g.flatten() for g in ref])
    ref_norm = ref_flat.norm() + 1e-8

    infl = []
    for i in range(len(syn_imgs)):
        x = syn_imgs[i:i + 1].float().div(255.0).to(device)
        m = syn_masks[i:i + 1].float().to(device)
        grads = torch.autograd.grad(dice_bce_loss(model(x), m), model.parameters())
        syn_flat = torch.cat([g.flatten() for g in grads])
        infl.append(((syn_flat * ref_flat).sum() / ref_norm).item())
    return np.array(infl)
