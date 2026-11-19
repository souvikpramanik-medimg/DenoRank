import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


@torch.no_grad()
def visualize_predictions(model, test_imgs, test_masks, device, save_path,
                          n_samples: int = 6, title: str = "Predictions"):
    """Image / ground truth / prediction / error panels for a few test patches."""
    model.eval()
    n = min(n_samples, len(test_imgs))
    idx = np.linspace(0, len(test_imgs) - 1, n).astype(int)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.4 * n))
    if n == 1:
        axes = axes[None, :]
    for row, i in enumerate(idx):
        x = test_imgs[i:i + 1].float().div(255.0).to(device)
        gt = test_masks[i:i + 1].float().cpu().numpy()[0, 0]
        pred = (torch.sigmoid(model(x)).cpu().numpy()[0, 0] > 0.5).astype(np.uint8)
        img = test_imgs[i].permute(1, 2, 0).cpu().numpy()

        ov = img.copy(); mm = gt > 0.5
        ov[mm] = (0.4 * ov[mm] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
        ov2 = img.copy(); mm = pred > 0
        ov2[mm] = (0.4 * ov2[mm] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
        err = np.zeros((*gt.shape, 3), dtype=np.uint8)
        err[(pred > 0) & (gt > 0.5)] = [0, 200, 0]
        err[(pred > 0) & (gt <= 0.5)] = [220, 0, 0]
        err[(pred == 0) & (gt > 0.5)] = [0, 100, 220]

        for col, (im, name) in enumerate([(img, "Image"), (ov, "GT"), (ov2, "Pred"), (err, "Error")]):
            axes[row, col].imshow(im); axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(name)
    plt.suptitle(title, fontsize=14); plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight"); plt.close()


def plot_influence(influences, selected_idx, k, save_path, title=""):
    """Bar plot of TracIn influence per synthetic sample (green = selected)."""
    order = np.argsort(-influences)
    sel = set(int(i) for i in selected_idx)
    colors = ["green" if int(i) in sel else "red" for i in order]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(influences)), influences[order], color=colors, alpha=0.7)
    ax.axvline(k, color="black", linestyle="--", label=f"k={k}")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title(title); ax.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=120); plt.close()


def plot_rank_agreement(influences, den_scores, save_path, title=""):
    """Scatter of TracIn influence vs. denoising residual with Spearman rho."""
    from scipy.stats import spearmanr
    rho, p = spearmanr(influences, den_scores)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(den_scores, influences, s=10, alpha=0.6)
    ax.set_xlabel("denoising residual (lower = better)")
    ax.set_ylabel("TracIn influence")
    ax.set_title(f"{title}  rho={rho:+.2f}, p={p:.1e}")
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()
