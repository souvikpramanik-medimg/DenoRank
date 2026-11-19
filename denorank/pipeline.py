"""End-to-end experiment: generate -> score -> select -> train -> evaluate."""
import gc
import os
import time

import numpy as np
import pandas as pd
import torch

from denorank.data.dataset import build_patch_tensors, choose_labeled_tiles, load_tiles
from denorank.engine import evaluate_model, fit_segmenter
from denorank.models.segmenter import build_segmenter
from denorank.scoring.agreement import rank_agreement
from denorank.scoring.copy_guard import pixel_copy_guard
from denorank.scoring.ensemble import ensemble_scores
from denorank.scoring.tracin import compute_tracin_influence
from denorank.selection import build_arms
from denorank.utils.seed import seed_everything
from denorank.utils.viz import plot_influence, plot_rank_agreement, visualize_predictions


def _load_splits(paths, seed):
    tr_i, tr_m = load_tiles(paths["train_img"], paths["train_mask"])
    if paths.get("val_img"):
        va_i, va_m = load_tiles(paths["val_img"], paths["val_mask"])
    else:  # no provided validation split: hold out 20% of the training tiles
        perm = np.random.RandomState(seed).permutation(len(tr_i))
        n_val = max(1, int(0.2 * len(tr_i)))
        va_idx, tr_idx = perm[:n_val], perm[n_val:]
        va_i, va_m = [tr_i[j] for j in va_idx], [tr_m[j] for j in va_idx]
        tr_i, tr_m = [tr_i[j] for j in tr_idx], [tr_m[j] for j in tr_idx]
    te_i, te_m = load_tiles(paths["test_img"], paths["test_mask"])
    return (tr_i, tr_m), (va_i, va_m), (te_i, te_m)


def run_one(cfg, dataset_name, paths, frac, seed, device, deadline=None):
    """Run one (dataset, label fraction, seed) setting and return result rows."""
    from denorank.generation.lora import load_sd_lora
    from denorank.generation.synthesize import generate_from_masks, random_deform
    from denorank.scoring.denoising import denoise_residual_scores

    run_root = cfg.out_dir
    run_dir = os.path.join(run_root, f"{dataset_name}_frac{frac}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    P = lambda n: os.path.join(run_dir, n)  # noqa: E731
    print(f"\n{'=' * 70}\n[run] {dataset_name} frac={frac} seed={seed}\n{'=' * 70}")
    seed_everything(seed)

    # ---- data ----
    (tr_i, tr_m), (va_i, va_m), (te_i, te_m) = _load_splits(paths, seed)
    lab = choose_labeled_tiles(len(tr_i), frac, seed)
    real_imgs, real_masks = build_patch_tensors([tr_i[j] for j in lab], [tr_m[j] for j in lab],
                                                cfg.tile, cfg.patch)
    all_imgs, _ = build_patch_tensors(tr_i, tr_m, cfg.tile, cfg.patch)
    val_imgs, val_masks = build_patch_tensors(va_i, va_m, cfg.tile, cfg.patch)
    test_imgs, test_masks = build_patch_tensors(te_i, te_m, cfg.tile, cfg.patch)
    print(f"[data] labelled={len(real_imgs)} unlabelled-pool={len(all_imgs)} "
          f"val={len(val_imgs)} test={len(test_imgs)}")

    # ---- generator (LoRA is trained on the unlabelled image pool: images only, no masks) ----
    pipe = load_sd_lora(all_imgs, os.path.join(run_root, f"_sd_lora_{dataset_name}"), cfg, device)

    # ---- synthetic pool ----
    syn_path = P("synthetic.pt")
    if os.path.exists(syn_path):
        d = torch.load(syn_path)
        syn_imgs, syn_masks = d["imgs"], d["masks"]
        print(f"[synthetic] loaded {len(syn_imgs)}")
    else:
        masks = random_deform(real_masks.repeat(cfg.gen_mult, 1, 1, 1), seed=seed)
        masks = masks[masks.float().mean(dim=(1, 2, 3)) >= cfg.min_mask_frac]
        print(f"[synthetic] generating {len(masks)} ...")
        syn_imgs, syn_masks = generate_from_masks(pipe, masks, cfg), masks
        torch.save({"imgs": syn_imgs, "masks": syn_masks}, syn_path)

    # ---- reference ensemble (used for TracIn and the ensemble baselines) ----
    ens = []
    for i in range(cfg.ens_k):
        pth = P(f"ens_{i}.pt")
        if os.path.exists(pth):
            m = build_segmenter(cfg.segmenter, cfg.seg_base).to(device)
            m.load_state_dict(torch.load(pth, map_location=device))
            m.eval()
        else:
            m, _ = fit_segmenter(real_imgs, real_masks, val_imgs, val_masks, device,
                                 name=cfg.segmenter, base=cfg.seg_base, epochs=cfg.ens_epochs,
                                 bs=cfg.seg_bs, lr=cfg.seg_lr, seed=seed + 1 + i, tag=f"ens{i}")
            torch.save(m.state_dict(), pth)
        ens.append(m)

    # ---- scores ----
    sc_path = P("scores.npz")
    if os.path.exists(sc_path):
        z = np.load(sc_path)
        keep, raw = z["keep"], {k: z[k] for k in z.files if k != "keep"}
        print("[scores] loaded")
    else:
        cons, epi = ensemble_scores(ens, syn_imgs, syn_masks, device, bs=cfg.seg_bs)
        raw = {"cons": cons, "epi": epi}
        keep, _ = pixel_copy_guard(syn_imgs, real_imgs, cfg.copy_l2_thresh)
        print(f"[scores] copy guard dropped {int((~keep).sum())}/{len(keep)}")
        t0 = time.time()
        raw["den"] = denoise_residual_scores(pipe, syn_imgs, cfg, device, seed=seed)
        print(f"[scores] denoising residual: {len(syn_imgs)} images in {time.time() - t0:.1f}s "
              f"({len(cfg.den_timesteps)} forward passes each)")
        np.savez(sc_path, keep=keep, **raw)

    kt = torch.from_numpy(keep)
    syn_imgs, syn_masks = syn_imgs[kt], syn_masks[kt]
    raw = {k: v[keep] for k, v in raw.items()}
    n_syn = len(syn_imgs)

    # ---- TracIn influence (reference signal) ----
    infl_path = P("influence.npy")
    if os.path.exists(infl_path):
        influences = np.load(infl_path)
    else:
        print(f"[influence] computing TracIn for {n_syn} samples ...")
        influences = compute_tracin_influence(ens[0], val_imgs, val_masks, syn_imgs, syn_masks,
                                              device, bs=cfg.seg_bs)
        np.save(infl_path, influences)

    agree = {}
    for name, v in raw.items():
        rho, p = rank_agreement(influences, v)
        agree[name] = (rho, p)
        print(f"[agreement] TracIn vs {name}: rho={rho:+.3f} p={p:.2e}")
    rows_agree = [{"dataset": dataset_name, "frac": frac, "seed": seed, "score": n,
                   "spearman_rho": round(r, 4), "p_value": p} for n, (r, p) in agree.items()]
    pd.DataFrame(rows_agree).to_csv(P("rank_agreement.csv"), index=False)
    if "den" in raw:
        plot_rank_agreement(influences, raw["den"], P("rank_agreement.png"), f"{dataset_name} {frac}")

    # ---- arms ----
    prev = []
    master = os.path.join(run_root, "results.csv")
    if os.path.exists(master):
        df = pd.read_csv(master)
        prev = df[(df["dataset"] == dataset_name) & (df["frac"] == frac) &
                  (df["seed"] == seed)].to_dict("records")

    rng = np.random.RandomState(seed)
    arms = build_arms(n_syn, raw, influences, rng, cfg) if n_syn > 0 else {}
    k = int(arms.pop("_k")[0]) if arms else 0
    rows, models = [], {}

    def run_arm(arm, idx):
        for r in prev:
            if r["arm"] == arm:
                print(f"[arm] {arm} reused (dice={r['test_dice']:.4f})")
                rows.append({kk: r[kk] for kk in ("dataset", "frac", "seed", "arm", "n_syn",
                                                  "val_dice", "test_dice", "test_iou", "test_hd95")})
                return
        if deadline is not None and time.time() > deadline:
            print(f"[time] budget exhausted, skipping {arm}")
            return
        tr_x = real_imgs if idx is None else torch.cat([real_imgs, syn_imgs[idx]])
        tr_m = real_masks if idx is None else torch.cat([real_masks, syn_masks[idx]])
        print(f"[arm] {arm} (+{0 if idx is None else len(idx)} synthetic)")
        m, vd = fit_segmenter(tr_x, tr_m, val_imgs, val_masks, device, name=cfg.segmenter,
                              base=cfg.seg_base, epochs=cfg.final_epochs, bs=cfg.seg_bs,
                              lr=cfg.seg_lr, seed=seed + 100, tag=arm)
        td, ti, thd = evaluate_model(m, test_imgs, test_masks, device, bs=cfg.seg_bs)
        rows.append({"dataset": dataset_name, "frac": frac, "seed": seed, "arm": arm,
                     "n_syn": 0 if idx is None else len(idx), "val_dice": round(vd, 4),
                     "test_dice": round(td, 4), "test_iou": round(ti, 4), "test_hd95": round(thd, 2)})
        models[arm] = m
        torch.save(m.state_dict(), P(f"final_{arm}.pt"))

    run_arm("real_only", None)
    for arm, idx in arms.items():
        run_arm(arm, idx)

    if arms:
        plot_influence(influences, arms["influence_topk"], k, P("influence_scores.png"),
                       f"{dataset_name} {frac} — TracIn influence")
    best = "den_topk" if "den_topk" in models else None
    if cfg.visualize and best:
        visualize_predictions(models[best], test_imgs, test_masks, device,
                              P("predictions_den_topk.png"), title=f"{dataset_name} {frac}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows
