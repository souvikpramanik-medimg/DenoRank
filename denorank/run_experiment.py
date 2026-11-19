"""Command-line entry point.

    python -m denorank.run_experiment --dataset CPM17 --data_root data/CPM17 \
        --fractions 0.05 0.10 0.25 --seeds 42 --out_dir runs
"""
import argparse
import os
import time

import pandas as pd
import torch

from denorank.config import Config
from denorank.data.dataset import resolve_paths
from denorank.pipeline import run_one


def parse_args():
    p = argparse.ArgumentParser(description="DenoRank experiment runner")
    p.add_argument("--dataset", required=True, help="name used for output folders, e.g. CPM17 or MoNuSeg")
    p.add_argument("--data_root", required=True,
                   help="folder with train/{images,masks}, [validation/{images,masks}], test/{images,masks}")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10, 0.25])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--segmenter", default="unet")
    p.add_argument("--den_unet", choices=["lora", "base"], default="lora")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--time_budget_hours", type=float, default=0.0)
    p.add_argument("--no_viz", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = Config(tile=a.tile, patch=a.patch, label_fractions=a.fractions, seeds=a.seeds,
                 segmenter=a.segmenter, den_unet=a.den_unet, out_dir=a.out_dir,
                 time_budget_hours=a.time_budget_hours, visualize=not a.no_viz)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cfg] device={device}  tile={cfg.tile} patch={cfg.patch}")
    os.makedirs(cfg.out_dir, exist_ok=True)
    paths = resolve_paths(a.data_root)
    deadline = time.time() + cfg.time_budget_hours * 3600 if cfg.time_budget_hours > 0 else None

    all_rows = []
    for frac in cfg.label_fractions:
        for seed in cfg.seeds:
            if deadline is not None and time.time() > deadline - 600:
                print("[time] budget nearly exhausted; stopping")
                break
            all_rows += run_one(cfg, a.dataset, paths, frac, seed, device, deadline)

    master = os.path.join(cfg.out_dir, "results.csv")
    if all_rows:
        new = pd.DataFrame(all_rows)
        if os.path.exists(master):
            new = pd.concat([pd.read_csv(master), new])
        new = new.drop_duplicates(subset=["dataset", "frac", "seed", "arm"], keep="last")
        new.to_csv(master, index=False)
    if os.path.exists(master):
        from denorank.summarize import summarize
        print(summarize(master))


if __name__ == "__main__":
    main()
