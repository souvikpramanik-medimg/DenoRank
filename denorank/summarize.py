"""Aggregate results.csv into mean/std test metrics per (dataset, fraction, arm)."""
import sys

import pandas as pd


def summarize(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    agg = (df.groupby(["dataset", "frac", "arm"])
             .agg(n_syn=("n_syn", "mean"), dice=("test_dice", "mean"), dice_std=("test_dice", "std"),
                  iou=("test_iou", "mean"), hd95=("test_hd95", "mean"), seeds=("seed", "nunique"))
             .reset_index())
    agg.to_csv(csv_path.replace("results.csv", "results_aggregated.csv"), index=False)
    return agg.round(4).to_string(index=False)


if __name__ == "__main__":
    print(summarize(sys.argv[1] if len(sys.argv) > 1 else "runs/results.csv"))
