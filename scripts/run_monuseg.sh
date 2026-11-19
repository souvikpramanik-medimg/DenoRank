#!/usr/bin/env bash
# MoNuSeg sweep. Images are larger than CPM-17, so tile/patch are kept configurable.
set -euo pipefail
python -m denorank.run_experiment \
  --dataset MoNuSeg \
  --data_root data/MoNuSeg \
  --fractions 0.05 0.10 0.25 \
  --seeds 42 \
  --tile 512 --patch 128 \
  --out_dir runs
