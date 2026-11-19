#!/usr/bin/env bash
# Reproduce the CPM-17 sweep (5% / 10% / 25% labels).
set -euo pipefail
python -m denorank.run_experiment \
  --dataset CPM17 \
  --data_root data/CPM17 \
  --fractions 0.05 0.10 0.25 \
  --seeds 42 \
  --tile 512 --patch 128 \
  --out_dir runs
