"""Subset-selection rules over scored synthetic samples."""
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def topk(scores: np.ndarray, k: int, higher_is_better: bool = True) -> np.ndarray:
    order = np.argsort(-scores if higher_is_better else scores)
    return order[:k]


def influence_gap_k(influences: np.ndarray, min_frac: float, max_frac: float,
                    default_k: Optional[int] = None) -> int:
    """Choose k at the largest gap between consecutive sorted influences."""
    n = len(influences)
    srt = np.sort(influences)[::-1]
    best_k, best_sep = None, -np.inf
    for k in range(max(1, int(min_frac * n)), int(max_frac * n) + 1, max(1, n // 40)):
        sep = srt[k - 1] - srt[k] if k < n else 0.0
        if sep > best_sep:
            best_sep, best_k = sep, k
    return best_k if best_k is not None else (default_k or n // 2)


def build_arms(n_syn: int, raw: Dict[str, np.ndarray], influences: np.ndarray,
               rng: np.random.RandomState, cfg) -> "OrderedDict[str, np.ndarray]":
    """Index sets for every training arm (name -> indices into the synthetic pool).

    raw["den"]  : denoising residual           (lower = better, gradient-free)
    raw["cons"] : ensemble consistency         (higher = better)
    raw["epi"]  : ensemble epistemic uncertainty (lower = better)
    """
    k_def = int(cfg.default_keep_frac * n_syn)
    k = influence_gap_k(influences, cfg.min_keep_frac, cfg.max_keep_frac, k_def)

    # ensemble-evidence baseline, oriented so that higher = better
    ev = [zscore(raw[n]) * s for n, s in (("cons", 1.0), ("epi", -1.0)) if n in raw]
    ev_mean = np.mean(ev, axis=0) if ev else np.zeros(n_syn)
    ev_min = np.min(ev, axis=0) if ev else np.zeros(n_syn)

    arms = OrderedDict()
    arms["all_syn"] = np.arange(n_syn)
    arms["random"] = np.sort(rng.choice(n_syn, size=k_def, replace=False))
    arms["mean"] = topk(ev_mean, k_def)
    arms["min"] = topk(ev_min, k_def)
    arms[f"random_k{k}"] = np.sort(rng.choice(n_syn, size=k, replace=False))
    arms[f"mean_k{k}"] = topk(ev_mean, k)
    arms["influence_topk"] = topk(influences, k)
    if "den" in raw:
        arms["den_topk"] = topk(raw["den"], k, higher_is_better=False)
        # fully gradient-free variant: fixed keep fraction, no influence needed to pick k
        arms["den_topk_fixed"] = topk(raw["den"], k_def, higher_is_better=False)
    arms["_k"] = np.array([k])
    return arms
