import numpy as np

from denorank.config import Config
from denorank.selection import build_arms, influence_gap_k, topk


def test_topk_orientation():
    s = np.array([0.3, 0.1, 0.9, 0.5])
    assert list(topk(s, 2, higher_is_better=True)) == [2, 3]
    assert list(topk(s, 2, higher_is_better=False)) == [1, 0]


def test_influence_gap_k_in_range():
    infl = np.concatenate([np.linspace(5, 3, 40), np.linspace(0.1, -1, 60)])
    k = influence_gap_k(infl, 0.2, 0.7)
    assert 20 <= k <= 70
    assert k == 40  # largest gap sits right after the 40 strongest samples


def test_build_arms_shapes():
    rng = np.random.RandomState(0)
    n = 100
    raw = {"den": rng.rand(n), "cons": rng.rand(n), "epi": rng.rand(n)}
    infl = -raw["den"] + 0.05 * rng.randn(n)
    arms = build_arms(n, raw, infl, rng, Config())
    k = int(arms.pop("_k")[0])
    assert len(arms["den_topk"]) == k == len(arms["influence_topk"])
    assert len(arms["den_topk_fixed"]) == len(arms["random"]) == 50
    assert all(len(set(v)) == len(v) for v in arms.values())
