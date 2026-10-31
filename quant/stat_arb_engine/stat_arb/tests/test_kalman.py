"""
tests/test_kalman.py
--------------------
Synthetic-data checks that the Kalman spread is a real mean-reverting level,
not the one-day noise the old raw-price filter produced.

Setup: B is a geometric random walk around $100; A is cointegrated with it in
log space through a known OU spread:

    log A[t] = 1.2 * log B[t] + 0.3 + s[t],   s[t] = 0.95 * s[t-1] + N(0, 0.01^2)

(half-life ~13.5 days, stationary std ~3.2%). After fitting on the first 252
days, the filter's spread must track s[t] out of sample.

Recovery is checked as a MEDIAN over 10 seeds: with only 252 autocorrelated
training days the OLS-seeded beta is occasionally off by enough that one
unlucky path drifts (min corr ~0.77) — that's estimation error, not a filter
bug, and a single-seed test would just be testing the seed.

Run with `python -m pytest tests` or just `python tests/test_kalman.py`.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.kalman import KalmanHedgeRatio  # noqa: E402

TRAIN = 252


def _synthetic_pair(n=1000, phi=0.95, sigma=0.01, beta=1.2, alpha=0.3, seed=7):
    rng = np.random.default_rng(seed)
    log_b = np.log(100.0) + np.cumsum(rng.normal(0.0, 0.015, n))
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0.0, sigma)
    log_a = beta * log_b + alpha + s
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(np.exp(log_a), idx), pd.Series(np.exp(log_b), idx), pd.Series(s, idx)


def test_spread_recovers_ou():
    corrs, ac1s = [], []
    for seed in range(10):
        a, b, s_true = _synthetic_pair(seed=seed)
        res = KalmanHedgeRatio().fit(a, b, train_size=TRAIN)
        est = res.spread.iloc[TRAIN:]
        corrs.append(np.corrcoef(est, s_true.iloc[TRAIN:])[0, 1])
        ac1s.append(est.autocorr(lag=1))

    corr, ac1 = np.median(corrs), np.median(ac1s)
    assert corr > 0.9, f"median spread/true-spread corr {corr:.3f} <= 0.9"
    # the old filter's "spread" had autocorr ~ -0.02 (it was a daily change)
    assert ac1 > 0.8, f"median spread lag-1 autocorr {ac1:.3f} — looks like daily noise"


def test_fast_filter_absorbs_spread():
    """
    Sanity check on the failure mode itself: the old default numbers
    (q=1e-4, R=1e-3, q/R=0.1 — a ~3-day memory) let the state learn the
    spread away. If this ever passes the >0.9 bar, the test above is moot.
    """
    a, b, s_true = _synthetic_pair()
    res = KalmanHedgeRatio(delta=1e-4, obs_noise=1e-3).fit(a, b, train_size=TRAIN)
    corr = np.corrcoef(res.spread.iloc[TRAIN:], s_true.iloc[TRAIN:])[0, 1]
    assert corr < 0.5, f"fast filter corr {corr:.3f} — expected the spread to be absorbed"


def test_hedge_ratio_recovered():
    a, b, _ = _synthetic_pair()
    res = KalmanHedgeRatio().fit(a, b, train_size=TRAIN)
    beta_oos = res.hedge_ratio.iloc[TRAIN:]
    assert abs(beta_oos.mean() - 1.2) < 0.1, f"mean beta {beta_oos.mean():.3f}, true 1.2"


def test_innovation_z_is_standardised():
    a, b, _ = _synthetic_pair()
    res = KalmanHedgeRatio().fit(a, b, train_size=TRAIN)
    z = res.innovation_z.iloc[TRAIN:]
    # e/sqrt(S) on a persistent spread isn't white, but its scale should be ~1
    assert 0.5 < z.std() < 2.0, f"innovation z std {z.std():.2f}"


def test_scale_invariant():
    """Same pair at 10x the price level must give the same spread (log space)."""
    a, b, _ = _synthetic_pair()
    s1 = KalmanHedgeRatio().fit(a, b, train_size=TRAIN).spread
    s2 = KalmanHedgeRatio().fit(a * 10, b * 10, train_size=TRAIN).spread
    assert np.corrcoef(s1.iloc[TRAIN:], s2.iloc[TRAIN:])[0, 1] > 0.99


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
