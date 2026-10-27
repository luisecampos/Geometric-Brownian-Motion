"""
signals/cointegration.py
------------------------
Test all ticker pairs for cointegration using the Engle-Granger two-step method.

What is cointegration?
  Two price series are cointegrated if a linear combination of them is
  stationary (mean-reverting). This is stronger than correlation — correlated
  series can drift apart forever, but cointegrated ones are bound by an
  equilibrium relationship.

Engle-Granger steps:
  1. Regress series A on series B (OLS) to find the hedge ratio β
  2. Test the residuals (spread = A - β*B) for stationarity using ADF
  3. If ADF p-value < cutoff → pair is cointegrated
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import coint

from signals.cache import parallel_map


def _coint_chunk(args):
    """Worker: Engle-Granger on a block of (i, j) column pairs of `values`."""
    values, idx_pairs = args
    out = []
    for i, j in idx_pairs:
        # coint() runs Engle-Granger: returns (test_stat, p_value, crit_values)
        stat, pval, _ = coint(values[:, i], values[:, j])
        out.append((i, j, float(stat), float(pval)))
    return out


def coint_tests(
    prices : pd.DataFrame,
    pairs  : list[tuple[str, str]] | None = None,
    n_jobs : int = 1,
    chunk  : int = 150,
) -> dict[tuple[str, str], tuple[float, float]]:
    """
    Engle-Granger test for each pair in `pairs` (default: every pair of
    columns). Returns {(ticker_a, ticker_b): (adf_stat, p_value)}.
    n_jobs > 1 spreads the tests over a process pool.
    """
    tickers = prices.columns.tolist()
    if pairs is None:
        pairs = list(combinations(tickers, 2))
    pos = {t: k for k, t in enumerate(tickers)}
    values = prices.values.astype(float)
    idx_pairs = [(pos[a], pos[b]) for a, b in pairs]
    blocks = [(values, idx_pairs[k:k + chunk]) for k in range(0, len(idx_pairs), chunk)]
    out = {}
    for block in parallel_map(_coint_chunk, blocks, n_jobs):
        for i, j, stat, pval in block:
            out[(tickers[i], tickers[j])] = (stat, pval)
    return out


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    pvalue_cutoff: float | None = 0.05,
    verbose: bool = True,
    pairs: list[tuple[str, str]] | None = None,
    n_jobs: int = 1,
) -> list[tuple[str, str, float]]:
    """
    Test every pair of tickers (or just `pairs`) for cointegration.

    Returns a list of (ticker_a, ticker_b, p_value) tuples sorted by p-value,
    filtered to those below pvalue_cutoff. pvalue_cutoff=None returns every
    pair — needed for multiple-testing corrections (see walkforward.py),
    which have to see the whole family of p-values, not just the "winners".

    verbose=False silences progress printing — useful when this is called
    many times in a loop (e.g. walk-forward re-fitting every rebalance date).
    """
    if pairs is None:
        pairs = list(combinations(prices.columns.tolist(), 2))
    if verbose:
        print(f"      Testing {len(pairs)} pairs...", end="", flush=True)

    tests = coint_tests(prices, pairs, n_jobs)
    results = [(a, b, p) for (a, b), (_, p) in tests.items()
               if pvalue_cutoff is None or p < pvalue_cutoff]

    if verbose:
        print(" done.")

    # Sort best pairs first
    results.sort(key=lambda x: x[2])
    return results


def spread_half_life(series_a: pd.Series, series_b: pd.Series) -> float:
    """
    Mean-reversion half-life (days) of the OLS spread a - beta*b - c.

    Fits the discrete OU / AR(1) form  ds[t] = lam * s[t-1] + eps  and
    converts:  half_life = -ln(2) / ln(1 + lam).  Returns inf if the spread
    isn't mean-reverting (lam >= 0) on this sample.

    Why it matters: a pair can pass Engle-Granger with a half-life of 200
    days — statistically "cointegrated", but it won't revert inside a
    63-day trading window, so it's dead capital. And a 1-day half-life is
    usually bid-ask bounce / stale prints, not a tradeable edge.
    """
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    X = np.column_stack([b, np.ones_like(b)])
    coef, *_ = np.linalg.lstsq(X, a, rcond=None)
    s = a - X @ coef

    lag, ds = s[:-1], np.diff(s)
    lag = lag - lag.mean()
    lam = float(lag @ (ds - ds.mean()) / (lag @ lag))
    if lam >= 0 or lam <= -1:
        return float("inf") if lam >= 0 else 0.0
    return float(-np.log(2) / np.log(1 + lam))
