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


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    pvalue_cutoff: float | None = 0.05,
    verbose: bool = True,
) -> list[tuple[str, str, float]]:
    """
    Test every pair of tickers for cointegration.

    Returns a list of (ticker_a, ticker_b, p_value) tuples sorted by p-value,
    filtered to those below pvalue_cutoff. pvalue_cutoff=None returns every
    pair — needed for multiple-testing corrections (see walkforward.py),
    which have to see the whole family of p-values, not just the "winners".

    verbose=False silences progress printing — useful when this is called
    many times in a loop (e.g. walk-forward re-fitting every rebalance date).
    """
    tickers = prices.columns.tolist()
    results = []

    total = len(list(combinations(tickers, 2)))
    if verbose:
        print(f"      Testing {total} pairs...", end="", flush=True)

    for t1, t2 in combinations(tickers, 2):
        s1 = prices[t1].values
        s2 = prices[t2].values

        # coint() runs Engle-Granger: returns (test_stat, p_value, crit_values)
        _, pval, _ = coint(s1, s2)

        if pvalue_cutoff is None or pval < pvalue_cutoff:
            results.append((t1, t2, pval))

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
