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

import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import coint


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    pvalue_cutoff: float = 0.05,
    verbose: bool = True,
) -> list[tuple[str, str, float]]:
    """
    Test every pair of tickers for cointegration.

    Returns a list of (ticker_a, ticker_b, p_value) tuples sorted by p-value,
    filtered to those below pvalue_cutoff.

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

        if pval < pvalue_cutoff:
            results.append((t1, t2, pval))

    if verbose:
        print(" done.")

    # Sort best pairs first
    results.sort(key=lambda x: x[2])
    return results
