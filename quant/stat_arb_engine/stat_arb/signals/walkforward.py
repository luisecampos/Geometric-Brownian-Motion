"""
signals/walkforward.py
-----------------------
Roll a cointegration re-fit forward through time instead of testing once on
the full history.

Why this matters:
  The original pipeline picked its pair(s) with one Engle-Granger test over
  the *entire* backtest window, then traded that same pair for the entire
  window. That's in-sample selection — you're using information ("this pair
  stayed cointegrated the whole time") that you couldn't have known on day 1.

  Walk-forward fixes this: at each rebalance date, only look at the trailing
  `train_window` days to decide which pairs are tradeable, then trade those
  pairs for the next `test_window` days without peeking further ahead. Roll
  forward and repeat. A pair can drop in and out of the active set over time,
  same as it would on a real desk re-running pair selection periodically.

Multiple testing:
  15 tickers = 105 pairs tested every quarter. At p < 0.05 with NO real
  cointegration anywhere you'd still expect ~5 "cointegrated" pairs per
  window — pure false positives that then get traded. Two corrections:
    - "bh"         : Benjamini-Hochberg, controls the false DISCOVERY rate at
                     `pvalue_cutoff` (of the pairs you pick, at most ~5% are
                     flukes on average). Default — less conservative, keeps power.
    - "bonferroni" : p < pvalue_cutoff / n_tests, controls the chance of even
                     ONE false positive per window. Very strict at n=105.
    - None         : the old raw cutoff.

Half-life filter:
  Even a genuinely cointegrated pair is useless if it reverts on a 200-day
  horizon inside a 63-day trading window. Pairs whose training-window spread
  half-life falls outside [min_half_life, max_half_life] are dropped.

Tests run on LOG prices, matching the log-space Kalman hedge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from signals.cointegration import find_cointegrated_pairs, spread_half_life


@dataclass
class PairPeriod:
    ticker_a     : str
    ticker_b     : str
    pvalue       : float
    period_start : pd.Timestamp   # first tradeable date (out-of-sample)
    period_end   : pd.Timestamp   # last tradeable date, exclusive of next period
    train_start  : pd.Timestamp = None   # first date of the selection window
    half_life    : float = float("nan")  # training-window spread half-life (days)


def select_pairs(
    pvals            : list[tuple[str, str, float]],
    alpha            : float,
    multiple_testing : str | None = "bh",
) -> list[tuple[str, str, float]]:
    """
    Apply a multiple-testing rule to the FULL family of (a, b, p) results.

    Benjamini-Hochberg: sort p ascending, find the largest k with
    p_(k) <= (k / m) * alpha, and accept the first k.
    """
    m = len(pvals)
    if m == 0:
        return []
    ranked = sorted(pvals, key=lambda x: x[2])

    if multiple_testing is None:
        return [r for r in ranked if r[2] < alpha]
    if multiple_testing == "bonferroni":
        return [r for r in ranked if r[2] < alpha / m]
    if multiple_testing == "bh":
        p = np.array([r[2] for r in ranked])
        passed = np.nonzero(p <= alpha * np.arange(1, m + 1) / m)[0]
        k = passed[-1] + 1 if len(passed) else 0
        return ranked[:k]
    raise ValueError(f"unknown multiple_testing: {multiple_testing!r}")


def generate_pair_schedule(
    prices           : pd.DataFrame,
    universe         : list[str],
    train_window     : int   = 252,
    test_window      : int   = 63,
    pvalue_cutoff    : float = 0.05,
    multiple_testing : str | None = "bh",
    min_half_life    : float = 5.0,
    max_half_life    : float = 60.0,
    verbose          : bool  = False,
) -> list[PairPeriod]:
    """
    Roll a `train_window`-day cointegration test forward in `test_window`-day
    steps across `prices`.

    At each step, tests every pair in `universe` for cointegration using only
    the trailing `train_window` days (data strictly before `period_start`),
    corrects for testing all of them at once, drops pairs whose half-life is
    outside [min_half_life, max_half_life], and makes the survivors tradeable
    for the following `test_window` days. No pair is ever selected using data
    from its own trading period.

    Returns a flat list of PairPeriod, one entry per (pair, active window).
    Multiple pairs can be active in the same period; the same pair can
    reappear in a later period if it re-qualifies.
    """
    prices = prices[universe].dropna()
    log_px = np.log(prices)
    dates  = prices.index
    n      = len(dates)

    schedule: list[PairPeriod] = []

    start = train_window
    while start < n:
        end = min(start + test_window, n)

        train_slice  = log_px.iloc[start - train_window : start]
        period_start = dates[start]
        period_end   = dates[end - 1]

        all_pvals = find_cointegrated_pairs(train_slice, pvalue_cutoff=None, verbose=False)
        passed    = select_pairs(all_pvals, pvalue_cutoff, multiple_testing)

        kept = 0
        for ticker_a, ticker_b, pval in passed:
            hl = spread_half_life(train_slice[ticker_a], train_slice[ticker_b])
            if not (min_half_life <= hl <= max_half_life):
                continue
            kept += 1
            schedule.append(PairPeriod(
                ticker_a     = ticker_a,
                ticker_b     = ticker_b,
                pvalue       = pval,
                period_start = period_start,
                period_end   = period_end,
                train_start  = dates[start - train_window],
                half_life    = hl,
            ))

        if verbose:
            raw = sum(p < pvalue_cutoff for *_, p in all_pvals)
            print(f"      {period_start.date()}: {raw:>3d} raw p<{pvalue_cutoff}"
                  f"  -> {len(passed):>2d} after {multiple_testing or 'no'} correction"
                  f"  -> {kept:>2d} after half-life filter")

        start = end

    return schedule
