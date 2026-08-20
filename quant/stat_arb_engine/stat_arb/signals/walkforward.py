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
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass

from signals.cointegration import find_cointegrated_pairs


@dataclass
class PairPeriod:
    ticker_a     : str
    ticker_b     : str
    pvalue       : float
    period_start : pd.Timestamp   # first tradeable date (out-of-sample)
    period_end   : pd.Timestamp   # last tradeable date, exclusive of next period


def generate_pair_schedule(
    prices        : pd.DataFrame,
    universe      : list[str],
    train_window  : int   = 252,
    test_window   : int   = 63,
    pvalue_cutoff : float = 0.05,
) -> list[PairPeriod]:
    """
    Roll a `train_window`-day cointegration test forward in `test_window`-day
    steps across `prices`.

    At each step, tests every pair in `universe` for cointegration using only
    the trailing `train_window` days (data strictly before `period_start`),
    and the pairs that pass become tradeable for the following `test_window`
    days. No pair is ever selected using data from its own trading period.

    Returns a flat list of PairPeriod, one entry per (pair, active window).
    Multiple pairs can be active in the same period; the same pair can
    reappear in a later period if it re-qualifies.
    """
    prices = prices[universe].dropna()
    dates  = prices.index
    n      = len(dates)

    schedule: list[PairPeriod] = []

    start = train_window
    while start < n:
        end = min(start + test_window, n)

        train_slice = prices.iloc[start - train_window : start]
        period_start = dates[start]
        period_end   = dates[end - 1]

        pairs = find_cointegrated_pairs(train_slice, pvalue_cutoff=pvalue_cutoff, verbose=False)
        for ticker_a, ticker_b, pval in pairs:
            schedule.append(PairPeriod(
                ticker_a     = ticker_a,
                ticker_b     = ticker_b,
                pvalue       = pval,
                period_start = period_start,
                period_end   = period_end,
            ))

        start = end

    return schedule
