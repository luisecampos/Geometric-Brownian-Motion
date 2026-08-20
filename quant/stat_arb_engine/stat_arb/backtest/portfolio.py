"""
backtest/portfolio.py
----------------------
Run every pair from a walk-forward schedule simultaneously and combine them
into one portfolio.

A single pair's backtest tells you if that one relationship was profitable.
It doesn't tell you what happens when you run a book of them together —
whether the pairs actually diversify each other or all blow up on the same
day because they share a common factor (e.g. two energy-sector pairs both
getting hit by the same oil shock). That's what `pair_pnl_corr` is for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from signals.kalman import KalmanHedgeRatio
from signals.zscore import compute_zscore
from signals.sizing import vol_target_size
from signals.walkforward import PairPeriod
from backtest.engine import run_backtest, BacktestResults


@dataclass
class PortfolioResults:
    equity_curve   : pd.Series
    daily_pnl      : pd.Series
    per_pair       : dict          # (ticker_a, ticker_b, period_start) -> BacktestResults
    pair_pnl       : pd.DataFrame  # daily PnL per pair-period, columns = pair labels
    pair_pnl_corr  : pd.DataFrame  # correlation matrix of pair_pnl columns


def run_portfolio_backtest(
    prices          : pd.DataFrame,
    pair_schedule   : list[PairPeriod],
    lookback_window : int = 60,
    entry_threshold : float = 2.0,
    exit_threshold  : float = 0.5,
    target_daily_vol: float = 0.01,
) -> PortfolioResults:
    """
    Run the full pipeline (Kalman hedge ratio -> z-score -> vol-target sizing
    -> backtest) for every PairPeriod in `pair_schedule`, restricted to each
    period's own active date range, then sum them into one portfolio.

    Each pair-period is fit independently on prices up to its own period —
    consistent with how `pair_schedule` was generated (see
    signals/walkforward.py), so nothing here re-introduces look-ahead.
    """
    per_pair  : dict = {}
    pair_pnl_cols: dict[str, pd.Series] = {}

    for pp in pair_schedule:
        label = f"{pp.ticker_a}/{pp.ticker_b} [{pp.period_start.date()}]"

        # Fit hedge ratio + z-score using history up to (and including) this
        # pair's active period, then slice down to just the active dates —
        # the Kalman/z-score warm-up naturally uses only causal, in-window data.
        series_a = prices.loc[:pp.period_end, pp.ticker_a]
        series_b = prices.loc[:pp.period_end, pp.ticker_b]

        kalman = KalmanHedgeRatio()
        hedge_ratios, spreads = kalman.fit(series_a, series_b)
        zscores = compute_zscore(spreads, window=lookback_window)
        size = vol_target_size(spreads, target_daily_vol=target_daily_vol).fillna(1.0)

        active = (series_a.index >= pp.period_start) & (series_a.index <= pp.period_end)

        results = run_backtest(
            series_a[active], series_b[active],
            hedge_ratios[active], zscores[active],
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            size=size[active],
        )

        per_pair[(pp.ticker_a, pp.ticker_b, pp.period_start)] = results
        pair_pnl_cols[label] = results.daily_pnl

    if not pair_pnl_cols:
        raise ValueError("No pair-periods to backtest — pair_schedule was empty.")

    pair_pnl = pd.concat(pair_pnl_cols, axis=1).fillna(0.0)
    daily_pnl = pair_pnl.sum(axis=1).rename("daily_pnl")
    equity_curve = daily_pnl.cumsum().rename("equity")
    pair_pnl_corr = pair_pnl.corr()

    return PortfolioResults(
        equity_curve  = equity_curve,
        daily_pnl     = daily_pnl,
        per_pair      = per_pair,
        pair_pnl      = pair_pnl,
        pair_pnl_corr = pair_pnl_corr,
    )
