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
    kalman_params  : dict = field(default_factory=dict)  # label -> fitted Kalman params


def run_portfolio_backtest(
    prices          : pd.DataFrame,
    pair_schedule   : list[PairPeriod],
    lookback_window : int = 60,
    entry_threshold : float = 2.0,
    exit_threshold  : float = 0.5,
    stop_threshold  : float | None = 3.5,
    target_daily_vol: float = 0.01,
    signal          : str = "rolling",
    train_window    : int = 252,
) -> PortfolioResults:
    """
    Run the full pipeline (Kalman hedge ratio -> z-score -> vol-target sizing
    -> backtest) for every PairPeriod in `pair_schedule`, restricted to each
    period's own active date range, then sum them into one portfolio.

    Each pair-period's Kalman filter is fit (centring, OLS seed, MLE noise
    parameters) on that pair's own selection window — the `train_window`
    days before period_start, the same data walk-forward used to pick it —
    and then filtered causally through the active period. So nothing here
    re-introduces look-ahead, and the z-score / vol warm-ups are complete by
    the first tradeable day.

    signal: "rolling"    -> rolling z-score of the spread (lookback_window)
            "innovation" -> Kalman innovation z = e / sqrt(S)

    PnL is a return on book capital: each pair's daily_pnl is its return on
    the whole book, so the portfolio return is their sum.
    """
    per_pair  : dict = {}
    pair_pnl_cols: dict[str, pd.Series] = {}
    kalman_params: dict = {}

    for pp in pair_schedule:
        label = f"{pp.ticker_a}/{pp.ticker_b} [{pp.period_start.date()}]"

        if pp.train_start is not None:
            window = prices.loc[pp.train_start:pp.period_end]
        else:
            start_pos = max(0, prices.index.get_loc(pp.period_start) - train_window)
            window = prices.loc[prices.index[start_pos]:pp.period_end]
        series_a = window[pp.ticker_a]
        series_b = window[pp.ticker_b]
        train_size = int((window.index < pp.period_start).sum())

        kalman = KalmanHedgeRatio()
        kf = kalman.fit(series_a, series_b, train_size=train_size)
        kalman_params[label] = kalman.fitted_params

        if signal == "rolling":
            zscores = compute_zscore(kf.spread, window=lookback_window)
        elif signal == "innovation":
            zscores = kf.innovation_z
        else:
            raise ValueError(f"unknown signal: {signal!r}")
        size = vol_target_size(series_a, series_b, kf.hedge_ratio, target_daily_vol=target_daily_vol)

        # Start the backtest one day BEFORE period_start, so day-1 decisions
        # can use the prior close's z / hedge / size (the engine always acts
        # on [t-1]). The engine can't enter on its own first row, so no
        # position is ever opened before period_start.
        first = max(0, train_size - 1)
        active = slice(first, None)

        results = run_backtest(
            series_a.iloc[active], series_b.iloc[active],
            kf.hedge_ratio.iloc[active], zscores.iloc[active],
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_threshold=stop_threshold,
            size=size.iloc[active],
        )

        per_pair[(pp.ticker_a, pp.ticker_b, pp.period_start)] = results
        pair_pnl_cols[label] = results.daily_pnl

    if not pair_pnl_cols:
        raise ValueError("No pair-periods to backtest — pair_schedule was empty.")

    # Reindex onto the full calendar so days with nothing active count as
    # flat (0 return) days — otherwise Sharpe/vol only see "in-market" days.
    calendar = prices.loc[min(pp.period_start for pp in pair_schedule):].index
    pair_pnl = pd.concat(pair_pnl_cols, axis=1).reindex(calendar).fillna(0.0)
    daily_pnl = pair_pnl.sum(axis=1).rename("daily_pnl")
    equity_curve = daily_pnl.cumsum().rename("equity")
    pair_pnl_corr = pair_pnl.corr()

    return PortfolioResults(
        equity_curve  = equity_curve,
        daily_pnl     = daily_pnl,
        per_pair      = per_pair,
        pair_pnl      = pair_pnl,
        pair_pnl_corr = pair_pnl_corr,
        kalman_params = kalman_params,
    )
