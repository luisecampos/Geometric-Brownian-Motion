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

The expensive part of a pair-period is the Kalman MLE fit, which doesn't
depend on entry/exit thresholds or the z-score lookback. `prepare_pair_periods`
does that work once; `run_portfolio_backtest` can then be re-run cheaply for
other thresholds (learn/tuning.py) or with per-pair size multipliers
(learn/metalabel.py, learn/allocation.py) on the same prepared data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from signals.kalman import KalmanHedgeRatio, KalmanResult
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


@dataclass
class PreparedPair:
    """One pair-period's causal inputs: selection window + active period."""
    pp         : PairPeriod
    label      : str
    series_a   : pd.Series      # prices from train_start through period_end
    series_b   : pd.Series
    train_size : int            # rows before period_start
    kf         : KalmanResult
    size       : pd.Series      # vol-target gross notional / capital
    kalman_params: dict
    _z_cache   : dict = field(default_factory=dict, repr=False)

    def zscores(self, lookback: int = 60, signal: str = "rolling") -> pd.Series:
        key = (signal, lookback)
        if key not in self._z_cache:
            if signal == "rolling":
                self._z_cache[key] = compute_zscore(self.kf.spread, window=lookback)
            elif signal == "innovation":
                self._z_cache[key] = self.kf.innovation_z
            else:
                raise ValueError(f"unknown signal: {signal!r}")
        return self._z_cache[key]

    @property
    def first(self) -> int:
        """Window position the engine starts on: the day before period_start."""
        return max(0, self.train_size - 1)


def pair_label(pp: PairPeriod) -> str:
    return f"{pp.ticker_a}/{pp.ticker_b} [{pp.period_start.date()}]"


def prepare_pair_periods(
    prices          : pd.DataFrame,
    pair_schedule   : list[PairPeriod],
    target_daily_vol: float = 0.01,
    train_window    : int = 252,
) -> list[PreparedPair]:
    """
    Fit each pair-period's Kalman filter (centring, OLS seed, MLE noise
    parameters) on that pair's own selection window — the `train_window`
    days before period_start, the same data walk-forward used to pick it —
    and filter causally through the active period. Also computes the
    vol-target size series.
    """
    prepared = []
    for pp in pair_schedule:
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
        size = vol_target_size(series_a, series_b, kf.hedge_ratio, target_daily_vol=target_daily_vol)

        prepared.append(PreparedPair(
            pp=pp, label=pair_label(pp), series_a=series_a, series_b=series_b,
            train_size=train_size, kf=kf, size=size,
            kalman_params=kalman.fitted_params,
        ))
    return prepared


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
    prepared        : list[PreparedPair] | None = None,
    size_multipliers: dict | None = None,
) -> PortfolioResults:
    """
    Run the full pipeline (Kalman hedge ratio -> z-score -> vol-target sizing
    -> backtest) for every PairPeriod in `pair_schedule`, restricted to each
    period's own active date range, then sum them into one portfolio.

    Nothing here re-introduces look-ahead: each Kalman filter is fit on its
    selection window only (see prepare_pair_periods), and the z-score / vol
    warm-ups are complete by the first tradeable day.

    signal: "rolling"    -> rolling z-score of the spread (lookback_window)
            "innovation" -> Kalman innovation z = e / sqrt(S)

    prepared: output of prepare_pair_periods (same schedule, same vol
        target), to skip re-fitting the Kalman filters.
    size_multipliers: optional {pair label: float | pd.Series}. A Series is
        indexed by DECISION date (the close the engine decides on) and
        multiplies size there; 0 means "don't enter". This is the hook the
        learning layer uses to skip or resize trades.

    PnL is a return on book capital: each pair's daily_pnl is its return on
    the whole book, so the portfolio return is their sum.
    """
    if prepared is None:
        prepared = prepare_pair_periods(prices, pair_schedule, target_daily_vol, train_window)
    size_multipliers = size_multipliers or {}

    per_pair  : dict = {}
    pair_pnl_cols: dict[str, pd.Series] = {}
    kalman_params: dict = {}

    for prep in prepared:
        pp = prep.pp
        zscores = prep.zscores(lookback_window, signal)
        size = prep.size
        mult = size_multipliers.get(prep.label)
        if mult is not None:
            if isinstance(mult, pd.Series):
                mult = mult.reindex(size.index).fillna(1.0)
            size = size * mult

        # Start the backtest one day BEFORE period_start, so day-1 decisions
        # can use the prior close's z / hedge / size (the engine always acts
        # on [t-1]). The engine can't enter on its own first row, so no
        # position is ever opened before period_start.
        active = slice(prep.first, None)

        results = run_backtest(
            prep.series_a.iloc[active], prep.series_b.iloc[active],
            prep.kf.hedge_ratio.iloc[active], zscores.iloc[active],
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_threshold=stop_threshold,
            size=size.iloc[active],
        )

        per_pair[(pp.ticker_a, pp.ticker_b, pp.period_start)] = results
        pair_pnl_cols[prep.label] = results.daily_pnl
        kalman_params[prep.label] = prep.kalman_params

    if not pair_pnl_cols:
        raise ValueError("No pair-periods to backtest — pair_schedule was empty.")

    calendar = prices.loc[min(prep.pp.period_start for prep in prepared):].index
    return combine_pair_results(per_pair, pair_pnl_cols, calendar, kalman_params)


def combine_pair_results(
    per_pair      : dict,
    pair_pnl_cols : dict[str, pd.Series],
    calendar      : pd.DatetimeIndex,
    kalman_params : dict | None = None,
) -> PortfolioResults:
    """Sum per-pair-period daily PnL into one book on a common calendar."""
    # Reindex onto the full calendar so days with nothing active count as
    # flat (0 return) days — otherwise Sharpe/vol only see "in-market" days.
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
        kalman_params = kalman_params or {},
    )
