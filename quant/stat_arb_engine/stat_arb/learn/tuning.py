"""
learn/tuning.py
---------------
Nested walk-forward parameter choice.

The outer loop is the engine's own walk-forward: pick pairs at T, trade
them for a quarter. The inner loop picks (entry, exit, lookback) at each T
from a small grid by the Sharpe each setting achieved over the trailing
`train_days` before T. That Sharpe comes from the grid's own out-of-sample
pair-periods, all of which finished before T, so the choice at T uses
nothing from T onward. The chosen setting is then used for the pair-periods
starting at T.

Every grid point is also a "configuration tried" for the deflated Sharpe
ratio, because a researcher could have just picked the best static one.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from backtest.portfolio import run_portfolio_backtest, combine_pair_results
from backtest.report import sharpe_ratio


GRID = {
    "entry"   : (1.5, 2.0, 2.5),
    "exit"    : (0.0, 0.5),
    "lookback": (40, 60),
}


def grid_keys(grid: dict = GRID) -> list[tuple]:
    return list(product(grid["entry"], grid["exit"], grid["lookback"]))


def key_name(key: tuple) -> str:
    e, x, l = key
    return f"static e={e:g} x={x:g} lb={l}"


def run_grid(prices, schedule, prepared, grid=GRID, stop_threshold=3.5, signal="rolling") -> dict:
    """Static backtest for every grid point on the same prepared pair-periods."""
    out = {}
    for key in grid_keys(grid):
        entry, exit_, lookback = key
        out[key] = run_portfolio_backtest(
            prices, schedule,
            lookback_window=lookback, entry_threshold=entry, exit_threshold=exit_,
            stop_threshold=stop_threshold, signal=signal, prepared=prepared,
        )
    return out


def nested_walk_forward(
    grid_results : dict,
    prepared     : list,
    default_key  : tuple,
    train_days   : int = 252,
    min_days     : int = 126,
):
    """
    Returns (tuned PortfolioResults, {T: chosen key}).

    Before `min_days` of out-of-sample history exist, the default setting is
    used. Ties go to the default.
    """
    periods = sorted({p.pp.period_start for p in prepared})
    calendar = next(iter(grid_results.values())).daily_pnl.index

    chosen = {}
    for T in periods:
        hist_idx = calendar[calendar < T][-train_days:]
        if len(hist_idx) < min_days:
            chosen[T] = default_key
            continue
        scores = {k: sharpe_ratio(r.daily_pnl.loc[hist_idx]) for k, r in grid_results.items()}
        best = max(scores, key=scores.get)
        chosen[T] = best if scores[best] > scores[default_key] + 1e-12 else default_key

    per_pair, cols, kparams = {}, {}, {}
    for prep in prepared:
        pp = prep.pp
        res = grid_results[chosen[pp.period_start]]
        key = (pp.ticker_a, pp.ticker_b, pp.period_start)
        per_pair[key] = res.per_pair[key]
        cols[prep.label] = res.pair_pnl[prep.label]
        kparams[prep.label] = prep.kalman_params
    return combine_pair_results(per_pair, cols, calendar, kparams), chosen
